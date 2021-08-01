package glopl

import (
	"NN-512/internal/compile/author/act"
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/bn"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/cov"
	"NN-512/internal/compile/author/mod"
	"NN-512/internal/compile/author/threader"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
	"fmt"
)

type Ctx struct {
	prefix   string
	platform raw.Platform
	lanes    int
	nms      nmsrc.Src
	tc       *threader.Ctx
	ac       *act.Ctx
	bc       *bn.Ctx
	dedup    map[string]string
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src, tc *threader.Ctx, ac *act.Ctx, bc *bn.Ctx) *Ctx {
	var lanes int
	switch pl.Config.Platform {
	case raw.AVX512Float32:
		lanes = 16
	default:
		panic("bug")
	}
	return &Ctx{
		prefix:   pl.Config.Prefix + "Glopl",
		platform: pl.Config.Platform,
		lanes:    lanes,
		nms:      nms,
		tc:       tc,
		ac:       ac,
		bc:       bc,
		dedup:    make(map[string]string),
	}
}

func (c *Ctx) name(s string) string {
	return c.nms.Name(s)
}

type Spec struct {
	Kind      raw.PoolingKind
	Channels  int
	ElemBytes int
	From      SpecFrom
	To        SpecTo
}

type SpecFrom struct {
	Height      int
	Width       int
	Pitch1Bytes []int
	Pitch2Bytes []int
	Ops         []mod.Op
}

type SpecTo struct {
	Ops []mod.Op
	Cnt int
}

func ceilQuo(n, d int) int {
	return (n + d - 1) / d
}

func vb(s string) cgen.Gen {
	return cgen.Vb(s)
}

func il(i int) cgen.Gen {
	return cgen.IntLit(i)
}

func cast(pitch int) cgen.Gen {
	return cgen.Cast{
		Type: cgen.PtrdiffT,
		Expr: il(pitch),
	}
}

func addr(ptr, pitch, idx cgen.Gen) cgen.Gen {
	return cgen.Add{
		Expr1: ptr,
		Expr2: cgen.Mul{
			Expr1: pitch,
			Expr2: idx,
		},
	}
}

func mix(a ...cgen.Stmts) cgen.Stmts {
	if len(a) == 1 {
		return a[0]
	}
	tot := 0
	for i := range a {
		tot += len(a[i])
	}
	var (
		ret = make(cgen.Stmts, tot)
		n   = 0
	)
	for i := 0; n < tot; i++ {
		for _, aa := range a {
			if i < len(aa) {
				ret[n] = aa[i]
				n++
			}
		}
	}
	return ret
}

type Call struct {
	*Ctx
	*Spec
	Team     cgen.Gen
	Tensors  []cgen.Gen
	funcName string
}

func (c *Call) Prep() cgen.Gen {
	sig := fmt.Sprintf("%v", c.Spec)
	if prior, ok := c.dedup[sig]; ok {
		c.funcName = prior
		return nil
	}
	c.funcName = c.name(c.prefix)
	c.dedup[sig] = c.funcName
	return cgen.Gens{
		&funcDefs{
			Ctx:      c.Ctx,
			Spec:     c.Spec,
			FuncName: c.funcName,
		},
		cgen.Newline,
	}
}

func (c *Call) Append(to []byte) []byte {
	var (
		tensors = vb(c.name("tensors"))
		ptrs    = cgen.CommaLines(c.Tensors)
	)
	return cgen.Stmts{
		cgen.Var{
			Type: cgen.PtrChar,
			What: cgen.Elem{Arr: tensors},
			Init: cgen.Brace{Inner: ptrs},
		},
		cgen.Call{
			Func: vb(c.funcName),
			Args: cgen.CommaSpaced{
				c.Team, tensors,
			},
		},
	}.Append(to)
}

type funcDefs struct {
	*Ctx
	*Spec
	FuncName  string
	unpacked  bool
	chanTile  int
	chanTiles int
	chanScrap int
	funcName  string
	datPtrs   []cgen.Gen
	bnPtrs    []cgen.Gen
	datSplit  int
	bnSplit   int
}

func (f *funcDefs) Append(to []byte) []byte {
	var (
		elemCost   = len(f.From.Pitch1Bytes)
		threadVecs int
	)
	switch f.platform {
	case raw.AVX512Float32:
		threadVecs = 512 / elemCost
		if min := 8; threadVecs < min {
			threadVecs = min
		}
	default:
		panic("bug")
	}
	var (
		width    = f.From.Width
		tight    = width * f.ElemBytes
		chanVecs int
	)
	for _, pitch := range f.From.Pitch1Bytes {
		if pitch != tight {
			f.unpacked = true
			break
		}
	}
	if f.unpacked {
		widthVecs := ceilQuo(width, f.lanes)
		chanVecs = f.From.Height * widthVecs
	} else {
		chanElems := f.From.Height * width
		chanVecs = ceilQuo(chanElems, f.lanes)
	}
	f.chanTile = ceilQuo(threadVecs, chanVecs)
	f.chanTiles = f.Channels / f.chanTile
	f.chanScrap = f.Channels % f.chanTile
	f.funcName = f.name(f.FuncName + "Callee")
	var (
		team     = vb(f.name("team"))
		tensors  = vb(f.name("tensors"))
		chanHull = f.chanTiles
	)
	if f.chanScrap > 0 {
		chanHull++
	}
	return cgen.Gens{
		f.calleeFunc(),
		cgen.Newline,
		cgen.StaticFuncDef{
			ReturnType: cgen.Void,
			Name:       f.FuncName,
			Params: cgen.CommaSpaced{
				cgen.Param{
					Type: f.tc.PtrTeam,
					What: team,
				},
				cgen.Param{
					Type: cgen.PtrPtrChar,
					What: tensors,
				},
			},
			Body: &threader.Do{
				Ctx:    f.tc,
				Callee: vb(f.funcName),
				Any:    tensors,
				Hull: []cgen.Gen{
					il(chanHull),
				},
				Team: team,
			},
		},
	}.Append(to)
}

func (f *funcDefs) calleeFunc() cgen.Gen {
	var (
		body    = make(cgen.Stmts, 5)
		tensors = vb(f.name("tensors"))
		c       = vb(f.name("c"))
	)
	callee := &threader.Callee{
		Ctx:  f.tc,
		Name: f.funcName,
		Task: vb(f.name("task")),
		Pt:   vb(f.name("pt")),
	}
	body[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: tensors,
		Init: callee.Any(),
	}
	body[1] = cgen.Var{
		Type: cgen.PtrdiffT, What: c,
		Init: cgen.Elem{Arr: callee.Pt, Idx: cgen.Zero},
	}
	body[2] = f.ptrs(tensors, c)
	if f.chanTiles > 0 {
		kern := f.kernel(f.chanTile)
		if f.chanScrap > 0 {
			body[3] = cgen.If{
				Cond: cgen.CmpL{
					Expr1: c,
					Expr2: il(f.chanTiles),
				},
				Then: cgen.Stmts{
					kern,
					cgen.Return{},
				},
			}
		} else {
			body[3] = kern
		}
	}
	if f.chanScrap > 0 {
		body[4] = f.kernel(f.chanScrap)
	}
	return callee.Func(body)
}

func (f *funcDefs) ptrs(tensors, c cgen.Gen) cgen.Gen {
	var (
		stmts     cgen.Stmts
		pitch2Idx = 0
		tensorIdx = 0
	)
	pitch2 := func() int {
		i := pitch2Idx
		pitch2Idx++
		if i < len(f.From.Pitch2Bytes) {
			return f.From.Pitch2Bytes[i]
		}
		return f.ElemBytes
	}
	tensor := func() cgen.Gen {
		i := tensorIdx
		tensorIdx++
		return cgen.Elem{
			Arr: tensors, Idx: il(i),
		}
	}
	datPtr := func() {
		var (
			ptr    = vb(f.name("ptr"))
			cPitch = cast(pitch2() * f.chanTile)
			cAddr  = addr(tensor(), cPitch, c)
		)
		f.datPtrs = append(f.datPtrs, ptr)
		stmts = append(stmts, cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptr, Init: cAddr,
		})
	}
	ndp := func(n int) {
		for ; n > 0; n-- {
			datPtr()
		}
	}
	bnPtr := func() {
		ptr := vb(f.name("ptr"))
		f.bnPtrs = append(f.bnPtrs, ptr)
		stmts = append(stmts, cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptr,
			Init: &bn.Offset{
				Ctx: f.bc,
				Mas: tensor(),
				Channel: cgen.Mul{
					Expr1: il(f.chanTile),
					Expr2: c,
				},
			},
		})
	}
	do := func(from bool) {
		var ops []mod.Op
		if from {
			datPtr()
			ops = f.From.Ops
		} else {
			ops = f.To.Ops
		}
		for i := range ops {
			switch op := &ops[i]; op.Kind {
			case mod.Add:
				ndp(op.Int)
			case mod.Bn:
				bnPtr()
			case mod.ReLU:
			default:
				panic("bug")
			}
		}
		if from {
			f.datSplit = len(f.datPtrs)
			f.bnSplit = len(f.bnPtrs)
		} else {
			ndp(f.To.Cnt)
		}
	}
	do(true)
	do(false)
	return stmts
}

func (f *funcDefs) kernel(chans int) cgen.Gen {
	switch f.platform {
	case raw.AVX512Float32:
		return f.m512(chans)
	default:
		panic("bug")
	}
}

func (f *funcDefs) m512(chans int) cgen.Gen {
	if f.unpacked {
		return f.m512Unpacked(chans)
	}
	return f.m512Semipacked(chans)
}

func (f *funcDefs) m512Unpacked(chans int) cgen.Gen {
	const (
		lanes     = 16
		laneBytes = 4
	)
	var (
		unroll = 1
		height = f.From.Height
		width  = f.From.Width
		vecs   = ceilQuo(width, lanes)
	)
	if f.datSplit == 1 {
		unroll = 4
	}
	iUnroll, jUnroll, kUnroll := cov.Box(
		unroll, 4, chans, height, vecs,
	)
	var (
		buf      = vb(f.name("buf"))
		mask     = vb(f.name("mask"))
		bufChans = lanes - lanes%iUnroll
		maskFull = il(1<<uint(bufChans) - 1)
		iIters   = chans / iUnroll
		iAfter   = chans % iUnroll
		bnMuls   = make([][]cgen.Gen, iUnroll)
		bnAdds   = make([][]cgen.Gen, iUnroll)
		jIters   int
		jAfter   int
		kIters   int
		kAfter   int
		accs     []cgen.Gen
		accLanes []int
	)
	leaf := func(i, j, k cgen.Gen, ii, jj, kk, l, a int) cgen.Stmts {
		var (
			acc   = accs[a]
			ldAcc = false
		)
		if acc == nil {
			acc = vb(f.name("acc"))
			ldAcc = true
			accs[a] = acc
			accLanes[a] = l
		}
		pull := &m512Pull{
			Ctx:    f.Ctx,
			Spec:   f.Spec,
			Lanes:  l,
			Ptrs:   make([]cgen.Gen, f.datSplit),
			BnMuls: bnMuls[ii],
			BnAdds: bnAdds[ii],
			Acc:    acc,
			LdAcc:  ldAcc,
		}
		for x, ptr := range f.datPtrs[:f.datSplit] {
			var (
				iiPitch = f.From.Pitch2Bytes[x]
				jjPitch = f.From.Pitch1Bytes[x]
				kkPitch = lanes * laneBytes
				iPitch  = iiPitch * iUnroll
				jPitch  = jjPitch * jUnroll
				kPitch  = kkPitch * kUnroll
			)
			ptr = cgen.Add{
				Expr1: ptr,
				Expr2: cast(iiPitch*ii + jjPitch*jj + kkPitch*kk),
			}
			if iIters > 0 {
				ptr = addr(ptr, cast(iPitch), i)
			}
			if jIters > 0 {
				ptr = addr(ptr, cast(jPitch), j)
			}
			if kIters > 0 {
				ptr = addr(ptr, cast(kPitch), k)
			}
			pull.Ptrs[x] = ptr
		}
		return pull.Stmts()
	}
	layer5 := func(i, j cgen.Gen, iCnt, jCnt int, first bool) cgen.Gen {
		stmts := make(cgen.Stmts, 3)
		if kIters > 0 {
			peeled := 0
			if first {
				peeled = 1
				var (
					peel = make([]cgen.Stmts, len(accs))
					k    = cgen.Zero
				)
				for ii := 0; ii < iCnt; ii++ {
					a := ii * jUnroll * kUnroll
					for jj := 0; jj < jCnt; jj++ {
						for kk := 0; kk < kUnroll; kk++ {
							peel[a] = leaf(i, j, k, ii, jj, kk, lanes, a)
							a++
						}
					}
				}
				stmts[0] = mix(peel...)
			}
			if peeled < kIters {
				var (
					body = make([]cgen.Stmts, len(accs))
					k    = vb(f.name("k"))
				)
				for ii := 0; ii < iCnt; ii++ {
					a := ii * jUnroll * kUnroll
					for jj := 0; jj < jCnt; jj++ {
						for kk := 0; kk < kUnroll; kk++ {
							body[a] = leaf(i, j, k, ii, jj, kk, lanes, a)
							a++
						}
					}
				}
				stmts[1] = cgen.For{
					Init: cgen.Var{
						Type: cgen.PtrdiffT, What: k,
						Init: il(peeled),
					},
					Cond: cgen.CmpL{
						Expr1: k, Expr2: il(kIters),
					},
					Post: cgen.IncPre{Expr: k},
					Body: mix(body...),
				}
			}
		}
		if kAfter > 0 {
			var (
				full = kAfter / lanes
				part = kAfter % lanes
				tail = make([]cgen.Stmts, len(accs))
				k    = il(kIters)
			)
			for ii := 0; ii < iCnt; ii++ {
				for jj := 0; jj < jCnt; jj++ {
					a := (ii*jUnroll + jj) * kUnroll
					for kk := 0; kk <= full; kk++ {
						l := lanes
						if kk == full {
							if l = part; l == 0 {
								break
							}
						}
						tail[a] = leaf(i, j, k, ii, jj, kk, l, a)
						a++
					}
				}
			}
			stmts[2] = mix(tail...)
		}
		return stmts
	}
	layer4 := func(i cgen.Gen, iCnt int) cgen.Gen {
		stmts := make(cgen.Stmts, 3)
		if jIters > 0 {
			j := cgen.Zero
			stmts[0] = layer5(i, j, iCnt, jUnroll, true)
			if jIters > 1 {
				j := vb(f.name("j"))
				stmts[1] = cgen.For{
					Init: cgen.Var{
						Type: cgen.PtrdiffT, What: j,
						Init: cgen.One,
					},
					Cond: cgen.CmpL{
						Expr1: j, Expr2: il(jIters),
					},
					Post: cgen.IncPre{Expr: j},
					Body: layer5(i, j, iCnt, jUnroll, false),
				}
			}
		}
		if jAfter > 0 {
			var (
				j     = il(jIters)
				first = jIters == 0
			)
			stmts[2] = layer5(i, j, iCnt, jAfter, first)
		}
		return stmts
	}
	layer3 := func(i cgen.Gen, iCnt int) cgen.Gen {
		jIters = height / jUnroll
		jAfter = height % jUnroll
		kIters = width / (kUnroll * lanes)
		kAfter = width % (kUnroll * lanes)
		accs = make([]cgen.Gen, iCnt*jUnroll*kUnroll)
		accLanes = make([]int, len(accs))
		var (
			stmts = layer4(i, iCnt)
			n     = 0
		)
		for x, acc := range accs {
			if acc != nil {
				accs[n] = acc
				accLanes[n] = accLanes[x]
				n++
			}
		}
		accs = accs[:n]
		accLanes = accLanes[:n]
		return stmts
	}
	layer2 := func(i cgen.Gen, iCnt int) (stmts cgen.Stmts) {
		stmts = make(cgen.Stmts, 5)
		stmts[0] = layer3(i, iCnt)
		fold := &m512Fold{
			Ctx:   f.Ctx,
			Spec:  f.Spec,
			Chans: iCnt,
			Frame: iUnroll,
			Accs:  accs,
			Lanes: accLanes,
		}
		var folded cgen.Gen
		stmts[1], folded = fold.Gens()
		stmts[2] = cgen.Assign{
			Expr1: buf,
			Expr2: avx.Mm512MaskMovPs{
				buf, mask, folded,
			},
		}
		if iCnt != iUnroll {
			return
		}
		stmts[3] = cgen.AndAssign{
			Expr1: mask,
			Expr2: cgen.ShiftHigh{
				Expr1: mask,
				Expr2: il(iUnroll),
			},
		}
		if chans < bufChans {
			return
		}
		ch := cgen.Paren{Inner: cgen.Sub{
			Expr1: cgen.Mul{
				Expr1: cast(iUnroll),
				Expr2: i,
			},
			Expr2: il(bufChans - iUnroll),
		}}
		stmts[4] = cgen.If{
			Cond: cgen.Unlikely{
				Cond: cgen.IsZero{Expr: mask},
			},
			Then: cgen.Stmts{
				cgen.Assign{
					Expr1: mask,
					Expr2: maskFull,
				},
				&m512Push{
					Ctx:     f.Ctx,
					Spec:    f.Spec,
					DatPtrs: f.datPtrs[f.datSplit:],
					BnPtrs:  f.bnPtrs[f.bnSplit:],
					Buf:     buf,
					Chan:    ch,
					ChanCnt: bufChans,
				},
			},
		}
		return
	}
	layer1 := func(i cgen.Gen, iCnt int) cgen.Gen {
		var (
			bnLds = make([]cgen.Stmts, iCnt)
			bnCnt = f.bnSplit
		)
		for ii := 0; ii < iCnt; ii++ {
			var (
				ch   = il(ii)
				muls = make([]cgen.Gen, bnCnt)
				adds = make([]cgen.Gen, bnCnt)
				lds  = make(cgen.Stmts, bnCnt)
			)
			if iIters > 0 {
				ch = cgen.Paren{Inner: cgen.Add{
					Expr1: ch,
					Expr2: cgen.Mul{
						Expr1: cast(iUnroll),
						Expr2: i,
					},
				}}
			}
			for x, ptr := range f.bnPtrs[:bnCnt] {
				var (
					bnMul = vb(f.name("bnMul"))
					bnAdd = vb(f.name("bnAdd"))
				)
				muls[x] = bnMul
				adds[x] = bnAdd
				lds[x] = &bn.Load{
					Ctx:     f.bc,
					Mas:     ptr,
					Channel: ch,
					Mul:     bnMul,
					Add:     bnAdd,
				}
			}
			bnMuls[ii] = muls
			bnAdds[ii] = adds
			bnLds[ii] = lds
		}
		return cgen.Gens{
			mix(bnLds...),
			layer2(i, iCnt),
		}
	}
	stmts := make(cgen.Stmts, 5)
	stmts[0] = cgen.Var{
		Type: avx.M512, What: buf,
		Init: avx.Mm512SetzeroPs,
	}
	stmts[1] = cgen.Var{
		Type: avx.Mmask16, What: mask,
		Init: maskFull,
	}
	if iIters > 0 {
		i := vb(f.name("i"))
		stmts[2] = cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT, What: i,
				Init: cgen.Zero,
			},
			Cond: cgen.CmpL{
				Expr1: i, Expr2: il(iIters),
			},
			Post: cgen.IncPre{Expr: i},
			Body: layer1(i, iUnroll),
		}
	}
	if iAfter > 0 {
		var (
			i        = il(iIters)
			jkUnroll = unroll / iAfter
		)
		jUnroll, kUnroll = cov.Rect(
			jkUnroll, jkUnroll, height, vecs,
		)
		stmts[3] = layer1(i, iAfter)
	}
	if rem := chans % bufChans; rem > 0 {
		stmts[4] = &m512Push{
			Ctx:     f.Ctx,
			Spec:    f.Spec,
			DatPtrs: f.datPtrs[f.datSplit:],
			BnPtrs:  f.bnPtrs[f.bnSplit:],
			Buf:     buf,
			Chan:    il(chans - rem),
			ChanCnt: rem,
		}
	}
	return stmts
}

func (f *funcDefs) m512Semipacked(chans int) cgen.Gen {
	const (
		lanes     = 16
		laneBytes = 4
	)
	var (
		unroll = 1
		elems  = f.From.Height * f.From.Width
		vecs   = ceilQuo(elems, lanes)
	)
	switch f.datSplit {
	case 1:
		unroll = 8
	case 2:
		unroll = 2
	}
	iUnroll, _ := cov.Rect(
		unroll, 4, chans, vecs,
	)
	var (
		buf      = vb(f.name("buf"))
		mask     = vb(f.name("mask"))
		bufChans = lanes - lanes%iUnroll
		maskFull = il(1<<uint(bufChans) - 1)
		iIters   = chans / iUnroll
		iAfter   = chans % iUnroll
		bnMuls   = make([][]cgen.Gen, iUnroll)
		bnAdds   = make([][]cgen.Gen, iUnroll)
		jUnroll  int
		jIters   int
		accs     []cgen.Gen
		accLanes []int
	)
	leaf := func(i, j cgen.Gen, ii, jj, l, a int) cgen.Stmts {
		var (
			acc   = accs[a]
			ldAcc = false
		)
		if acc == nil {
			acc = vb(f.name("acc"))
			ldAcc = true
			accs[a] = acc
			accLanes[a] = l
		}
		pull := &m512Pull{
			Ctx:    f.Ctx,
			Spec:   f.Spec,
			Lanes:  l,
			Ptrs:   make([]cgen.Gen, f.datSplit),
			BnMuls: bnMuls[ii],
			BnAdds: bnAdds[ii],
			Acc:    acc,
			LdAcc:  ldAcc,
		}
		for x, ptr := range f.datPtrs[:f.datSplit] {
			var (
				iiPitch = f.From.Pitch2Bytes[x]
				jjPitch = lanes * laneBytes
				iPitch  = iiPitch * iUnroll
				jPitch  = jjPitch * jUnroll
			)
			ptr = cgen.Add{
				Expr1: ptr,
				Expr2: cast(iiPitch*ii + jjPitch*jj),
			}
			if iIters > 0 {
				ptr = addr(ptr, cast(iPitch), i)
			}
			if jIters > 0 {
				ptr = addr(ptr, cast(jPitch), j)
			}
			pull.Ptrs[x] = ptr
		}
		return pull.Stmts()
	}
	layer3 := func(i cgen.Gen, iCnt int) cgen.Gen {
		stmts := make(cgen.Stmts, 3)
		jUnroll = unroll / iCnt
		jIter := jUnroll * lanes
		jIters = elems / jIter
		jAfter := elems % jIter
		if jIters > 0 {
			n := iCnt * jUnroll
			accs = make([]cgen.Gen, n)
			accLanes = make([]int, n)
			var (
				peel = make([]cgen.Stmts, n)
				j    = cgen.Zero
			)
			for a, ii := 0, 0; ii < iCnt; ii++ {
				for jj := 0; jj < jUnroll; jj++ {
					peel[a] = leaf(i, j, ii, jj, lanes, a)
					a++
				}
			}
			stmts[0] = mix(peel...)
			if jIters > 1 {
				var (
					body = make([]cgen.Stmts, n)
					j    = vb(f.name("j"))
				)
				for a, ii := 0, 0; ii < iCnt; ii++ {
					for jj := 0; jj < jUnroll; jj++ {
						body[a] = leaf(i, j, ii, jj, lanes, a)
						a++
					}
				}
				stmts[1] = cgen.For{
					Init: cgen.Var{
						Type: cgen.PtrdiffT, What: j,
						Init: cgen.One,
					},
					Cond: cgen.CmpL{
						Expr1: j, Expr2: il(jIters),
					},
					Post: cgen.IncPre{Expr: j},
					Body: mix(body...),
				}
			}
		}
		if jAfter > 0 {
			var (
				full = jAfter / lanes
				part = jAfter % lanes
				jCnt = jUnroll
			)
			if jIters == 0 {
				if jCnt = full; part > 0 {
					jCnt++
				}
				accs = make([]cgen.Gen, iCnt*jCnt)
				accLanes = make([]int, len(accs))
			}
			var (
				tail = make([]cgen.Stmts, len(accs))
				j    = il(jIters)
			)
			for ii := 0; ii < iCnt; ii++ {
				a := ii * jCnt
				for jj := 0; jj <= full; jj++ {
					l := lanes
					if jj == full {
						if l = part; l == 0 {
							break
						}
					}
					tail[a] = leaf(i, j, ii, jj, l, a)
					a++
				}
			}
			stmts[2] = mix(tail...)
		}
		return stmts
	}
	layer2 := func(i cgen.Gen, iCnt int) (stmts cgen.Stmts) {
		stmts = make(cgen.Stmts, 5)
		stmts[0] = layer3(i, iCnt)
		fold := &m512Fold{
			Ctx:   f.Ctx,
			Spec:  f.Spec,
			Chans: iCnt,
			Frame: iUnroll,
			Accs:  accs,
			Lanes: accLanes,
		}
		var folded cgen.Gen
		stmts[1], folded = fold.Gens()
		stmts[2] = cgen.Assign{
			Expr1: buf,
			Expr2: avx.Mm512MaskMovPs{
				buf, mask, folded,
			},
		}
		if iCnt != iUnroll {
			return
		}
		stmts[3] = cgen.AndAssign{
			Expr1: mask,
			Expr2: cgen.ShiftHigh{
				Expr1: mask,
				Expr2: il(iUnroll),
			},
		}
		if chans < bufChans {
			return
		}
		ch := cgen.Paren{Inner: cgen.Sub{
			Expr1: cgen.Mul{
				Expr1: cast(iUnroll),
				Expr2: i,
			},
			Expr2: il(bufChans - iUnroll),
		}}
		stmts[4] = cgen.If{
			Cond: cgen.Unlikely{
				Cond: cgen.IsZero{Expr: mask},
			},
			Then: cgen.Stmts{
				cgen.Assign{
					Expr1: mask,
					Expr2: maskFull,
				},
				&m512Push{
					Ctx:     f.Ctx,
					Spec:    f.Spec,
					DatPtrs: f.datPtrs[f.datSplit:],
					BnPtrs:  f.bnPtrs[f.bnSplit:],
					Buf:     buf,
					Chan:    ch,
					ChanCnt: bufChans,
				},
			},
		}
		return
	}
	layer1 := func(i cgen.Gen, iCnt int) cgen.Gen {
		var (
			bnLds = make([]cgen.Stmts, iCnt)
			bnCnt = f.bnSplit
		)
		for ii := 0; ii < iCnt; ii++ {
			var (
				ch   = il(ii)
				muls = make([]cgen.Gen, bnCnt)
				adds = make([]cgen.Gen, bnCnt)
				lds  = make(cgen.Stmts, bnCnt)
			)
			if iIters > 0 {
				ch = cgen.Paren{Inner: cgen.Add{
					Expr1: ch,
					Expr2: cgen.Mul{
						Expr1: cast(iUnroll),
						Expr2: i,
					},
				}}
			}
			for x, ptr := range f.bnPtrs[:bnCnt] {
				var (
					bnMul = vb(f.name("bnMul"))
					bnAdd = vb(f.name("bnAdd"))
				)
				muls[x] = bnMul
				adds[x] = bnAdd
				lds[x] = &bn.Load{
					Ctx:     f.bc,
					Mas:     ptr,
					Channel: ch,
					Mul:     bnMul,
					Add:     bnAdd,
				}
			}
			bnMuls[ii] = muls
			bnAdds[ii] = adds
			bnLds[ii] = lds
		}
		return cgen.Gens{
			mix(bnLds...),
			layer2(i, iCnt),
		}
	}
	stmts := make(cgen.Stmts, 5)
	stmts[0] = cgen.Var{
		Type: avx.M512, What: buf,
		Init: avx.Mm512SetzeroPs,
	}
	stmts[1] = cgen.Var{
		Type: avx.Mmask16, What: mask,
		Init: maskFull,
	}
	if iIters > 0 {
		i := vb(f.name("i"))
		stmts[2] = cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT, What: i,
				Init: cgen.Zero,
			},
			Cond: cgen.CmpL{
				Expr1: i, Expr2: il(iIters),
			},
			Post: cgen.IncPre{Expr: i},
			Body: layer1(i, iUnroll),
		}
	}
	if iAfter > 0 {
		i := il(iIters)
		stmts[3] = layer1(i, iAfter)
	}
	if rem := chans % bufChans; rem > 0 {
		stmts[4] = &m512Push{
			Ctx:     f.Ctx,
			Spec:    f.Spec,
			DatPtrs: f.datPtrs[f.datSplit:],
			BnPtrs:  f.bnPtrs[f.bnSplit:],
			Buf:     buf,
			Chan:    il(chans - rem),
			ChanCnt: rem,
		}
	}
	return stmts
}

type m512Pull struct {
	*Ctx
	*Spec
	Lanes  int
	Ptrs   []cgen.Gen
	BnMuls []cgen.Gen
	BnAdds []cgen.Gen
	Acc    cgen.Gen
	LdAcc  bool
	mask   cgen.Gen
	ptrIdx int
	bnIdx  int
	lds    cgen.Stmts
	nonlds cgen.Stmts
}

func (m *m512Pull) ld() (dat cgen.Gen) {
	if m.ptrIdx == 0 && m.LdAcc {
		dat = m.Acc
	} else {
		dat = vb(m.name("dat"))
	}
	m.lds = append(m.lds, cgen.Var{
		Type: avx.M512, What: dat,
		Init: avx.Mm512MaskzLoaduPs{
			m.mask,
			m.Ptrs[m.ptrIdx],
		},
	})
	m.ptrIdx++
	return
}

func (m *m512Pull) nonld(a cgen.Gen) {
	m.nonlds = append(m.nonlds, a)
}

func (m *m512Pull) adder(a ...cgen.Gen) {
	for n := len(a); n > 1; {
		fold := n >> 1
		n -= fold
		for i := 0; i < fold; i++ {
			to := a[i]
			m.nonld(cgen.Assign{
				Expr1: to,
				Expr2: avx.Mm512MaskAddPs{
					to, m.mask,
					to, a[n+i],
				},
			})
		}
	}
}

func (m *m512Pull) apply(dat cgen.Gen, ops []mod.Op) bool {
	last := len(ops) - 1
	for i := range ops {
		switch op := &ops[i]; op.Kind {
		case mod.Add:
			n := op.Int
			if i == last &&
				!m.LdAcc && m.Kind == raw.AvgGlobal {
				dats := make([]cgen.Gen, 2+n)
				dats[0] = m.Acc
				dats[1] = dat
				for j := 0; j < n; j++ {
					dats[2+j] = m.ld()
				}
				m.adder(dats...)
				return true
			}
			dats := make([]cgen.Gen, 1+n)
			dats[0] = dat
			for j := 1; j <= n; j++ {
				dats[j] = m.ld()
			}
			m.adder(dats...)
		case mod.Bn:
			j := m.bnIdx
			m.bnIdx++
			m.nonld(&bn.Apply{
				Ctx: m.bc,
				Mul: m.BnMuls[j],
				Add: m.BnAdds[j],
				To:  dat,
			})
		case mod.ReLU:
			ns := op.Float
			if i == last && ns == 0 &&
				!m.LdAcc && m.Kind == raw.MaxGlobal {
				return false
			}
			m.nonld(&act.ReLU{
				Ctx:      m.ac,
				NegSlope: ns,
				Var:      dat,
			})
		default:
			panic("bug")
		}
	}
	return m.LdAcc
}

func (m *m512Pull) Stmts() cgen.Stmts {
	m.mask = il(1<<uint(m.Lanes) - 1)
	var (
		dat  = m.ld()
		done = m.apply(dat, m.From.Ops)
	)
	if !done {
		switch m.Kind {
		case raw.AvgGlobal:
			m.adder(m.Acc, dat)
		case raw.MaxGlobal:
			m.nonld(cgen.Assign{
				Expr1: m.Acc,
				Expr2: avx.Mm512MaskMaxPs{
					m.Acc, m.mask,
					m.Acc, dat,
				},
			})
		default:
			panic("bug")
		}
	}
	return append(
		m.lds, m.nonlds...,
	)
}

type m512Fold struct {
	*Ctx
	*Spec
	Chans int
	Frame int
	Accs  []cgen.Gen
	Lanes []int
	pm1lo cgen.Gen
	pm1hi cgen.Gen
	pm4lo cgen.Gen
	pm4hi cgen.Gen
}

func (m *m512Fold) combine(a ...cgen.Gen) cgen.Gen {
	switch m.Kind {
	case raw.AvgGlobal:
		return avx.Mm512MaskAddPs(a)
	case raw.MaxGlobal:
		return avx.Mm512MaskMaxPs(a)
	default:
		panic("bug")
	}
}

func (m *m512Fold) chanwise() cgen.Stmts {
	var (
		n     = m.Chans
		each  = len(m.Accs) / n
		stmts = make([]cgen.Stmts, n)
	)
	for i := 0; i < n; i++ {
		j := i * each
		for cnt := each; cnt > 1; {
			fold := cnt >> 1
			cnt -= fold
			for k := j; k < j+fold; k++ {
				var (
					a1, a2 = &m.Accs[k], &m.Accs[k+cnt]
					l1, l2 = &m.Lanes[k], &m.Lanes[k+cnt]
				)
				if *l1 < *l2 {
					*a1, *a2 = *a2, *a1
					*l1, *l2 = *l2, *l1
				}
				stmts[i] = append(stmts[i], cgen.Assign{
					Expr1: *a1,
					Expr2: m.combine(
						*a1, il(1<<uint(*l2)-1),
						*a1, *a2,
					),
				})
			}
		}
		m.Accs[i] = m.Accs[j]
		m.Lanes[i] = m.Lanes[j]
	}
	return mix(stmts...)
}

func (m *m512Fold) funnel(xs []int, fit int) (stmts cgen.Stmts) {
	n := len(xs)
	if n > 1 {
		var (
			n2  = n >> 1
			n1  = n - n2
			xs1 = make([]int, n1)
			xs2 = make([]int, n2)
		)
		for i, x := range xs {
			if ii := i >> 1; i&1 == 0 {
				xs1[ii] = x
			} else {
				xs2[ii] = x
			}
		}
		stmts = mix(
			m.funnel(xs1, fit*2),
			m.funnel(xs2, fit*2),
		)
	} else {
		if fit > 1 {
			if m.Lanes[xs[0]] <= fit {
				return
			}
		}
		stmts = m.funnel(xs, fit*2)
	}
	var (
		acc1 = m.Accs[xs[0]]
		acc2 = acc1
	)
	if n > 1 {
		acc2 = m.Accs[xs[1]]
	}
	permute := func(pm cgen.Gen) cgen.Gen {
		if n > 1 {
			return avx.Mm512Permutex2varPs{
				acc1, pm, acc2,
			}
		}
		return avx.Mm512PermutexvarPs{
			pm, acc1,
		}
	}
	inner := func(ctrl int) cgen.Gen {
		return avx.Mm512ShufflePs{
			acc1, acc2, il(ctrl),
		}
	}
	outer := func(ctrl int) cgen.Gen {
		return avx.Mm512ShuffleF32x4{
			acc1, acc2, il(ctrl),
		}
	}
	var hi cgen.Gen
	for _, x := range xs {
		if m.Lanes[x] > fit {
			hi = vb(m.name("hi"))
			break
		}
	}
	if hi != nil {
		var call cgen.Gen
		switch fit {
		case 1:
			m.pm1hi = vb(m.name("pm1hi"))
			call = permute(m.pm1hi)
		case 2:
			call = inner(0xee)
		case 4:
			if n > 1 {
				if m.pm4hi == nil {
					m.pm4hi = vb(m.name("pm4hi"))
				}
				call = permute(m.pm4hi)
			} else {
				call = outer(0x01)
			}
		case 8:
			call = outer(0xee)
		}
		stmts = append(stmts, cgen.Var{
			Type: avx.M512, What: hi,
			Init: call,
		})
	}
	if n > 1 || fit == 1 {
		var call cgen.Gen
		switch fit {
		case 1:
			m.pm1lo = vb(m.name("pm1lo"))
			call = permute(m.pm1lo)
		case 2:
			call = inner(0x44)
		case 4:
			if m.pm4lo == nil {
				m.pm4lo = vb(m.name("pm4lo"))
			}
			call = permute(m.pm4lo)
		case 8:
			call = outer(0x44)
		}
		stmts = append(stmts, cgen.Assign{
			Expr1: acc1, Expr2: call,
		})
	}
	if hi != nil {
		mask := 0
		for i := n - 1; i >= 0; i-- {
			mask <<= uint(fit)
			if l := &m.Lanes[xs[i]]; *l > fit {
				mask |= 1<<uint(*l-fit) - 1
				*l = fit
			}
		}
		if fit == 1 {
			for bits := m.Frame; bits < 16; {
				mask |= mask << uint(bits)
				bits *= 2
			}
			mask &= 0xffff
		}
		stmts = append(stmts, cgen.Assign{
			Expr1: acc1,
			Expr2: m.combine(
				acc1, il(mask),
				acc1, hi,
			),
		})
	}
	return
}

func (m *m512Fold) pms() cgen.Stmts {
	const lanes = 16
	stmts := make(cgen.Stmts, 0, 4)
	decl := func(pm cgen.Gen, fn func(int) int) {
		if pm == nil {
			return
		}
		set := make(avx.Mm512SetEpi32, lanes)
		for i := 0; i < lanes; i++ {
			set[lanes-1-i] = il(fn(i))
		}
		stmts = append(stmts, cgen.Var{
			Type: avx.M512i, What: pm,
			Init: set,
		})
	}
	decl(m.pm1lo, func(i int) int {
		i %= m.Frame
		return i&-2 + i&1*lanes
	})
	decl(m.pm1hi, func(i int) int {
		i %= m.Frame
		return i | 1 + i&1*lanes
	})
	decl(m.pm4lo, func(i int) int {
		return i&^4 + i&4*(lanes/4)
	})
	decl(m.pm4hi, func(i int) int {
		return i | 4 + i&4*(lanes/4)
	})
	return stmts
}

func (m *m512Fold) Gens() (stmts, acc cgen.Gen) {
	var (
		stmts1 = m.chanwise()
		xs     = make([]int, m.Chans)
	)
	for i := range xs {
		xs[i] = i
	}
	var (
		stmts3 = m.funnel(xs, 1)
		stmts2 = m.pms()
	)
	stmts = cgen.Gens{
		stmts1, stmts2, stmts3,
	}
	acc = m.Accs[0]
	return
}

type m512Push struct {
	*Ctx
	*Spec
	DatPtrs []cgen.Gen
	BnPtrs  []cgen.Gen
	Buf     cgen.Gen
	Chan    cgen.Gen
	ChanCnt int
}

func (m *m512Push) Append(to []byte) []byte {
	var (
		lds    cgen.Stmts
		nonlds cgen.Stmts
		pitch  = cast(m.ElemBytes)
		mask   = il(1<<uint(m.ChanCnt) - 1)
	)
	ld := func(a cgen.Gen) {
		lds = append(lds, a)
	}
	nonld := func(a cgen.Gen) {
		nonlds = append(nonlds, a)
	}
	datPtr := func() cgen.Gen {
		ptr := m.DatPtrs[0]
		m.DatPtrs = m.DatPtrs[1:]
		return addr(ptr, pitch, m.Chan)
	}
	if m.Kind == raw.AvgGlobal {
		var (
			hw  = m.From.Height * m.From.Width
			rcp = 1 / avx.Mm512Set1PsLit(hw)
		)
		nonld(cgen.Assign{
			Expr1: m.Buf,
			Expr2: avx.Mm512MulPs{m.Buf, rcp},
		})
	}
	for i := range m.To.Ops {
		op := &m.To.Ops[i]
		switch op.Kind {
		case mod.Add:
			var (
				n    = 1 + op.Int
				dats = make([]cgen.Gen, n)
			)
			dats[0] = m.Buf
			for j := 1; j < n; j++ {
				dats[j] = vb(m.name("dat"))
				ld(cgen.Var{
					Type: avx.M512, What: dats[j],
					Init: avx.Mm512MaskzLoaduPs{
						mask, datPtr(),
					},
				})
			}
			for n > 1 {
				fold := n >> 1
				n -= fold
				for j := 0; j < fold; j++ {
					keep := dats[n-1-j]
					nonld(cgen.Assign{
						Expr1: keep,
						Expr2: avx.Mm512AddPs{
							keep, dats[n+j],
						},
					})
				}
			}
		case mod.Bn:
			var (
				bnMul = vb(m.name("bnMul"))
				bnAdd = vb(m.name("bnAdd"))
			)
			ld(&bn.Load{
				Ctx:     m.bc,
				Mas:     m.BnPtrs[0],
				Channel: m.Chan,
				Mul:     bnMul,
				Add:     bnAdd,
				Cnt:     m.ChanCnt,
			})
			m.BnPtrs = m.BnPtrs[1:]
			nonld(&bn.Apply{
				Ctx: m.bc,
				Mul: bnMul,
				Add: bnAdd,
				To:  m.Buf,
			})
		case mod.ReLU:
			nonld(&act.ReLU{
				Ctx:      m.ac,
				NegSlope: op.Float,
				Var:      m.Buf,
			})
		default:
			panic("bug")
		}
	}
	for n := m.To.Cnt; n > 0; n-- {
		nonld(avx.Mm512MaskStoreuPs{
			datPtr(), mask, m.Buf,
		})
	}
	to = lds.Append(to)
	to = nonlds.Append(to)
	return to
}
