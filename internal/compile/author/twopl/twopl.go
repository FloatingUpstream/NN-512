package twopl

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
		prefix:   pl.Config.Prefix + "Twopl",
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
	Kind     raw.PoolingKind
	PaddingH int
	PaddingW int
	Channels int
	From     SpecFrom
	To       SpecTo
}

type SpecFrom struct {
	Height      int
	Width       int
	Pitch1Bytes []int
	Pitch2Bytes []int
	Ops         []mod.Op
}

type SpecTo struct {
	Pitch1Bytes []int
	Pitch2Bytes []int
	Ops         []mod.Op
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

func mask(n int) cgen.Gen {
	return il(1<<uint(n) - 1)
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

func mix(a []cgen.Stmts) cgen.Stmts {
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
	bandTile  int
	bandTiles int
	bandScrap int
	bandHull  int
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
		costFrom   = len(f.From.Pitch1Bytes)
		costTo     = len(f.To.Pitch1Bytes)
		cost       = costFrom*4 + costTo
		threadVecs int
	)
	switch f.platform {
	case raw.AVX512Float32:
		threadVecs = ceilQuo(5<<9, cost)
	default:
		panic("bug")
	}
	var (
		widthVecs = ceilQuo(f.From.Width, f.lanes)
		bandVecs  = widthVecs * 2
		chanBands = (f.From.Height + f.PaddingH*2) / 2
		chanVecs  = chanBands * bandVecs
	)
	if threadVecs < chanVecs {
		var (
			bands = ceilQuo(threadVecs, bandVecs)
			fit   = chanBands / bands
		)
		bands = chanBands / fit
		f.bandTile = bands
		f.bandTiles = fit
		f.bandScrap = chanBands - bands*fit
		if f.bandScrap > 0 {
			f.bandTiles--
			f.bandScrap += bands
		}
		f.bandHull = fit
		f.chanTile = 1
		f.chanTiles = f.Channels
		f.chanScrap = 0
	} else {
		chans := ceilQuo(threadVecs, chanVecs)
		f.bandTile = chanBands
		f.bandTiles = 1
		f.bandScrap = 0
		f.bandHull = 1
		f.chanTile = chans
		f.chanTiles = f.Channels / chans
		f.chanScrap = f.Channels % chans
	}
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
					il(f.bandHull),
					il(chanHull),
				},
				Team: team,
			},
		},
	}.Append(to)
}

func (f *funcDefs) calleeFunc() cgen.Gen {
	var (
		body    = make(cgen.Stmts, 6)
		tensors = vb(f.name("tensors"))
		b       = vb(f.name("b"))
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
		Type: cgen.PtrdiffT, What: b,
		Init: cgen.Elem{Arr: callee.Pt, Idx: cgen.Zero},
	}
	body[2] = cgen.Var{
		Type: cgen.PtrdiffT, What: c,
		Init: cgen.Elem{Arr: callee.Pt, Idx: cgen.One},
	}
	body[3] = f.ptrs(tensors, b, c)
	inner := func(chans, bands, cases int, lo, hi bool) cgen.Gen {
		if f.PaddingH == 0 {
			return f.kernel(chans, bands, false, false)
		}
		hiPadH := f.From.Height%2 == 0
		if f.bandHull == 1 {
			return f.kernel(chans, bands, true, hiPadH)
		}
		stmts := make(cgen.Stmts, 3)
		if lo {
			stmts[0] = f.kernel(chans, bands, true, false)
			if cases--; cases == 0 {
				return stmts
			}
			stmts[0] = cgen.If{
				Cond: cgen.IsZero{Expr: b},
				Then: cgen.Stmts{
					stmts[0],
					cgen.Return{},
				},
			}
		}
		if hi && hiPadH {
			stmts[1] = f.kernel(chans, bands, false, true)
			if cases--; cases == 0 {
				return stmts
			}
			stmts[1] = cgen.If{
				Cond: cgen.CmpE{
					Expr1: b,
					Expr2: il(f.bandHull - 1),
				},
				Then: cgen.Stmts{
					stmts[1],
					cgen.Return{},
				},
			}
		}
		stmts[2] = f.kernel(chans, bands, false, false)
		return stmts
	}
	outer := func(chans int) cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if f.bandTiles > 0 {
			if f.bandScrap > 0 {
				stmts[0] = cgen.If{
					Cond: cgen.CmpL{
						Expr1: b,
						Expr2: il(f.bandTiles),
					},
					Then: cgen.Stmts{
						inner(chans, f.bandTile, f.bandTiles, true, false),
						cgen.Return{},
					},
				}
			} else {
				stmts[0] = inner(chans, f.bandTile, f.bandTiles, true, true)
			}
		}
		if f.bandScrap > 0 {
			lo := f.bandTiles == 0
			stmts[1] = inner(chans, f.bandScrap, 1, lo, true)
		}
		return stmts
	}
	if f.chanTiles > 0 {
		body[4] = outer(f.chanTile)
		if f.chanScrap > 0 {
			body[4] = cgen.If{
				Cond: cgen.CmpL{
					Expr1: c,
					Expr2: il(f.chanTiles),
				},
				Then: cgen.Stmts{
					body[4],
					cgen.Return{},
				},
			}
		}
	}
	if f.chanScrap > 0 {
		body[5] = outer(f.chanScrap)
	}
	return callee.Func(body)
}

func (f *funcDefs) ptrs(tensors, b, c cgen.Gen) cgen.Gen {
	var (
		stmts     cgen.Stmts
		tensorIdx = 0
		datPtrIdx = 0
	)
	tensor := func() cgen.Gen {
		i := tensorIdx
		tensorIdx++
		return cgen.Elem{
			Arr: tensors,
			Idx: il(i),
		}
	}
	datPtr := func() {
		var (
			ptr    = vb(f.name("ptr"))
			expr   = tensor()
			i      = datPtrIdx
			n      = len(f.From.Pitch1Bytes)
			bPitch = f.bandTile
			cPitch = f.chanTile
		)
		if datPtrIdx++; i < n {
			pitch1 := f.From.Pitch1Bytes[i]
			if f.PaddingH == 1 {
				expr = cgen.Sub{
					Expr1: expr,
					Expr2: cast(pitch1),
				}
			}
			bPitch *= pitch1 * 2
			cPitch *= f.From.Pitch2Bytes[i]
		} else {
			bPitch *= f.To.Pitch1Bytes[i-n]
			cPitch *= f.To.Pitch2Bytes[i-n]
		}
		expr = addr(expr, cast(bPitch), b)
		expr = addr(expr, cast(cPitch), c)
		f.datPtrs = append(f.datPtrs, ptr)
		stmts = append(stmts, cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptr, Init: expr,
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
			n := len(f.From.Pitch1Bytes)
			n += len(f.To.Pitch1Bytes)
			ndp(n - datPtrIdx)
		}
	}
	do(true)
	do(false)
	return stmts
}

func (f *funcDefs) kernel(chans, bands int, loPadH, hiPadH bool) cgen.Gen {
	switch f.platform {
	case raw.AVX512Float32:
		return f.m512(chans, bands, loPadH, hiPadH)
	default:
		panic("bug")
	}
}

func (f *funcDefs) m512(chans, bands int, loPadH, hiPadH bool) cgen.Gen {
	const (
		lanes     = 16
		laneBytes = 4
	)
	var (
		elems   = (f.From.Width + f.PaddingW*2) & -2
		cells   = ceilQuo(elems, lanes*2)
		unroll  = 1
		iUnroll int
		jUnroll int
		kUnroll int
	)
	if f.PaddingW == 0 {
		iUnroll, jUnroll, kUnroll = cov.Box(
			unroll, unroll, chans, bands, cells,
		)
	} else {
		iUnroll, jUnroll = cov.Rect(
			unroll, unroll, chans, bands,
		)
		kUnroll = 1
	}
	var (
		i       cgen.Gen
		iCnt    int
		bnMuls  = make([][]cgen.Gen, iUnroll)
		bnAdds  = make([][]cgen.Gen, iUnroll)
		j       cgen.Gen
		jCnt    int
		jLoPad  bool
		jHiPad  bool
		carries [][]cgen.Gen
		carryIn bool
		k       cgen.Gen
	)
	meld := func(a *[3][]cgen.Stmts) cgen.Gen {
		return cgen.Gens{
			mix(a[0]),
			mix(a[1]),
			mix(a[2]),
		}
	}
	layer5 := func(ii, jj, kk, l int) [3]cgen.Stmts {
		cell := &m512Cell{
			Ctx:  f.Ctx,
			Spec: f.Spec,
			Pre: m512CellPre{
				BnMuls: bnMuls[ii][:f.bnSplit],
				BnAdds: bnAdds[ii][:f.bnSplit],
			},
			Height: 2,
			Width:  l,
			Post: m512CellPost{
				Ptrs:   make([]cgen.Gen, len(f.To.Pitch1Bytes)),
				BnMuls: bnMuls[ii][f.bnSplit:],
				BnAdds: bnAdds[ii][f.bnSplit:],
			},
		}
		hh := 0
		if jLoPad && jj == 0 {
			hh = 1
			cell.Height = 1
		} else if jHiPad && jj == jCnt-1 {
			cell.Height = 1
		}
		if f.PaddingW == 1 {
			cell.Carry = carries[ii][jj]
			cell.CarryIn = carryIn
		}
		const stride = lanes * laneBytes
		for h := 0; h < cell.Height; h++ {
			for w := 0; w*lanes < l; w++ {
				ptrs := make([]cgen.Gen, f.datSplit)
				for x := range ptrs {
					var (
						pitch2 = f.From.Pitch2Bytes[x]
						pitch1 = f.From.Pitch1Bytes[x]
						ptr    = f.datPtrs[x]
					)
					ptr = cgen.Add{
						Expr1: ptr,
						Expr2: cast(pitch2*ii +
							pitch1*(jj*2+h+hh) +
							stride*(kk*2+w)),
					}
					ptr = addr(ptr, cast(pitch2*iUnroll), i)
					ptr = addr(ptr, cast(pitch1*2*jUnroll), j)
					ptr = addr(ptr, cast(stride*2*kUnroll), k)
					ptrs[x] = ptr
				}
				cell.Pre.Ptrs[h*2+w] = ptrs
			}
		}
		for x := range cell.Post.Ptrs {
			var (
				pitch2 = f.To.Pitch2Bytes[x]
				pitch1 = f.To.Pitch1Bytes[x]
				ptr    = f.datPtrs[f.datSplit+x]
			)
			ptr = cgen.Add{
				Expr1: ptr,
				Expr2: cast(pitch2*ii + pitch1*jj + stride*kk),
			}
			ptr = addr(ptr, cast(pitch2*iUnroll), i)
			ptr = addr(ptr, cast(pitch1*jUnroll), j)
			ptr = addr(ptr, cast(stride*kUnroll), k)
			cell.Post.Ptrs[x] = ptr
		}
		return cell.Stmts()
	}
	layer4NoPadW := func() cgen.Gen {
		var (
			stmts  = make(cgen.Stmts, 2)
			kIters = f.From.Width / (kUnroll * lanes * 2)
			kAfter = f.From.Width % (kUnroll * lanes * 2) & -2
		)
		if kIters > 0 {
			k = vb(f.name("k"))
			var body [3][]cgen.Stmts
			for ii := 0; ii < iCnt; ii++ {
				for jj := 0; jj < jCnt; jj++ {
					for kk := 0; kk < kUnroll; kk++ {
						for x, s := range layer5(ii, jj, kk, lanes*2) {
							body[x] = append(body[x], s)
						}
					}
				}
			}
			stmts[0] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT, What: k,
					Init: il(0),
				},
				Cond: cgen.CmpL{
					Expr1: k, Expr2: il(kIters),
				},
				Post: cgen.IncPre{Expr: k},
				Body: meld(&body),
			}
		}
		if kAfter > 0 {
			k = il(kIters)
			var (
				full = kAfter / (lanes * 2)
				part = kAfter % (lanes * 2)
				tail [3][]cgen.Stmts
			)
			for ii := 0; ii < iCnt; ii++ {
				for jj := 0; jj < jCnt; jj++ {
					for kk := 0; kk <= full; kk++ {
						l := lanes * 2
						if kk == full {
							if l = part; l == 0 {
								break
							}
						}
						for x, s := range layer5(ii, jj, kk, l) {
							tail[x] = append(tail[x], s)
						}
					}
				}
			}
			stmts[1] = meld(&tail)
		}
		return stmts
	}
	layer4PadW := func() cgen.Gen {
		var (
			stmts  = make(cgen.Stmts, 3)
			kIters = f.From.Width / (lanes * 2)
			kAfter = f.From.Width % (lanes * 2)
		)
		if kIters > 0 {
			k = il(0)
			var peel [3][]cgen.Stmts
			for ii := 0; ii < iCnt; ii++ {
				for jj := 0; jj < jCnt; jj++ {
					carries[ii][jj] = vb(f.name("carry"))
					for x, s := range layer5(ii, jj, 0, lanes*2) {
						peel[x] = append(peel[x], s)
					}
				}
			}
			stmts[0] = meld(&peel)
			carryIn = true
			if kIters > 1 {
				k = vb(f.name("k"))
				var body [3][]cgen.Stmts
				for ii := 0; ii < iCnt; ii++ {
					for jj := 0; jj < jCnt; jj++ {
						for x, s := range layer5(ii, jj, 0, lanes*2) {
							body[x] = append(body[x], s)
						}
					}
				}
				stmts[1] = cgen.For{
					Init: cgen.Var{
						Type: cgen.PtrdiffT, What: k,
						Init: il(1),
					},
					Cond: cgen.CmpL{
						Expr1: k, Expr2: il(kIters),
					},
					Post: cgen.IncPre{Expr: k},
					Body: meld(&body),
				}
			}
		}
		k = il(kIters)
		var tail [3][]cgen.Stmts
		for ii := 0; ii < iCnt; ii++ {
			for jj := 0; jj < jCnt; jj++ {
				for x, s := range layer5(ii, jj, 0, kAfter) {
					tail[x] = append(tail[x], s)
				}
			}
		}
		stmts[2] = meld(&tail)
		return stmts
	}
	layer3 := func() cgen.Gen {
		if f.PaddingW == 0 {
			return layer4NoPadW()
		}
		if carries == nil {
			carries = make([][]cgen.Gen, iUnroll)
		}
		for ii := 0; ii < iCnt; ii++ {
			carries[ii] = make([]cgen.Gen, jCnt)
		}
		carryIn = false
		return layer4PadW()
	}
	layer2 := func() cgen.Gen {
		var (
			stmts  = make(cgen.Stmts, 4)
			jIters = bands / jUnroll
			jAfter = bands % jUnroll
		)
		if jIters > 0 {
			jCnt = jUnroll
			first, past := 0, jIters
			if loPadH {
				j = il(first)
				first++
				jLoPad = true
				jHiPad = hiPadH && bands == jUnroll
				stmts[0] = layer3()
			}
			if hiPadH && first < past && jAfter == 0 {
				past--
				j = il(past)
				jLoPad, jHiPad = false, true
				stmts[2] = layer3()
			}
			if first < past {
				j = vb(f.name("j"))
				jLoPad, jHiPad = false, false
				stmts[1] = cgen.For{
					Init: cgen.Var{
						Type: cgen.PtrdiffT, What: j,
						Init: il(first),
					},
					Cond: cgen.CmpL{
						Expr1: j, Expr2: il(past),
					},
					Post: cgen.IncPre{Expr: j},
					Body: layer3(),
				}
			}
		}
		if jAfter > 0 {
			j, jCnt = il(jIters), jAfter
			jLoPad = loPadH && jIters == 0
			jHiPad = hiPadH
			stmts[3] = layer3()
		}
		return stmts
	}
	layer1 := func() cgen.Gen {
		var (
			bnLds = make([]cgen.Stmts, iCnt)
			bnCnt = len(f.bnPtrs)
		)
		c1 := cgen.Mul{
			Expr1: cast(iUnroll),
			Expr2: i,
		}
		for ii := 0; ii < iCnt; ii++ {
			var (
				muls = make([]cgen.Gen, bnCnt)
				adds = make([]cgen.Gen, bnCnt)
				lds  = make(cgen.Stmts, bnCnt)
			)
			c2 := cgen.Paren{Inner: cgen.Add{
				Expr1: c1,
				Expr2: il(ii),
			}}
			for x, ptr := range f.bnPtrs {
				var (
					bnMul = vb(f.name("bnMul"))
					bnAdd = vb(f.name("bnAdd"))
				)
				muls[x] = bnMul
				adds[x] = bnAdd
				lds[x] = &bn.Load{
					Ctx:     f.bc,
					Mas:     ptr,
					Channel: c2,
					Mul:     bnMul,
					Add:     bnAdd,
				}
			}
			bnMuls[ii] = muls
			bnAdds[ii] = adds
			bnLds[ii] = lds
		}
		return cgen.Gens{
			mix(bnLds),
			layer2(),
		}
	}
	var (
		stmts  = make(cgen.Stmts, 2)
		iIters = chans / iUnroll
		iAfter = chans % iUnroll
	)
	if iIters > 0 {
		i, iCnt = vb(f.name("i")), iUnroll
		stmts[0] = cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT, What: i,
				Init: il(0),
			},
			Cond: cgen.CmpL{
				Expr1: i, Expr2: il(iIters),
			},
			Post: cgen.IncPre{Expr: i},
			Body: layer1(),
		}
	}
	if iAfter > 0 {
		u := unroll / iAfter
		if f.PaddingW == 0 {
			jUnroll, kUnroll = cov.Rect(
				u, u, bands, cells,
			)
		} else {
			jUnroll = u
		}
		i, iCnt = il(iIters), iAfter
		stmts[1] = layer1()
	}
	return stmts
}

type m512LdSt struct {
	*Ctx
	*Spec
	Lanes  int
	Ptrs   []cgen.Gen
	BnMuls []cgen.Gen
	BnAdds []cgen.Gen
	Dat    cgen.Gen
	LdDat  bool
	mask   cgen.Gen
	stmts  [3]cgen.Stmts
}

func (m *m512LdSt) ptr() cgen.Gen {
	p := m.Ptrs[0]
	m.Ptrs = m.Ptrs[1:]
	return p
}

func (m *m512LdSt) stmt(i int, s cgen.Gen) {
	m.stmts[i] = append(m.stmts[i], s)
}

func (m *m512LdSt) ld(dat cgen.Gen) {
	m.stmt(0, cgen.Var{
		Type: avx.M512, What: dat,
		Init: avx.Mm512MaskzLoaduPs{
			m.mask, m.ptr(),
		},
	})
}

func (m *m512LdSt) adder(dats []cgen.Gen) {
	for n := len(dats); n > 1; {
		fold := n >> 1
		n -= fold
		for i := 0; i < fold; i++ {
			to := dats[n-1-i]
			m.stmt(1, cgen.Assign{
				Expr1: to,
				Expr2: avx.Mm512AddPs{
					to, dats[n+i],
				},
			})
		}
	}
}

func (m *m512LdSt) apply(ops []mod.Op) {
	i := 0
	for j := range ops {
		switch op := &ops[j]; op.Kind {
		case mod.Add:
			n := 1 + op.Int
			dats := make([]cgen.Gen, n)
			dats[0] = m.Dat
			for k := 1; k < n; k++ {
				dat := vb(m.name("dat"))
				m.ld(dat)
				dats[k] = dat
			}
			m.adder(dats)
		case mod.Bn:
			m.stmt(1, &bn.Apply{
				Ctx: m.bc,
				Mul: m.BnMuls[i],
				Add: m.BnAdds[i],
				To:  m.Dat,
			})
			i++
		case mod.ReLU:
			m.stmt(1, &act.ReLU{
				Ctx:      m.ac,
				NegSlope: op.Float,
				Var:      m.Dat,
			})
		default:
			panic("bug")
		}
	}
}

func (m *m512LdSt) st() {
	for range m.Ptrs {
		m.stmt(2, avx.Mm512MaskStoreuPs{
			m.ptr(), m.mask, m.Dat,
		})
	}
}

func (m *m512LdSt) Stmts() [3]cgen.Stmts {
	m.mask = mask(m.Lanes)
	if m.LdDat {
		m.ld(m.Dat)
		m.apply(m.From.Ops)
	} else {
		m.apply(m.To.Ops)
		m.st()
	}
	return m.stmts
}

type m512NoPadW struct {
	*Ctx
	*Spec
	Dats  [4]cgen.Gen
	Width int
}

func (m *m512NoPadW) combine(to, from cgen.Gen) cgen.Gen {
	if from == nil {
		return nil
	}
	assn := cgen.Assign{Expr1: to}
	switch m.Kind {
	case raw.Avg2x2Stride2:
		assn.Expr2 = avx.Mm512AddPs{to, from}
	case raw.Max2x2Stride2:
		assn.Expr2 = avx.Mm512MaxPs{to, from}
	default:
		panic("bug")
	}
	return assn
}

func (m *m512NoPadW) permute(pm cgen.Gen) cgen.Gen {
	a, b := m.Dats[0], m.Dats[1]
	if b == nil {
		return avx.Mm512PermutexvarPs{pm, a}
	}
	return avx.Mm512Permutex2varPs{a, pm, b}
}

func (m *m512NoPadW) Stmts() cgen.Stmts {
	var (
		stmts = make(cgen.Stmts, 8)
		lo    = m.Dats[0]
		hi    = vb(m.name("hi"))
	)
	for i := 0; i < 2; i++ {
		to, from := m.Dats[i], m.Dats[i+2]
		stmts[i] = m.combine(to, from)
	}
	switch m.Width {
	case 2:
		stmts[4] = avx.Mm512ShufflePs{
			lo, lo, il(0x01),
		}
	case 4:
		stmts[4] = avx.Mm512ShufflePs{
			lo, lo, il(0x0d),
		}
		stmts[5] = avx.Mm512ShufflePs{
			lo, lo, il(0x08),
		}
	default:
		var (
			setLo = make(avx.Mm512SetEpi32, 16)
			setHi = make(avx.Mm512SetEpi32, 16)
			pmLo  = vb(m.name("pmLo"))
			pmHi  = vb(m.name("pmHi"))
		)
		for i := 0; i <= 15; i++ {
			j, k := 15-i, i*2
			setLo[j] = il(k)
			setHi[j] = il(k + 1)
		}
		stmts[2] = cgen.Var{
			Type: avx.M512i, What: pmLo,
			Init: setLo,
		}
		stmts[3] = cgen.Var{
			Type: avx.M512i, What: pmHi,
			Init: setHi,
		}
		stmts[4] = m.permute(pmHi)
		stmts[5] = m.permute(pmLo)
	}
	stmts[4] = cgen.Var{
		Type: avx.M512, What: hi,
		Init: stmts[4],
	}
	if stmts[5] != nil {
		stmts[5] = cgen.Assign{
			Expr1: lo,
			Expr2: stmts[5],
		}
	}
	stmts[6] = m.combine(lo, hi)
	if m.Kind == raw.Avg2x2Stride2 {
		var rcp cgen.Gen
		if m.Dats[2] == nil {
			rcp = avx.Mm512Set1PsLit(0.5)
		} else {
			rcp = avx.Mm512Set1PsLit(0.25)
		}
		stmts[7] = cgen.Assign{
			Expr1: lo,
			Expr2: avx.Mm512MulPs{
				lo, rcp,
			},
		}
	}
	return stmts
}

func (m *m512NoPadW) Dat() (cgen.Gen, int) {
	return m.Dats[0], m.Width / 2
}

type m512PadW struct {
	*Ctx
	*Spec
	Dats    [4]cgen.Gen
	Height  int
	Width   int
	Carry   cgen.Gen
	CarryIn bool
	stmts   cgen.Stmts
}

func (m *m512PadW) stmt(a cgen.Gen) {
	m.stmts = append(m.stmts, a)
}

func (m *m512PadW) combine(to, from cgen.Gen) {
	assn := cgen.Assign{Expr1: to}
	switch m.Kind {
	case raw.Avg2x2Stride2:
		assn.Expr2 = avx.Mm512AddPs{to, from}
	case raw.Max2x2Stride2:
		assn.Expr2 = avx.Mm512MaxPs{to, from}
	default:
		panic("bug")
	}
	m.stmt(assn)
}

func (m *m512PadW) heightwise() {
	if m.Height == 1 || m.Width == 0 {
		return
	}
	m.combine(m.Dats[0], m.Dats[2])
	if m.Width <= 16 {
		return
	}
	m.combine(m.Dats[1], m.Dats[3])
}

func (m *m512PadW) shuffle(i3, i2, i1, i0 int) cgen.Gen {
	var (
		from = m.Dats[0]
		ctrl = i3<<6 | i2<<4 | i1<<2 | i0
	)
	return avx.Mm512ShufflePs{
		from, from, il(ctrl),
	}
}

func (m *m512PadW) permute(pm cgen.Gen) cgen.Gen {
	a, b := m.Dats[0], m.Dats[1]
	if b == nil {
		return avx.Mm512PermutexvarPs{pm, a}
	}
	return avx.Mm512Permutex2varPs{a, pm, b}
}

func (m *m512PadW) widthwise() {
	var (
		pm    bool
		genLo cgen.Gen
		genHi cgen.Gen
	)
	if m.CarryIn {
		if m.Width == 0 {
			m.Dats[0] = m.Carry
		} else if m.Width <= 16 {
			m.Dats[1] = m.Carry
		}
		pm = true
	} else {
		switch m.Width {
		case 1, 2:
		case 3:
			genHi = m.shuffle(0, 0, 2, 0)
		case 4:
			genLo = m.shuffle(0, 3, 1, 0)
			genHi = m.shuffle(0, 3, 2, 0)
		case 5:
			genLo = m.shuffle(0, 3, 1, 0)
			pm = true
		default:
			pm = true
		}
	}
	if pm {
		var (
			pmLo cgen.Gen
			pmHi cgen.Gen
		)
		if genLo == nil {
			pmLo = vb(m.name("pmLo"))
			genLo = m.permute(pmLo)
		}
		switch m.Width {
		case 0:
		case 1, 2:
			genHi = m.Dats[0]
		case 3:
			genHi = m.shuffle(0, 0, 2, 0)
		case 4:
			genHi = m.shuffle(0, 3, 2, 0)
		default:
			pmHi = vb(m.name("pmHi"))
			genHi = m.permute(pmHi)
		}
		var (
			setLo = make(avx.Mm512SetEpi32, 16)
			setHi = make(avx.Mm512SetEpi32, 16)
		)
		if m.CarryIn {
			setLo[15] = il(31)
		} else {
			setLo[15] = il(0)
		}
		setHi[15] = il(0)
		for i := 1; i <= 15; i++ {
			j, k := 15-i, i*2
			setLo[j] = il(k - 1)
			if k == m.Width {
				setHi[j] = setLo[j]
			} else {
				setHi[j] = il(k)
			}
		}
		if pmLo != nil {
			m.stmt(cgen.Var{
				Type: avx.M512i, What: pmLo,
				Init: setLo,
			})
		}
		if pmHi != nil {
			m.stmt(cgen.Var{
				Type: avx.M512i, What: pmHi,
				Init: setHi,
			})
		}
	}
	var hi cgen.Gen
	if genHi != nil {
		hi = vb(m.name("hi"))
		m.stmt(cgen.Var{
			Type: avx.M512, What: hi,
			Init: genHi,
		})
	}
	if m.CarryIn {
		if m.Width > 16 {
			put := cgen.Assign{
				Expr1: m.Dats[1],
				Expr2: avx.Mm512MaskMovPs{
					m.Dats[1], il(1 << 15),
					m.Carry,
				},
			}
			if m.Width == 32 {
				hold := vb(m.name("hold"))
				m.stmt(cgen.Var{
					Type: avx.M512, What: hold,
					Init: m.Dats[1],
				})
				m.stmt(put)
				m.stmt(cgen.Assign{
					Expr1: m.Carry,
					Expr2: hold,
				})
			} else {
				m.stmt(put)
			}
		}
	} else if m.Width == 32 {
		m.stmt(cgen.Var{
			Type: avx.M512, What: m.Carry,
			Init: m.Dats[1],
		})
	}
	if genLo != nil {
		m.stmt(cgen.Assign{
			Expr1: m.Dats[0],
			Expr2: genLo,
		})
	}
	if hi != nil {
		m.combine(m.Dats[0], hi)
	}
	if m.Kind == raw.Avg2x2Stride2 {
		denom := m.Height
		if hi != nil {
			denom *= 2
		}
		if denom > 1 {
			rcp := 1 / avx.Mm512Set1PsLit(denom)
			m.stmt(cgen.Assign{
				Expr1: m.Dats[0],
				Expr2: avx.Mm512MulPs{
					m.Dats[0], rcp,
				},
			})
		}
	}
}

func (m *m512PadW) Stmts() cgen.Stmts {
	m.heightwise()
	m.widthwise()
	return m.stmts
}

func (m *m512PadW) Dat() (cgen.Gen, int) {
	var (
		result = m.Dats[0]
		lanes  = (1 + m.Width + 1) / 2
	)
	if lanes > 16 {
		lanes = 16
	}
	return result, lanes
}

type m512Cell struct {
	*Ctx
	*Spec
	Pre     m512CellPre
	Height  int
	Width   int
	Carry   cgen.Gen
	CarryIn bool
	Post    m512CellPost
}

type m512CellPre struct {
	Ptrs   [4][]cgen.Gen
	BnMuls []cgen.Gen
	BnAdds []cgen.Gen
}

type m512CellPost struct {
	Ptrs   []cgen.Gen
	BnMuls []cgen.Gen
	BnAdds []cgen.Gen
}

func (m *m512Cell) Stmts() [3]cgen.Stmts {
	var (
		stmts [3]cgen.Stmts
		toMix [3][4]cgen.Stmts
		dats  [4]cgen.Gen
	)
	for i, ptrs := range &m.Pre.Ptrs {
		if ptrs == nil {
			continue
		}
		var (
			dat   = vb(m.name("dat"))
			lanes = m.Width - i&1*16
		)
		if lanes > 16 {
			lanes = 16
		}
		pre := &m512LdSt{
			Ctx:    m.Ctx,
			Spec:   m.Spec,
			Lanes:  lanes,
			Ptrs:   ptrs,
			BnMuls: m.Pre.BnMuls,
			BnAdds: m.Pre.BnAdds,
			Dat:    dat,
			LdDat:  true,
		}
		for j, ss := range pre.Stmts() {
			toMix[j][i] = ss
		}
		dats[i] = dat
	}
	for i := range &stmts {
		stmts[i] = mix(toMix[i][:])
	}
	var core interface {
		Stmts() cgen.Stmts
		Dat() (cgen.Gen, int)
	}
	if m.PaddingW == 0 {
		core = &m512NoPadW{
			Ctx:   m.Ctx,
			Spec:  m.Spec,
			Dats:  dats,
			Width: m.Width,
		}
	} else {
		core = &m512PadW{
			Ctx:     m.Ctx,
			Spec:    m.Spec,
			Dats:    dats,
			Height:  m.Height,
			Width:   m.Width,
			Carry:   m.Carry,
			CarryIn: m.CarryIn,
		}
	}
	stmts[1] = append(
		stmts[1], core.Stmts()...,
	)
	post := &m512LdSt{
		Ctx:    m.Ctx,
		Spec:   m.Spec,
		Ptrs:   m.Post.Ptrs,
		BnMuls: m.Post.BnMuls,
		BnAdds: m.Post.BnAdds,
	}
	post.Dat, post.Lanes = core.Dat()
	for i, ss := range post.Stmts() {
		stmts[i] = append(
			stmts[i], ss...,
		)
	}
	return stmts
}
