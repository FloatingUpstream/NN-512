package elwi

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
	nms      nmsrc.Src
	tc       *threader.Ctx
	ac       *act.Ctx
	bc       *bn.Ctx
	dedup    map[string]string
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src, tc *threader.Ctx, ac *act.Ctx, bc *bn.Ctx) *Ctx {
	return &Ctx{
		prefix:   pl.Config.Prefix + "Elwi",
		platform: pl.Config.Platform,
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

func (c *Ctx) lanes() int {
	switch c.platform {
	case raw.AVX512Float32:
		return 16
	default:
		panic("bug")
	}
}

type Spec struct {
	Channels    int
	Height      int
	Width       int
	ElemBytes   int
	Pitch1Bytes []int
	Pitch2Bytes []int
	Ops         [][]mod.Op
}

func enough(ctx *Ctx, spec *Spec) int {
	var (
		cost = len(spec.Pitch1Bytes)
		mul  int
	)
	switch ctx.platform {
	case raw.AVX512Float32:
		const lo = 8
		if mul = 512 / cost; mul < lo {
			mul = lo
		}
	default:
		panic("bug")
	}
	return ctx.lanes() * mul
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
	const (
		formPacked = iota
		formSemipacked
		formUnpacked
	)
	var (
		form   = formPacked
		tight1 = c.Width * c.ElemBytes
		tight2 = c.Height * tight1
		funcs  cgen.Gen
	)
	for i, pitch1 := range c.Pitch1Bytes {
		if pitch1 != tight1 {
			form = formUnpacked
			break
		}
		if c.Pitch2Bytes[i] != tight2 {
			form = formSemipacked
		}
	}
	if form == formPacked {
	outer:
		for _, ops := range c.Ops {
			for i := range ops {
				if ops[i].Kind == mod.Bn {
					form = formSemipacked
					break outer
				}
			}
		}
	}
	switch form {
	case formPacked:
		funcs = &packed{
			Ctx:      c.Ctx,
			Spec:     c.Spec,
			FuncName: c.funcName,
		}
	case formSemipacked:
		funcs = &semipacked{
			Ctx:      c.Ctx,
			Spec:     c.Spec,
			FuncName: c.funcName,
		}
	case formUnpacked:
		funcs = &unpacked{
			Ctx:      c.Ctx,
			Spec:     c.Spec,
			FuncName: c.funcName,
		}
	}
	return cgen.Gens{
		funcs, cgen.Newline,
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

type unpacked struct {
	*Ctx
	*Spec
	FuncName string
	wTile    int
	wTiles   int
	wScrap   int
	hTile    int
	hTiles   int
	hScrap   int
	cTile    int
	cTiles   int
	cScrap   int
	funcName string
	datPtrs  []cgen.Gen
	bnPtrs   []cgen.Gen
}

func (u *unpacked) Append(to []byte) []byte {
	elems := enough(u.Ctx, u.Spec)
	if u.Width >= elems {
		fit := u.Width / elems
		elems = u.Width / fit
		elems -= elems % u.lanes()
		u.wTile = elems
		u.wTiles = fit
		u.wScrap = u.Width - elems*fit
		if u.wScrap > 0 {
			u.wTiles--
			u.wScrap += elems
		}
		u.hTile = 1
		u.hTiles = u.Height
		u.hScrap = 0
		u.cTile = 1
		u.cTiles = u.Channels
		u.cScrap = 0
	} else if hw := u.Height * u.Width; hw >= elems {
		var (
			rows = (elems + u.Width - 1) / u.Width
			fit  = u.Height / rows
		)
		rows = u.Height / fit
		u.wTile = u.Width
		u.wTiles = 1
		u.wScrap = 0
		u.hTile = rows
		u.hTiles = fit
		u.hScrap = u.Height - rows*fit
		if u.hScrap > 0 {
			u.hTiles--
			u.hScrap += rows
		}
		u.cTile = 1
		u.cTiles = u.Channels
		u.cScrap = 0
	} else {
		chans := (elems + hw - 1) / hw
		u.wTile = u.Width
		u.wTiles = 1
		u.wScrap = 0
		u.hTile = u.Height
		u.hTiles = 1
		u.hScrap = 0
		u.cTile = chans
		u.cTiles = u.Channels / chans
		u.cScrap = u.Channels % chans
	}
	u.funcName = u.name(u.FuncName + "Callee")
	var (
		team    = vb(u.name("team"))
		tensors = vb(u.name("tensors"))
		wHull   = u.wTiles
		hHull   = u.hTiles
		cHull   = u.cTiles
	)
	if u.wScrap > 0 {
		wHull++
	}
	if u.hScrap > 0 {
		hHull++
	}
	if u.cScrap > 0 {
		cHull++
	}
	return cgen.Gens{
		u.calleeFunc(),
		cgen.Newline,
		cgen.StaticFuncDef{
			ReturnType: cgen.Void,
			Name:       u.FuncName,
			Params: cgen.CommaSpaced{
				cgen.Param{
					Type: u.tc.PtrTeam,
					What: team,
				},
				cgen.Param{
					Type: cgen.PtrPtrChar,
					What: tensors,
				},
			},
			Body: &threader.Do{
				Ctx:    u.tc,
				Callee: vb(u.funcName),
				Any:    tensors,
				Hull: []cgen.Gen{
					il(wHull),
					il(hHull),
					il(cHull),
				},
				Team: team,
			},
		},
	}.Append(to)
}

func (u *unpacked) calleeFunc() cgen.Gen {
	var (
		body    = make(cgen.Stmts, 7)
		tensors = vb(u.name("tensors"))
		w       = vb(u.name("w"))
		h       = vb(u.name("h"))
		c       = vb(u.name("c"))
	)
	callee := &threader.Callee{
		Ctx:  u.tc,
		Name: u.funcName,
		Task: vb(u.name("task")),
		Pt:   vb(u.name("pt")),
	}
	body[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: tensors,
		Init: callee.Any(),
	}
	body[1] = cgen.Var{
		Type: cgen.PtrdiffT, What: w,
		Init: cgen.Elem{Arr: callee.Pt, Idx: il(0)},
	}
	body[2] = cgen.Var{
		Type: cgen.PtrdiffT, What: h,
		Init: cgen.Elem{Arr: callee.Pt, Idx: il(1)},
	}
	body[3] = cgen.Var{
		Type: cgen.PtrdiffT, What: c,
		Init: cgen.Elem{Arr: callee.Pt, Idx: il(2)},
	}
	body[4] = u.ptrs(tensors, w, h, c)
	doIf := func(do, i cgen.Gen, n int) cgen.Gen {
		return cgen.If{
			Cond: cgen.CmpL{
				Expr1: i,
				Expr2: il(n),
			},
			Then: cgen.Stmts{
				do,
				cgen.Return{},
			},
		}
	}
	wSplit := func(chans, rows int) cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if u.wTiles > 0 {
			k := u.kernel(chans, rows, u.wTile)
			if u.wScrap > 0 {
				stmts[0] = doIf(k, w, u.wTiles)
			} else {
				stmts[0] = k
			}
		}
		if u.wScrap > 0 {
			stmts[1] = u.kernel(chans, rows, u.wScrap)
		}
		return stmts
	}
	hSplit := func(chans int) cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if u.hTiles > 0 {
			ws := wSplit(chans, u.hTile)
			if u.hScrap > 0 {
				stmts[0] = doIf(ws, h, u.hTiles)
			} else {
				stmts[0] = ws
			}
		}
		if u.hScrap > 0 {
			stmts[1] = wSplit(chans, u.hScrap)
		}
		return stmts
	}
	if u.cTiles > 0 {
		hs := hSplit(u.cTile)
		if u.cScrap > 0 {
			body[5] = doIf(hs, c, u.cTiles)
		} else {
			body[5] = hs
		}
	}
	if u.cScrap > 0 {
		body[6] = hSplit(u.cScrap)
	}
	return callee.Func(body)
}

func (u *unpacked) ptrs(tensors, w, h, c cgen.Gen) cgen.Gen {
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
		ptr := vb(u.name("ptr"))
		u.datPtrs = append(u.datPtrs, ptr)
		var (
			wPitch = u.wTile * u.ElemBytes
			hPitch = u.hTile * u.Pitch1Bytes[datPtrIdx]
			cPitch = u.cTile * u.Pitch2Bytes[datPtrIdx]
			a1     = tensor()
			a2     = addr(a1, cast(wPitch), w)
			a3     = addr(a2, cast(hPitch), h)
			a4     = addr(a3, cast(cPitch), c)
		)
		stmts = append(stmts, cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptr, Init: a4,
		})
		datPtrIdx++
	}
	ndp := func(n int) {
		for ; n > 0; n-- {
			datPtr()
		}
	}
	bnPtr := func() {
		ptr := vb(u.name("ptr"))
		u.bnPtrs = append(u.bnPtrs, ptr)
		stmts = append(stmts, cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptr,
			Init: &bn.Offset{
				Ctx: u.bc,
				Mas: tensor(),
				Channel: cgen.Mul{
					Expr1: il(u.cTile),
					Expr2: c,
				},
			},
		})
	}
	for i, ops := range u.Ops {
		if i < len(u.Ops)-1 {
			datPtr()
		}
		for j := range ops {
			switch op := &ops[j]; op.Kind {
			case mod.Add:
				ndp(op.Int)
			case mod.Bn:
				bnPtr()
			case mod.ReLU:
			default:
				panic("bug")
			}
		}
	}
	ndp(len(u.Pitch1Bytes) - datPtrIdx)
	return stmts
}

func (u *unpacked) kernel(chans, rows, elems int) cgen.Gen {
	switch u.platform {
	case raw.AVX512Float32:
		return u.m512(chans, rows, elems)
	default:
		panic("bug")
	}
}

func (u *unpacked) m512(chans, rows, elems int) cgen.Gen {
	const (
		lanes     = 16
		laneBytes = 4
	)
	unroll := 6 - len(u.datPtrs)
	if unroll < 1 {
		unroll = 1
	}
	iUnroll, jUnroll, kUnroll := cov.Box(
		unroll, unroll, chans, rows, (elems+lanes-1)/lanes,
	)
	var (
		iIters = chans / iUnroll
		iAfter = chans % iUnroll
		bnMuls = make([][]cgen.Gen, iUnroll)
		bnAdds = make([][]cgen.Gen, iUnroll)
		jIters = rows / jUnroll
		jAfter = rows % jUnroll
		kIters = elems / (kUnroll * lanes)
		kAfter = elems % (kUnroll * lanes)
	)
	leaf := func(i, j, k cgen.Gen, ii, jj, kk, l int) cgen.Stmts {
		cell := &m512Cell{
			Ctx:    u.Ctx,
			Spec:   u.Spec,
			Lanes:  l,
			Ptrs:   make([]cgen.Gen, len(u.datPtrs)),
			BnMuls: bnMuls[ii],
			BnAdds: bnAdds[ii],
		}
		for x, ptr := range u.datPtrs {
			var (
				iiPitch = u.Pitch2Bytes[x]
				jjPitch = u.Pitch1Bytes[x]
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
			cell.Ptrs[x] = ptr
		}
		return cell.Stmts()
	}
	kSplit := func(i, j cgen.Gen, iCnt, jCnt int) cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if kIters > 0 {
			var (
				body = make([]cgen.Stmts, iCnt*jCnt*kUnroll)
				k    = vb(u.name("k"))
			)
			for ii := 0; ii < iCnt; ii++ {
				for jj := 0; jj < jCnt; jj++ {
					for kk := 0; kk < kUnroll; kk++ {
						x := (ii*jCnt+jj)*kUnroll + kk
						body[x] = leaf(i, j, k, ii, jj, kk, lanes)
					}
				}
			}
			stmts[0] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT, What: k,
					Init: cgen.Zero,
				},
				Cond: cgen.CmpL{
					Expr1: k, Expr2: il(kIters),
				},
				Post: cgen.IncPre{Expr: k},
				Body: mix(body),
			}
		}
		if kAfter > 0 {
			var (
				full = kAfter / lanes
				part = kAfter % lanes
				tail = make([]cgen.Stmts, iCnt*jCnt*kUnroll)
				k    = il(kIters)
			)
			for ii := 0; ii < iCnt; ii++ {
				for jj := 0; jj < jCnt; jj++ {
					for kk := 0; kk <= full; kk++ {
						var (
							x = (ii*jCnt+jj)*kUnroll + kk
							l = lanes
						)
						if kk == full {
							l = part
						}
						if l > 0 {
							tail[x] = leaf(i, j, k, ii, jj, kk, l)
						}
					}
				}
			}
			stmts[1] = mix(tail)
		}
		return stmts
	}
	jSplit := func(i cgen.Gen, iCnt int) cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if jIters > 0 {
			j := vb(u.name("j"))
			stmts[0] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT, What: j,
					Init: cgen.Zero,
				},
				Cond: cgen.CmpL{
					Expr1: j, Expr2: il(jIters),
				},
				Post: cgen.IncPre{Expr: j},
				Body: kSplit(i, j, iCnt, jUnroll),
			}
		}
		if jAfter > 0 {
			j := il(jIters)
			stmts[1] = kSplit(i, j, iCnt, jAfter)
		}
		return stmts
	}
	iBlock := func(i cgen.Gen, iCnt int) cgen.Gen {
		var (
			bnLds = make([]cgen.Stmts, iCnt)
			bnCnt = len(u.bnPtrs)
		)
		for ii := 0; ii < iCnt; ii++ {
			var (
				ch   = il(ii)
				muls = make([]cgen.Gen, bnCnt)
				adds = make([]cgen.Gen, bnCnt)
				lds  = make(cgen.Stmts, bnCnt)
			)
			if iIters > 0 {
				ch = cgen.Paren{
					Inner: addr(ch, cast(iUnroll), i),
				}
			}
			for x, ptr := range u.bnPtrs {
				var (
					bnMul = vb(u.name("bnMul"))
					bnAdd = vb(u.name("bnAdd"))
				)
				muls[x] = bnMul
				adds[x] = bnAdd
				lds[x] = &bn.Load{
					Ctx:     u.bc,
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
			mix(bnLds), jSplit(i, iCnt),
		}
	}
	stmts := make(cgen.Stmts, 2)
	if iIters > 0 {
		i := vb(u.name("i"))
		stmts[0] = cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT, What: i,
				Init: cgen.Zero,
			},
			Cond: cgen.CmpL{
				Expr1: i, Expr2: il(iIters),
			},
			Post: cgen.IncPre{Expr: i},
			Body: iBlock(i, iUnroll),
		}
	}
	if iAfter > 0 {
		i := il(iIters)
		stmts[1] = iBlock(i, iAfter)
	}
	return stmts
}

type semipacked struct {
	*Ctx
	*Spec
	FuncName  string
	elemTile  int
	elemTiles int
	elemScrap int
	chanTile  int
	chanTiles int
	chanScrap int
	funcName  string
	datPtrs   []cgen.Gen
	bnPtrs    []cgen.Gen
}

func (s *semipacked) Append(to []byte) []byte {
	var (
		hw    = s.Height * s.Width
		elems = enough(s.Ctx, s.Spec)
	)
	if hw >= elems {
		fit := hw / elems
		elems = hw / fit
		elems -= elems % s.lanes()
		s.elemTile = elems
		s.elemTiles = fit
		s.elemScrap = hw - elems*fit
		if s.elemScrap > 0 {
			s.elemTiles--
			s.elemScrap += elems
		}
		s.chanTile = 1
		s.chanTiles = s.Channels
		s.chanScrap = 0
	} else {
		chans := elems / hw
		if chans*hw < elems {
			chans++
		}
		s.elemTile = hw
		s.elemTiles = 1
		s.elemScrap = 0
		s.chanTile = chans
		s.chanTiles = s.Channels / chans
		s.chanScrap = s.Channels % chans
	}
	s.funcName = s.name(s.FuncName + "Callee")
	var (
		team     = vb(s.name("team"))
		tensors  = vb(s.name("tensors"))
		elemHull = s.elemTiles
		chanHull = s.chanTiles
	)
	if s.elemScrap > 0 {
		elemHull++
	}
	if s.chanScrap > 0 {
		chanHull++
	}
	return cgen.Gens{
		s.calleeFunc(),
		cgen.Newline,
		cgen.StaticFuncDef{
			ReturnType: cgen.Void,
			Name:       s.FuncName,
			Params: cgen.CommaSpaced{
				cgen.Param{
					Type: s.tc.PtrTeam,
					What: team,
				},
				cgen.Param{
					Type: cgen.PtrPtrChar,
					What: tensors,
				},
			},
			Body: &threader.Do{
				Ctx:    s.tc,
				Callee: vb(s.funcName),
				Any:    tensors,
				Hull: []cgen.Gen{
					il(elemHull),
					il(chanHull),
				},
				Team: team,
			},
		},
	}.Append(to)
}

func (s *semipacked) calleeFunc() cgen.Gen {
	var (
		body    = make(cgen.Stmts, 6)
		tensors = vb(s.name("tensors"))
		e       = vb(s.name("e"))
		c       = vb(s.name("c"))
	)
	callee := &threader.Callee{
		Ctx:  s.tc,
		Name: s.funcName,
		Task: vb(s.name("task")),
		Pt:   vb(s.name("pt")),
	}
	body[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: tensors,
		Init: callee.Any(),
	}
	body[1] = cgen.Var{
		Type: cgen.PtrdiffT, What: e,
		Init: cgen.Elem{Arr: callee.Pt, Idx: cgen.Zero},
	}
	body[2] = cgen.Var{
		Type: cgen.PtrdiffT, What: c,
		Init: cgen.Elem{Arr: callee.Pt, Idx: cgen.One},
	}
	body[3] = s.ptrs(tensors, e, c)
	doIf := func(do, i cgen.Gen, n int) cgen.Gen {
		return cgen.If{
			Cond: cgen.CmpL{
				Expr1: i,
				Expr2: il(n),
			},
			Then: cgen.Stmts{
				do,
				cgen.Return{},
			},
		}
	}
	kernels := func(chans int) cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if s.elemTiles > 0 {
			k := s.kernel(chans, s.elemTile)
			if s.elemScrap > 0 {
				stmts[0] = doIf(k, e, s.elemTiles)
			} else {
				stmts[0] = k
			}
		}
		if s.elemScrap > 0 {
			stmts[1] = s.kernel(chans, s.elemScrap)
		}
		return stmts
	}
	if s.chanTiles > 0 {
		ks := kernels(s.chanTile)
		if s.chanScrap > 0 {
			body[4] = doIf(ks, c, s.chanTiles)
		} else {
			body[4] = ks
		}
	}
	if s.chanScrap > 0 {
		body[5] = kernels(s.chanScrap)
	}
	return callee.Func(body)
}

func (s *semipacked) ptrs(tensors, e, c cgen.Gen) cgen.Gen {
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
	pitch := func() int {
		i := datPtrIdx
		datPtrIdx++
		return s.Pitch2Bytes[i]
	}
	datPtr := func() {
		ptr := vb(s.name("ptr"))
		s.datPtrs = append(s.datPtrs, ptr)
		var (
			ePitch = s.elemTile * s.ElemBytes
			cPitch = s.chanTile * pitch()
			a1     = tensor()
			a2     = addr(a1, cast(ePitch), e)
			a3     = addr(a2, cast(cPitch), c)
		)
		stmts = append(stmts, cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptr, Init: a3,
		})
	}
	ndp := func(n int) {
		for ; n > 0; n-- {
			datPtr()
		}
	}
	bnPtr := func() {
		ptr := vb(s.name("ptr"))
		s.bnPtrs = append(s.bnPtrs, ptr)
		stmts = append(stmts, cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptr,
			Init: &bn.Offset{
				Ctx: s.bc,
				Mas: tensor(),
				Channel: cgen.Mul{
					Expr1: il(s.chanTile),
					Expr2: c,
				},
			},
		})
	}
	for i, ops := range s.Ops {
		if i < len(s.Ops)-1 {
			datPtr()
		}
		for j := range ops {
			switch op := &ops[j]; op.Kind {
			case mod.Add:
				ndp(op.Int)
			case mod.Bn:
				bnPtr()
			case mod.ReLU:
			default:
				panic("bug")
			}
		}
	}
	ndp(len(s.Pitch2Bytes) - datPtrIdx)
	return stmts
}

func (s *semipacked) kernel(chans, elems int) cgen.Gen {
	switch s.platform {
	case raw.AVX512Float32:
		return s.m512(chans, elems)
	default:
		panic("bug")
	}
}

func (s *semipacked) m512(chans, elems int) cgen.Gen {
	const (
		lanes     = 16
		laneBytes = 4
	)
	unroll := 6 - len(s.datPtrs)
	if unroll < 1 {
		unroll = 1
	}
	iUnroll, jUnroll := cov.Rect(
		unroll, unroll, chans, (elems+lanes-1)/lanes,
	)
	var (
		iIters = chans / iUnroll
		iAfter = chans % iUnroll
		bnMuls = make([][]cgen.Gen, iUnroll)
		bnAdds = make([][]cgen.Gen, iUnroll)
		jIters = elems / (jUnroll * lanes)
		jAfter = elems % (jUnroll * lanes)
	)
	leaf := func(i, j cgen.Gen, ii, jj, l int) cgen.Stmts {
		cell := &m512Cell{
			Ctx:    s.Ctx,
			Spec:   s.Spec,
			Lanes:  l,
			Ptrs:   make([]cgen.Gen, len(s.datPtrs)),
			BnMuls: bnMuls[ii],
			BnAdds: bnAdds[ii],
		}
		for k, ptr := range s.datPtrs {
			var (
				iiPitch = s.Pitch2Bytes[k]
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
			cell.Ptrs[k] = ptr
		}
		return cell.Stmts()
	}
	inner := func(i cgen.Gen, iCnt int) cgen.Gen {
		jSplit := make(cgen.Stmts, 2)
		if jIters > 0 {
			var (
				body = make([]cgen.Stmts, iCnt*jUnroll)
				j    = vb(s.name("j"))
			)
			for ii := 0; ii < iCnt; ii++ {
				for jj := 0; jj < jUnroll; jj++ {
					k := ii*jUnroll + jj
					body[k] = leaf(i, j, ii, jj, lanes)
				}
			}
			jSplit[0] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT, What: j,
					Init: cgen.Zero,
				},
				Cond: cgen.CmpL{
					Expr1: j, Expr2: il(jIters),
				},
				Post: cgen.IncPre{Expr: j},
				Body: mix(body),
			}
		}
		if jAfter > 0 {
			var (
				full = jAfter / lanes
				part = jAfter % lanes
				tail = make([]cgen.Stmts, iCnt*jUnroll)
				j    = il(jIters)
			)
			for ii := 0; ii < iCnt; ii++ {
				for jj := 0; jj <= full; jj++ {
					var (
						k = ii*jUnroll + jj
						l = lanes
					)
					if jj == full {
						l = part
					}
					if l > 0 {
						tail[k] = leaf(i, j, ii, jj, l)
					}
				}
			}
			jSplit[1] = mix(tail)
		}
		return jSplit
	}
	outer := func(i cgen.Gen, iCnt int) cgen.Gen {
		var (
			bnLds = make([]cgen.Stmts, iCnt)
			bnCnt = len(s.bnPtrs)
		)
		for ii := 0; ii < iCnt; ii++ {
			var (
				ch   = il(ii)
				muls = make([]cgen.Gen, bnCnt)
				adds = make([]cgen.Gen, bnCnt)
				lds  = make(cgen.Stmts, bnCnt)
			)
			if iIters > 0 {
				ch = cgen.Paren{
					Inner: addr(ch, cast(iUnroll), i),
				}
			}
			for k, ptr := range s.bnPtrs {
				var (
					bnMul = vb(s.name("bnMul"))
					bnAdd = vb(s.name("bnAdd"))
				)
				muls[k] = bnMul
				adds[k] = bnAdd
				lds[k] = &bn.Load{
					Ctx:     s.bc,
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
			mix(bnLds), inner(i, iCnt),
		}
	}
	iSplit := make(cgen.Stmts, 2)
	if iIters > 0 {
		i := vb(s.name("i"))
		iSplit[0] = cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT, What: i,
				Init: cgen.Zero,
			},
			Cond: cgen.CmpL{
				Expr1: i, Expr2: il(iIters),
			},
			Post: cgen.IncPre{Expr: i},
			Body: outer(i, iUnroll),
		}
	}
	if iAfter > 0 {
		i := il(iIters)
		iSplit[1] = outer(i, iAfter)
	}
	return iSplit
}

type packed struct {
	*Ctx
	*Spec
	FuncName string
	grain    int
	grains   int
	remain   int
	funcName string
	ptrs     []cgen.Gen
}

func (p *packed) Append(to []byte) []byte {
	p.grain = enough(p.Ctx, p.Spec)
	elems := p.Channels * p.Height * p.Width
	p.grains = elems / p.grain
	p.remain = elems % p.grain
	p.funcName = p.name(p.FuncName + "Callee")
	var (
		team    = vb(p.name("team"))
		tensors = vb(p.name("tensors"))
		hull    = p.grains
	)
	if p.remain > 0 {
		hull++
	}
	return cgen.Gens{
		p.calleeFunc(),
		cgen.Newline,
		cgen.StaticFuncDef{
			ReturnType: cgen.Void,
			Name:       p.FuncName,
			Params: cgen.CommaSpaced{
				cgen.Param{
					Type: p.tc.PtrTeam, What: team,
				},
				cgen.Param{
					Type: cgen.PtrPtrChar, What: tensors,
				},
			},
			Body: &threader.Do{
				Ctx:    p.tc,
				Callee: vb(p.funcName),
				Any:    tensors,
				Hull:   []cgen.Gen{il(hull)},
				Team:   team,
			},
		},
	}.Append(to)
}

func (p *packed) calleeFunc() cgen.Gen {
	var (
		body    = make(cgen.Stmts, 5)
		tensors = vb(p.name("tensors"))
		i       = vb(p.name("i"))
	)
	callee := &threader.Callee{
		Ctx:  p.tc,
		Name: p.funcName,
		Task: vb(p.name("task")),
		Pt:   vb(p.name("pt")),
	}
	body[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: tensors,
		Init: callee.Any(),
	}
	body[1] = cgen.Var{
		Type: cgen.PtrdiffT, What: i,
		Init: cgen.Elem{Arr: callee.Pt, Idx: cgen.Zero},
	}
	body[2] = p.loadPtrs(tensors, i)
	if p.grains > 0 {
		body[3] = p.kernel(p.grain)
		if p.remain > 0 {
			body[3] = cgen.If{
				Cond: cgen.CmpL{
					Expr1: i,
					Expr2: il(p.grains),
				},
				Then: cgen.Stmts{
					body[3],
					cgen.Return{},
				},
			}
		}
	}
	if p.remain > 0 {
		body[4] = p.kernel(p.remain)
	}
	return callee.Func(body)
}

func (p *packed) loadPtrs(tensors, i cgen.Gen) cgen.Gen {
	var (
		n     = len(p.Pitch1Bytes)
		stmts = make(cgen.Stmts, n)
		pitch = cast(p.grain * p.ElemBytes)
	)
	p.ptrs = make([]cgen.Gen, n)
	for j := range p.ptrs {
		p.ptrs[j] = vb(p.name("ptr"))
		var (
			a1 = cgen.Elem{Arr: tensors, Idx: il(j)}
			a2 = addr(a1, pitch, i)
		)
		stmts[j] = cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: p.ptrs[j], Init: a2,
		}
	}
	return stmts
}

func (p *packed) kernel(elems int) cgen.Gen {
	switch p.platform {
	case raw.AVX512Float32:
		return p.m512(elems)
	default:
		panic("bug")
	}
}

func (p *packed) m512(elems int) cgen.Gen {
	const (
		unroll    = 4
		lanes     = 16
		iterElems = unroll * lanes
	)
	var (
		iters = elems / iterElems
		after = elems % iterElems
	)
	code := func(j cgen.Gen, k, l int) cgen.Stmts {
		cell := &m512Cell{
			Ctx:   p.Ctx,
			Spec:  p.Spec,
			Lanes: l,
			Ptrs:  make([]cgen.Gen, len(p.ptrs)),
		}
		for i, ptr := range p.ptrs {
			const (
				laneBytes = 4
				kPitch    = lanes * laneBytes
				jPitch    = unroll * kPitch
			)
			ptr = addr(ptr, cast(kPitch), il(k))
			if iters > 0 {
				ptr = addr(ptr, cast(jPitch), j)
			}
			cell.Ptrs[i] = ptr
		}
		return cell.Stmts()
	}
	stmts := make(cgen.Stmts, 2)
	if iters > 0 {
		var (
			inner = make([]cgen.Stmts, unroll)
			j     = vb(p.name("j"))
		)
		for k := 0; k < unroll; k++ {
			inner[k] = code(j, k, lanes)
		}
		stmts[0] = cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT, What: j,
				Init: cgen.Zero,
			},
			Cond: cgen.CmpL{
				Expr1: j, Expr2: il(iters),
			},
			Post: cgen.IncPre{Expr: j},
			Body: mix(inner),
		}
	}
	if after > 0 {
		var (
			full  = after / lanes
			part  = after % lanes
			outer = make([]cgen.Stmts, full, full+1)
			j     = il(iters)
		)
		for k := 0; k < full; k++ {
			outer[k] = code(j, k, lanes)
		}
		if part > 0 {
			last := code(j, full, part)
			outer = append(outer, last)
		}
		stmts[1] = mix(outer)
	}
	return stmts
}

type m512Cell struct {
	*Ctx
	*Spec
	Lanes    int
	Ptrs     []cgen.Gen
	BnMuls   []cgen.Gen
	BnAdds   []cgen.Gen
	mask     cgen.Gen
	nextPtr  int
	nextBn   int
	loads    cgen.Stmts
	nonloads cgen.Stmts
}

func (m *m512Cell) ptr() cgen.Gen {
	i := m.nextPtr
	m.nextPtr = i + 1
	return m.Ptrs[i]
}

func (m *m512Cell) load() cgen.Gen {
	dat := vb(m.name("dat"))
	m.loads = append(m.loads, cgen.Var{
		Type: avx.M512, What: dat,
		Init: avx.Mm512MaskzLoaduPs{
			m.mask, m.ptr(),
		},
	})
	return dat
}

func (m *m512Cell) nonload(a cgen.Gen) {
	m.nonloads = append(m.nonloads, a)
}

func (m *m512Cell) adder(dats []cgen.Gen) {
	for n := len(dats); n > 1; {
		fold := n >> 1
		n -= fold
		for i := 0; i < fold; i++ {
			to := dats[n-1-i]
			m.nonload(cgen.Assign{
				Expr1: to,
				Expr2: avx.Mm512AddPs{
					to, dats[n+i],
				},
			})
		}
	}
}

func (m *m512Cell) apply(dat cgen.Gen, ops []mod.Op) {
	for i := range ops {
		switch op := &ops[i]; op.Kind {
		case mod.Add:
			n := op.Int
			dats := make([]cgen.Gen, 1+n)
			dats[0] = dat
			for j := 1; j <= n; j++ {
				dats[j] = m.load()
			}
			m.adder(dats)
		case mod.Bn:
			j := m.nextBn
			m.nextBn = j + 1
			m.nonload(&bn.Apply{
				Ctx: m.bc,
				Mul: m.BnMuls[j],
				Add: m.BnAdds[j],
				To:  dat,
			})
		case mod.ReLU:
			m.nonload(&act.ReLU{
				Ctx:      m.ac,
				NegSlope: op.Float,
				Var:      dat,
			})
		default:
			panic("bug")
		}
	}
}

func (m *m512Cell) Stmts() cgen.Stmts {
	m.mask = il(1<<uint(m.Lanes) - 1)
	var (
		last     = len(m.Ops) - 1
		dats     = make([]cgen.Gen, last)
		loads    = make([]cgen.Stmts, last)
		nonloads = make([]cgen.Stmts, last)
	)
	for i := 0; i < last; i++ {
		m.loads = nil
		m.nonloads = nil
		dat := m.load()
		m.apply(dat, m.Ops[i])
		dats[i] = dat
		loads[i] = m.loads
		nonloads[i] = m.nonloads
	}
	if last > 1 {
		m.loads = mix(loads)
		m.nonloads = mix(nonloads)
		m.adder(dats)
	}
	dat := dats[0]
	m.apply(dat, m.Ops[last])
	for range m.Ptrs[m.nextPtr:] {
		m.nonload(avx.Mm512MaskStoreuPs{
			m.ptr(), m.mask, dat,
		})
	}
	return append(
		m.loads, m.nonloads...,
	)
}
