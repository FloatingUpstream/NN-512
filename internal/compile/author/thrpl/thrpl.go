package thrpl

import (
	"NN-512/internal/compile/author/act"
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/bn"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/mod"
	"NN-512/internal/compile/author/threader"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
	"fmt"
)

type Ctx struct {
	prefix     string
	platform   raw.Platform
	cacheBytes int
	nms        nmsrc.Src
	tc         *threader.Ctx
	ac         *act.Ctx
	bc         *bn.Ctx
	dedup      map[string]string
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src, tc *threader.Ctx, ac *act.Ctx, bc *bn.Ctx) *Ctx {
	return &Ctx{
		prefix:     pl.Config.Prefix + "Thrpl",
		platform:   pl.Config.Platform,
		cacheBytes: pl.Config.L1DataCachePerThread + pl.Config.L2CachePerThreadExL1,
		nms:        nms,
		tc:         tc,
		ac:         ac,
		bc:         bc,
		dedup:      make(map[string]string),
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

func fl(f float32) cgen.Gen {
	return cgen.FloatLit(f)
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
	elemBytes int
	elemTile  int
	elemTiles int
	elemScrap int
	elemHull  int
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
	chans     int
	bands     int
	elems     int
	loEdgeH   bool
	hiEdgeH   bool
	loEdgeW   bool
	hiEdgeW   bool
}

func (f *funcDefs) Append(to []byte) []byte {
	var (
		vecElems    int
		threadCells int
	)
	switch f.platform {
	case raw.AVX512Float32:
		f.elemBytes = 4
		vecElems = 16
		threadCells = 128
	default:
		panic("bug")
	}
	var (
		fromPtrs = len(f.From.Pitch1Bytes)
		toPtrs   = len(f.To.Pitch1Bytes)
	)
	threadCells = ceilQuo(threadCells*5, fromPtrs*4+toPtrs)
	var (
		cacheVecs  = f.cacheBytes / (vecElems * f.elemBytes)
		cellVecs   = fromPtrs*6 + toPtrs
		bandCells  = cacheVecs / cellVecs / 2
		widthCells = ceilQuo(f.From.Width, vecElems*2)
	)
	if bandCells < 2 {
		bandCells = 2
	}
	if bandCells < widthCells {
		n := ceilQuo(widthCells, bandCells)
		bandCells = widthCells / n
		if bandCells < 2 {
			bandCells = 2
		}
		f.elemTile = bandCells * vecElems * 2
		f.elemTiles = f.From.Width / f.elemTile
		f.elemScrap = f.From.Width % f.elemTile
		f.elemHull = f.elemTiles
		if f.elemScrap > 0 {
			if f.elemScrap > f.elemTile-vecElems*2 {
				f.elemHull++
			} else {
				f.elemTiles--
				f.elemScrap += f.elemTile
			}
		}
	} else {
		bandCells = widthCells
		f.elemTile = f.From.Width
		f.elemTiles = 1
		f.elemScrap = 0
		f.elemHull = 1
	}
	var (
		threadBands = ceilQuo(threadCells, bandCells)
		heightBands = (f.From.Height+f.PaddingH*2-3)/2 + 1
	)
	if threadBands < heightBands {
		fit := heightBands / threadBands
		f.bandTile = heightBands / fit
		f.bandTiles = fit
		f.bandScrap = heightBands - f.bandTile*fit
		f.bandHull = fit
		if f.bandScrap > 0 {
			f.bandTiles--
			f.bandScrap += f.bandTile
		}
		f.chanTile = 1
		f.chanTiles = f.Channels
		f.chanScrap = 0
	} else {
		f.bandTile = heightBands
		f.bandTiles = 1
		f.bandScrap = 0
		f.bandHull = 1
		f.chanTile = ceilQuo(threadBands, heightBands)
		f.chanTiles = f.Channels / f.chanTile
		f.chanScrap = f.Channels % f.chanTile
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
					il(f.elemHull),
					il(chanHull),
				},
				Team: team,
			},
		},
	}.Append(to)
}

func (f *funcDefs) calleeFunc() cgen.Gen {
	var (
		body    = make(cgen.Stmts, 7)
		tensors = vb(f.name("tensors"))
		b       = vb(f.name("b"))
		e       = vb(f.name("e"))
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
		Init: cgen.Elem{Arr: callee.Pt, Idx: il(0)},
	}
	body[2] = cgen.Var{
		Type: cgen.PtrdiffT, What: e,
		Init: cgen.Elem{Arr: callee.Pt, Idx: il(1)},
	}
	body[3] = cgen.Var{
		Type: cgen.PtrdiffT, What: c,
		Init: cgen.Elem{Arr: callee.Pt, Idx: il(2)},
	}
	body[4] = f.ptrs(tensors, b, e, c)
	layer4 := func(cases int, lo, hi bool) cgen.Gen {
		var (
			stmts  = make(cgen.Stmts, 3)
			pad    = f.PaddingH
			loEdge = lo && pad > 0
			hiEdge = hi && (f.From.Height+pad*2-3)%2 < pad
		)
		if loEdge {
			f.loEdgeH = true
			f.hiEdgeH = hiEdge && cases == 1
			stmts[0] = f.kernel()
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
		if hiEdge {
			f.loEdgeH = false
			f.hiEdgeH = true
			stmts[1] = f.kernel()
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
		f.loEdgeH = false
		f.hiEdgeH = false
		stmts[2] = f.kernel()
		return stmts
	}
	layer3 := func() cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if f.bandTiles > 0 {
			f.bands = f.bandTile
			if f.bandScrap > 0 {
				stmts[0] = cgen.If{
					Cond: cgen.CmpL{
						Expr1: b,
						Expr2: il(f.bandTiles),
					},
					Then: cgen.Stmts{
						layer4(f.bandTiles, true, false),
						cgen.Return{},
					},
				}
			} else {
				stmts[0] = layer4(f.bandTiles, true, true)
			}
		}
		if f.bandScrap > 0 {
			f.bands = f.bandScrap
			lo := f.bandTiles == 0
			stmts[1] = layer4(1, lo, true)
		}
		return stmts
	}
	layer2 := func(cases int, lo, hi bool) cgen.Gen {
		var (
			stmts  = make(cgen.Stmts, 3)
			loEdge = lo && f.PaddingW > 0
			hiEdge = hi
		)
		if loEdge {
			f.loEdgeW = true
			f.hiEdgeW = hiEdge && cases == 1
			stmts[0] = layer3()
			if cases--; cases == 0 {
				return stmts
			}
			stmts[0] = cgen.If{
				Cond: cgen.IsZero{Expr: e},
				Then: cgen.Stmts{
					stmts[0],
					cgen.Return{},
				},
			}
		}
		if hiEdge {
			f.loEdgeW = false
			f.hiEdgeW = true
			stmts[1] = layer3()
			if cases--; cases == 0 {
				return stmts
			}
			stmts[1] = cgen.If{
				Cond: cgen.CmpE{
					Expr1: e,
					Expr2: il(f.elemHull - 1),
				},
				Then: cgen.Stmts{
					stmts[1],
					cgen.Return{},
				},
			}
		}
		f.loEdgeW = false
		f.hiEdgeW = false
		stmts[2] = layer3()
		return stmts
	}
	layer1 := func() cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if f.elemTiles > 0 {
			f.elems = f.elemTile
			if f.elemScrap > 0 {
				stmts[0] = cgen.If{
					Cond: cgen.CmpL{
						Expr1: e,
						Expr2: il(f.elemTiles),
					},
					Then: cgen.Stmts{
						layer2(f.elemTiles, true, false),
						cgen.Return{},
					},
				}
			} else {
				stmts[0] = layer2(f.elemTiles, true, true)
			}
		}
		if f.elemScrap > 0 {
			f.elems = f.elemScrap
			lo := f.elemTiles == 0
			stmts[1] = layer2(1, lo, true)
		}
		return stmts
	}
	if f.chanTiles > 0 {
		f.chans = f.chanTile
		body[5] = layer1()
		if f.chanScrap > 0 {
			body[5] = cgen.If{
				Cond: cgen.CmpL{
					Expr1: c,
					Expr2: il(f.chanTiles),
				},
				Then: cgen.Stmts{
					body[5],
					cgen.Return{},
				},
			}
		}
	}
	if f.chanScrap > 0 {
		f.chans = f.chanScrap
		body[6] = layer1()
	}
	return callee.Func(body)
}

func (f *funcDefs) ptrs(tensors, b, e, c cgen.Gen) cgen.Gen {
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
			ePitch = f.elemTile * f.elemBytes
			cPitch = f.chanTile
		)
		if datPtrIdx++; i < n {
			pitch1 := f.From.Pitch1Bytes[i]
			expr = cgen.Sub{
				Expr1: expr,
				Expr2: cast(pitch1 * f.PaddingH),
			}
			bPitch *= pitch1 * 2
			cPitch *= f.From.Pitch2Bytes[i]
		} else {
			bPitch *= f.To.Pitch1Bytes[i-n]
			ePitch /= 2
			cPitch *= f.To.Pitch2Bytes[i-n]
		}
		expr = addr(expr, cast(bPitch), b)
		expr = addr(expr, cast(ePitch), e)
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

func (f *funcDefs) kernel() cgen.Gen {
	var (
		i      = vb(f.name("i"))
		bnMuls []cgen.Gen
		bnAdds []cgen.Gen
		j      cgen.Gen
		loEdge bool
		hiEdge bool
	)
	layer3 := func() cgen.Gen {
		switch f.platform {
		case raw.AVX512Float32:
			return &m512Band{
				Ctx:      f.Ctx,
				Spec:     f.Spec,
				Elems:    f.elems,
				LoEdgeH:  loEdge,
				HiEdgeH:  hiEdge,
				LoEdgeW:  f.loEdgeW,
				HiEdgeW:  f.hiEdgeW,
				Ptrs:     f.datPtrs,
				PtrSplit: f.datSplit,
				ChanIdx:  i,
				BandIdx:  j,
				BnMuls:   bnMuls,
				BnAdds:   bnAdds,
				BnSplit:  f.bnSplit,
			}
		default:
			panic("bug")
		}
	}
	layer2 := func() cgen.Gen {
		var (
			stmts = make(cgen.Stmts, 3)
			first = 0
			past  = f.bands
		)
		if f.loEdgeH {
			j = il(first)
			first++
			loEdge = true
			hiEdge = f.hiEdgeH && first == past
			stmts[0] = layer3()
		}
		if f.hiEdgeH && first < past {
			past--
			j = il(past)
			loEdge = false
			hiEdge = true
			stmts[2] = layer3()
		}
		if first < past {
			j = vb(f.name("j"))
			loEdge = false
			hiEdge = false
			stmts[1] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT,
					What: j,
					Init: il(first),
				},
				Cond: cgen.CmpL{
					Expr1: j,
					Expr2: il(past),
				},
				Post: cgen.IncPre{
					Expr: j,
				},
				Body: layer3(),
			}
		}
		return stmts
	}
	layer1 := func() cgen.Gen {
		n := len(f.bnPtrs)
		bnMuls = make([]cgen.Gen, n)
		bnAdds = make([]cgen.Gen, n)
		stmts := make(cgen.Stmts, n+1)
		for x, ptr := range f.bnPtrs {
			var (
				bnMul = vb(f.name("bnMul"))
				bnAdd = vb(f.name("bnAdd"))
			)
			bnMuls[x] = bnMul
			bnAdds[x] = bnAdd
			stmts[x] = &bn.Load{
				Ctx:     f.bc,
				Mas:     ptr,
				Channel: i,
				Mul:     bnMul,
				Add:     bnAdd,
			}
		}
		stmts[n] = layer2()
		return stmts
	}
	return cgen.For{
		Init: cgen.Var{
			Type: cgen.PtrdiffT,
			What: i,
			Init: il(0),
		},
		Cond: cgen.CmpL{
			Expr1: i,
			Expr2: il(f.chans),
		},
		Post: cgen.IncPre{
			Expr: i,
		},
		Body: layer1(),
	}
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

func (m *m512LdSt) stmt(i int, s cgen.Gen) {
	m.stmts[i] = append(m.stmts[i], s)
}

func (m *m512LdSt) ld(dat cgen.Gen) {
	ptr := m.Ptrs[0]
	m.Ptrs = m.Ptrs[1:]
	m.stmt(0, cgen.Var{
		Type: avx.M512, What: dat,
		Init: avx.Mm512MaskzLoaduPs{
			m.mask, ptr,
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
	for _, ptr := range m.Ptrs {
		m.stmt(2, avx.Mm512MaskStoreuPs{
			ptr, m.mask, m.Dat,
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

func m512Combine(s *Spec, to, from cgen.Gen) cgen.Gen {
	assn := cgen.Assign{Expr1: to}
	switch s.Kind {
	case raw.Avg3x3Stride2:
		assn.Expr2 = avx.Mm512AddPs{to, from}
	case raw.Max3x3Stride2:
		assn.Expr2 = avx.Mm512MaxPs{to, from}
	default:
		panic("bug")
	}
	return assn
}

type m512Flatten struct {
	*Ctx
	*Spec
	LoEdgeH bool
	HiEdgeH bool
	Lanes   int
	Ptrs    []cgen.Gen
	ChanIdx cgen.Gen
	BandIdx cgen.Gen
	CellIdx cgen.Gen
	VecIdx  int
	BnMuls  []cgen.Gen
	BnAdds  []cgen.Gen
	Dat     cgen.Gen
}

func (m *m512Flatten) Stmts() [3]cgen.Stmts {
	var (
		stmts [3]cgen.Stmts
		toMix [3][3]cgen.Stmts
		first = 0
		past  = 3
	)
	if m.LoEdgeH {
		first = m.PaddingH
	}
	if m.HiEdgeH {
		pad := m.PaddingH
		cut := (m.From.Height+pad*2-3)%2 - pad
		if cut < 0 {
			past += cut
		}
	}
	for h := first; h < past; h++ {
		dat := m.Dat
		if h > first {
			dat = vb(m.name("dat"))
		}
		ld := &m512LdSt{
			Ctx:    m.Ctx,
			Spec:   m.Spec,
			Lanes:  m.Lanes,
			Ptrs:   make([]cgen.Gen, len(m.Ptrs)),
			BnMuls: m.BnMuls,
			BnAdds: m.BnAdds,
			Dat:    dat,
			LdDat:  true,
		}
		for i, ptr := range m.Ptrs {
			const vecBytes = 64
			var (
				pitch1 = m.From.Pitch1Bytes[i]
				pitch2 = m.From.Pitch2Bytes[i]
			)
			ptr = cgen.Add{
				Expr1: ptr,
				Expr2: cast(pitch1*h + vecBytes*m.VecIdx),
			}
			ptr = addr(ptr, cast(pitch2), m.ChanIdx)
			ptr = addr(ptr, cast(pitch1*2), m.BandIdx)
			ptr = addr(ptr, cast(vecBytes*2), m.CellIdx)
			ld.Ptrs[i] = ptr
		}
		for i, ss := range ld.Stmts() {
			toMix[i][h] = ss
		}
		if h > first {
			ss := &toMix[1][h]
			*ss = append(*ss, m512Combine(
				m.Spec, m.Dat, dat,
			))
		}
	}
	for i := range &stmts {
		stmts[i] = mix(
			toMix[i][first:past],
		)
	}
	return stmts
}

type m512Emit struct {
	*Ctx
	*Spec
	LoEdgeH bool
	HiEdgeH bool
	LoEdgeW bool
	HiEdgeW bool
	Ptrs    []cgen.Gen
	ChanIdx cgen.Gen
	BandIdx cgen.Gen
	CellIdx cgen.Gen
	BnMuls  []cgen.Gen
	BnAdds  []cgen.Gen
	Dat     cgen.Gen
}

func (m *m512Emit) height() int {
	ret := 3
	if m.LoEdgeH {
		ret -= m.PaddingH
	}
	if m.HiEdgeH {
		pad := m.PaddingH
		cut := (m.From.Height+pad*2-3)%2 - pad
		if cut < 0 {
			ret += cut
		}
	}
	return ret
}

func (m *m512Emit) width() int {
	if m.HiEdgeW {
		var (
			padded = m.From.Width + m.PaddingW*2
			rem    = ((padded-3)/2 + 1) % 16
		)
		if rem > 0 {
			return rem
		}
	}
	return 16
}

func (m *m512Emit) divide() cgen.Gen {
	var (
		h  = m.height()
		w  = m.width()
		ns = make([]int, w)
	)
	for i := range ns {
		ns[i] = 3
	}
	if m.LoEdgeW {
		ns[0] -= m.PaddingW
	}
	if m.HiEdgeW {
		pad := m.PaddingW
		cut := (m.From.Width+pad*2-3)%2 - pad
		if cut < 0 {
			ns[w-1] += cut
		}
	}
	n := ns[0]
	for i := 1; i < w; i++ {
		if ns[i] == n {
			continue
		}
		var (
			rcp = vb(m.name("rcp"))
			set = make(avx.Mm512SetPs, 16)
		)
		for j := 0; j < w; j++ {
			denom := float32(ns[j] * h)
			set[15-j] = fl(1 / denom)
		}
		for j := 15 - w; j >= 0; j-- {
			set[j] = cgen.Zero
		}
		return cgen.Stmts{
			cgen.Var{
				Type: avx.M512, What: rcp,
				Init: set,
			},
			cgen.Assign{
				Expr1: m.Dat,
				Expr2: avx.Mm512MulPs{
					m.Dat, rcp,
				},
			},
		}
	}
	if n *= h; n == 1 {
		return nil
	}
	denom := avx.Mm512Set1PsLit(n)
	return cgen.Assign{
		Expr1: m.Dat,
		Expr2: avx.Mm512MulPs{
			m.Dat, 1 / denom,
		},
	}
}

func (m *m512Emit) Stmts() [3]cgen.Stmts {
	st := &m512LdSt{
		Ctx:    m.Ctx,
		Spec:   m.Spec,
		Lanes:  m.width(),
		Ptrs:   make([]cgen.Gen, len(m.Ptrs)),
		BnMuls: m.BnMuls,
		BnAdds: m.BnAdds,
		Dat:    m.Dat,
	}
	for i, ptr := range m.Ptrs {
		var (
			pitch2 = cast(m.To.Pitch2Bytes[i])
			pitch1 = cast(m.To.Pitch1Bytes[i])
			stride = cast(64)
		)
		ptr = addr(ptr, pitch2, m.ChanIdx)
		ptr = addr(ptr, pitch1, m.BandIdx)
		ptr = addr(ptr, stride, m.CellIdx)
		st.Ptrs[i] = ptr
	}
	ret := st.Stmts()
	if m.Kind == raw.Avg3x3Stride2 {
		if div := m.divide(); div != nil {
			ss := ret[1]
			ss = append(ss, nil)
			copy(ss[1:], ss)
			ss[0] = div
			ret[1] = ss
		}
	}
	return ret
}

type m512Band struct {
	*Ctx
	*Spec
	Elems    int
	LoEdgeH  bool
	HiEdgeH  bool
	LoEdgeW  bool
	HiEdgeW  bool
	Ptrs     []cgen.Gen
	PtrSplit int
	ChanIdx  cgen.Gen
	BandIdx  cgen.Gen
	BnMuls   []cgen.Gen
	BnAdds   []cgen.Gen
	BnSplit  int
}

func (m *m512Band) Append(to []byte) []byte {
	var gen cgen.Gen
	switch m.PaddingW {
	case 0:
		gen = &m512Pad0W{m512Band: m}
	case 1:
		gen = &m512Pad1W{m512Band: m}
	case 2:
		gen = &m512Pad2W{m512Band: m}
	}
	return gen.Append(to)
}

type m512Pad0W struct {
	*m512Band
	at      int
	last    int
	avail   int
	take    int
	window  int
	cellIdx cgen.Gen
	in      [3]cgen.Gen
	out     cgen.Gen
	pm1     cgen.Gen
	pm2     cgen.Gen
	pm3     cgen.Gen
}

func (m *m512Pad0W) part1() (prep, eval cgen.Gen) {
	var (
		w = m.window
		x = m.in[0]
	)
	switch {
	case w < 5:
		eval = x
	case w < 7:
		eval = avx.Mm512ShufflePs{
			x, x, il(2<<2 | 0),
		}
	default:
		if m.pm1 == nil {
			m.pm1 = vb(m.name("pm"))
			set := make(avx.Mm512SetEpi32, 16)
			for i := 0; i < 16; i++ {
				set[15-i] = il(i * 2)
			}
			prep = cgen.Var{
				Type: avx.M512i, What: m.pm1,
				Init: set,
			}
		}
		if w < 19 {
			eval = avx.Mm512PermutexvarPs{
				m.pm1, x,
			}
		} else {
			eval = avx.Mm512Permutex2varPs{
				x, m.pm1, m.in[1],
			}
		}
	}
	return
}

func (m *m512Pad0W) part2() (prep, eval cgen.Gen) {
	var (
		w = m.window
		x = m.in[0]
	)
	switch {
	case w < 7:
		eval = avx.Mm512ShufflePs{
			x, x, il(3<<2 | 1),
		}
	default:
		if m.pm2 == nil {
			m.pm2 = vb(m.name("pm"))
			set := make(avx.Mm512SetEpi32, 16)
			for i := 0; i < 16; i++ {
				set[15-i] = il(1 + i*2)
			}
			prep = cgen.Var{
				Type: avx.M512i, What: m.pm2,
				Init: set,
			}
		}
		if w < 19 {
			eval = avx.Mm512PermutexvarPs{
				m.pm2, x,
			}
		} else {
			eval = avx.Mm512Permutex2varPs{
				x, m.pm2, m.in[1],
			}
		}
	}
	return
}

func (m *m512Pad0W) part3() (prep, eval cgen.Gen) {
	var (
		w = m.window
		x = m.in[0]
	)
	switch {
	case w < 5:
		eval = avx.Mm512ShufflePs{
			x, x, il(2),
		}
	default:
		stmts := make(cgen.Stmts, 2)
		if m.pm3 == nil {
			m.pm3 = vb(m.name("pm"))
			set := make(avx.Mm512SetEpi32, 16)
			for i := 0; i < 15; i++ {
				set[15-i] = il(2 + i*2)
			}
			set[0] = il(0)
			stmts[0] = cgen.Var{
				Type: avx.M512i, What: m.pm3,
				Init: set,
			}
		}
		if w < 17 {
			eval = avx.Mm512PermutexvarPs{
				m.pm3, x,
			}
		} else {
			blend := x
			if w > 32 {
				blend = vb(m.name("blend"))
				stmts[1] = cgen.Var{
					Type: avx.M512,
					What: blend,
					Init: avx.Mm512MaskMovPs{
						x, il(1), m.in[2],
					},
				}
			}
			eval = avx.Mm512Permutex2varPs{
				blend, m.pm3, m.in[1],
			}
		}
		prep = stmts
	}
	return
}

func (m *m512Pad0W) core() cgen.Gen {
	var (
		stmts = make(cgen.Stmts, 8)
		evals [3]cgen.Gen
	)
	stmts[0], evals[0] = m.part1()
	stmts[1], evals[1] = m.part2()
	stmts[2], evals[2] = m.part3()
	for i := 0; i < 3; i++ {
		pack := m.out
		if i > 0 {
			pack = vb(m.name("pack"))
		}
		stmts[3+i] = cgen.Var{
			Type: avx.M512, What: pack,
			Init: evals[i],
		}
		if i > 0 {
			stmts[5+i] = m512Combine(
				m.Spec, m.out, pack,
			)
		}
	}
	return stmts
}

func (m *m512Pad0W) cell() cgen.Gen {
	var (
		stmts [3]cgen.Stmts
		toMix [3][3]cgen.Stmts
		start = 0
	)
	if m.at > 0 {
		start = 1
	}
	for i := start; m.take > 0; i++ {
		lanes := m.take
		if lanes > 16 {
			lanes = 16
		}
		m.take -= lanes
		m.in[i] = vb(m.name("in"))
		flatten := &m512Flatten{
			Ctx:     m.Ctx,
			Spec:    m.Spec,
			LoEdgeH: m.LoEdgeH,
			HiEdgeH: m.HiEdgeH,
			Lanes:   lanes,
			Ptrs:    m.Ptrs[:m.PtrSplit],
			ChanIdx: m.ChanIdx,
			BandIdx: m.BandIdx,
			CellIdx: m.cellIdx,
			VecIdx:  i,
			BnMuls:  m.BnMuls[:m.BnSplit],
			BnAdds:  m.BnAdds[:m.BnSplit],
			Dat:     m.in[i],
		}
		for j, ss := range flatten.Stmts() {
			toMix[j][i] = ss
		}
	}
	for i := range &stmts {
		stmts[i] = mix(toMix[i][:])
	}
	m.out = vb(m.name("out"))
	stmts[1] = append(
		stmts[1], m.core(),
	)
	if m.at < m.last {
		stmts[1] = append(
			stmts[1], cgen.Assign{
				Expr1: m.in[0],
				Expr2: m.in[2],
			},
		)
	}
	emit := &m512Emit{
		Ctx:     m.Ctx,
		Spec:    m.Spec,
		LoEdgeH: m.LoEdgeH,
		HiEdgeH: m.HiEdgeH,
		LoEdgeW: m.LoEdgeW && m.at == 0,
		HiEdgeW: m.HiEdgeW && m.at == m.last,
		Ptrs:    m.Ptrs[m.PtrSplit:],
		ChanIdx: m.ChanIdx,
		BandIdx: m.BandIdx,
		CellIdx: m.cellIdx,
		BnMuls:  m.BnMuls[m.BnSplit:],
		BnAdds:  m.BnAdds[m.BnSplit:],
		Dat:     m.out,
	}
	for i, ss := range emit.Stmts() {
		stmts[i] = append(
			stmts[i], ss,
		)
	}
	return cgen.Gens{
		stmts[0],
		stmts[1],
		stmts[2],
	}
}

func (m *m512Pad0W) shift() bool {
	m.at++
	if m.at > m.last {
		return false
	}
	n := 48
	if m.at > 0 {
		n = 32
		m.window -= n
	}
	if m.avail < n {
		m.take = m.avail
	} else {
		m.take = n
	}
	m.avail -= m.take
	m.window += m.take
	return true
}

func (m *m512Pad0W) pre() cgen.Gen {
	m.cellIdx = il(0)
	return m.cell()
}

func (m *m512Pad0W) loop() cgen.Gen {
	if !m.shift() || m.window != 48 {
		return nil
	}
	m.cellIdx = vb(m.name("k"))
	body := m.cell()
	for m.shift() && m.window == 48 {
	}
	return cgen.For{
		Init: cgen.Var{
			Type: cgen.PtrdiffT,
			What: m.cellIdx,
			Init: il(1),
		},
		Cond: cgen.CmpL{
			Expr1: m.cellIdx,
			Expr2: il(m.at),
		},
		Post: cgen.IncPre{
			Expr: m.cellIdx,
		},
		Body: body,
	}
}

func (m *m512Pad0W) post() cgen.Gen {
	if m.at > m.last {
		return nil
	}
	ret := make(cgen.Gens, 2)
	m.cellIdx = il(m.at)
	ret[0] = m.cell()
	if m.shift() {
		m.cellIdx = il(m.at)
		ret[1] = m.cell()
	}
	return ret
}

func (m *m512Pad0W) Append(to []byte) []byte {
	m.at = -1
	if m.HiEdgeW {
		yield := (m.Elems-3)/2 + 1
		m.last = ceilQuo(yield, 16) - 1
		m.avail = m.Elems
	} else {
		m.last = m.Elems/32 - 1
		m.avail = m.Elems + 16
	}
	m.shift()
	return cgen.Stmts{
		m.pre(),
		m.loop(),
		m.post(),
	}.Append(to)
}

type m512Pad1W struct {
	*m512Band
	cells   int
	final   int
	head    bool
	tail    bool
	cellIdx cgen.Gen
	in      [3]cgen.Gen
	out     cgen.Gen
	pm1     cgen.Gen
	pm2     cgen.Gen
	pm3     cgen.Gen
}

func (m *m512Pad1W) nFwd() int {
	if m.tail {
		return m.final
	}
	return 32
}

func (m *m512Pad1W) part1() (prep, eval cgen.Gen) {
	var (
		n = m.nFwd()
		y = m.in[1]
	)
	switch {
	case n < 3:
		eval = y
	case n < 5:
		eval = avx.Mm512ShufflePs{
			y, y, il(2<<2 | 0),
		}
	default:
		if m.pm1 == nil {
			m.pm1 = vb(m.name("pm"))
			set := make(avx.Mm512SetEpi32, 16)
			for i := 0; i < 16; i++ {
				set[15-i] = il(i * 2)
			}
			prep = cgen.Var{
				Type: avx.M512i, What: m.pm1,
				Init: set,
			}
		}
		if n < 17 {
			eval = avx.Mm512PermutexvarPs{
				m.pm1, y,
			}
		} else {
			eval = avx.Mm512Permutex2varPs{
				y, m.pm1, m.in[2],
			}
		}
	}
	return
}

func (m *m512Pad1W) part2() (prep, eval cgen.Gen) {
	var (
		n = m.nFwd()
		y = m.in[1]
	)
	switch {
	case n < 2:
	case n < 6:
		eval = avx.Mm512ShufflePs{
			y, y, il(3<<2 | 1),
		}
	default:
		if m.pm2 == nil {
			m.pm2 = vb(m.name("pm"))
			set := make(avx.Mm512SetEpi32, 16)
			for i := 0; i < 16; i++ {
				set[15-i] = il(1 + i*2)
			}
			prep = cgen.Var{
				Type: avx.M512i, What: m.pm2,
				Init: set,
			}
		}
		if n < 18 {
			eval = avx.Mm512PermutexvarPs{
				m.pm2, y,
			}
		} else {
			eval = avx.Mm512Permutex2varPs{
				y, m.pm2, m.in[2],
			}
		}
	}
	return
}

func (m *m512Pad1W) part3() (prep, eval cgen.Gen) {
	var (
		n = m.nFwd()
		x = m.in[0]
		y = m.in[1]
	)
	if x == nil {
		switch {
		case n < 3:
			return
		case n < 7:
			eval = avx.Mm512ShufflePs{
				y, y, il(3<<4 | 1<<2),
			}
			return
		}
	}
	var (
		stmts = make(cgen.Stmts, 2)
		short = false
		blend cgen.Gen
	)
	if m.pm3 == nil {
		m.pm3 = vb(m.name("pm"))
		set := make(avx.Mm512SetEpi32, 16)
		set[15] = il(31)
		for i := 1; i < 16; i++ {
			set[15-i] = il(i*2 - 1)
		}
		stmts[0] = cgen.Var{
			Type: avx.M512i, What: m.pm3,
			Init: set,
		}
	}
	if x == nil {
		if n < 19 {
			short = true
			blend = y
		} else {
			blend = m.in[2]
		}
	} else {
		var mov cgen.Gen
		switch {
		case n < 3:
			short = true
			blend = x
		case n < 17:
			short = true
			mov = avx.Mm512MaskMovPs{
				y, il(1 << 15), x,
			}
		case n < 19:
			blend = x
		default:
			mov = avx.Mm512MaskMovPs{
				m.in[2], il(1 << 15), x,
			}
		}
		if mov != nil {
			blend = vb(m.name("blend"))
			stmts[1] = cgen.Var{
				Type: avx.M512, What: blend,
				Init: mov,
			}
		}
	}
	prep = stmts
	if short {
		eval = avx.Mm512PermutexvarPs{
			m.pm3, blend,
		}
	} else {
		eval = avx.Mm512Permutex2varPs{
			y, m.pm3, blend,
		}
	}
	return
}

func (m *m512Pad1W) core() cgen.Gen {
	var (
		stmts = make(cgen.Stmts, 8)
		evals [3]cgen.Gen
	)
	stmts[0], evals[0] = m.part1()
	stmts[1], evals[1] = m.part2()
	stmts[2], evals[2] = m.part3()
	for i := 0; i < 3; i++ {
		var pack cgen.Gen
		if i == 0 {
			pack = m.out
		} else if evals[i] == nil {
			continue
		} else {
			pack = vb(m.name("pack"))
		}
		stmts[3+i] = cgen.Var{
			Type: avx.M512, What: pack,
			Init: evals[i],
		}
		if i == 0 {
			continue
		}
		var (
			bits int
			call cgen.Gen
		)
		if i == 1 {
			n := m.nFwd() / 2
			bits = 1<<uint(n) - 1
		} else {
			n := ceilQuo(m.nFwd(), 2)
			bits = 1<<uint(n) - 1
			if m.in[0] == nil {
				bits--
			}
		}
		args := []cgen.Gen{
			m.out, il(bits),
			m.out, pack,
		}
		switch m.Kind {
		case raw.Avg3x3Stride2:
			call = avx.Mm512MaskAddPs(args)
		case raw.Max3x3Stride2:
			call = avx.Mm512MaskMaxPs(args)
		default:
			panic("bug")
		}
		stmts[5+i] = cgen.Assign{
			Expr1: m.out,
			Expr2: call,
		}
	}
	return stmts
}

func (m *m512Pad1W) cell() cgen.Gen {
	var (
		stmts [3]cgen.Stmts
		toMix [3][3]cgen.Stmts
		start = 1
		take  = m.nFwd()
	)
	if m.head && !m.LoEdgeW {
		start--
		take += 16
	}
	for i := start; take > 0; i++ {
		lanes := take
		if lanes > 16 {
			lanes = 16
		}
		take -= lanes
		m.in[i] = vb(m.name("in"))
		flatten := &m512Flatten{
			Ctx:     m.Ctx,
			Spec:    m.Spec,
			LoEdgeH: m.LoEdgeH,
			HiEdgeH: m.HiEdgeH,
			Lanes:   lanes,
			Ptrs:    m.Ptrs[:m.PtrSplit],
			ChanIdx: m.ChanIdx,
			BandIdx: m.BandIdx,
			CellIdx: m.cellIdx,
			VecIdx:  i - 1,
			BnMuls:  m.BnMuls[:m.BnSplit],
			BnAdds:  m.BnAdds[:m.BnSplit],
			Dat:     m.in[i],
		}
		for j, ss := range flatten.Stmts() {
			toMix[j][i] = ss
		}
	}
	for i := range &stmts {
		stmts[i] = mix(toMix[i][start:])
	}
	m.out = vb(m.name("out"))
	stmts[1] = append(
		stmts[1], m.core(),
	)
	if !m.tail {
		if m.head {
			m.in[0] = m.in[2]
		} else {
			stmts[1] = append(
				stmts[1], cgen.Assign{
					Expr1: m.in[0],
					Expr2: m.in[2],
				},
			)
		}
	}
	emit := &m512Emit{
		Ctx:     m.Ctx,
		Spec:    m.Spec,
		LoEdgeH: m.LoEdgeH,
		HiEdgeH: m.HiEdgeH,
		LoEdgeW: m.LoEdgeW && m.head,
		HiEdgeW: m.HiEdgeW && m.tail,
		Ptrs:    m.Ptrs[m.PtrSplit:],
		ChanIdx: m.ChanIdx,
		BandIdx: m.BandIdx,
		CellIdx: m.cellIdx,
		BnMuls:  m.BnMuls[m.BnSplit:],
		BnAdds:  m.BnAdds[m.BnSplit:],
		Dat:     m.out,
	}
	for i, ss := range emit.Stmts() {
		stmts[i] = append(
			stmts[i], ss,
		)
	}
	return cgen.Gens{
		stmts[0],
		stmts[1],
		stmts[2],
	}
}

func (m *m512Pad1W) pre() cgen.Gen {
	m.head = true
	m.tail = m.cells == 1
	m.cellIdx = il(0)
	return m.cell()
}

func (m *m512Pad1W) loop() cgen.Gen {
	if m.cells < 3 {
		return nil
	}
	stop := m.cells
	if m.final < 32 {
		stop--
	}
	m.head = false
	m.tail = false
	m.cellIdx = vb(m.name("k"))
	return cgen.For{
		Init: cgen.Var{
			Type: cgen.PtrdiffT,
			What: m.cellIdx,
			Init: il(1),
		},
		Cond: cgen.CmpL{
			Expr1: m.cellIdx,
			Expr2: il(stop),
		},
		Post: cgen.IncPre{
			Expr: m.cellIdx,
		},
		Body: m.cell(),
	}
}

func (m *m512Pad1W) post() cgen.Gen {
	if m.cells == 1 {
		return nil
	}
	if m.cells >= 3 && m.final == 32 {
		return nil
	}
	m.head = false
	m.tail = true
	m.cellIdx = il(m.cells - 1)
	return m.cell()
}

func (m *m512Pad1W) Append(to []byte) []byte {
	var (
		full = m.Elems / 32
		part = m.Elems % 32
	)
	if part == 0 {
		m.cells = full
		m.final = 32
	} else {
		m.cells = full + 1
		m.final = part
	}
	return cgen.Stmts{
		m.pre(),
		m.loop(),
		m.post(),
	}.Append(to)
}

type m512Pad2W struct {
	*m512Band
	curr    int
	last    int
	at      int
	cellIdx cgen.Gen
	in      [3]cgen.Gen
	out     cgen.Gen
	blend   [2]cgen.Gen
	pm1     cgen.Gen
	pm2     cgen.Gen
	pm3     cgen.Gen
}

func (m *m512Pad2W) nFwd() int {
	n := m.Elems - m.at
	if n < 0 {
		n = 0
	} else if n > 32 {
		n = 32
	}
	return n
}

func (m *m512Pad2W) have(offset int) bool {
	i := m.at + offset
	if i < 0 {
		return !m.LoEdgeW
	}
	return i < m.Elems
}

func (m *m512Pad2W) mask(offset int) int {
	ret := 0
	for i := 0; i < 16; i++ {
		if m.have(offset + i*2) {
			ret |= 1 << uint(i)
		}
	}
	return ret
}

func (m *m512Pad2W) part1() (prep cgen.Gen) {
	var (
		x = m.in[0]
		y = m.in[1]
		z = m.in[2]
	)
	if x != nil {
		switch n := m.nFwd(); {
		case n < 1:
			y = x
		case n < 17:
			z = x
		default:
			b := vb(m.name("blend"))
			prep = cgen.Var{
				Type: avx.M512, What: b,
				Init: avx.Mm512MaskMovPs{
					z, il(3 << 14), x,
				},
			}
			z = b
		}
	}
	m.blend[0] = y
	m.blend[1] = z
	return
}

func (m *m512Pad2W) part2() (prep, eval cgen.Gen) {
	var (
		x = m.in[0]
		n = m.nFwd()
		y = m.blend[0]
	)
	if x == nil {
		if n < 5 {
			eval = avx.Mm512ShufflePs{
				y, y, il(2<<4 | 0<<2),
			}
			return
		}
	}
	if m.pm1 == nil {
		m.pm1 = vb(m.name("pm"))
		set := make(avx.Mm512SetEpi32, 16)
		set[15] = il(30)
		for i := 1; i < 16; i++ {
			set[15-i] = il(i*2 - 2)
		}
		prep = cgen.Var{
			Type: avx.M512i, What: m.pm1,
			Init: set,
		}
	}
	var short bool
	if x == nil {
		short = n < 17
	} else {
		short = n < 1
	}
	if short {
		eval = avx.Mm512PermutexvarPs{
			m.pm1, y,
		}
	} else {
		eval = avx.Mm512Permutex2varPs{
			y, m.pm1, m.blend[1],
		}
	}
	return
}

func (m *m512Pad2W) part3() (prep, eval cgen.Gen) {
	n := m.nFwd()
	if n < 1 {
		return
	}
	y := m.in[1]
	switch {
	case n < 3:
		eval = y
	case n < 5:
		eval = avx.Mm512ShufflePs{
			y, y, il(2<<2 | 0<<0),
		}
	default:
		if m.pm2 == nil {
			m.pm2 = vb(m.name("pm"))
			set := make(avx.Mm512SetEpi32, 16)
			for i := 0; i < 16; i++ {
				set[15-i] = il(i*2 - 0)
			}
			prep = cgen.Var{
				Type: avx.M512i, What: m.pm2,
				Init: set,
			}
		}
		if n < 17 {
			eval = avx.Mm512PermutexvarPs{
				m.pm2, y,
			}
		} else {
			eval = avx.Mm512Permutex2varPs{
				y, m.pm2, m.in[2],
			}
		}
	}
	return
}

func (m *m512Pad2W) part4() (prep, eval cgen.Gen) {
	var (
		x = m.in[0]
		n = m.nFwd()
		y = m.blend[0]
	)
	if x == nil {
		switch {
		case n < 2:
			return
		case n < 6:
			eval = avx.Mm512ShufflePs{
				y, y, il(3<<4 | 1<<2),
			}
			return
		}
	} else if !m.have(-1) {
		return
	}
	if m.pm3 == nil {
		m.pm3 = vb(m.name("pm"))
		set := make(avx.Mm512SetEpi32, 16)
		set[15] = il(31)
		for i := 1; i < 16; i++ {
			set[15-i] = il(i*2 - 1)
		}
		prep = cgen.Var{
			Type: avx.M512i, What: m.pm3,
			Init: set,
		}
	}
	var (
		short bool
		z     = m.blend[1]
	)
	if x == nil {
		short = n < 18
	} else {
		short = n < 2
		switch n {
		case 1:
			y = x
		case 17:
			z = x
		}
	}
	if short {
		eval = avx.Mm512PermutexvarPs{
			m.pm3, y,
		}
	} else {
		eval = avx.Mm512Permutex2varPs{
			y, m.pm3, z,
		}
	}
	return
}

func (m *m512Pad2W) core() cgen.Gen {
	var (
		stmts = make(cgen.Stmts, 10)
		evals [3]cgen.Gen
	)
	stmts[0] = m.part1()
	stmts[1], evals[0] = m.part2()
	stmts[2], evals[1] = m.part3()
	stmts[3], evals[2] = m.part4()
	for i := 0; i < 3; i++ {
		var pack cgen.Gen
		if i == 0 {
			pack = m.out
		} else if evals[i] == nil {
			continue
		} else {
			pack = vb(m.name("pack"))
		}
		stmts[4+i] = cgen.Var{
			Type: avx.M512, What: pack,
			Init: evals[i],
		}
		if i == 0 {
			continue
		}
		var both int
		if i == 1 {
			var (
				this = m.mask(0)
				prev = m.mask(-2)
				fill = this &^ prev
			)
			if fill != 0 {
				stmts[7] = cgen.Assign{
					Expr1: m.out,
					Expr2: avx.Mm512MaskMovPs{
						m.out, il(fill), pack,
					},
				}
			}
			both = this & prev
			if both == 0 {
				continue
			}
		} else {
			both = m.mask(-1)
		}
		args := []cgen.Gen{
			m.out, il(both),
			m.out, pack,
		}
		var call cgen.Gen
		switch m.Kind {
		case raw.Avg3x3Stride2:
			call = avx.Mm512MaskAddPs(args)
		case raw.Max3x3Stride2:
			call = avx.Mm512MaskMaxPs(args)
		default:
			panic("bug")
		}
		stmts[7+i] = cgen.Assign{
			Expr1: m.out,
			Expr2: call,
		}
	}
	return stmts
}

func (m *m512Pad2W) cell() cgen.Gen {
	var (
		stmts [3]cgen.Stmts
		toMix [3][3]cgen.Stmts
		start = 1
		take  = m.nFwd()
	)
	if m.curr == 0 && !m.LoEdgeW {
		start--
		take += 16
	}
	for i := start; take > 0; i++ {
		lanes := take
		if lanes > 16 {
			lanes = 16
		}
		take -= lanes
		m.in[i] = vb(m.name("in"))
		flatten := &m512Flatten{
			Ctx:     m.Ctx,
			Spec:    m.Spec,
			LoEdgeH: m.LoEdgeH,
			HiEdgeH: m.HiEdgeH,
			Lanes:   lanes,
			Ptrs:    m.Ptrs[:m.PtrSplit],
			ChanIdx: m.ChanIdx,
			BandIdx: m.BandIdx,
			CellIdx: m.cellIdx,
			VecIdx:  i - 1,
			BnMuls:  m.BnMuls[:m.BnSplit],
			BnAdds:  m.BnAdds[:m.BnSplit],
			Dat:     m.in[i],
		}
		for j, ss := range flatten.Stmts() {
			toMix[j][i] = ss
		}
	}
	for i := range &stmts {
		stmts[i] = mix(toMix[i][start:])
	}
	m.out = vb(m.name("out"))
	stmts[1] = append(
		stmts[1], m.core(),
	)
	if m.curr < m.last {
		if m.curr == 0 {
			m.in[0] = m.in[2]
		} else {
			stmts[1] = append(
				stmts[1], cgen.Assign{
					Expr1: m.in[0],
					Expr2: m.in[2],
				},
			)
		}
	}
	emit := &m512Emit{
		Ctx:     m.Ctx,
		Spec:    m.Spec,
		LoEdgeH: m.LoEdgeH,
		HiEdgeH: m.HiEdgeH,
		LoEdgeW: m.LoEdgeW && m.curr == 0,
		HiEdgeW: m.HiEdgeW && m.curr == m.last,
		Ptrs:    m.Ptrs[m.PtrSplit:],
		ChanIdx: m.ChanIdx,
		BandIdx: m.BandIdx,
		CellIdx: m.cellIdx,
		BnMuls:  m.BnMuls[m.BnSplit:],
		BnAdds:  m.BnAdds[m.BnSplit:],
		Dat:     m.out,
	}
	for i, ss := range emit.Stmts() {
		stmts[i] = append(
			stmts[i], ss,
		)
	}
	return cgen.Gens{
		stmts[0],
		stmts[1],
		stmts[2],
	}
}

func (m *m512Pad2W) shift() bool {
	m.curr++
	if m.curr > m.last {
		return false
	}
	m.at += 32
	return true
}

func (m *m512Pad2W) pre() cgen.Gen {
	m.cellIdx = il(0)
	return m.cell()
}

func (m *m512Pad2W) loop() cgen.Gen {
	if !m.shift() || m.nFwd() != 32 {
		return nil
	}
	m.cellIdx = vb(m.name("k"))
	body := m.cell()
	for m.shift() && m.nFwd() == 32 {
	}
	return cgen.For{
		Init: cgen.Var{
			Type: cgen.PtrdiffT,
			What: m.cellIdx,
			Init: il(1),
		},
		Cond: cgen.CmpL{
			Expr1: m.cellIdx,
			Expr2: il(m.curr),
		},
		Post: cgen.IncPre{
			Expr: m.cellIdx,
		},
		Body: body,
	}
}

func (m *m512Pad2W) post() cgen.Gen {
	if m.curr > m.last {
		return nil
	}
	ret := make(cgen.Gens, 2)
	for i := range ret {
		m.cellIdx = il(m.curr)
		ret[i] = m.cell()
		if !m.shift() {
			break
		}
	}
	return ret
}

func (m *m512Pad2W) Append(to []byte) []byte {
	padded := 2 + m.Elems
	if m.HiEdgeW {
		padded += 2
	}
	yield := (padded-3)/2 + 1
	m.last = ceilQuo(yield, 16) - 1
	return cgen.Stmts{
		m.pre(),
		m.loop(),
		m.post(),
	}.Append(to)
}
