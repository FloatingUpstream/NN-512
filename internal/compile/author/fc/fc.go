package fc

import (
	"NN-512/internal/compile/author/act"
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/bn"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/mod"
	"NN-512/internal/compile/author/sumr"
	"NN-512/internal/compile/author/threader"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
	"fmt"
)

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

func min(x, y int) int {
	if x <= y {
		return x
	}
	return y
}

func max(x, y int) int {
	if x >= y {
		return x
	}
	return y
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

func loMask(n int) cgen.Gen {
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
		prefix:   pl.Config.Prefix + "Fc",
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

func (c *Ctx) newLayout(toC, fromC, fromH, fromW int) *layout {
	var y *layout
	pad := func(n int) int {
		n += y.alignment - 1
		n &= -y.alignment
		return n
	}
	switch c.platform {
	case raw.AVX512Float32:
		y = &layout{
			cellWeights1: 16,
			groupCells1:  16,
			alignment:    64,
			weightBytes1: 4,
			weightBytes2: 2,
			biasBytes:    4,
			datBytes:     4,
		}
	default:
		panic("bug")
	}
	y.fromHW = fromH * fromW
	y.fromCHW = fromC * y.fromHW
	y.cellWeights2 = y.fromCHW % y.cellWeights1
	y.stripGroups1 = y.fromCHW / y.cellWeights1
	y.stripGroups2 = y.stripGroups1 + btoi(y.cellWeights2 > 0)
	y.groupCells2 = toC % y.groupCells1
	y.strips1 = toC / y.groupCells1
	y.strips2 = y.strips1 + btoi(y.groupCells2 > 0)
	y.cellBytes = y.cellWeights1 * y.weightBytes2
	y.groupBytes1 = y.groupCells1 * y.cellBytes
	y.groupBytes2 = y.groupCells2 * y.cellBytes
	y.stripBytes1 = pad(y.stripGroups2 * y.groupBytes1)
	y.stripBytes2 = pad(y.stripGroups2 * y.groupBytes2)
	y.biasOffset = y.strips1*y.stripBytes1 + y.stripBytes2
	y.totalBytes = y.biasOffset + toC*y.biasBytes
	return y
}

type layout struct {
	fromHW       int
	fromCHW      int
	cellWeights1 int
	cellWeights2 int
	stripGroups1 int
	stripGroups2 int
	groupCells1  int
	groupCells2  int
	strips1      int
	strips2      int
	alignment    int
	weightBytes1 int
	weightBytes2 int
	biasBytes    int
	datBytes     int
	cellBytes    int
	groupBytes1  int
	groupBytes2  int
	stripBytes1  int
	stripBytes2  int
	biasOffset   int
	totalBytes   int
}

type Arrange struct {
	*Ctx
	ToC     int
	FromC   int
	FromH   int
	FromW   int
	BnPre   int
	BnPost  int
	Team    cgen.Gen
	Tensors []cgen.Gen
	*layout
	callerName string
}

func (a *Arrange) Prep() cgen.Gen {
	a.layout = a.newLayout(
		a.ToC, a.FromC, a.FromH, a.FromW,
	)
	const affix = "Arrange"
	sig := fmt.Sprint(
		affix, " ",
		a.ToC, a.FromC, a.FromH, a.FromW,
		a.BnPre, a.BnPost,
	)
	if prior, ok := a.dedup[sig]; ok {
		a.callerName = prior
		return nil
	}
	a.callerName = a.name(a.prefix + affix)
	a.dedup[sig] = a.callerName
	return cgen.Gens{
		&arrange{Arrange: a},
		cgen.Newline,
	}
}

func (a *Arrange) Bytes() int {
	return a.totalBytes
}

func (a *Arrange) Append(to []byte) []byte {
	var (
		tensors = vb(a.name("tensors"))
		ptrs    = cgen.CommaLines(a.Tensors)
	)
	return cgen.Stmts{
		cgen.Var{
			Type: cgen.PtrChar,
			What: cgen.Elem{Arr: tensors},
			Init: cgen.Brace{Inner: ptrs},
		},
		cgen.Call{
			Func: vb(a.callerName),
			Args: cgen.CommaSpaced{
				a.Team, tensors,
			},
		},
	}.Append(to)
}

type arrange struct {
	*Arrange
	tile       int
	tiles      int
	scrap      int
	hull1      int
	hull2      int
	calleeName string
	weights1   cgen.Gen
	biases1    cgen.Gen
	bnPtrs     []cgen.Gen
	weights2   cgen.Gen
	biases2    cgen.Gen
	strips     int
	groupCells int
}

func (a *arrange) Append(to []byte) []byte {
	var (
		threadVecs int
		stripVecs  = a.stripGroups2 * a.groupCells1
		team       = vb(a.name("team"))
		tensors    = vb(a.name("tensors"))
	)
	switch a.platform {
	case raw.AVX512Float32:
		threadVecs = 512
	default:
		panic("bug")
	}
	a.tile = ceilQuo(threadVecs, stripVecs)
	a.tiles = a.strips1 / a.tile
	a.scrap = a.strips1 % a.tile
	a.hull1 = a.tiles + btoi(a.scrap > 0)
	a.hull2 = a.hull1 + btoi(a.strips1 < a.strips2)
	a.calleeName = a.name(a.callerName + "Callee")
	return cgen.Gens{
		a.calleeFunc(),
		cgen.Newline,
		cgen.StaticFuncDef{
			ReturnType: cgen.Void,
			Name:       a.callerName,
			Params: cgen.CommaSpaced{
				cgen.Param{
					Type: a.tc.PtrTeam,
					What: team,
				},
				cgen.Param{
					Type: cgen.PtrPtrChar,
					What: tensors,
				},
			},
			Body: &threader.Do{
				Ctx:    a.tc,
				Callee: vb(a.calleeName),
				Any:    tensors,
				Hull: []cgen.Gen{
					il(a.hull2),
				},
				Team: team,
			},
		},
	}.Append(to)
}

func (a *arrange) calleeFunc() cgen.Gen {
	var (
		body    = make(cgen.Stmts, 6)
		tensors = vb(a.name("tensors"))
		t       = vb(a.name("t"))
	)
	callee := &threader.Callee{
		Ctx:  a.tc,
		Name: a.calleeName,
		Task: vb(a.name("task")),
		Pt:   vb(a.name("pt")),
	}
	body[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: tensors,
		Init: callee.Any(),
	}
	body[1] = cgen.Var{
		Type: cgen.PtrdiffT, What: t,
		Init: cgen.Elem{Arr: callee.Pt, Idx: il(0)},
	}
	body[2] = a.ptrs(tensors, t)
	part := func(i, n int) {
		body[i] = a.kernel()
		if n < a.hull2 {
			body[i] = cgen.If{
				Cond: cgen.CmpL{
					Expr1: t,
					Expr2: il(n),
				},
				Then: cgen.Stmts{
					body[i],
					cgen.Return{},
				},
			}
		}
	}
	if 0 < a.tiles {
		a.strips = a.tile
		a.groupCells = a.groupCells1
		part(3, a.tiles)
	}
	if a.tiles < a.hull1 {
		a.strips = a.scrap
		a.groupCells = a.groupCells1
		part(4, a.hull1)
	}
	if a.hull1 < a.hull2 {
		a.strips = 1
		a.groupCells = a.groupCells2
		body[5] = a.kernel()
	}
	return callee.Func(body)
}

func (a *arrange) ptrs(tensors, t cgen.Gen) cgen.Gen {
	var (
		bnCnt = a.BnPre + a.BnPost
		stmts = make(cgen.Stmts, 3+bnCnt+2)
		s     = t
	)
	if a.tile > 1 {
		s = vb(a.name("s"))
		var strip cgen.Gen = cgen.Mul{
			Expr1: cast(a.tile),
			Expr2: t,
		}
		if i := a.tiles + 1; i < a.hull2 {
			fix := cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpE{
						Expr1: t,
						Expr2: il(i),
					},
					Then: il(a.tile - a.scrap),
					Else: il(0),
				},
			}
			strip = cgen.Sub{
				Expr1: strip,
				Expr2: fix,
			}
		}
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT, What: s,
			Init: strip,
		}
	}
	var (
		tensorIdx    = 0
		n            = a.groupCells1
		weightPitch1 = cast(n * a.fromCHW * a.weightBytes1)
		weightPitch2 = cast(a.stripBytes1)
		biasPitch    = cast(n * a.biasBytes)
	)
	tensor := func() cgen.Gen {
		i := tensorIdx
		tensorIdx++
		return cgen.Elem{
			Arr: tensors,
			Idx: il(i),
		}
	}
	a.weights1 = vb(a.name("weights"))
	stmts[1] = cgen.Var{
		Type: cgen.RestrictPtrChar, What: a.weights1,
		Init: addr(tensor(), weightPitch1, s),
	}
	a.biases1 = vb(a.name("biases"))
	stmts[2] = cgen.Var{
		Type: cgen.RestrictPtrChar, What: a.biases1,
		Init: addr(tensor(), biasPitch, s),
	}
	a.bnPtrs = make([]cgen.Gen, bnCnt)
	for i := range a.bnPtrs {
		var (
			bnPtr = vb(a.name("bnPtr"))
			expr  = tensor()
		)
		if i >= a.BnPre {
			expr = &bn.Offset{
				Ctx: a.bc,
				Mas: expr,
				Channel: cgen.Mul{
					Expr1: il(n),
					Expr2: s,
				},
			}
		}
		stmts[3+i] = cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: bnPtr, Init: expr,
		}
		a.bnPtrs[i] = bnPtr
	}
	var (
		arranged1 = tensor()
		arranged2 = cgen.Add{
			Expr1: arranged1,
			Expr2: cast(a.biasOffset),
		}
	)
	a.weights2 = vb(a.name("weights"))
	stmts[3+bnCnt] = cgen.Var{
		Type: cgen.RestrictPtrChar, What: a.weights2,
		Init: addr(arranged1, weightPitch2, s),
	}
	a.biases2 = vb(a.name("biases"))
	stmts[3+bnCnt+1] = cgen.Var{
		Type: cgen.RestrictPtrChar, What: a.biases2,
		Init: addr(arranged2, biasPitch, s),
	}
	return stmts
}

func (a *arrange) kernel() cgen.Gen {
	switch a.platform {
	case raw.AVX512Float32:
		return a.m512()
	default:
		panic("bug")
	}
}

func (a *arrange) m512() cgen.Gen {
	if a.BnPre == 0 {
		return a.m512NoBnPre()
	}
	if a.cellWeights1%a.fromHW == 0 {
		return a.m512BnPreSpecial()
	}
	return a.m512BnPreGeneral()
}

func (a *arrange) m512NoBnPre() cgen.Gen {
	const (
		lanes  = 16
		unroll = 16
	)
	var (
		i      = vb(a.name("i"))
		gc     = a.groupCells
		bnMuls []cgen.Gen
		j      cgen.Gen
		jg     = 1 + gc%2
	)
	if cells := jg * gc; cells < unroll {
		jg *= unroll / cells
	}
	ch := func(cell int) cgen.Gen {
		return cgen.Paren{
			Inner: cgen.Add{
				Expr1: il(cell),
				Expr2: cgen.Mul{
					Expr1: il(gc),
					Expr2: i,
				},
			},
		}
	}
	ld := func(wt cgen.Gen, pair, side, elems int) cgen.Gen {
		var (
			from       = a.weights1
			k          = pair*2 + side
			group      = k / gc
			cell       = k % gc
			groupPitch = a.cellWeights1 * a.weightBytes1
			cellPitch  = a.fromCHW * a.weightBytes1
			iPitch     = gc * cellPitch
			jPitch     = jg * groupPitch
		)
		from = cgen.Add{
			Expr1: from,
			Expr2: cast(group*groupPitch + cell*cellPitch),
		}
		from = addr(from, cast(iPitch), i)
		from = addr(from, cast(jPitch), j)
		return cgen.Var{
			Type: avx.M512, What: wt,
			Init: avx.Mm512MaskzLoaduPs{
				loMask(elems), from,
			},
		}
	}
	mul := func(wt cgen.Gen, pair, side int) cgen.Gen {
		if bnMuls == nil {
			return nil
		}
		var (
			k     = pair*2 + side
			bnMul = bnMuls[k%gc]
		)
		return cgen.Assign{
			Expr1: wt,
			Expr2: avx.Mm512MulPs{
				bnMul, wt,
			},
		}
	}
	cvt := func(half, wt cgen.Gen) cgen.Gen {
		return cgen.Var{
			Type: avx.M256i, What: half,
			Init: avx.Mm512CvtpsPh{
				wt, avx.FroundToNearestIntNoExc,
			},
		}
	}
	st := func(yield cgen.Gen, pair, elems int) cgen.Gen {
		var (
			to     = a.weights2
			iPitch = a.stripBytes1
			jPitch = jg * gc * a.cellBytes
		)
		to = cgen.Add{
			Expr1: to,
			Expr2: cast(pair * 2 * a.cellBytes),
		}
		to = addr(to, cast(iPitch), i)
		to = addr(to, cast(jPitch), j)
		return avx.Mm512MaskStoreuEpi32{
			to, loMask(elems), yield,
		}
	}
	two := func(pair, elemsLo, elemsHi int) cgen.Stmts {
		var (
			wtLo   = vb(a.name("wtLo"))
			wtHi   = vb(a.name("wtHi"))
			halfLo = vb(a.name("halfLo"))
			halfHi = vb(a.name("halfHi"))
			yield  = vb(a.name("yield"))
		)
		return cgen.Stmts{
			cgen.Stmts{
				ld(wtLo, pair, 0, elemsLo),
				ld(wtHi, pair, 1, elemsHi),
			},
			cgen.Stmts{
				mul(wtLo, pair, 0),
				mul(wtHi, pair, 1),
			},
			cgen.Stmts{
				cvt(halfLo, wtLo),
				cvt(halfHi, wtHi),
			},
			cgen.Var{
				Type: avx.M512i, What: yield,
				Init: avx.Mm512Inserti64x4{
					avx.Mm512Castsi256Si512{halfLo},
					halfHi, il(1),
				},
			},
			st(yield, pair, lanes),
		}
	}
	one := func(pair, elemsLo int) cgen.Stmts {
		var (
			wtLo   = vb(a.name("wtLo"))
			halfLo = vb(a.name("halfLo"))
			yield  = vb(a.name("yield"))
		)
		return cgen.Stmts{
			ld(wtLo, pair, 0, elemsLo),
			mul(wtLo, pair, 0),
			cvt(halfLo, wtLo),
			cgen.Var{
				Type: avx.M512i, What: yield,
				Init: avx.Mm512Castsi256Si512{halfLo},
			},
			st(yield, pair, lanes/2),
		}
	}
	layer4 := func(cells1, cells2 int) cgen.Gen {
		var (
			n1    = cells2 / 2
			n2    = n1 + cells2%2
			toMix = make([]cgen.Stmts, n2)
		)
		for pair := 0; pair < n1; pair++ {
			var (
				k       = pair*2 + 1
				elemsLo = a.cellWeights1
				elemsHi = elemsLo
			)
			if k >= cells1 {
				elemsHi = a.cellWeights2
				if k-1 >= cells1 {
					elemsLo = elemsHi
				}
			}
			toMix[pair] = two(pair, elemsLo, elemsHi)
		}
		if n1 < n2 {
			elemsLo := a.cellWeights1
			if n1*2 >= cells1 {
				elemsLo = a.cellWeights2
			}
			toMix[n1] = one(n1, elemsLo)
		}
		const bundle = unroll / 2
		var (
			bundles = ceilQuo(n2, bundle)
			ret     = make(cgen.Gens, bundles)
		)
		for x := range ret {
			var (
				first = x * bundle
				past  = min(first+bundle, n2)
			)
			ret[x] = mix(toMix[first:past])
		}
		return ret
	}
	layer3 := func() cgen.Gen {
		var (
			stmts = make(cgen.Stmts, 2)
			iters = a.stripGroups1 / jg
			after = a.stripGroups1 % jg
			n1    = jg * gc
			n2    = after * gc
			n3    = n2
		)
		if a.stripGroups1 < a.stripGroups2 {
			after++
			n3 += gc
		}
		if iters > 0 {
			j = vb(a.name("j"))
			stmts[0] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT, What: j,
					Init: il(0),
				},
				Cond: cgen.CmpL{
					Expr1: j,
					Expr2: il(iters),
				},
				Post: cgen.IncPre{
					Expr: j,
				},
				Body: layer4(n1, n1),
			}
		}
		if after > 0 {
			j = il(iters)
			stmts[1] = layer4(n2, n3)
		}
		return stmts
	}
	layer2 := func() cgen.Gen {
		var (
			parts = ceilQuo(gc, lanes)
			toMix = make([]cgen.Stmts, parts)
		)
		for part := range toMix {
			var (
				bnCnt = a.BnPost
				stmts = make(cgen.Stmts, 1+bnCnt*2+1)
				bias  = vb(a.name("bias"))
				first = part * lanes
				cnt   = min(gc-first, lanes)
				mask  = loMask(cnt)
				bnCh  = ch(first)
			)
			offset := cgen.Add{
				Expr1: cast(first * a.biasBytes),
				Expr2: cgen.Mul{
					Expr1: cast(gc * a.biasBytes),
					Expr2: i,
				},
			}
			from := cgen.Add{
				Expr1: a.biases1,
				Expr2: offset,
			}
			to := cgen.Add{
				Expr1: a.biases2,
				Expr2: offset,
			}
			stmts[0] = cgen.Var{
				Type: avx.M512, What: bias,
				Init: avx.Mm512MaskzLoaduPs{
					mask, from,
				},
			}
			for x := 0; x < bnCnt; x++ {
				var (
					bnPtr = a.bnPtrs[x]
					bnMul = vb(a.name("bnMul"))
					bnAdd = vb(a.name("bnAdd"))
				)
				stmts[1+x*2] = &bn.Load{
					Ctx:     a.bc,
					Mas:     bnPtr,
					Channel: bnCh,
					Mul:     bnMul,
					Add:     bnAdd,
					Cnt:     cnt,
				}
				stmts[1+x*2+1] = &bn.Apply{
					Ctx: a.bc,
					Mul: bnMul,
					Add: bnAdd,
					To:  bias,
				}
			}
			stmts[1+bnCnt*2] = avx.Mm512MaskStoreuPs{
				to, mask, bias,
			}
			toMix[part] = stmts
		}
		return cgen.Gens{
			layer3(),
			mix(toMix),
		}
	}
	layer1 := func() cgen.Gen {
		var (
			bnPrep cgen.Gen
			bnCnt  = a.BnPost
		)
		if bnCnt > 0 {
			bnMuls = make([]cgen.Gen, gc)
			toMix := make([]cgen.Stmts, gc)
			for cell := range toMix {
				var (
					stmts = make(cgen.Stmts, bnCnt*2)
					bnCh  = ch(cell)
				)
				for x := 0; x < bnCnt; x++ {
					var (
						bnPtr = a.bnPtrs[x]
						bnMul = vb(a.name("bnMul"))
					)
					stmts[x*2] = &bn.Load{
						Ctx:     a.bc,
						Mas:     bnPtr,
						Channel: bnCh,
						Mul:     bnMul,
					}
					if x == 0 {
						bnMuls[cell] = bnMul
						continue
					}
					prod := bnMuls[cell]
					stmts[x*2+1] = cgen.Assign{
						Expr1: prod,
						Expr2: avx.Mm512MulPs{
							prod, bnMul,
						},
					}
				}
				toMix[cell] = stmts
			}
			bnPrep = mix(toMix)
		}
		return cgen.Gens{
			bnPrep,
			layer2(),
		}
	}
	return cgen.For{
		Init: cgen.Var{
			Type: cgen.PtrdiffT, What: i,
			Init: il(0),
		},
		Cond: cgen.CmpL{
			Expr1: i,
			Expr2: il(a.strips),
		},
		Post: cgen.IncPre{
			Expr: i,
		},
		Body: layer1(),
	}
}

func (a *arrange) m512BnPreSpecial() cgen.Gen {
	var (
		i        = vb(a.name("i"))
		gc       = a.groupCells
		jUnroll  = 8
		j        cgen.Gen
		cells    int
		postMuls []cgen.Gen
		sums     []cgen.Gen
		k        cgen.Gen
		elems    int
		preMul1  cgen.Gen
		preAdd1  cgen.Gen
	)
	if a.BnPost == 0 {
		jUnroll = 16
	}
	ch := func(cell int) cgen.Gen {
		return cgen.Paren{
			Inner: cgen.Add{
				Expr1: il(cell),
				Expr2: cgen.Add{
					Expr1: cgen.Mul{
						Expr1: il(gc),
						Expr2: i,
					},
					Expr2: cgen.Mul{
						Expr1: il(jUnroll),
						Expr2: j,
					},
				},
			},
		}
	}
	ld := func(wt cgen.Gen, pair, side int) cgen.Gen {
		var (
			from      = a.weights1
			cell      = pair*2 + side
			cellPitch = a.fromCHW * a.weightBytes1
			iPitch    = gc * cellPitch
			jPitch    = jUnroll * cellPitch
			kPitch    = a.cellWeights1 * a.weightBytes1
		)
		from = cgen.Add{
			Expr1: from,
			Expr2: cast(cell * cellPitch),
		}
		from = addr(from, cast(iPitch), i)
		from = addr(from, cast(jPitch), j)
		from = addr(from, cast(kPitch), k)
		return cgen.Var{
			Type: avx.M512, What: wt,
			Init: avx.Mm512MaskzLoaduPs{
				loMask(elems), from,
			},
		}
	}
	madd := func(wt cgen.Gen, pair, side int) cgen.Gen {
		var (
			cell = pair*2 + side
			sum  = sums[cell]
		)
		return cgen.Assign{
			Expr1: sum,
			Expr2: avx.Mm512FmaddPs{
				wt, preAdd1, sum,
			},
		}
	}
	muls := func(wt cgen.Gen, pair, side int) cgen.Gen {
		inner := wt
		if postMuls != nil {
			cell := pair*2 + side
			inner = avx.Mm512MulPs{
				wt, postMuls[cell],
			}
		}
		return cgen.Assign{
			Expr1: wt,
			Expr2: avx.Mm512MulPs{
				inner, preMul1,
			},
		}
	}
	cvt := func(half, wt cgen.Gen) cgen.Gen {
		return cgen.Var{
			Type: avx.M256i, What: half,
			Init: avx.Mm512CvtpsPh{
				wt, avx.FroundToNearestIntNoExc,
			},
		}
	}
	st := func(yield cgen.Gen, pair, have int) cgen.Gen {
		var (
			to     = a.weights2
			iPitch = a.stripBytes1
			jPitch = jUnroll * a.cellBytes
			kPitch = gc * a.cellBytes
			mask   = loMask(have * 8)
		)
		to = cgen.Add{
			Expr1: to,
			Expr2: cast(pair * 2 * a.cellBytes),
		}
		to = addr(to, cast(iPitch), i)
		to = addr(to, cast(jPitch), j)
		to = addr(to, cast(kPitch), k)
		return avx.Mm512MaskStoreuEpi32{
			to, mask, yield,
		}
	}
	two := func(pair int) cgen.Stmts {
		var (
			wtLo   = vb(a.name("wtLo"))
			wtHi   = vb(a.name("wtHi"))
			halfLo = vb(a.name("halfLo"))
			halfHi = vb(a.name("halfHi"))
			yield  = vb(a.name("yield"))
		)
		return cgen.Stmts{
			cgen.Stmts{
				ld(wtLo, pair, 0),
				ld(wtHi, pair, 1),
			},
			cgen.Stmts{
				madd(wtLo, pair, 0),
				madd(wtHi, pair, 1),
			},
			cgen.Stmts{
				muls(wtLo, pair, 0),
				muls(wtHi, pair, 1),
			},
			cgen.Stmts{
				cvt(halfLo, wtLo),
				cvt(halfHi, wtHi),
			},
			cgen.Var{
				Type: avx.M512i, What: yield,
				Init: avx.Mm512Inserti64x4{
					avx.Mm512Castsi256Si512{halfLo},
					halfHi, il(1),
				},
			},
			st(yield, pair, 2),
		}
	}
	one := func(pair int) cgen.Stmts {
		var (
			wtLo   = vb(a.name("wtLo"))
			halfLo = vb(a.name("halfLo"))
			yield  = vb(a.name("yield"))
		)
		return cgen.Stmts{
			ld(wtLo, pair, 0),
			madd(wtLo, pair, 0),
			muls(wtLo, pair, 0),
			cvt(halfLo, wtLo),
			cgen.Var{
				Type: avx.M512i, What: yield,
				Init: avx.Mm512Castsi256Si512{halfLo},
			},
			st(yield, pair, 1),
		}
	}
	layer7 := func() cgen.Gen {
		var (
			n1    = cells / 2
			n2    = n1 + cells%2
			toMix = make([]cgen.Stmts, n2)
		)
		for pair := 0; pair < n1; pair++ {
			toMix[pair] = two(pair)
		}
		if n1 < n2 {
			toMix[n1] = one(n1)
		}
		const bundle = 4
		var (
			n3  = ceilQuo(n2, bundle)
			ret = make(cgen.Gens, n3)
		)
		for x := range ret {
			var (
				first = x * bundle
				past  = min(first+bundle, n2)
			)
			ret[x] = mix(toMix[first:past])
		}
		return ret
	}
	layer6 := func() cgen.Gen {
		var (
			preCnt = a.BnPre
			stmts  = make(cgen.Stmts, preCnt*3)
			chans  = elems / a.fromHW
		)
		preCh := cgen.Mul{
			Expr1: il(a.cellWeights1 / a.fromHW),
			Expr2: k,
		}
		for x := 0; x < preCnt; x++ {
			var (
				prePtr  = a.bnPtrs[x]
				preMul2 = vb(a.name("preMul"))
				preAdd2 = vb(a.name("preAdd"))
			)
			if chans == 1 {
				stmts[x*3] = &bn.Load{
					Ctx:     a.bc,
					Mas:     prePtr,
					Channel: preCh,
					Mul:     preMul2,
					Add:     preAdd2,
				}
			} else {
				stmts[x*3] = &bn.Load{
					Ctx:     a.bc,
					Mas:     prePtr,
					Channel: preCh,
					Mul:     preMul2,
					Add:     preAdd2,
					Cnt:     chans,
					Spread:  a.fromHW,
				}
			}
			if x == 0 {
				preMul1 = preMul2
				preAdd1 = preAdd2
				continue
			}
			stmts[x*3+1] = cgen.Assign{
				Expr1: preMul1,
				Expr2: avx.Mm512MulPs{
					preMul1, preMul2,
				},
			}
			stmts[x*3+2] = &bn.Apply{
				Ctx: a.bc,
				Mul: preMul2,
				Add: preAdd2,
				To:  preAdd1,
			}
		}
		return cgen.Gens{
			stmts,
			layer7(),
		}
	}
	layer5 := func() cgen.Gen {
		var (
			stmts = make(cgen.Stmts, 2)
			iters = a.stripGroups1
		)
		if iters > 0 {
			k = vb(a.name("k"))
			elems = a.cellWeights1
			stmts[0] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT, What: k,
					Init: il(0),
				},
				Cond: cgen.CmpL{
					Expr1: k,
					Expr2: il(iters),
				},
				Post: cgen.IncPre{
					Expr: k,
				},
				Body: layer6(),
			}
		}
		if iters < a.stripGroups2 {
			k = il(iters)
			elems = a.cellWeights2
			stmts[1] = layer6()
		}
		return stmts
	}
	layer4 := func() cgen.Gen {
		var (
			postCnt = a.BnPost
			stmts   = make(cgen.Stmts, 3+postCnt*2+1)
			bias    = vb(a.name("bias"))
			mask    = loMask(cells)
			iPitch  = cast(gc * a.biasBytes)
			jPitch  = cast(jUnroll * a.biasBytes)
			from    = addr(a.biases1, iPitch, i)
			to      = addr(a.biases2, iPitch, i)
			postCh  = ch(0)
		)
		stmts[0] = &sumr.Pack{
			Platform: a.platform,
			Nms:      a.nms,
			Vars:     sums,
		}
		stmts[1] = cgen.Var{
			Type: avx.M512, What: bias,
			Init: avx.Mm512MaskzLoaduPs{
				mask, addr(from, jPitch, j),
			},
		}
		stmts[2] = cgen.Assign{
			Expr1: bias,
			Expr2: avx.Mm512AddPs{
				sums[0], bias,
			},
		}
		for x := 0; x < postCnt; x++ {
			var (
				postPtr = a.bnPtrs[a.BnPre+x]
				postMul = vb(a.name("postMul"))
				postAdd = vb(a.name("postAdd"))
			)
			stmts[3+x*2] = &bn.Load{
				Ctx:     a.bc,
				Mas:     postPtr,
				Channel: postCh,
				Mul:     postMul,
				Add:     postAdd,
				Cnt:     cells,
			}
			stmts[3+x*2+1] = &bn.Apply{
				Ctx: a.bc,
				Mul: postMul,
				Add: postAdd,
				To:  bias,
			}
		}
		stmts[3+postCnt*2] = avx.Mm512MaskStoreuPs{
			addr(to, jPitch, j), mask, bias,
		}
		return cgen.Gens{
			layer5(),
			stmts,
		}
	}
	layer3 := func() cgen.Gen {
		sums = make([]cgen.Gen, cells)
		stmts := make(cgen.Stmts, cells)
		for cell := range stmts {
			sum := vb(a.name("sum"))
			sums[cell] = sum
			stmts[cell] = cgen.Var{
				Type: avx.M512, What: sum,
				Init: avx.Mm512SetzeroPs,
			}
		}
		return cgen.Gens{
			stmts,
			layer4(),
		}
	}
	layer2 := func() cgen.Gen {
		var (
			postPrep cgen.Gen
			postCnt  = a.BnPost
		)
		if postCnt > 0 {
			postMuls = make([]cgen.Gen, cells)
			toMix := make([]cgen.Stmts, cells)
			for cell := range toMix {
				var (
					stmts  = make(cgen.Stmts, postCnt*2)
					postCh = ch(cell)
				)
				for x := 0; x < postCnt; x++ {
					var (
						postPtr = a.bnPtrs[a.BnPre+x]
						postMul = vb(a.name("postMul"))
					)
					stmts[x*2] = &bn.Load{
						Ctx:     a.bc,
						Mas:     postPtr,
						Channel: postCh,
						Mul:     postMul,
					}
					if x == 0 {
						postMuls[cell] = postMul
						continue
					}
					prod := postMuls[cell]
					stmts[x*2+1] = cgen.Assign{
						Expr1: prod,
						Expr2: avx.Mm512MulPs{
							prod, postMul,
						},
					}
				}
				toMix[cell] = stmts
			}
			postPrep = mix(toMix)
		}
		return cgen.Gens{
			postPrep,
			layer3(),
		}
	}
	layer1 := func() cgen.Gen {
		var (
			stmts = make(cgen.Stmts, 2)
			iters = gc / jUnroll
			after = gc % jUnroll
		)
		if iters > 0 {
			j = vb(a.name("j"))
			cells = jUnroll
			stmts[0] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT, What: j,
					Init: il(0),
				},
				Cond: cgen.CmpL{
					Expr1: j,
					Expr2: il(iters),
				},
				Post: cgen.IncPre{
					Expr: j,
				},
				Body: layer2(),
			}
		}
		if after > 0 {
			j = il(iters)
			cells = after
			stmts[1] = layer2()
		}
		return stmts
	}
	return cgen.For{
		Init: cgen.Var{
			Type: cgen.PtrdiffT, What: i,
			Init: il(0),
		},
		Cond: cgen.CmpL{
			Expr1: i,
			Expr2: il(a.strips),
		},
		Post: cgen.IncPre{
			Expr: i,
		},
		Body: layer1(),
	}
}

func (a *arrange) m512BnPreGeneral() cgen.Gen {
	var (
		i        = vb(a.name("i"))
		gc       = a.groupCells
		jUnroll  = 8
		j        cgen.Gen
		cells    int
		postMuls []cgen.Gen
		sums     []cgen.Gen
		cw       = a.cellWeights1
		kUnroll  int
		k        cgen.Gen
		preChans int
		group    int
		run      int
		preMul1  cgen.Gen
		preAdd1  cgen.Gen
		preMul2  cgen.Gen
		preAdd2  cgen.Gen
		elems    int
		l        cgen.Gen
	)
	postChan := func(cell int) cgen.Gen {
		return cgen.Paren{
			Inner: cgen.Add{
				Expr1: il(cell),
				Expr2: cgen.Add{
					Expr1: cgen.Mul{
						Expr1: il(gc),
						Expr2: i,
					},
					Expr2: cgen.Mul{
						Expr1: il(jUnroll),
						Expr2: j,
					},
				},
			},
		}
	}
	ld := func(wt cgen.Gen, pair, side int) cgen.Gen {
		var (
			from      = a.weights1
			cell      = pair*2 + side
			cellPitch = a.fromCHW * a.weightBytes1
			iPitch    = gc * cellPitch
			jPitch    = jUnroll * cellPitch
			kPitch    = kUnroll * a.fromHW * a.weightBytes1
			lPitch    = cw * a.weightBytes1
		)
		from = cgen.Add{
			Expr1: from,
			Expr2: cast(cell * cellPitch),
		}
		from = addr(from, cast(iPitch), i)
		from = addr(from, cast(jPitch), j)
		from = addr(from, cast(kPitch), k)
		from = addr(from, cast(lPitch), l)
		return cgen.Var{
			Type: avx.M512, What: wt,
			Init: avx.Mm512MaskzLoaduPs{
				loMask(elems), from,
			},
		}
	}
	madd := func(wt cgen.Gen, pair, side int) cgen.Gen {
		var (
			cell = pair*2 + side
			sum  = sums[cell]
		)
		return cgen.Assign{
			Expr1: sum,
			Expr2: avx.Mm512FmaddPs{
				wt, preAdd1, sum,
			},
		}
	}
	muls := func(wt cgen.Gen, pair, side int) cgen.Gen {
		inner := preMul1
		if postMuls != nil {
			var (
				cell    = pair*2 + side
				postMul = postMuls[cell]
			)
			inner = avx.Mm512MulPs{
				postMul, preMul1,
			}
		}
		return cgen.Assign{
			Expr1: wt,
			Expr2: avx.Mm512MulPs{
				wt, inner,
			},
		}
	}
	cvt := func(half, wt cgen.Gen) cgen.Gen {
		return cgen.Var{
			Type: avx.M256i, What: half,
			Init: avx.Mm512CvtpsPh{
				wt, avx.FroundToNearestIntNoExc,
			},
		}
	}
	st := func(yield cgen.Gen, pair, have int) cgen.Gen {
		var (
			to         = a.weights2
			iPitch     = a.stripBytes1
			jPitch     = jUnroll * a.cellBytes
			groupPitch = gc * a.cellBytes
			kGroups    = kUnroll * a.fromHW / cw
			kPitch     = kGroups * groupPitch
			lPitch     = groupPitch
			mask       = loMask(have * 8)
		)
		to = cgen.Add{
			Expr1: to,
			Expr2: cast(pair * 2 * a.cellBytes),
		}
		to = addr(to, cast(iPitch), i)
		to = addr(to, cast(jPitch), j)
		to = addr(to, cast(kPitch), k)
		to = addr(to, cast(lPitch), l)
		return avx.Mm512MaskStoreuEpi32{
			to, mask, yield,
		}
	}
	two := func(pair int) cgen.Stmts {
		var (
			wtLo   = vb(a.name("wtLo"))
			wtHi   = vb(a.name("wtHi"))
			halfLo = vb(a.name("halfLo"))
			halfHi = vb(a.name("halfHi"))
			yield  = vb(a.name("yield"))
		)
		return cgen.Stmts{
			cgen.Stmts{
				ld(wtLo, pair, 0),
				ld(wtHi, pair, 1),
			},
			cgen.Stmts{
				madd(wtLo, pair, 0),
				madd(wtHi, pair, 1),
			},
			cgen.Stmts{
				muls(wtLo, pair, 0),
				muls(wtHi, pair, 1),
			},
			cgen.Stmts{
				cvt(halfLo, wtLo),
				cvt(halfHi, wtHi),
			},
			cgen.Var{
				Type: avx.M512i, What: yield,
				Init: avx.Mm512Inserti64x4{
					avx.Mm512Castsi256Si512{halfLo},
					halfHi, il(1),
				},
			},
			st(yield, pair, 2),
		}
	}
	one := func(pair int) cgen.Stmts {
		var (
			wtLo   = vb(a.name("wtLo"))
			halfLo = vb(a.name("halfLo"))
			yield  = vb(a.name("yield"))
		)
		return cgen.Stmts{
			ld(wtLo, pair, 0),
			madd(wtLo, pair, 0),
			muls(wtLo, pair, 0),
			cvt(halfLo, wtLo),
			cgen.Var{
				Type: avx.M512i, What: yield,
				Init: avx.Mm512Castsi256Si512{halfLo},
			},
			st(yield, pair, 1),
		}
	}
	layer9 := func() cgen.Gen {
		var (
			n1    = cells / 2
			n2    = n1 + cells%2
			toMix = make([]cgen.Stmts, n2)
		)
		for pair := 0; pair < n1; pair++ {
			toMix[pair] = two(pair)
		}
		if n1 < n2 {
			toMix[n1] = one(n1)
		}
		const bundle = 2
		var (
			n3  = ceilQuo(n2, bundle)
			ret = make(cgen.Gens, n3)
		)
		for x := range ret {
			var (
				first = x * bundle
				past  = min(first+bundle, n2)
			)
			ret[x] = mix(toMix[first:past])
		}
		return ret
	}
	layer8 := func() cgen.Gen {
		if run == 1 {
			l = il(group)
			return layer9()
		}
		l = vb(a.name("l"))
		return cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT, What: l,
				Init: il(group),
			},
			Cond: cgen.CmpL{
				Expr1: l,
				Expr2: il(group + run),
			},
			Post: cgen.IncPre{
				Expr: l,
			},
			Body: layer9(),
		}
	}
	layer7 := func() cgen.Gen {
		var (
			prePrep cgen.Gens
			shift   = 0
			before  = group * cw
			preChan = before / a.fromHW
			seen    = before % a.fromHW
			remain  = a.fromHW - seen
		)
		for {
			var do cgen.Gen
			if shift != 0 || remain == a.fromHW {
				var (
					preCnt = a.BnPre
					stmts  = make(cgen.Stmts, preCnt*3)
				)
				preCh := cgen.Paren{
					Inner: cgen.Add{
						Expr1: il(preChan),
						Expr2: cgen.Mul{
							Expr1: cast(kUnroll),
							Expr2: k,
						},
					},
				}
				for x := 0; x < preCnt; x++ {
					var (
						prePtr  = a.bnPtrs[x]
						preMul3 = vb(a.name("preMul"))
						preAdd3 = vb(a.name("preAdd"))
					)
					stmts[x*3] = &bn.Load{
						Ctx:     a.bc,
						Mas:     prePtr,
						Channel: preCh,
						Mul:     preMul3,
						Add:     preAdd3,
					}
					if x == 0 {
						preMul2 = preMul3
						preAdd2 = preAdd3
						continue
					}
					stmts[x*3+1] = cgen.Assign{
						Expr1: preMul2,
						Expr2: avx.Mm512MulPs{
							preMul2, preMul3,
						},
					}
					stmts[x*3+2] = &bn.Apply{
						Ctx: a.bc,
						Mul: preMul3,
						Add: preAdd3,
						To:  preAdd2,
					}
				}
				do = stmts
			}
			if shift == 0 {
				preMul1 = preMul2
				preAdd1 = preAdd2
			} else {
				var (
					n    = min(cw-shift, remain)
					bits = 1<<uint(n) - 1
					mask = il(bits << uint(shift))
				)
				do = cgen.Stmts{
					do,
					cgen.Assign{
						Expr1: preMul1,
						Expr2: avx.Mm512MaskMovPs{
							preMul1, mask, preMul2,
						},
					},
					cgen.Assign{
						Expr1: preAdd1,
						Expr2: avx.Mm512MaskMovPs{
							preAdd1, mask, preAdd2,
						},
					},
				}
			}
			if do != nil {
				prePrep = append(prePrep, do)
			}
			if shift += remain; shift >= cw {
				elems = cw
				break
			}
			if preChan++; preChan == preChans {
				elems = shift
				break
			}
			remain = a.fromHW
		}
		return cgen.Stmts{
			prePrep,
			layer8(),
		}
	}
	layer6 := func() cgen.Gen {
		var (
			ret cgen.Gens
			n   = ceilQuo(preChans*a.fromHW, cw)
		)
		for group = 0; group < n; group += run {
			var (
				before = group * cw
				seen   = before % a.fromHW
				remain = a.fromHW - seen
			)
			run = max(remain/cw, 1)
			ret = append(ret, layer7())
		}
		return ret
	}
	layer5 := func() cgen.Gen {
		if cw&(cw-1) != 0 {
			panic("bug")
		}
		kUnroll = 1
		for n := a.fromHW; n%cw != 0; n *= 2 {
			kUnroll *= 2
		}
		var (
			stmts = make(cgen.Stmts, 2)
			iters = a.FromC / kUnroll
			after = a.FromC % kUnroll
		)
		if iters > 0 {
			k = vb(a.name("k"))
			preChans = kUnroll
			stmts[0] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT, What: k,
					Init: il(0),
				},
				Cond: cgen.CmpL{
					Expr1: k,
					Expr2: il(iters),
				},
				Post: cgen.IncPre{
					Expr: k,
				},
				Body: layer6(),
			}
		}
		if after > 0 {
			k = il(iters)
			preChans = after
			stmts[1] = layer6()
		}
		return stmts
	}
	layer4 := func() cgen.Gen {
		var (
			postCnt = a.BnPost
			stmts   = make(cgen.Stmts, 3+postCnt*2+1)
			bias    = vb(a.name("bias"))
			mask    = loMask(cells)
			iPitch  = cast(gc * a.biasBytes)
			jPitch  = cast(jUnroll * a.biasBytes)
			from    = addr(a.biases1, iPitch, i)
			to      = addr(a.biases2, iPitch, i)
			postCh  = postChan(0)
		)
		stmts[0] = &sumr.Pack{
			Platform: a.platform,
			Nms:      a.nms,
			Vars:     sums,
		}
		stmts[1] = cgen.Var{
			Type: avx.M512, What: bias,
			Init: avx.Mm512MaskzLoaduPs{
				mask, addr(from, jPitch, j),
			},
		}
		stmts[2] = cgen.Assign{
			Expr1: bias,
			Expr2: avx.Mm512AddPs{
				sums[0], bias,
			},
		}
		for x := 0; x < postCnt; x++ {
			var (
				postPtr = a.bnPtrs[a.BnPre+x]
				postMul = vb(a.name("postMul"))
				postAdd = vb(a.name("postAdd"))
			)
			stmts[3+x*2] = &bn.Load{
				Ctx:     a.bc,
				Mas:     postPtr,
				Channel: postCh,
				Mul:     postMul,
				Add:     postAdd,
				Cnt:     cells,
			}
			stmts[3+x*2+1] = &bn.Apply{
				Ctx: a.bc,
				Mul: postMul,
				Add: postAdd,
				To:  bias,
			}
		}
		stmts[3+postCnt*2] = avx.Mm512MaskStoreuPs{
			addr(to, jPitch, j), mask, bias,
		}
		return cgen.Gens{
			layer5(),
			stmts,
		}
	}
	layer3 := func() cgen.Gen {
		sums = make([]cgen.Gen, cells)
		stmts := make(cgen.Stmts, cells)
		for cell := range stmts {
			sum := vb(a.name("sum"))
			sums[cell] = sum
			stmts[cell] = cgen.Var{
				Type: avx.M512, What: sum,
				Init: avx.Mm512SetzeroPs,
			}
		}
		return cgen.Gens{
			stmts,
			layer4(),
		}
	}
	layer2 := func() cgen.Gen {
		var (
			postPrep cgen.Gen
			postCnt  = a.BnPost
		)
		if postCnt > 0 {
			postMuls = make([]cgen.Gen, cells)
			toMix := make([]cgen.Stmts, cells)
			for cell := range toMix {
				var (
					stmts  = make(cgen.Stmts, postCnt*2)
					postCh = postChan(cell)
				)
				for x := 0; x < postCnt; x++ {
					var (
						postPtr = a.bnPtrs[a.BnPre+x]
						postMul = vb(a.name("postMul"))
					)
					stmts[x*2] = &bn.Load{
						Ctx:     a.bc,
						Mas:     postPtr,
						Channel: postCh,
						Mul:     postMul,
					}
					if x == 0 {
						postMuls[cell] = postMul
						continue
					}
					prod := postMuls[cell]
					stmts[x*2+1] = cgen.Assign{
						Expr1: prod,
						Expr2: avx.Mm512MulPs{
							prod, postMul,
						},
					}
				}
				toMix[cell] = stmts
			}
			postPrep = mix(toMix)
		}
		return cgen.Gens{
			postPrep,
			layer3(),
		}
	}
	layer1 := func() cgen.Gen {
		var (
			stmts = make(cgen.Stmts, 2)
			iters = gc / jUnroll
			after = gc % jUnroll
		)
		if iters > 0 {
			j = vb(a.name("j"))
			cells = jUnroll
			stmts[0] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT, What: j,
					Init: il(0),
				},
				Cond: cgen.CmpL{
					Expr1: j,
					Expr2: il(iters),
				},
				Post: cgen.IncPre{
					Expr: j,
				},
				Body: layer2(),
			}
		}
		if after > 0 {
			j = il(iters)
			cells = after
			stmts[1] = layer2()
		}
		return stmts
	}
	return cgen.For{
		Init: cgen.Var{
			Type: cgen.PtrdiffT, What: i,
			Init: il(0),
		},
		Cond: cgen.CmpL{
			Expr1: i,
			Expr2: il(a.strips),
		},
		Post: cgen.IncPre{
			Expr: i,
		},
		Body: layer1(),
	}
}

type Apply struct {
	*Ctx
	ToC     int
	FromC   int
	FromH   int
	FromW   int
	Ops     []mod.Op
	Team    cgen.Gen
	Tensors []cgen.Gen
	*layout
	callerName string
}

func (a *Apply) Prep() cgen.Gen {
	a.layout = a.newLayout(
		a.ToC, a.FromC, a.FromH, a.FromW,
	)
	const affix = "Apply"
	sig := fmt.Sprint(
		affix, " ",
		a.ToC, a.FromC, a.FromH, a.FromW,
		a.Ops, len(a.Tensors),
	)
	if prior, ok := a.dedup[sig]; ok {
		a.callerName = prior
		return nil
	}
	a.callerName = a.name(a.prefix + affix)
	a.dedup[sig] = a.callerName
	return cgen.Gens{
		&apply{Apply: a},
		cgen.Newline,
	}
}

func (a *Apply) Append(to []byte) []byte {
	var (
		tensors = vb(a.name("tensors"))
		ptrs    = cgen.CommaLines(a.Tensors)
	)
	return cgen.Stmts{
		cgen.Var{
			Type: cgen.PtrChar,
			What: cgen.Elem{Arr: tensors},
			Init: cgen.Brace{Inner: ptrs},
		},
		cgen.Call{
			Func: vb(a.callerName),
			Args: cgen.CommaSpaced{
				a.Team, tensors,
			},
		},
	}.Append(to)
}

type apply struct {
	*Apply
	tile       int
	tiles      int
	scrap      int
	hull1      int
	hull2      int
	calleeName string
	wtPtr      cgen.Gen
	biasPtr    cgen.Gen
	seq        []cgen.Gen
	strips     int
	groupCells int
}

func (a *apply) Append(to []byte) []byte {
	var (
		threadVecs int
		stripVecs  = a.stripGroups2 * a.groupCells1
		team       = vb(a.name("team"))
		tensors    = vb(a.name("tensors"))
	)
	switch a.platform {
	case raw.AVX512Float32:
		threadVecs = 512
	default:
		panic("bug")
	}
	a.tile = ceilQuo(threadVecs, stripVecs)
	a.tiles = a.strips1 / a.tile
	a.scrap = a.strips1 % a.tile
	a.hull1 = a.tiles + btoi(a.scrap > 0)
	a.hull2 = a.hull1 + btoi(a.strips1 < a.strips2)
	a.calleeName = a.name(a.callerName + "Callee")
	return cgen.Gens{
		a.calleeFunc(),
		cgen.Newline,
		cgen.StaticFuncDef{
			ReturnType: cgen.Void,
			Name:       a.callerName,
			Params: cgen.CommaSpaced{
				cgen.Param{
					Type: a.tc.PtrTeam,
					What: team,
				},
				cgen.Param{
					Type: cgen.PtrPtrChar,
					What: tensors,
				},
			},
			Body: &threader.Do{
				Ctx:    a.tc,
				Callee: vb(a.calleeName),
				Any:    tensors,
				Hull: []cgen.Gen{
					il(a.hull2),
				},
				Team: team,
			},
		},
	}.Append(to)
}

func (a *apply) calleeFunc() cgen.Gen {
	var (
		body    = make(cgen.Stmts, 6)
		tensors = vb(a.name("tensors"))
		t       = vb(a.name("t"))
	)
	callee := &threader.Callee{
		Ctx:  a.tc,
		Name: a.calleeName,
		Task: vb(a.name("task")),
		Pt:   vb(a.name("pt")),
	}
	body[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: tensors,
		Init: callee.Any(),
	}
	body[1] = cgen.Var{
		Type: cgen.PtrdiffT, What: t,
		Init: cgen.Elem{Arr: callee.Pt, Idx: il(0)},
	}
	body[2] = a.ptrs(tensors, t)
	part := func(i, n int) {
		body[i] = a.kernel()
		if n < a.hull2 {
			body[i] = cgen.If{
				Cond: cgen.CmpL{
					Expr1: t,
					Expr2: il(n),
				},
				Then: cgen.Stmts{
					body[i],
					cgen.Return{},
				},
			}
		}
	}
	if 0 < a.tiles {
		a.strips = a.tile
		a.groupCells = a.groupCells1
		part(3, a.tiles)
	}
	if a.tiles < a.hull1 {
		a.strips = a.scrap
		a.groupCells = a.groupCells1
		part(4, a.hull1)
	}
	if a.hull1 < a.hull2 {
		a.strips = 1
		a.groupCells = a.groupCells2
		body[5] = a.kernel()
	}
	return callee.Func(body)
}

func (a *apply) ptrs(tensors, t cgen.Gen) cgen.Gen {
	var (
		stmts     cgen.Stmts
		s         = t
		tensorIdx = 0
		wtPitch   = cast(a.stripBytes1)
		gc        = a.groupCells1
		biasPitch = cast(gc * a.biasBytes)
		datPitch  = cast(gc * a.datBytes)
	)
	if a.tile > 1 {
		s = vb(a.name("s"))
		var expr cgen.Gen = cgen.Mul{
			Expr1: il(a.tile),
			Expr2: t,
		}
		if i := a.tiles + 1; i < a.hull2 {
			fix := cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpE{
						Expr1: t,
						Expr2: il(i),
					},
					Then: il(a.tile - a.scrap),
					Else: il(0),
				},
			}
			expr = cgen.Sub{
				Expr1: expr,
				Expr2: fix,
			}
		}
		stmts = append(stmts, cgen.Var{
			Type: cgen.PtrdiffT, What: s,
			Init: expr,
		})
	}
	tensor := func() cgen.Gen {
		i := tensorIdx
		tensorIdx++
		return cgen.Elem{
			Arr: tensors,
			Idx: il(i),
		}
	}
	var (
		arranged1 = tensor()
		arranged2 = cgen.Add{
			Expr1: arranged1,
			Expr2: cast(a.biasOffset),
		}
	)
	a.wtPtr = vb(a.name("wtPtr"))
	stmts = append(stmts, cgen.Var{
		Type: cgen.RestrictPtrChar, What: a.wtPtr,
		Init: addr(arranged1, wtPitch, s),
	})
	a.biasPtr = vb(a.name("biasPtr"))
	stmts = append(stmts, cgen.Var{
		Type: cgen.RestrictPtrChar, What: a.biasPtr,
		Init: addr(arranged2, biasPitch, s),
	})
	dp := func() {
		var (
			datPtr = vb(a.name("datPtr"))
			expr   = tensor()
		)
		if len(a.seq) > 0 {
			expr = addr(expr, datPitch, s)
		}
		a.seq = append(a.seq, datPtr)
		stmts = append(stmts, cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: datPtr, Init: expr,
		})
	}
	ndp := func(n int) {
		for ; n > 0; n-- {
			dp()
		}
	}
	bp := func() {
		bnPtr := vb(a.name("bnPtr"))
		a.seq = append(a.seq, bnPtr)
		stmts = append(stmts, cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: bnPtr,
			Init: &bn.Offset{
				Ctx: a.bc,
				Mas: tensor(),
				Channel: cgen.Mul{
					Expr1: il(gc),
					Expr2: s,
				},
			},
		})
	}
	dp()
	for i := range a.Ops {
		op := &a.Ops[i]
		switch op.Kind {
		case mod.Add:
			ndp(op.Int)
		case mod.Bn:
			bp()
		case mod.ReLU:
		default:
			panic("bug")
		}
	}
	ndp(len(a.Tensors) - tensorIdx)
	return stmts
}

func (a *apply) kernel() cgen.Gen {
	switch a.platform {
	case raw.AVX512Float32:
		return a.m512()
	default:
		panic("bug")
	}
}

func (a *apply) m512() cgen.Gen {
	const (
		lanes  = 16
		bundle = 4
	)
	var (
		i       = vb(a.name("i"))
		gc      = a.groupCells
		sums    []cgen.Gen
		jUnroll = 1 + gc%2
		j       cgen.Gen
		groups  int
		edge    int
		dats    []cgen.Gen
	)
	if cells := jUnroll * gc; cells < bundle*2 {
		jUnroll *= bundle * 2 / cells
	}
	ldWts := func(wts cgen.Gen, pair, sides int) cgen.Gen {
		var (
			from   = a.wtPtr
			iPitch = a.stripBytes1
			jPitch = jUnroll * gc * a.cellBytes
			mask   = loMask(lanes / 2 * sides)
		)
		from = cgen.Add{
			Expr1: from,
			Expr2: cast(pair * 2 * a.cellBytes),
		}
		from = addr(from, cast(iPitch), i)
		from = addr(from, cast(jPitch), j)
		return cgen.Var{
			Type: avx.M512i, What: wts,
			Init: avx.Mm512MaskzLoaduEpi32{
				mask, from,
			},
		}
	}
	ldDat := func(pair, side int) cgen.Gen {
		var (
			k     = pair*2 + side
			group = k / gc
		)
		if dats[group] != nil {
			return nil
		}
		dat := vb(a.name("dat"))
		dats[group] = dat
		var (
			elems      = a.cellWeights1
			from       = a.seq[0]
			groupPitch = a.cellWeights1 * a.datBytes
			jPitch     = jUnroll * groupPitch
		)
		if group == groups-1 {
			elems = edge
		}
		from = cgen.Add{
			Expr1: from,
			Expr2: cast(group * groupPitch),
		}
		from = addr(from, cast(jPitch), j)
		return cgen.Var{
			Type: avx.M512, What: dat,
			Init: avx.Mm512MaskzLoaduPs{
				loMask(elems), from,
			},
		}
	}
	cvt := func(wt, half cgen.Gen) cgen.Gen {
		return cgen.Var{
			Type: avx.M512, What: wt,
			Init: avx.Mm512CvtphPs{half},
		}
	}
	madd := func(wt cgen.Gen, pair, side int) cgen.Gen {
		var (
			k   = pair*2 + side
			dat = dats[k/gc]
			sum = sums[k%gc]
		)
		return cgen.Assign{
			Expr1: sum,
			Expr2: avx.Mm512FmaddPs{
				wt, dat, sum,
			},
		}
	}
	two := func(pair int) cgen.Stmts {
		var (
			wts  = vb(a.name("wts"))
			wtLo = vb(a.name("wtLo"))
			wtHi = vb(a.name("wtHi"))
		)
		return cgen.Stmts{
			cgen.Stmts{
				ldWts(wts, pair, 2),
				ldDat(pair, 0),
				ldDat(pair, 1),
			},
			cgen.Stmts{
				cvt(wtLo, avx.Mm512Castsi512Si256{wts}),
				cvt(wtHi, avx.Mm512Extracti64x4Epi64{
					wts, il(1),
				}),
			},
			cgen.Stmts{
				madd(wtLo, pair, 0),
				madd(wtHi, pair, 1),
			},
		}
	}
	one := func(pair int) cgen.Stmts {
		var (
			wts  = vb(a.name("wts"))
			wtLo = vb(a.name("wtLo"))
		)
		return cgen.Stmts{
			cgen.Stmts{
				ldWts(wts, pair, 1),
				ldDat(pair, 0),
			},
			cvt(wtLo, avx.Mm512Castsi512Si256{wts}),
			madd(wtLo, pair, 0),
		}
	}
	layer4 := func() cgen.Gen {
		dats = make([]cgen.Gen, groups)
		var (
			cells = groups * gc
			pairs = cells / 2
			slots = pairs + cells%2
			toMix = make([]cgen.Stmts, slots)
		)
		for pair := 0; pair < pairs; pair++ {
			toMix[pair] = two(pair)
		}
		if pairs < slots {
			toMix[pairs] = one(pairs)
		}
		var (
			n   = ceilQuo(slots, bundle)
			ret = make(cgen.Gens, n)
		)
		for x := range ret {
			var (
				first = x * bundle
				past  = min(first+bundle, slots)
			)
			ret[x] = mix(toMix[first:past])
		}
		return ret
	}
	layer3 := func() cgen.Gen {
		var (
			stmts = make(cgen.Stmts, 2)
			iters = a.stripGroups1 / jUnroll
			after = a.stripGroups1 % jUnroll
			edge1 = a.cellWeights1
			edge2 = edge1
		)
		if a.stripGroups1 < a.stripGroups2 {
			after++
			edge2 = a.cellWeights2
		}
		if iters > 0 {
			j = vb(a.name("j"))
			groups = jUnroll
			edge = edge1
			stmts[0] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT, What: j,
					Init: il(0),
				},
				Cond: cgen.CmpL{
					Expr1: j,
					Expr2: il(iters),
				},
				Post: cgen.IncPre{
					Expr: j,
				},
				Body: layer4(),
			}
		}
		if after > 0 {
			j = il(iters)
			groups = after
			edge = edge2
			stmts[1] = layer4()
		}
		return stmts
	}
	layer2 := func() cgen.Gen {
		var (
			parts = ceilQuo(gc, lanes)
			toMix = make([]cgen.Stmts, parts)
		)
		for part := range toMix {
			var (
				stmts [2]cgen.Stmts
				first = part * lanes
				cnt   = min(gc-first, lanes)
				bias  = vb(a.name("bias"))
				mask  = loMask(cnt)
				yield = sums[first]
				at    = 0
			)
			stmt := func(x int, s cgen.Gen) {
				stmts[x] = append(stmts[x], s)
			}
			ae := func(ptr cgen.Gen, each int) cgen.Gen {
				ptr = cgen.Add{
					Expr1: ptr,
					Expr2: cast(first * each),
				}
				return addr(ptr, cast(gc*each), i)
			}
			next := func() cgen.Gen {
				if at++; at >= len(a.seq) {
					return nil
				}
				return a.seq[at]
			}
			stmt(1, &sumr.Pack{
				Platform: a.platform,
				Nms:      a.nms,
				Vars:     sums[first : first+cnt],
			})
			stmt(0, cgen.Var{
				Type: avx.M512, What: bias,
				Init: avx.Mm512MaskzLoaduPs{
					mask, ae(a.biasPtr, a.biasBytes),
				},
			})
			stmt(1, cgen.Assign{
				Expr1: yield,
				Expr2: avx.Mm512AddPs{
					yield, bias,
				},
			})
			for op := range a.Ops {
				op := &a.Ops[op]
				switch op.Kind {
				case mod.Add:
					var (
						n  = 1 + op.Int
						ds = make([]cgen.Gen, n)
					)
					ds[0] = yield
					for x := 1; x < n; x++ {
						var (
							dat  = vb(a.name("dat"))
							from = ae(next(), a.datBytes)
						)
						ds[x] = dat
						stmt(0, cgen.Var{
							Type: avx.M512, What: dat,
							Init: avx.Mm512MaskzLoaduPs{
								mask, from,
							},
						})
					}
					for n > 1 {
						fold := n >> 1
						n -= fold
						for x := 0; x < fold; x++ {
							keep := ds[x]
							stmt(1, cgen.Assign{
								Expr1: keep,
								Expr2: avx.Mm512AddPs{
									keep, ds[n+x],
								},
							})
						}
					}
				case mod.Bn:
					var (
						bnPtr = next()
						bnMul = vb(a.name("bnMul"))
						bnAdd = vb(a.name("bnAdd"))
					)
					bnChan := cgen.Paren{
						Inner: cgen.Add{
							Expr1: il(first),
							Expr2: cgen.Mul{
								Expr1: il(gc),
								Expr2: i,
							},
						},
					}
					stmt(0, &bn.Load{
						Ctx:     a.bc,
						Mas:     bnPtr,
						Channel: bnChan,
						Mul:     bnMul,
						Add:     bnAdd,
						Cnt:     cnt,
					})
					stmt(1, &bn.Apply{
						Ctx: a.bc,
						Mul: bnMul,
						Add: bnAdd,
						To:  yield,
					})
				case mod.ReLU:
					stmt(1, &act.ReLU{
						Ctx:      a.ac,
						NegSlope: op.Float,
						Var:      yield,
					})
				default:
					panic("bug")
				}
			}
			for {
				datPtr := next()
				if datPtr == nil {
					break
				}
				stmt(1, avx.Mm512MaskStoreuPs{
					ae(datPtr, a.datBytes),
					mask, yield,
				})
			}
			toMix[part] = append(
				stmts[0], stmts[1]...,
			)
		}
		return cgen.Gens{
			layer3(),
			mix(toMix),
		}
	}
	layer1 := func() cgen.Gen {
		sums = make([]cgen.Gen, gc)
		stmts := make(cgen.Stmts, gc)
		for cell := range stmts {
			sum := vb(a.name("sum"))
			sums[cell] = sum
			stmts[cell] = cgen.Var{
				Type: avx.M512, What: sum,
				Init: avx.Mm512SetzeroPs,
			}
		}
		return cgen.Gens{
			stmts,
			layer2(),
		}
	}
	return cgen.For{
		Init: cgen.Var{
			Type: cgen.PtrdiffT, What: i,
			Init: il(0),
		},
		Cond: cgen.CmpL{
			Expr1: i,
			Expr2: il(a.strips),
		},
		Post: cgen.IncPre{
			Expr: i,
		},
		Body: layer1(),
	}
}
