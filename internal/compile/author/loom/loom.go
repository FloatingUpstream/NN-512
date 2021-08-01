package loom

import (
	"NN-512/internal/compile/author/act"
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/bn"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/mod"
	"NN-512/internal/compile/author/threader"
	"NN-512/internal/compile/author/trans"
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

func addMul(a, b, c cgen.Gen) cgen.Gen {
	return cgen.Add{
		Expr1: a,
		Expr2: cgen.Mul{
			Expr1: b,
			Expr2: c,
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
	prefix      string
	platform    raw.Platform
	cacheBytes1 int
	cacheBytes2 int
	nms         nmsrc.Src
	tc          *threader.Ctx
	ac          *act.Ctx
	bc          *bn.Ctx
	dedup       map[string]interface{}
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src, tc *threader.Ctx, ac *act.Ctx, bc *bn.Ctx) *Ctx {
	return &Ctx{
		prefix:      pl.Config.Prefix + "Loom",
		platform:    pl.Config.Platform,
		cacheBytes1: pl.Config.L1DataCachePerThread,
		cacheBytes2: pl.Config.L2CachePerThreadExL1,
		nms:         nms,
		tc:          tc,
		ac:          ac,
		bc:          bc,
		dedup:       make(map[string]interface{}),
	}
}

func (c *Ctx) name(s string) string {
	return c.nms.Name(s)
}

type Spec struct {
	From      SpecFrom
	Filts     []SpecFilts
	To        SpecTo
	FilterH   int
	FilterW   int
	StrideH   int
	StrideW   int
	PaddingH  int
	PaddingW  int
	DilationH int
	DilationW int
	Groups    int
}

type SpecFrom struct {
	Chans       int
	Height      int
	Width       int
	Pitch1Bytes []int
	Pitch2Bytes []int
	Ops         []mod.Op
}

type SpecFilts struct {
	Cnt    int
	BnPre  int
	BnPost int
}

type SpecTo struct {
	Pitch1Bytes []int
	Pitch2Bytes []int
	Ops         []mod.Op
}

type spans struct {
	spanH1 int
	spanH2 int
	spanH3 int
	spanW1 int
	spanW2 int
	spanW3 int
}

type loopW struct {
	fromW    int
	fromStep int
	blkFirst int
	blkPast  int
	spans
}

type loopH struct {
	fromH    int
	fromStep int
	blkFirst int
	blkStep  int
	blkPast  int
	lws      []*loopW
}

type blocks struct {
	cnt int
	lhs []*loopH
}

func newBlocks(ctx *Ctx, spec *Spec, blkVecs, vecLanes int) *blocks {
	var (
		blks   blocks
		fromH1 int
		lw1    loopW
		fromH2 int
		lw2    loopW
		lh1    loopH
		lh2    loopH
	)
	layer6 := func() {
		lh := lh2
		blks.lhs = append(
			blks.lhs, &lh,
		)
	}
	layer5 := func(flush bool) {
		split := true
		if len(lh2.lws) == len(lh1.lws) {
			split = false
			for x, lw := range lh2.lws {
				if *lw != *lh1.lws[x] {
					split = true
					break
				}
			}
		}
		switch {
		case split:
			if lh2.blkFirst < lh2.blkPast {
				layer6()
			}
			lh2 = lh1
			lh2.lws = make([]*loopW, len(lh1.lws))
			for x, lw := range lh1.lws {
				lw := *lw
				lh2.lws[x] = &lw
			}
		default:
			if lh2.fromStep == 0 {
				lh2.fromStep = lh1.fromH - lh2.fromH
				lh2.blkStep = lh1.blkFirst - lh2.blkFirst
			}
			lh2.blkPast = lh1.blkPast
		}
		if flush {
			layer6()
		}
	}
	layer4 := func(flush bool) {
		if lw2.fromW == 0 {
			if lh1.blkFirst < lh1.blkPast {
				layer5(false)
			}
			lh1.fromH = fromH2
			lh1.blkFirst = lw2.blkFirst
			lh1.lws = lh1.lws[:0]
		}
		lh1.blkPast = lw2.blkPast
		lw2.blkFirst -= lh1.blkFirst
		lw2.blkPast -= lh1.blkFirst
		x := len(lh1.lws)
		switch x {
		case cap(lh1.lws):
			lh1.lws = append(
				lh1.lws, new(loopW),
			)
		default:
			lh1.lws = lh1.lws[:x+1]
			if lh1.lws[x] == nil {
				lh1.lws[x] = new(loopW)
			}
		}
		*lh1.lws[x] = lw2
		if flush {
			layer5(true)
		}
	}
	layer3 := func(flush bool) {
		if flush {
			layer4(true)
			return
		}
		switch {
		case lw1.fromW == 0:
		case lw1.spans != lw2.spans:
		default:
			if lw2.fromStep == 0 {
				lw2.fromStep = lw1.fromW - lw2.fromW
			}
			lw2.blkPast = lw1.blkPast
			return
		}
		if lw2.blkFirst < lw2.blkPast {
			layer4(false)
		}
		fromH2 = fromH1
		lw2 = lw1
	}
	layer2 := func() {
		var (
			h1    = spec.PaddingH
			h2    = h1 + spec.From.Height
			h3    = h2 + spec.PaddingH
			w1    = spec.PaddingW
			w2    = w1 + spec.From.Width
			w3    = w2 + spec.PaddingW
			stepH = blkVecs * spec.StrideH
			stepW = vecLanes * spec.StrideW
		)
		for h := 0; h < h3; h += stepH {
			for w := 0; w < w3; w += stepW {
				fromH1 = h
				lw1.fromW = w
				lw1.blkFirst = blks.cnt
				lw1.blkPast = blks.cnt + 1
				blks.cnt++
				lw1.spanH1 = min(max(h1-h, 0), stepH)
				lw1.spanH2 = min(max(h2-h, 0), stepH)
				lw1.spanH3 = min(h3-h, stepH)
				lw1.spanW1 = min(max(w1-w, 0), stepW)
				lw1.spanW2 = min(max(w2-w, 0), stepW)
				lw1.spanW3 = min(w3-w, stepW)
				var (
					datH = lw1.spanH2 - lw1.spanH1
					datW = lw1.spanW2 - lw1.spanW1
				)
				if datH == 0 || datW == 0 {
					lw1.spanH1 = lw1.spanH3
					lw1.spanH2 = lw1.spanH3
					lw1.spanW1 = lw1.spanW3
					lw1.spanW2 = lw1.spanW3
				}
				layer3(false)
			}
		}
		layer3(true)
	}
	layer1 := func() *blocks {
		sig := fmt.Sprint(
			"newBlocks",
			" ",
			spec.From.Height,
			spec.From.Width,
			spec.StrideH,
			spec.StrideW,
			spec.PaddingH,
			spec.PaddingW,
			blkVecs,
			vecLanes,
		)
		if prior, ok := ctx.dedup[sig]; ok {
			return prior.(*blocks)
		}
		ctx.dedup[sig] = &blks
		layer2()
		return &blks
	}
	return layer1()
}

type node struct {
	filtH int
	filtW int
	deck  int
	pile  int
	base  bool
}

type field struct {
	sboxH     int
	sboxW     int
	nodeFirst int
	nodeStep  int
}

type layout struct {
	fromChans      int
	toChans        int
	slices1        int
	slices2        int
	epochs1        int
	epochs2        int
	biasBytes      int
	biasGroupBytes int
	biasEpochBytes int
	biasTotalBytes int
	lifts          []int
	shifts         []int
	nodes          []*node
	fields         []*field
	wtBytes        int
	wtSliceWts1    int
	wtSliceWts2    int
	wtSliceBytes1  int
	wtSliceBytes2  int
	wtCores1       int
	wtCores2       int
	wtCoreBytes11  int
	wtCoreBytes12  int
	wtCoreBytes21  int
	wtCoreBytes22  int
	wtNodeBytes1   int
	wtNodeBytes2   int
	wtGroupBytes1  int
	wtGroupBytes2  int
	wtEpochBytes1  int
	wtEpochBytes2  int
	wtTotalBytes   int
	blks           *blocks
	blkStep        int
	datBytes       int
	datVecDats     int
	datVecBytes    int
	datSliceVecs   int
	datSliceBytes  int
	datCores       int
	datCoreBytes1  int
	datCoreBytes2  int
	datGroupBytes1 int
	datGroupBytes2 int
	datFieldBytes1 int
	datFieldBytes2 int
	datEpochBytes1 int
	datEpochBytes2 int
	datTotalBytes  int
	sumSiteBytes1  int
	sumSiteBytes2  int
	sumPileBytes   int
	sumCores       int
	sumCoreBytes   int
	sumGroupBytes  int
	sumTotalBytes  int
}

func newLayout(ctx *Ctx, spec *Spec) *layout {
	var (
		y layout
	)
	layer10 := func() {
		y.datCoreBytes1 = y.slices1 * y.datSliceBytes
		y.datCoreBytes2 = y.slices2 * y.datSliceBytes
		y.datGroupBytes1 = y.datCores * y.datCoreBytes1
		y.datGroupBytes2 = y.datCores * y.datCoreBytes2
		y.datFieldBytes1 = spec.Groups * y.datGroupBytes1
		y.datFieldBytes2 = spec.Groups * y.datGroupBytes2
		y.datEpochBytes1 = len(y.fields) * y.datFieldBytes1
		y.datEpochBytes2 = len(y.fields) * y.datFieldBytes2
		y.datTotalBytes = y.epochs1*y.datEpochBytes1 + y.datEpochBytes2
	}
	layer9 := func() {
		y.wtCoreBytes11 = y.slices1 * y.wtSliceBytes1
		y.wtCoreBytes12 = y.slices1 * y.wtSliceBytes2
		y.wtCoreBytes21 = y.slices2 * y.wtSliceBytes1
		y.wtCoreBytes22 = y.slices2 * y.wtSliceBytes2
		y.wtNodeBytes1 = y.wtCores1*y.wtCoreBytes11 + y.wtCoreBytes12
		y.wtNodeBytes2 = y.wtCores1*y.wtCoreBytes21 + y.wtCoreBytes22
		y.wtGroupBytes1 = len(y.nodes) * y.wtNodeBytes1
		y.wtGroupBytes2 = len(y.nodes) * y.wtNodeBytes2
		y.wtEpochBytes1 = spec.Groups * y.wtGroupBytes1
		y.wtEpochBytes2 = spec.Groups * y.wtGroupBytes2
		y.wtTotalBytes = y.epochs1*y.wtEpochBytes1 + y.wtEpochBytes2
		layer10()
	}
	layer8 := func() {
		y.biasGroupBytes = y.toChans * y.biasBytes
		y.biasEpochBytes = spec.Groups * y.biasGroupBytes
		y.biasTotalBytes = y.epochs2 * y.biasEpochBytes
		layer9()
	}
	layer7 := func() {
		wtSliceBytes := y.wtSliceBytes1
		if y.wtCores1 == 0 {
			wtSliceBytes = y.wtSliceBytes2
		}
		switch ctx.platform {
		case raw.AVX512Float32:
			var (
				sliceBytes = 2*wtSliceBytes + y.datSliceBytes
				cacheBytes = ctx.cacheBytes1 + ctx.cacheBytes2
			)
			const (
				empirical1 = 4
				empirical2 = 512
				empirical3 = 4
			)
			y.slices1 = cacheBytes / empirical1 / sliceBytes
			y.slices1 = max(y.slices1, empirical2)
			y.slices2 = y.fromChans % y.slices1
			y.epochs1 = y.fromChans / y.slices1
			y.epochs2 = y.epochs1 + btoi(y.slices2 > 0)
			if y.epochs1 > 0 && y.epochs1 < y.epochs2 {
				if y.slices2*empirical3 < y.slices1 {
					y.slices2 += y.slices1
					y.epochs1--
					y.epochs2--
				}
			}
		default:
			panic("bug")
		}
		layer8()
	}
	layer6 := func() {
		y.sumSiteBytes1 = y.wtSliceWts1 * y.datSliceBytes
		y.sumSiteBytes2 = y.wtSliceWts2 * y.datSliceBytes
		y.sumPileBytes = y.wtCores1*y.sumSiteBytes1 + y.sumSiteBytes2
		var (
			lift = y.lifts[len(y.lifts)-1]
			cut1 = lift / y.datSliceVecs
			cut2 = cut1 - btoi(cut1 > 0)
			cut3 = cut2 * y.blkStep
		)
		y.sumCores = y.datCores - cut3
		y.sumCoreBytes = len(y.shifts) * y.sumPileBytes
		y.sumGroupBytes = y.sumCores * y.sumCoreBytes
		y.sumTotalBytes = spec.Groups * y.sumGroupBytes
		layer7()
	}
	layer5 := func() {
		y.blks = newBlocks(ctx, spec, y.datSliceVecs, y.datVecDats)
		switch lh := y.blks.lhs[0]; lh.blkStep {
		case 0:
			y.blkStep = lh.blkPast - lh.blkFirst
		default:
			y.blkStep = lh.blkStep
		}
		y.datVecBytes = y.datVecDats * y.datBytes
		y.datSliceBytes = y.datSliceVecs * y.datVecBytes
		y.datCores = y.blks.cnt
		layer6()
	}
	layer4 := func() {
		y.wtSliceWts2 = y.toChans % y.wtSliceWts1
		y.wtSliceBytes1 = y.wtSliceWts1 * y.wtBytes
		y.wtSliceBytes2 = y.wtSliceWts2 * y.wtBytes
		y.wtCores1 = y.toChans / y.wtSliceWts1
		y.wtCores2 = y.wtCores1 + btoi(y.wtSliceWts2 > 0)
		layer5()
	}
	layer3 := func() {
		if len(spec.Filts) > 1 && spec.Groups > 1 {
			panic("bug")
		}
		filts := 0
		for i := range spec.Filts {
			filts += spec.Filts[i].Cnt
		}
		y.fromChans = spec.From.Chans / spec.Groups
		y.toChans = filts / spec.Groups
		layer4()
	}
	layer2 := func() {
		nds := make([][][]*node, spec.StrideH)
		for sboxH := range nds {
			nds[sboxH] = make([][]*node, spec.StrideW)
		}
		for filtH := 0; filtH < spec.FilterH; filtH++ {
			var (
				dilaH = filtH * spec.DilationH
				sboxH = dilaH % spec.StrideH
				nds   = nds[sboxH]
				lift  = dilaH / spec.StrideH
				deck  = -1
			)
			for at, is := range y.lifts {
				if is == lift {
					deck = at
					break
				}
			}
			if deck == -1 {
				deck = len(y.lifts)
				y.lifts = append(
					y.lifts, lift,
				)
			}
			for filtW := 0; filtW < spec.FilterW; filtW++ {
				var (
					dilaW = filtW * spec.DilationW
					sboxW = dilaW % spec.StrideW
					shift = dilaW / spec.StrideW
				)
				nd := &node{
					filtH: filtH,
					filtW: filtW,
					deck:  deck,
					pile:  -1,
					base:  false,
				}
				for at, is := range y.shifts {
					if is == shift {
						nd.pile = at
						break
					}
				}
				if nd.pile == -1 {
					nd.pile = len(y.shifts)
					nd.base = true
					y.shifts = append(
						y.shifts, shift,
					)
				}
				nds[sboxW] = append(
					nds[sboxW], nd,
				)
			}
		}
		for sboxH, nds := range nds {
			for sboxW, nds := range nds {
				if nds == nil {
					continue
				}
				fld := &field{
					sboxH:     sboxH,
					sboxW:     sboxW,
					nodeFirst: len(y.nodes),
					nodeStep:  0,
				}
				for _, nd := range nds {
					if nd.filtH == nds[0].filtH {
						fld.nodeStep++
					}
					y.nodes = append(
						y.nodes, nd,
					)
				}
				y.fields = append(
					y.fields, fld,
				)
			}
		}
		layer3()
	}
	layer1 := func() *layout {
		switch ctx.platform {
		case raw.AVX512Float32:
			y.biasBytes = 4
			y.wtBytes = 4
			y.wtSliceWts1 = 6
			y.datBytes = 4
			y.datVecDats = 16
			y.datSliceVecs = 4
		default:
			panic("bug")
		}
		layer2()
		return &y
	}
	return layer1()
}

type ArrangeFilts struct {
	*Ctx
	*Spec
	Team    cgen.Gen
	Tensors []cgen.Gen
	*layout
	callerName string
}

func (a *ArrangeFilts) Prep() cgen.Gen {
	a.layout = newLayout(a.Ctx, a.Spec)
	const affix = "ArrangeFilts"
	sig := fmt.Sprint(affix, " ", a.Spec)
	if prior, ok := a.dedup[sig]; ok {
		a.callerName = prior.(string)
		return nil
	}
	a.callerName = a.name(a.prefix + affix)
	a.dedup[sig] = a.callerName
	return cgen.Gens{
		&arrangeFilts{ArrangeFilts: a},
		cgen.Newline,
	}
}

func (a *ArrangeFilts) Bytes() int {
	return a.biasTotalBytes + a.wtTotalBytes
}

func (a *ArrangeFilts) Append(to []byte) []byte {
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

type arrangeFilts struct {
	*ArrangeFilts
	bundleFilts int
	bundleTile  int
	bundleTiles int
	bundleScrap int
	bundleHull  int
	groupTile   int
	groupTiles  int
	groupScrap  int
	groupHull   int
	calleeName  string
	tensors     cgen.Gen
	bundleCoord cgen.Gen
	groupCoord  cgen.Gen
	epochCoord  cgen.Gen
	slices      int
	coreBytes   int
	nodeBytes   int
	groupBytes  int
	epochFirst  int
	epochCnt    int
	arrangedB   cgen.Gen
	arrangedW   cgen.Gen
	filtsIdx    int
	wtPtr       cgen.Gen
	biasPtr     cgen.Gen
	bnPtrs      []cgen.Gen
	groupIdx    cgen.Gen
	bundleIdx   cgen.Gen
	bundleLast  cgen.Gen
	baseFilt    int
	baseBundle  int
	filts1      int
	filts2      int
	bundleFirst int
	bundlePast  int
}

func (a *arrangeFilts) Append(to []byte) []byte {
	var (
		vecWts     int
		threadVecs int
	)
	switch a.platform {
	case raw.AVX512Float32:
		vecWts = 16
		a.bundleFilts = vecWts
		threadVecs = 512
	default:
		panic("bug")
	}
	var (
		epochChans = ceilQuo(a.fromChans, a.epochs2)
		spatialWts = a.FilterH * a.FilterW
		filtVecs   int
	)
	switch {
	case spatialWts <= vecWts:
		filtVecs = ceilQuo(epochChans, vecWts/spatialWts)
	default:
		filtVecs = epochChans * ceilQuo(spatialWts, vecWts)
	}
	var (
		bundleVecs   = a.bundleFilts * filtVecs
		groupVecs    = a.toChans * filtVecs
		groupBundles int
	)
	switch len(a.Filts) {
	case 1:
		groupBundles = ceilQuo(a.toChans, a.bundleFilts)
	default:
		for i := range a.Filts {
			filts := a.Filts[i].Cnt
			groupBundles += ceilQuo(filts, a.bundleFilts)
		}
	}
	switch {
	case threadVecs <= groupVecs:
		var (
			tile  = ceilQuo(threadVecs, bundleVecs)
			tiles = max(groupBundles/tile, 1)
		)
		a.bundleTile = groupBundles / tiles
		a.bundleTiles = tiles
		a.bundleScrap = groupBundles - tiles*a.bundleTile
		a.bundleHull = tiles
		if a.bundleScrap > 0 {
			a.bundleTiles--
			a.bundleScrap += a.bundleTile
		}
		a.groupTile = 1
		a.groupTiles = a.Groups
		a.groupScrap = 0
		a.groupHull = a.Groups
	default:
		a.bundleTile = groupBundles
		a.bundleTiles = 1
		a.bundleScrap = 0
		a.bundleHull = 1
		var (
			tile  = ceilQuo(threadVecs, groupVecs)
			tiles = max(a.Groups/tile, 1)
		)
		a.groupTile = a.Groups / tiles
		a.groupTiles = tiles
		a.groupScrap = a.Groups - tiles*a.groupTile
		a.groupHull = tiles
		if a.groupScrap > 0 {
			a.groupTiles--
			a.groupScrap += a.groupTile
		}
	}
	a.calleeName = a.name(a.callerName + "Callee")
	var (
		team    = vb(a.name("team"))
		tensors = vb(a.name("tensors"))
	)
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
					il(a.bundleHull),
					il(a.groupHull),
					il(a.epochs2),
				},
				Team: team,
			},
		},
	}.Append(to)
}

func (a *arrangeFilts) calleeFunc() cgen.Gen {
	callee := &threader.Callee{
		Ctx:  a.tc,
		Name: a.calleeName,
		Task: vb(a.name("task")),
		Pt:   vb(a.name("pt")),
	}
	var (
		body   = make(cgen.Stmts, 7)
		usedPt = false
	)
	a.tensors = vb(a.name("tensors"))
	body[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: a.tensors,
		Init: callee.Any(),
	}
	coord := func(nm string, hull, i int) cgen.Gen {
		var (
			ret  = vb(a.name(nm))
			expr cgen.Gen
		)
		switch hull {
		case 1:
			expr = il(0)
		default:
			expr = cgen.Elem{
				Arr: callee.Pt, Idx: il(i),
			}
			usedPt = true
		}
		body[1+i] = cgen.Var{
			Type: cgen.PtrdiffT, What: ret,
			Init: expr,
		}
		return ret
	}
	a.bundleCoord = coord("b", a.bundleHull, 0)
	a.groupCoord = coord("g", a.groupHull, 1)
	a.epochCoord = coord("e", a.epochs2, 2)
	if !usedPt {
		body[4] = cgen.Cast{
			Type: cgen.Void,
			Expr: callee.Pt,
		}
	}
	kernel := func() cgen.Gen {
		var assn cgen.Gen
		if a.epochs2 > 1 && a.epochCnt == 1 {
			assn = cgen.Assign{
				Expr1: a.epochCoord,
				Expr2: il(a.epochFirst),
			}
		}
		return cgen.Stmts{
			assn,
			a.kernel1(),
		}
	}
	if a.epochs1 > 0 {
		a.slices = a.slices1
		a.coreBytes = a.wtCoreBytes11
		a.nodeBytes = a.wtNodeBytes1
		a.groupBytes = a.wtGroupBytes1
		a.epochFirst = 0
		a.epochCnt = a.epochs1
		put := kernel()
		if a.epochs1 < a.epochs2 {
			put = cgen.If{
				Cond: cgen.CmpL{
					Expr1: a.epochCoord,
					Expr2: il(a.epochs1),
				},
				Then: cgen.Stmts{
					put,
					cgen.Return{},
				},
			}
		}
		body[5] = put
	}
	if a.epochs1 < a.epochs2 {
		a.slices = a.slices2
		a.coreBytes = a.wtCoreBytes21
		a.nodeBytes = a.wtNodeBytes2
		a.groupBytes = a.wtGroupBytes2
		a.epochFirst = a.epochs1
		a.epochCnt = 1
		body[6] = kernel()
	}
	return callee.Func(body)
}

func (a *arrangeFilts) kernel1() cgen.Gen {
	var (
		n              = len(a.Filts)
		savedFiltsIdx  = 0
		savedTensorIdx = 0
	)
	tensor := func(filtsIdx, off int) cgen.Gen {
		if savedFiltsIdx != filtsIdx {
			savedFiltsIdx = filtsIdx
			at := 0
			for x := 0; x < filtsIdx; x++ {
				at += 2
				at += a.Filts[x].BnPre
				at += a.Filts[x].BnPost
			}
			savedTensorIdx = at
		}
		return cgen.Elem{
			Arr: a.tensors,
			Idx: il(savedTensorIdx + off),
		}
	}
	ptrDecls := func(filtsIdx int) cgen.Gen {
		wtDecl := func() cgen.Gen {
			a.wtPtr = vb(a.name("wtPtr"))
			filtHW := a.FilterH * a.FilterW
			return cgen.Var{
				Type: cgen.RestrictPtrChar,
				What: a.wtPtr,
				Init: addMul(
					tensor(filtsIdx, 0),
					il(a.slices1*filtHW*a.wtBytes),
					a.epochCoord,
				),
			}
		}
		biasDecl := func() cgen.Gen {
			if a.epochFirst == 0 {
				a.biasPtr = vb(a.name("biasPtr"))
				return cgen.Var{
					Type: cgen.RestrictPtrChar,
					What: a.biasPtr,
					Init: tensor(filtsIdx, 1),
				}
			}
			a.biasPtr = nil
			return nil
		}
		bnDecls := func() cgen.Gen {
			var (
				pre  = a.Filts[filtsIdx].BnPre
				post = a.Filts[filtsIdx].BnPost
				ret  = make(cgen.Stmts, pre+post)
			)
			a.bnPtrs = make([]cgen.Gen, pre+post)
			for x := range a.bnPtrs {
				var (
					bnPtr = vb(a.name("bnPtr"))
					expr  = tensor(filtsIdx, 2+x)
				)
				if x < pre {
					expr = &bn.Offset{
						Ctx: a.bc,
						Mas: expr,
						Channel: cgen.Mul{
							Expr1: il(a.slices1),
							Expr2: a.epochCoord,
						},
					}
				}
				ret[x] = cgen.Var{
					Type: cgen.RestrictPtrChar,
					What: bnPtr, Init: expr,
				}
				a.bnPtrs[x] = bnPtr
			}
			return ret
		}
		a.filtsIdx = filtsIdx
		return cgen.Stmts{
			wtDecl(),
			biasDecl(),
			bnDecls(),
		}
	}
	layer5 := func() cgen.Gen {
		if n == 1 {
			a.baseFilt = 0
			a.baseBundle = 0
			return a.kernel2()
		}
		var (
			atFilt   = make([]int, n+1)
			atBundle = make([]int, n+1)
		)
		for x := 0; x < n; x++ {
			var (
				filts   = a.Filts[x].Cnt
				bundles = ceilQuo(filts, a.bundleFilts)
			)
			atFilt[x+1] = atFilt[x] + filts
			atBundle[x+1] = atBundle[x] + bundles
		}
		leaf := func(x int) cgen.Stmts {
			a.baseFilt = atFilt[x]
			a.baseBundle = atBundle[x]
			var assn cgen.Gen
			if x+1 < n {
				assn = cgen.Assign{
					Expr1: a.bundleIdx,
					Expr2: il(atBundle[x+1]),
				}
			}
			return cgen.Stmts{
				ptrDecls(x),
				a.kernel2(),
				assn,
			}
		}
		var tree func(int, int) cgen.Stmts
		tree = func(first, last int) cgen.Stmts {
			if first == last {
				return leaf(first)
			}
			var (
				start = atBundle[first]
				stop  = atBundle[last+1]
				split = start + (stop-start)/2
				x     = first + 1
			)
			for atBundle[x+1] <= split {
				x++
			}
			return cgen.Stmts{
				cgen.If{
					Cond: cgen.CmpL{
						Expr1: a.bundleIdx,
						Expr2: il(atBundle[x]),
					},
					Then: tree(first, x-1),
				},
				tree(x, last),
			}
		}
		return tree(0, n-1)
	}
	layer4 := func() cgen.Gen {
		a.bundleIdx = vb(a.name("j"))
		switch a.bundleHull {
		case 1:
			a.bundleLast = nil
		default:
			a.bundleLast = vb(a.name("jj"))
		}
		stmts := make(cgen.Stmts, 3)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: a.bundleIdx,
			Init: cgen.Mul{
				Expr1: il(a.bundleTile),
				Expr2: a.bundleCoord,
			},
		}
		if a.bundleLast != nil {
			var expr cgen.Gen
			switch a.bundleTiles {
			case a.bundleHull:
				expr = il(a.bundleTile - 1)
			case 0:
				expr = il(a.bundleScrap - 1)
			default:
				expr = cgen.Paren{
					Inner: cgen.Ternary{
						Cond: cgen.CmpL{
							Expr1: a.bundleCoord,
							Expr2: il(a.bundleTiles),
						},
						Then: il(a.bundleTile - 1),
						Else: il(a.bundleScrap - 1),
					},
				}
			}
			stmts[1] = cgen.Var{
				Type: cgen.PtrdiffT,
				What: a.bundleLast,
				Init: cgen.Add{
					Expr1: a.bundleIdx,
					Expr2: expr,
				},
			}
		}
		stmts[2] = layer5()
		return stmts
	}
	layer3 := func() cgen.Gen {
		a.groupIdx = vb(a.name("i"))
		var (
			stmts = make(cgen.Stmts, 3)
			iters = 0
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: a.groupIdx,
			Init: cgen.Mul{
				Expr1: il(a.groupTile),
				Expr2: a.groupCoord,
			},
		}
		switch a.groupTiles {
		case a.groupHull:
			iters = a.groupTile
		case 0:
			iters = a.groupScrap
		}
		switch iters {
		case 1:
			stmts[2] = layer4()
		default:
			var (
				last = vb(a.name("ii"))
				expr cgen.Gen
			)
			switch iters {
			case 0:
				expr = cgen.Paren{
					Inner: cgen.Ternary{
						Cond: cgen.CmpL{
							Expr1: a.groupCoord,
							Expr2: il(a.groupTiles),
						},
						Then: il(a.groupTile - 1),
						Else: il(a.groupScrap - 1),
					},
				}
			default:
				expr = il(iters - 1)
			}
			stmts[1] = cgen.Var{
				Type: cgen.PtrdiffT,
				What: last,
				Init: cgen.Add{
					Expr1: a.groupIdx,
					Expr2: expr,
				},
			}
			stmts[2] = cgen.For{
				Cond: cgen.CmpLE{
					Expr1: a.groupIdx,
					Expr2: last,
				},
				Post: cgen.IncPre{
					Expr: a.groupIdx,
				},
				Body: layer4(),
			}
		}
		return stmts
	}
	layer2 := func() cgen.Gen {
		var decls cgen.Gen
		if n == 1 {
			decls = ptrDecls(0)
		}
		return cgen.Gens{
			decls,
			layer3(),
		}
	}
	layer1 := func() cgen.Gen {
		a.arrangedB = vb(a.name("arrangedB"))
		a.arrangedW = vb(a.name("arrangedW"))
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.RestrictPtrChar,
				What: a.arrangedB,
				Init: addMul(
					tensor(n, 0),
					il(a.biasEpochBytes),
					a.epochCoord,
				),
			},
			cgen.Var{
				Type: cgen.RestrictPtrChar,
				What: a.arrangedW,
				Init: addMul(
					cgen.Add{
						Expr1: tensor(n, 0),
						Expr2: il(a.biasTotalBytes),
					},
					il(a.wtEpochBytes1),
					a.epochCoord,
				),
			},
			layer2(),
		}
	}
	return layer1()
}

func (a *arrangeFilts) kernel2() cgen.Gen {
	var (
		filts1 int
		filts2 int
	)
	layer3 := func() cgen.Gen {
		switch a.platform {
		case raw.AVX512Float32:
			return a.m512()
		default:
			panic("bug")
		}
	}
	layer2 := func() cgen.Gen {
		var (
			retIf cgen.Gen
			past  = a.baseBundle
		)
		if a.bundleLast != nil {
			retIf = cgen.If1{
				Cond: cgen.CmpGE{
					Expr1: a.bundleIdx,
					Expr2: a.bundleLast,
				},
				Then: cgen.Return{},
			}
		}
		do := func(bundles int) cgen.Gen {
			a.bundleFirst = past
			past += bundles
			a.bundlePast = past
			if bundles == 1 {
				return cgen.If{
					Cond: cgen.CmpE{
						Expr1: a.bundleIdx,
						Expr2: il(past - 1),
					},
					Then: cgen.Stmts{
						layer3(),
						retIf,
						cgen.Assign{
							Expr1: a.bundleIdx,
							Expr2: il(past),
						},
					},
				}
			}
			return cgen.If{
				Cond: cgen.CmpL{
					Expr1: a.bundleIdx,
					Expr2: il(past),
				},
				Then: cgen.Stmts{
					cgen.For{
						Cond: cgen.CmpNE{
							Expr1: a.bundleIdx,
							Expr2: il(past),
						},
						Post: cgen.IncPre{
							Expr: a.bundleIdx,
						},
						Body: cgen.Stmts{
							layer3(),
							retIf,
						},
					},
				},
			}
		}
		var (
			stmts = make(cgen.Stmts, 4)
			quo1  = filts1 / a.bundleFilts
			rem1  = filts1 - a.bundleFilts*quo1
			tail  = filts2 - a.bundleFilts*quo1
		)
		if quo1 > 0 {
			a.filts1 = a.bundleFilts
			a.filts2 = a.bundleFilts
			stmts[0] = do(quo1)
		}
		if rem1 > 0 {
			a.filts1 = rem1
			a.filts2 = min(tail, a.bundleFilts)
			tail -= a.filts2
			stmts[1] = do(1)
		}
		if tail > 0 {
			var (
				quo2 = tail / a.bundleFilts
				rem2 = tail - a.bundleFilts*quo2
			)
			if quo2 > 0 {
				a.filts1 = 0
				a.filts2 = a.bundleFilts
				stmts[2] = do(quo2)
			}
			if rem2 > 0 {
				a.filts1 = 0
				a.filts2 = rem2
				stmts[3] = do(1)
			}
		}
		return stmts
	}
	layer1 := func() cgen.Gen {
		switch len(a.Filts) {
		case 1:
			filts2 = a.toChans
		default:
			filts2 = a.Filts[a.filtsIdx].Cnt
		}
		var (
			past   = a.baseFilt + filts2
			split  = a.toChans - a.wtSliceWts2
			clamp1 = max(past-split, 0)
			clamp2 = min(clamp1, filts2)
		)
		filts1 = filts2 - clamp2
		return layer2()
	}
	return layer1()
}

func (a *arrangeFilts) m512() cgen.Gen {
	var (
		filtHW    int
		nodeIdxes []int
		preCnt    int
		postCnt   int
		postMul1  cgen.Gen
		postAdd1  cgen.Gen
		bias      cgen.Gen
		coreIdx1  cgen.Gen
		coreOff1  int
		stepIdx   cgen.Gen
		stepChans int
		stepWts   int
		haveWts   int
		tpOff     int
		tp        *trans.Pose
		colVec    cgen.Gen
		colQuo    int
		colRem    int
		preMul1   cgen.Gen
		preAdd1   cgen.Gen
		nodeIdx   int
		coreIdx2  int
		coreOff2  int
		emitLane  int
		emitLanes int
	)
	layer17 := func() cgen.Gen {
		var (
			ae         = a.arrangedW
			slicePitch = a.wtSliceBytes1
		)
		if emitLane == a.filts1 {
			slicePitch = a.wtSliceBytes2
		}
		var (
			stepPitch = stepChans * slicePitch
			mask1     = 1<<uint(emitLanes) - 1
			mask2     = mask1 << uint(emitLane)
		)
		ae = cgen.Add{
			Expr1: ae,
			Expr2: il(
				nodeIdx*a.nodeBytes +
					coreIdx2*a.coreBytes +
					colQuo*slicePitch +
					coreOff2*a.wtBytes -
					emitLane*a.wtBytes,
			),
		}
		ae = addMul(ae, il(a.groupBytes), a.groupIdx)
		ae = addMul(ae, il(a.coreBytes), coreIdx1)
		ae = addMul(ae, il(stepPitch), stepIdx)
		return avx.Mm512MaskStoreuPs{
			ae, il(mask2), colVec,
		}
	}
	layer16 := func() cgen.Gen {
		var stmts cgen.Stmts
		nodeIdx = nodeIdxes[colRem]
		coreIdx2 = 0
		coreOff2 = coreOff1
		emitLane = 0
		for emitLane < a.filts2 {
			var (
				lanes1 = a.filts2 - emitLane
				lanes2 = a.wtSliceWts1 - coreOff2
			)
			emitLanes = min(lanes1, lanes2)
			stmts = append(stmts, layer17())
			coreIdx2++
			coreOff2 = 0
			emitLane += emitLanes
		}
		return stmts
	}
	layer15 := func() cgen.Gen {
		if preCnt == 0 {
			return layer16()
		}
		return cgen.Stmts{
			cgen.Assign{
				Expr1: bias,
				Expr2: avx.Mm512FmaddPs{
					colVec, preAdd1,
					bias,
				},
			},
			cgen.Assign{
				Expr1: colVec,
				Expr2: avx.Mm512MulPs{
					colVec, preMul1,
				},
			},
			layer16(),
		}
	}
	layer14 := func() cgen.Gen {
		if preCnt == 0 ||
			colRem > 0 {
			return layer15()
		}
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		preCh := cgen.Paren{
			Inner: addMul(
				addMul(
					il(colQuo),
					il(a.fromChans),
					a.groupIdx,
				),
				il(stepChans),
				stepIdx,
			),
		}
		for x, prePtr := range a.bnPtrs[:preCnt] {
			var (
				preMul2 = vb(a.name("preMul"))
				preAdd2 = vb(a.name("preAdd"))
			)
			stmt(&bn.Load{
				Ctx:     a.bc,
				Mas:     prePtr,
				Channel: preCh,
				Mul:     preMul2,
				Add:     preAdd2,
			})
			if x == 0 {
				preMul1 = preMul2
				preAdd1 = preAdd2
				continue
			}
			stmt(cgen.Assign{
				Expr1: preMul1,
				Expr2: avx.Mm512MulPs{
					preMul1, preMul2,
				},
			})
			stmt(cgen.Assign{
				Expr1: preAdd1,
				Expr2: avx.Mm512FmaddPs{
					preAdd1, preMul2,
					preAdd2,
				},
			})
		}
		stmt(layer15())
		return stmts
	}
	layer13 := func() cgen.Gen {
		if postCnt == 0 {
			return layer14()
		}
		return cgen.Stmts{
			cgen.Assign{
				Expr1: colVec,
				Expr2: avx.Mm512MulPs{
					colVec, postMul1,
				},
			},
			layer14(),
		}
	}
	layer12 := func() cgen.Gen {
		var (
			n    = tp.Cols
			gens = make(cgen.Gens, n)
		)
		for x, wt := range tp.Vars[:n] {
			colVec = wt
			colQuo = (tpOff + x) / filtHW
			colRem = (tpOff + x) % filtHW
			gens[x] = layer13()
		}
		return gens
	}
	layer11 := func() cgen.Gen {
		var (
			n     = tp.Rows
			stmts = make(cgen.Stmts, n+2)
		)
		for x, wt := range tp.Vars[:n] {
			var (
				mask        = loMask(tp.Cols)
				ae          = a.wtPtr
				filtPitch   = a.fromChans * filtHW * a.wtBytes
				groupPitch  = a.toChans * filtPitch
				bundlePitch = a.bundleFilts * filtPitch
				stepPitch   = stepWts * a.wtBytes
			)
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(
					-a.baseBundle*bundlePitch +
						x*filtPitch +
						tpOff*a.wtBytes,
				),
			}
			ae = addMul(ae, il(groupPitch), a.groupIdx)
			ae = addMul(ae, il(bundlePitch), a.bundleIdx)
			ae = addMul(ae, il(stepPitch), stepIdx)
			stmts[x] = cgen.Var{
				Type: avx.M512, What: wt,
				Init: avx.Mm512MaskzLoaduPs{
					mask, ae,
				},
			}
		}
		stmts[n] = tp
		stmts[n+1] = layer12()
		return stmts
	}
	layer10 := func() cgen.Gen {
		var (
			cols1 = haveWts - tpOff
			cols2 = min(cols1, a.bundleFilts)
		)
		tp = &trans.Pose{
			Platform: a.platform,
			Nms:      a.nms,
			Rows:     a.filts2,
			Cols:     cols2,
		}
		tp.Vars = make(
			[]cgen.Gen,
			max(tp.Rows, tp.Cols),
		)
		for x := range tp.Vars {
			wt := vb(a.name("wt"))
			tp.Vars[x] = wt
		}
		return layer11()
	}
	layer9 := func() cgen.Gen {
		var (
			n    = ceilQuo(haveWts, a.bundleFilts)
			gens = make(cgen.Gens, n)
		)
		for x := range gens {
			tpOff = x * a.bundleFilts
			gens[x] = layer10()
		}
		return gens
	}
	layer8 := func() cgen.Gen {
		stepIdx = vb(a.name("k"))
		switch {
		case filtHW < a.bundleFilts:
			stepChans = a.bundleFilts / filtHW
			stepWts = stepChans * filtHW
		default:
			stepChans = 1
			stepWts = filtHW
		}
		var (
			stmts = make(cgen.Stmts, 3)
			iters = a.slices / stepChans
			after = a.slices % stepChans
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: stepIdx,
			Init: il(0),
		}
		if iters > 0 {
			haveWts = stepWts
			stmts[1] = cgen.For{
				Cond: cgen.CmpNE{
					Expr1: stepIdx,
					Expr2: il(iters),
				},
				Post: cgen.IncPre{
					Expr: stepIdx,
				},
				Body: layer9(),
			}
		}
		if after > 0 {
			haveWts = after * filtHW
			stmts[2] = layer9()
		}
		return stmts
	}
	layer7 := func() cgen.Gen {
		coreIdx1 = vb(a.name("c"))
		var (
			stmts = make(cgen.Stmts, 2)
			add   = a.baseFilt
			sub   = a.baseBundle * a.bundleFilts
			numer = cgen.Cast{
				Type: cgen.SizeT,
				Expr: cgen.Paren{
					Inner: addMul(
						il(add-sub),
						il(a.bundleFilts),
						a.bundleIdx,
					),
				},
			}
			denom  = il(a.wtSliceWts1)
			marks  = make([]bool, a.wtSliceWts1)
			marked = 0
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: coreIdx1,
			Init: cgen.Quo{
				Expr1: numer,
				Expr2: denom,
			},
		}
		for x1 := a.bundleFirst; x1 < a.bundlePast; x1++ {
			var (
				x2 = add - sub + x1*a.bundleFilts
				x3 = x2 % a.wtSliceWts1
			)
			if marks[x3] {
				break
			}
			marks[x3] = true
			marked++
		}
		switch marked {
		case 1:
			for off, mark := range marks {
				if mark {
					coreOff1 = off
					stmts[1] = layer8()
					break
				}
			}
		default:
			cases := make(cgen.Stmts, 0, marked)
			for off, mark := range marks {
				if mark {
					var expr cgen.Gen
					if len(cases)+1 < marked {
						expr = il(off)
					}
					coreOff1 = off
					cases = append(
						cases, cgen.Case{
							Expr: expr,
							Body: cgen.Stmts{
								layer8(),
								cgen.Break,
							},
						},
					)
				}
			}
			stmts[1] = cgen.Switch{
				Expr: cgen.Rem{
					Expr1: numer,
					Expr2: denom,
				},
				Cases: cases,
			}
		}
		return stmts
	}
	layer6 := func() cgen.Gen {
		store := func() cgen.Gen {
			var (
				ae          = a.arrangedB
				bundlePitch = a.bundleFilts * a.biasBytes
				mask        = loMask(a.filts2)
			)
			ae = cgen.Sub{
				Expr1: ae,
				Expr2: il(
					a.baseBundle*bundlePitch -
						a.baseFilt*a.biasBytes,
				),
			}
			ae = addMul(ae, il(a.biasGroupBytes), a.groupIdx)
			ae = addMul(ae, il(bundlePitch), a.bundleIdx)
			return avx.Mm512MaskStoreuPs{
				ae, mask, bias,
			}
		}
		if preCnt == 0 {
			return cgen.Stmts{
				store(),
				layer7(),
			}
		}
		return cgen.Stmts{
			layer7(),
			store(),
		}
	}
	layer5 := func() cgen.Gen {
		var stmt cgen.Gen
		switch a.epochFirst {
		case 0:
			load := func() cgen.Gen {
				var (
					mask        = loMask(a.filts2)
					ae          = a.biasPtr
					groupPitch  = a.toChans * a.biasBytes
					bundlePitch = a.bundleFilts * a.biasBytes
				)
				ae = cgen.Sub{
					Expr1: ae,
					Expr2: il(a.baseBundle * bundlePitch),
				}
				ae = addMul(ae, il(groupPitch), a.groupIdx)
				ae = addMul(ae, il(bundlePitch), a.bundleIdx)
				return cgen.Assign{
					Expr1: bias,
					Expr2: avx.Mm512MaskzLoaduPs{
						mask, ae,
					},
				}
			}
			post := func() cgen.Gen {
				if postCnt == 0 {
					return nil
				}
				return cgen.Assign{
					Expr1: bias,
					Expr2: avx.Mm512FmaddPs{
						postMul1, bias,
						postAdd1,
					},
				}
			}
			stmt = cgen.If{
				Cond: cgen.IsZero{
					Expr: a.epochCoord,
				},
				Then: cgen.Stmts{
					load(),
					post(),
				},
			}
		default:
			if postCnt > 0 {
				stmt = cgen.Cast{
					Type: cgen.Void,
					Expr: postAdd1,
				}
			}
		}
		return cgen.Stmts{
			stmt,
			layer6(),
		}
	}
	layer4 := func() cgen.Gen {
		bias = vb(a.name("bias"))
		return cgen.Stmts{
			cgen.Var{
				Type: avx.M512, What: bias,
				Init: avx.Mm512SetzeroPs,
			},
			layer5(),
		}
	}
	layer3 := func() cgen.Gen {
		if postCnt == 0 {
			return layer4()
		}
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		postCh := cgen.Paren{
			Inner: addMul(
				addMul(
					il(-a.baseBundle*a.bundleFilts),
					il(a.toChans),
					a.groupIdx,
				),
				il(a.bundleFilts),
				a.bundleIdx,
			),
		}
		for x, postPtr := range a.bnPtrs[preCnt:] {
			var (
				postMul2 = vb(a.name("postMul"))
				postAdd2 = vb(a.name("postAdd"))
			)
			stmt(&bn.Load{
				Ctx:     a.bc,
				Mas:     postPtr,
				Channel: postCh,
				Mul:     postMul2,
				Add:     postAdd2,
				Cnt:     a.filts2,
			})
			if x == 0 {
				postMul1 = postMul2
				postAdd1 = postAdd2
				continue
			}
			stmt(cgen.Assign{
				Expr1: postMul1,
				Expr2: avx.Mm512MulPs{
					postMul1, postMul2,
				},
			})
			stmt(cgen.Assign{
				Expr1: postAdd1,
				Expr2: avx.Mm512FmaddPs{
					postAdd1, postMul2,
					postAdd2,
				},
			})
		}
		stmt(layer4())
		return stmts
	}
	layer2 := func() cgen.Gen {
		preCnt = a.Filts[a.filtsIdx].BnPre
		postCnt = a.Filts[a.filtsIdx].BnPost
		return layer3()
	}
	layer1 := func() cgen.Gen {
		filtHW = a.FilterH * a.FilterW
		nodeIdxes = make([]int, filtHW)
		for x1, nd := range a.nodes {
			x2 := nd.filtH*a.FilterW + nd.filtW
			nodeIdxes[x2] = x1
		}
		return layer2()
	}
	return layer1()
}

type ArrangeDats struct {
	*Ctx
	*Spec
	Team    cgen.Gen
	Tensors []cgen.Gen
	*layout
	callerName string
}

func (a *ArrangeDats) Prep() cgen.Gen {
	a.layout = newLayout(a.Ctx, a.Spec)
	const affix = "ArrangeDats"
	sig := fmt.Sprint(affix, " ", a.Spec)
	if prior, ok := a.dedup[sig]; ok {
		a.callerName = prior.(string)
		return nil
	}
	a.callerName = a.name(a.prefix + affix)
	a.dedup[sig] = a.callerName
	return cgen.Gens{
		&arrangeDats{ArrangeDats: a},
		cgen.Newline,
	}
}

func (a *ArrangeDats) Bytes() int {
	return a.datTotalBytes
}

func (a *ArrangeDats) Append(to []byte) []byte {
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

type arrangeDats struct {
	*ArrangeDats
	sliceTile1  int
	sliceTile2  int
	sliceTiles  int
	sliceScrap1 int
	sliceScrap2 int
	sliceHull   int
	coreTile    int
	coreTiles   int
	coreScrap   int
	coreHull    int
	groupTile   int
	groupTiles  int
	groupScrap  int
	groupHull   int
	calleeName  string
	tensors     cgen.Gen
	sliceCoord  cgen.Gen
	coreCoord   cgen.Gen
	groupCoord  cgen.Gen
	epochCoord  cgen.Gen
	sliceTile   int
	sliceScrap  int
	coreBytes   int
	groupBytes  int
	fieldBytes  int
	datPtrs     []cgen.Gen
	bnPtrs      []cgen.Gen
	arranged    cgen.Gen
	groupIdx    cgen.Gen
	coreIdx     cgen.Gen
	coreLast    cgen.Gen
	coreH       cgen.Gen
	coreW       cgen.Gen
	*spans
	sliceIdx cgen.Gen
	bnMuls   []cgen.Gen
	bnAdds   []cgen.Gen
}

func (a *arrangeDats) Append(to []byte) []byte {
	var threadVecs int
	switch a.platform {
	case raw.AVX512Float32:
		threadVecs = 512
	default:
		panic("bug")
	}
	var (
		blkVecs    = len(a.fields) * a.datSliceVecs
		chanVecs   = a.datCores * blkVecs
		groupVecs1 = a.fromChans * chanVecs
		groupVecs2 = ceilQuo(groupVecs1, a.epochs2)
		coreVecs   = ceilQuo(groupVecs2, a.datCores)
	)
	a.sliceTile1 = a.slices1
	a.sliceTile2 = a.slices2
	a.sliceTiles = 1
	a.sliceScrap1 = 0
	a.sliceScrap2 = 0
	a.sliceHull = 1
	a.groupTile = 1
	a.groupTiles = a.Groups
	a.groupScrap = 0
	a.groupHull = a.Groups
	switch {
	case threadVecs <= coreVecs:
		minSlices := a.slices1
		switch {
		case a.epochs1 == a.epochs2:
		case a.epochs1 == 0 || a.slices1 > a.slices2:
			minSlices = a.slices2
		}
		var (
			tile  = ceilQuo(threadVecs, blkVecs)
			tiles = max(minSlices/tile, 1)
		)
		a.sliceTile1 = a.slices1 / tiles
		a.sliceTile2 = a.slices2 / tiles
		a.sliceTiles = tiles
		a.sliceScrap1 = a.slices1 - tiles*a.sliceTile1
		a.sliceScrap2 = a.slices2 - tiles*a.sliceTile2
		a.sliceHull = tiles
		if (a.epochs1 > 0 && a.sliceScrap1 > 0) ||
			(a.epochs1 < a.epochs2 && a.sliceScrap2 > 0) {
			a.sliceTiles--
			a.sliceScrap1 += a.sliceTile1
			a.sliceScrap2 += a.sliceTile2
		}
		a.coreTile = 1
		a.coreTiles = a.datCores
		a.coreScrap = 0
		a.coreHull = a.datCores
	case threadVecs <= groupVecs2:
		var (
			tile  = ceilQuo(threadVecs, coreVecs)
			tiles = max(a.datCores/tile, 1)
		)
		a.coreTile = a.datCores / tiles
		a.coreTiles = tiles
		a.coreScrap = a.datCores - tiles*a.coreTile
		a.coreHull = tiles
		if a.coreScrap > 0 {
			a.coreTiles--
			a.coreScrap += a.coreTile
		}
	default:
		a.coreTile = a.datCores
		a.coreTiles = 1
		a.coreScrap = 0
		a.coreHull = 1
		var (
			tile  = ceilQuo(threadVecs, groupVecs2)
			tiles = max(a.Groups/tile, 1)
		)
		a.groupTile = a.Groups / tiles
		a.groupTiles = tiles
		a.groupScrap = a.Groups - tiles*a.groupTile
		a.groupHull = tiles
		if a.groupScrap > 0 {
			a.groupTiles--
			a.groupScrap += a.groupTile
		}
	}
	a.calleeName = a.name(a.callerName + "Callee")
	var (
		team    = vb(a.name("team"))
		tensors = vb(a.name("tensors"))
	)
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
					il(a.sliceHull),
					il(a.coreHull),
					il(a.groupHull),
					il(a.epochs2),
				},
				Team: team,
			},
		},
	}.Append(to)
}

func (a *arrangeDats) calleeFunc() cgen.Gen {
	callee := &threader.Callee{
		Ctx:  a.tc,
		Name: a.calleeName,
		Task: vb(a.name("task")),
		Pt:   vb(a.name("pt")),
	}
	var (
		body   = make(cgen.Stmts, 8)
		usedPt = false
	)
	a.tensors = vb(a.name("tensors"))
	body[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: a.tensors,
		Init: callee.Any(),
	}
	coord := func(nm string, hull, i int) cgen.Gen {
		var (
			ret  = vb(a.name(nm))
			expr cgen.Gen
		)
		switch hull {
		case 1:
			expr = il(0)
		default:
			expr = cgen.Elem{
				Arr: callee.Pt, Idx: il(i),
			}
			usedPt = true
		}
		body[1+i] = cgen.Var{
			Type: cgen.PtrdiffT, What: ret,
			Init: expr,
		}
		return ret
	}
	a.sliceCoord = coord("s", a.sliceHull, 0)
	a.coreCoord = coord("c", a.coreHull, 1)
	a.groupCoord = coord("g", a.groupHull, 2)
	a.epochCoord = coord("e", a.epochs2, 3)
	if !usedPt {
		body[5] = cgen.Cast{
			Type: cgen.Void,
			Expr: callee.Pt,
		}
	}
	kernel := func(first, cnt int) cgen.Gen {
		var assn cgen.Gen
		if a.epochs2 > 1 && cnt == 1 {
			assn = cgen.Assign{
				Expr1: a.epochCoord,
				Expr2: il(first),
			}
		}
		return cgen.Stmts{
			assn,
			a.kernel1(),
		}
	}
	if a.epochs1 > 0 {
		a.sliceTile = a.sliceTile1
		a.sliceScrap = a.sliceScrap1
		a.coreBytes = a.datCoreBytes1
		a.groupBytes = a.datGroupBytes1
		a.fieldBytes = a.datFieldBytes1
		put := kernel(0, a.epochs1)
		if a.epochs1 < a.epochs2 {
			put = cgen.If{
				Cond: cgen.CmpL{
					Expr1: a.epochCoord,
					Expr2: il(a.epochs1),
				},
				Then: cgen.Stmts{
					put,
					cgen.Return{},
				},
			}
		}
		body[6] = put
	}
	if a.epochs1 < a.epochs2 {
		a.sliceTile = a.sliceTile2
		a.sliceScrap = a.sliceScrap2
		a.coreBytes = a.datCoreBytes2
		a.groupBytes = a.datGroupBytes2
		a.fieldBytes = a.datFieldBytes2
		body[7] = kernel(a.epochs1, 1)
	}
	return callee.Func(body)
}

func (a *arrangeDats) kernel1() cgen.Gen {
	a.datPtrs = a.datPtrs[:0]
	a.bnPtrs = a.bnPtrs[:0]
	var (
		stmts     cgen.Stmts
		tensorIdx = 0
	)
	decl := func(ptr, expr cgen.Gen) {
		stmts = append(
			stmts, cgen.Var{
				Type: cgen.RestrictPtrChar,
				What: ptr, Init: expr,
			},
		)
	}
	tensor := func() cgen.Gen {
		i := tensorIdx
		tensorIdx++
		return cgen.Elem{
			Arr: a.tensors,
			Idx: il(i),
		}
	}
	datPtr := func() {
		var (
			ptr    = vb(a.name("datPtr"))
			i      = len(a.datPtrs)
			pitch1 = a.From.Pitch1Bytes[i]
			pitch2 = a.From.Pitch2Bytes[i]
		)
		a.datPtrs = append(a.datPtrs, ptr)
		decl(
			ptr, addMul(
				cgen.Sub{
					Expr1: tensor(),
					Expr2: il(
						a.PaddingH*pitch1 +
							a.PaddingW*a.datBytes,
					),
				},
				il(a.slices1*pitch2),
				a.epochCoord,
			),
		)
	}
	datPtrs := func(n int) {
		for ; n > 0; n-- {
			datPtr()
		}
	}
	bnPtr := func() {
		ptr := vb(a.name("bnPtr"))
		a.bnPtrs = append(a.bnPtrs, ptr)
		decl(
			ptr, &bn.Offset{
				Ctx: a.bc,
				Mas: tensor(),
				Channel: cgen.Mul{
					Expr1: il(a.slices1),
					Expr2: a.epochCoord,
				},
			},
		)
	}
	datPtr()
	for op := range a.From.Ops {
		op := &a.From.Ops[op]
		switch op.Kind {
		case mod.Add:
			datPtrs(op.Int)
		case mod.Bn:
			bnPtr()
		case mod.ReLU:
		default:
			panic("bug")
		}
	}
	a.arranged = vb(a.name("arranged"))
	decl(
		a.arranged, addMul(
			tensor(),
			il(a.datEpochBytes1),
			a.epochCoord,
		),
	)
	return append(
		stmts,
		a.kernel2(),
	)
}

func (a *arrangeDats) kernel2() cgen.Gen {
	a.groupIdx = vb(a.name("i"))
	var (
		stmts = make(cgen.Stmts, 3)
		iters = 0
	)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: a.groupIdx,
		Init: cgen.Mul{
			Expr1: il(a.groupTile),
			Expr2: a.groupCoord,
		},
	}
	switch a.groupTiles {
	case a.groupHull:
		iters = a.groupTile
	case 0:
		iters = a.groupScrap
	}
	switch iters {
	case 1:
		stmts[2] = a.kernel3()
	default:
		var (
			last = vb(a.name("ii"))
			expr cgen.Gen
		)
		switch iters {
		case 0:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: a.groupCoord,
						Expr2: il(a.groupTiles),
					},
					Then: il(a.groupTile - 1),
					Else: il(a.groupScrap - 1),
				},
			}
		default:
			expr = il(iters - 1)
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: a.groupIdx,
				Expr2: expr,
			},
		}
		stmts[2] = cgen.For{
			Cond: cgen.CmpLE{
				Expr1: a.groupIdx,
				Expr2: last,
			},
			Post: cgen.IncPre{
				Expr: a.groupIdx,
			},
			Body: a.kernel3(),
		}
	}
	return stmts
}

func (a *arrangeDats) kernel3() cgen.Gen {
	a.coreIdx = vb(a.name("j"))
	switch a.coreHull {
	case 1:
		a.coreLast = nil
	default:
		a.coreLast = vb(a.name("last"))
	}
	stmts := make(cgen.Stmts, 3)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: a.coreIdx,
		Init: cgen.Mul{
			Expr1: il(a.coreTile),
			Expr2: a.coreCoord,
		},
	}
	if a.coreLast != nil {
		var expr cgen.Gen
		switch a.coreTiles {
		case a.coreHull:
			expr = il(a.coreTile - 1)
		case 0:
			expr = il(a.coreScrap - 1)
		default:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: a.coreCoord,
						Expr2: il(a.coreTiles),
					},
					Then: il(a.coreTile - 1),
					Else: il(a.coreScrap - 1),
				},
			}
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: a.coreLast,
			Init: cgen.Add{
				Expr1: a.coreIdx,
				Expr2: expr,
			},
		}
	}
	stmts[2] = a.kernel4()
	return stmts
}

func (a *arrangeDats) kernel4() cgen.Gen {
	var (
		lh  *loopH
		rel cgen.Gen
		lw  *loopW
	)
	layer7 := func() cgen.Gen {
		var retIf cgen.Gen
		if a.coreLast != nil {
			retIf = cgen.If1{
				Cond: cgen.CmpGE{
					Expr1: a.coreIdx,
					Expr2: a.coreLast,
				},
				Then: cgen.Return{},
			}
		}
		return cgen.Stmts{
			a.kernel5(),
			retIf,
			cgen.IncPre{
				Expr: a.coreIdx,
			},
		}
	}
	layer6 := func() cgen.Gen {
		if lw.fromStep == 0 {
			return layer7()
		}
		last := vb(a.name("jj"))
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: last,
				Init: cgen.Add{
					Expr1: cgen.Sub{
						Expr1: il(lw.blkPast - 1),
						Expr2: rel,
					},
					Expr2: a.coreIdx,
				},
			},
			cgen.For{
				Cond: cgen.CmpLE{
					Expr1: a.coreIdx,
					Expr2: last,
				},
				Post: cgen.AddAssign{
					Expr1: a.coreW,
					Expr2: il(lw.fromStep),
				},
				Body: layer7(),
			},
		}
	}
	layer5 := func() cgen.Gen {
		a.coreW = vb(a.name("w"))
		a.spans = &lw.spans
		var expr cgen.Gen
		switch lw.fromStep {
		case 0:
			expr = il(lw.fromW)
		default:
			expr = addMul(
				il(lw.fromW-lw.blkFirst*lw.fromStep),
				il(lw.fromStep),
				rel,
			)
		}
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: a.coreW,
				Init: expr,
			},
			layer6(),
		}
	}
	layer4 := func() cgen.Gen {
		var (
			lws  = lh.lws
			tree func(int, int) cgen.Stmts
		)
		leaf := func(x int) cgen.Stmts {
			lw = lws[x]
			var assn cgen.Gen
			if x+1 < len(lws) {
				assn = cgen.Assign{
					Expr1: rel,
					Expr2: il(lw.blkPast),
				}
			}
			return cgen.Stmts{
				layer5(),
				assn,
			}
		}
		tree = func(first, last int) cgen.Stmts {
			if first == last {
				return leaf(first)
			}
			var (
				start = lws[first].blkFirst
				stop  = lws[last].blkPast
				split = start + (stop-start)/2
				x     = first + 1
			)
			for lws[x].blkPast <= split {
				x++
			}
			return cgen.Stmts{
				cgen.If{
					Cond: cgen.CmpL{
						Expr1: rel,
						Expr2: il(lws[x].blkFirst),
					},
					Then: tree(first, x-1),
				},
				tree(x, last),
			}
		}
		return tree(0, len(lws)-1)
	}
	layer3 := func() cgen.Gen {
		if lh.fromStep == 0 {
			return layer4()
		}
		return cgen.For{
			Cond: cgen.CmpL{
				Expr1: a.coreIdx,
				Expr2: il(lh.blkPast),
			},
			Post: cgen.CommaSpaced{
				cgen.Assign{
					Expr1: rel,
					Expr2: il(0),
				},
				cgen.AddAssign{
					Expr1: a.coreH,
					Expr2: il(lh.fromStep),
				},
			},
			Body: layer4(),
		}
	}
	layer2 := func() cgen.Gen {
		rel = vb(a.name("rel"))
		a.coreH = vb(a.name("h"))
		var (
			relExpr cgen.Gen = cgen.Sub{
				Expr1: a.coreIdx,
				Expr2: il(lh.blkFirst),
			}
			hExpr = il(lh.fromH)
		)
		if lh.blkStep != 0 {
			var (
				numer cgen.Gen = cgen.Cast{
					Type: cgen.SizeT,
					Expr: cgen.Paren{
						Inner: relExpr,
					},
				}
				denom = il(lh.blkStep)
			)
			relExpr = cgen.Rem{
				Expr1: numer,
				Expr2: denom,
			}
			hExpr = addMul(
				hExpr,
				cgen.Quo{
					Expr1: numer,
					Expr2: denom,
				},
				il(lh.fromStep),
			)
		}
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: rel,
				Init: relExpr,
			},
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: a.coreH,
				Init: hExpr,
			},
			layer3(),
		}
	}
	layer1 := func() cgen.Gen {
		var (
			lhs  = a.blks.lhs
			tree func(int, int) cgen.Stmts
		)
		leaf := func(x int) cgen.Stmts {
			lh = lhs[x]
			var assn cgen.Gen
			if x+1 < len(lhs) {
				assn = cgen.Assign{
					Expr1: a.coreIdx,
					Expr2: il(lh.blkPast),
				}
			}
			return cgen.Stmts{
				layer2(),
				assn,
			}
		}
		tree = func(first, last int) cgen.Stmts {
			if first == last {
				return leaf(first)
			}
			var (
				start = lhs[first].blkFirst
				stop  = lhs[last].blkPast
				split = start + (stop-start)/2
				x     = first + 1
			)
			for lhs[x].blkPast <= split {
				x++
			}
			return cgen.Stmts{
				cgen.If{
					Cond: cgen.CmpL{
						Expr1: a.coreIdx,
						Expr2: il(lhs[x].blkFirst),
					},
					Then: tree(first, x-1),
				},
				tree(x, last),
			}
		}
		return tree(0, len(lhs)-1)
	}
	return layer1()
}

func (a *arrangeDats) kernel5() cgen.Gen {
	a.sliceIdx = vb(a.name("k"))
	var (
		stmts = make(cgen.Stmts, 3)
		iters = 0
	)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: a.sliceIdx,
		Init: cgen.Mul{
			Expr1: il(a.sliceTile),
			Expr2: a.sliceCoord,
		},
	}
	switch {
	case a.sliceTiles == a.sliceHull:
		iters = a.sliceTile
	case a.sliceTiles == 0:
		fallthrough
	case a.sliceTile == a.sliceScrap:
		iters = a.sliceScrap
	}
	switch iters {
	case 1:
		stmts[2] = a.kernel6()
	default:
		var (
			last = vb(a.name("kk"))
			expr cgen.Gen
		)
		switch iters {
		case 0:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: a.sliceCoord,
						Expr2: il(a.sliceTiles),
					},
					Then: il(a.sliceTile - 1),
					Else: il(a.sliceScrap - 1),
				},
			}
		default:
			expr = il(iters - 1)
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: a.sliceIdx,
				Expr2: expr,
			},
		}
		stmts[2] = cgen.For{
			Cond: cgen.CmpLE{
				Expr1: a.sliceIdx,
				Expr2: last,
			},
			Post: cgen.IncPre{
				Expr: a.sliceIdx,
			},
			Body: a.kernel6(),
		}
	}
	return stmts
}

func (a *arrangeDats) kernel6() cgen.Gen {
	layer2 := func() cgen.Gen {
		switch a.platform {
		case raw.AVX512Float32:
			return a.m512()
		default:
			panic("bug")
		}
	}
	layer1 := func() cgen.Gen {
		a.bnMuls = a.bnMuls[:0]
		a.bnAdds = a.bnAdds[:0]
		var (
			last = len(a.bnPtrs)
			gens = make(cgen.Gens, last+1)
		)
		ch := cgen.Paren{
			Inner: addMul(
				a.sliceIdx,
				il(a.fromChans),
				a.groupIdx,
			),
		}
		for x, bnPtr := range a.bnPtrs {
			var (
				bnMul = vb(a.name("bnMul"))
				bnAdd = vb(a.name("bnAdd"))
			)
			a.bnMuls = append(a.bnMuls, bnMul)
			a.bnAdds = append(a.bnAdds, bnAdd)
			gens[x] = &bn.Load{
				Ctx:     a.bc,
				Mas:     bnPtr,
				Channel: ch,
				Mul:     bnMul,
				Add:     bnAdd,
			}
		}
		gens[last] = layer2()
		return gens
	}
	return layer1()
}

func (a *arrangeDats) m512() cgen.Gen {
	var (
		sbox     [][]int
		locs     []int
		sboxH    int
		vecIdx   int
		dats     []cgen.Gen
		sboxW    int
		fieldIdx int
		pms      map[int]cgen.Gen
	)
	layer6 := func() cgen.Gen {
		var (
			stmts = cgen.Stmts{nil}
			zero  = avx.Mm512SetzeroPs
		)
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		anOff := func(from int) int {
			var (
				loc1 = 0
				loc2 = -1
				seek int
			)
			for x := 0; x < 16; x++ {
				for w, fld := range sbox[0] {
					if fld == 0 {
						continue
					}
					at := x*a.StrideW + w
					if at >= loc1+16 {
						loc1 = at
					}
					switch loc2 {
					case -1:
						if at == loc1+from {
							loc2 = loc1
							seek = at
							for seek < loc1+16 {
								seek += a.StrideW
							}
						}
					default:
						if at == seek {
							return loc1 - loc2
						}
					}
				}
			}
			return 16
		}
		ctrl := func(from, off int) cgen.Gen {
			if pms == nil {
				pms = make(map[int]cgen.Gen)
			}
			pm := pms[from+off*16]
			if pm == nil {
				pm = vb(a.name("pm"))
				if off == 0 {
					off = anOff(from)
				}
				var (
					set = make(avx.Mm512SetEpi32, 16)
					at  = from
				)
				for x := 15; x >= 0; x-- {
					set[x] = il(min(at, 31))
					was := at
					at += a.StrideW
					if was < 16 && at >= 16 {
						at -= off
						at += 16
					}
				}
				stmt(cgen.Var{
					Type: avx.M512i, What: pm,
					Init: set,
				})
				pms[from+off*16] = pm
				pms[from] = pm
			}
			return pm
		}
		store := func(to, lanes int, expr cgen.Gen) cgen.Gen {
			var (
				ae   = a.arranged
				mask = loMask(lanes)
			)
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(
					fieldIdx*a.fieldBytes +
						vecIdx*a.datVecBytes +
						to*a.datBytes,
				),
			}
			ae = addMul(ae, il(a.groupBytes), a.groupIdx)
			ae = addMul(ae, il(a.coreBytes), a.coreIdx)
			ae = addMul(ae, il(a.datSliceBytes), a.sliceIdx)
			return avx.Mm512MaskStoreuPs{
				ae, mask, expr,
			}
		}
		var (
			x = 0
			w = sboxW
		)
		for need := 16; need > 0; {
			var (
				lanes  = 0
				nonpad = 0
			)
			take := func() {
				lanes++
				h := vecIdx*a.StrideH + sboxH
				if h >= a.spanH1 && h < a.spanH2 &&
					w >= a.spanW1 && w < a.spanW2 {
					nonpad++
				}
				w += a.StrideW
			}
			for locs[x]+16 <= w {
				x++
			}
			var (
				ndats = 1
				loc1  = locs[x]
				dat1  = dats[x]
				from  = w - loc1
			)
			take()
			for lanes < need && w < loc1+16 {
				take()
			}
			var (
				loc2 = loc1
				dat2 cgen.Gen
			)
			if lanes < need {
				x++
				for locs[x]+16 <= w {
					x++
				}
				ndats = 2
				loc2 = locs[x]
				dat2 = dats[x]
				take()
				for lanes < need && w < loc2+16 {
					take()
				}
				if lanes < need && loc2+16 > a.spanW2 {
					lanes = need
				}
			}
			switch {
			case nonpad == 0:
				if stmts[0] == nil {
					stmts[0] = store(0, 16, zero)
				}
			case a.StrideW == 1:
				stmt(store(0, 16, dat1))
			default:
				var (
					to   = 16 - need
					off  = loc2 - loc1
					pm   = ctrl(from, off)
					expr cgen.Gen
				)
				switch ndats {
				case 1:
					expr = avx.Mm512PermutexvarPs{
						pm, dat1,
					}
				default:
					switch {
					case dat1 == nil:
						dat1 = zero
					case dat2 == nil:
						dat2 = zero
					}
					expr = avx.Mm512Permutex2varPs{
						dat1, pm, dat2,
					}
				}
				stmt(store(to, lanes, expr))
			}
			need -= lanes
		}
		return stmts
	}
	layer5 := func() cgen.Gen {
		var gens cgen.Gens
		for w, fld := range sbox[sboxH] {
			if fld == 0 {
				continue
			}
			sboxW = w
			fieldIdx = ^fld
			gens = append(
				gens, layer6(),
			)
		}
		return gens
	}
	layer4 := func() cgen.Gen {
		if dats == nil {
			dats = make([]cgen.Gen, len(locs))
		}
		h := vecIdx*a.StrideH + sboxH
		if h < a.spanH1 || h >= a.spanH2 {
			for x := range dats {
				dats[x] = nil
			}
			return layer5()
		}
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		load := func(mask cgen.Gen, ptr, w int) cgen.Gen {
			var (
				ae         = a.datPtrs[ptr]
				pitch1     = a.From.Pitch1Bytes[ptr]
				pitch2     = a.From.Pitch2Bytes[ptr]
				groupPitch = a.fromChans * pitch2
			)
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(h*pitch1 + w*a.datBytes),
			}
			ae = addMul(ae, il(groupPitch), a.groupIdx)
			ae = addMul(ae, il(pitch2), a.sliceIdx)
			ae = addMul(ae, il(pitch1), a.coreH)
			ae = addMul(ae, il(a.datBytes), a.coreW)
			return avx.Mm512MaskzLoaduPs{
				mask, ae,
			}
		}
		for x, loc := range locs {
			if loc+16 <= a.spanW1 || loc >= a.spanW2 {
				dats[x] = nil
				continue
			}
			dat := vb(a.name("dat"))
			dats[x] = dat
			var (
				lane   = max(a.spanW1-loc, 0)
				lanes  = min(a.spanW2-loc, 16) - lane
				mask1  = 1<<uint(lanes) - 1
				mask2  = mask1 << uint(lane)
				mask3  = il(mask2)
				datPtr = 0
				bnPtr  = 0
			)
			stmt(cgen.Var{
				Type: avx.M512, What: dat,
				Init: load(mask3, datPtr, loc),
			})
			for op := range a.From.Ops {
				op := &a.From.Ops[op]
				switch op.Kind {
				case mod.Add:
					for n := op.Int; n > 0; n-- {
						datPtr++
						ld := load(mask3, datPtr, loc)
						stmt(cgen.Assign{
							Expr1: dat,
							Expr2: avx.Mm512AddPs{
								dat, ld,
							},
						})
					}
				case mod.Bn:
					stmt(&bn.Apply{
						Ctx:  a.bc,
						Mul:  a.bnMuls[bnPtr],
						Add:  a.bnAdds[bnPtr],
						To:   dat,
						Mask: mask3,
					})
					bnPtr++
				case mod.ReLU:
					stmt(&act.ReLU{
						Ctx:      a.ac,
						NegSlope: op.Float,
						Var:      dat,
					})
				default:
					panic("bug")
				}
			}
		}
		stmt(layer5())
		return stmts
	}
	layer3 := func() cgen.Gen {
		var gens cgen.Gens
		for h := range sbox {
			if sbox[h] == nil {
				continue
			}
			sboxH = h
			for x := 0; x < a.datSliceVecs; x++ {
				vecIdx = x
				gens = append(
					gens, layer4(),
				)
			}
		}
		return gens
	}
	layer2 := func() cgen.Gen {
		past := 0
		for x := 0; x < 16; x++ {
			for w, fld := range sbox[0] {
				if fld == 0 {
					continue
				}
				at := x*a.StrideW + w
				if at < past {
					continue
				}
				locs = append(locs, at)
				past = at + 16
			}
		}
		return layer3()
	}
	layer1 := func() cgen.Gen {
		sbox = make([][]int, a.StrideH)
		for x, fld := range a.fields {
			var (
				h = fld.sboxH
				w = fld.sboxW
			)
			if sbox[h] == nil {
				sbox[h] = make([]int, a.StrideW)
			}
			sbox[h][w] = ^x
		}
		return layer2()
	}
	return layer1()
}

type ProduceSums struct {
	*Ctx
	*Spec
	Team    cgen.Gen
	Tensors []cgen.Gen
	*layout
	callerName string
}

func (p *ProduceSums) Prep() cgen.Gen {
	p.layout = newLayout(p.Ctx, p.Spec)
	const affix = "ProduceSums"
	sig := fmt.Sprint(affix, " ", p.Spec)
	if prior, ok := p.dedup[sig]; ok {
		p.callerName = prior.(string)
		return nil
	}
	p.callerName = p.name(p.prefix + affix)
	p.dedup[sig] = p.callerName
	return cgen.Gens{
		&produceSums{ProduceSums: p},
		cgen.Newline,
	}
}

func (p *ProduceSums) Bytes() int {
	return p.sumTotalBytes
}

func (p *ProduceSums) Append(to []byte) []byte {
	var (
		tensors = vb(p.name("tensors"))
		ptrs    = cgen.CommaLines(p.Tensors)
	)
	return cgen.Stmts{
		cgen.Var{
			Type: cgen.PtrChar,
			What: cgen.Elem{Arr: tensors},
			Init: cgen.Brace{Inner: ptrs},
		},
		cgen.Call{
			Func: vb(p.callerName),
			Args: cgen.CommaSpaced{
				p.Team, tensors,
			},
		},
	}.Append(to)
}

type produceSums struct {
	*ProduceSums
	nodeTbl       cgen.Gen
	epochFirst    int
	epochCnt      int
	slices        int
	wtCoreBytes   int
	wtNodeBytes   int
	wtGroupBytes  int
	datCoreBytes  int
	datGroupBytes int
	datFieldBytes int
	wtTile        int
	wtTiles       int
	wtScrap       int
	wtHull        int
	calleeName    string
	tensors       cgen.Gen
	epochCoord    cgen.Gen
	fieldCoord    cgen.Gen
	nodeFirst     cgen.Gen
	groupCoord    cgen.Gen
	toCoord       cgen.Gen
	nodeOff       cgen.Gen
	wtCoord       cgen.Gen
	nodeCoord     cgen.Gen
	lift          cgen.Gen
	pileCoord     cgen.Gen
	base          cgen.Gen
	fromCoord     cgen.Gen
	biasPtr       cgen.Gen
	wtPtr         cgen.Gen
	datPtr        cgen.Gen
	sumPtr        cgen.Gen
	vecs1         int
	vecs2         int
	bnPre         bool
	bias          bool
	rdwr          bool
	wtIdx         cgen.Gen
	wtShort       bool
}

func (p *produceSums) Append(to []byte) []byte {
	nm := func(s string) string {
		return p.name(p.callerName + s)
	}
	fieldTbl := vb(nm("FieldTbl"))
	p.nodeTbl = vb(nm("NodeTbl"))
	type fn func(int) cgen.Gen
	table := func(arr cgen.Gen, n int, line fn) cgen.Gen {
		lines := make(cgen.CommaLines, n)
		for x := range lines {
			lines[x] = line(x)
		}
		return cgen.Gens{
			cgen.Static{
				Tail: cgen.Var{
					Type: cgen.PtrdiffT,
					What: cgen.Elem{Arr: arr},
					Init: cgen.Brace{Inner: lines},
				},
			},
			cgen.Newline,
			cgen.Newline,
		}
	}
	callee := func(first, cnt int) cgen.Gen {
		p.epochFirst = first
		p.epochCnt = cnt
		switch {
		case first < p.epochs1:
			p.slices = p.slices1
			p.wtCoreBytes = p.wtCoreBytes11
			p.wtNodeBytes = p.wtNodeBytes1
			p.wtGroupBytes = p.wtGroupBytes1
			p.datCoreBytes = p.datCoreBytes1
			p.datGroupBytes = p.datGroupBytes1
			p.datFieldBytes = p.datFieldBytes1
		default:
			p.slices = p.slices2
			p.wtCoreBytes = p.wtCoreBytes21
			p.wtNodeBytes = p.wtNodeBytes2
			p.wtGroupBytes = p.wtGroupBytes2
			p.datCoreBytes = p.datCoreBytes2
			p.datGroupBytes = p.datGroupBytes2
			p.datFieldBytes = p.datFieldBytes2
		}
		var threadSlices int
		switch p.platform {
		case raw.AVX512Float32:
			threadSlices = 512
		default:
			panic("bug")
		}
		var (
			tile  = ceilQuo(threadSlices, p.slices)
			tiles = max(p.wtCores2/tile, 1)
		)
		p.wtTile = p.wtCores2 / tiles
		p.wtTiles = tiles
		p.wtScrap = p.wtCores2 - tiles*p.wtTile
		p.wtHull = tiles
		if p.wtScrap > 0 {
			p.wtTiles--
			p.wtScrap += p.wtTile
		}
		p.calleeName = nm("Callee")
		return cgen.Gens{
			p.calleeFunc(),
			cgen.Newline,
		}
	}
	var (
		team    = vb(p.name("team"))
		tensors = vb(p.name("tensors"))
		tuple   = vb(p.name("tuple"))
	)
	store := func(x int, expr cgen.Gen) cgen.Gen {
		return cgen.Assign{
			Expr1: cgen.Elem{
				Arr: tuple, Idx: il(x),
			},
			Expr2: cgen.Cast{
				Type: cgen.PtrVoid,
				Expr: expr,
			},
		}
	}
	loop3 := func(fieldCoord cgen.Gen) cgen.Gen {
		var (
			nodeFirst = vb(p.name("node"))
			nodeStep  = vb(p.name("step"))
			nodePast  = vb(p.name("past"))
		)
		load := func(x int, what cgen.Gen) cgen.Gen {
			return cgen.Var{
				Type: cgen.PtrdiffT,
				What: what,
				Init: cgen.Elem{
					Arr: fieldTbl,
					Idx: addMul(
						il(x),
						il(2),
						fieldCoord,
					),
				},
			}
		}
		return cgen.Stmts{
			load(0, nodeFirst),
			load(1, nodeStep),
			load(2, nodePast),
			cgen.For{
				Cond: cgen.CmpL{
					Expr1: nodeFirst,
					Expr2: nodePast,
				},
				Post: cgen.AddAssign{
					Expr1: nodeFirst,
					Expr2: nodeStep,
				},
				Body: cgen.Stmts{
					store(3, nodeFirst),
					&threader.Do{
						Ctx:    p.tc,
						Callee: vb(p.calleeName),
						Any:    tuple,
						Hull: []cgen.Gen{
							il(p.wtHull),
							nodeStep,
							il(p.sumCores),
							il(p.Groups),
						},
						Team: team,
					},
				},
			},
		}
	}
	loop2 := func() cgen.Gen {
		fieldCoord := vb(p.name("field"))
		return cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT,
				What: fieldCoord,
				Init: il(0),
			},
			Cond: cgen.CmpL{
				Expr1: fieldCoord,
				Expr2: il(len(p.fields)),
			},
			Post: cgen.IncPre{
				Expr: fieldCoord,
			},
			Body: cgen.Stmts{
				store(2, fieldCoord),
				loop3(fieldCoord),
			},
		}
	}
	loop1 := func() cgen.Gen {
		epochCoord := vb(p.name("epoch"))
		return cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT,
				What: epochCoord,
				Init: il(p.epochFirst),
			},
			Cond: cgen.CmpL{
				Expr1: epochCoord,
				Expr2: il(
					p.epochFirst + p.epochCnt,
				),
			},
			Post: cgen.IncPre{
				Expr: epochCoord,
			},
			Body: cgen.Stmts{
				store(1, epochCoord),
				loop2(),
			},
		}
	}
	var (
		prep = make(cgen.Gens, 4)
		body = make(cgen.Stmts, 4)
	)
	prep[0] = table(
		fieldTbl,
		len(p.fields),
		func(x int) cgen.Gen {
			var (
				fld  = p.fields[x]
				past cgen.Gen
			)
			if x+1 == len(p.fields) {
				past = il(len(p.nodes))
			}
			return cgen.CommaSpaced{
				il(fld.nodeFirst),
				il(fld.nodeStep),
				past,
			}
		},
	)
	prep[1] = table(
		p.nodeTbl,
		len(p.nodes),
		func(x int) cgen.Gen {
			nd := p.nodes[x]
			return cgen.CommaSpaced{
				il(p.lifts[nd.deck]),
				il(nd.pile),
				il(btoi(nd.base)),
			}
		},
	)
	body[0] = cgen.Var{
		Type: cgen.PtrVoid,
		What: cgen.Elem{
			Arr: tuple, Idx: il(4),
		},
	}
	body[1] = cgen.Assign{
		Expr1: cgen.Elem{
			Arr: tuple, Idx: il(0),
		},
		Expr2: tensors,
	}
	if p.epochs1 > 0 {
		prep[2] = callee(0, p.epochs1)
		body[2] = loop1()
	}
	if p.epochs1 < p.epochs2 {
		prep[3] = callee(p.epochs1, 1)
		body[3] = loop1()
	}
	return cgen.Gens{
		prep,
		cgen.StaticFuncDef{
			ReturnType: cgen.Void,
			Name:       p.callerName,
			Params: cgen.CommaSpaced{
				cgen.Param{
					Type: p.tc.PtrTeam,
					What: team,
				},
				cgen.Param{
					Type: cgen.PtrPtrChar,
					What: tensors,
				},
			},
			Body: body,
		},
	}.Append(to)
}

func (p *produceSums) calleeFunc() cgen.Gen {
	callee := &threader.Callee{
		Ctx:  p.tc,
		Name: p.calleeName,
		Task: vb(p.name("task")),
		Pt:   vb(p.name("pt")),
	}
	var stmts cgen.Stmts
	stmt := func(st cgen.Gen) {
		stmts = append(stmts, st)
	}
	tuple := vb(p.name("tuple"))
	stmt(cgen.Var{
		Type: cgen.PtrPtrVoid, What: tuple,
		Init: callee.Any(),
	})
	p.tensors = vb(p.name("tensors"))
	stmt(cgen.Var{
		Type: cgen.PtrPtrChar, What: p.tensors,
		Init: cgen.Elem{
			Arr: tuple, Idx: il(0),
		},
	})
	tupleIdx := 1
	tupleVar := func(nm string, expr cgen.Gen) cgen.Gen {
		ret := vb(p.name(nm))
		if expr == nil {
			expr = cgen.Cast{
				Type: cgen.PtrdiffT,
				Expr: cgen.Elem{
					Arr: tuple, Idx: il(tupleIdx),
				},
			}
		}
		tupleIdx++
		stmt(cgen.Var{
			Type: cgen.PtrdiffT, What: ret,
			Init: expr,
		})
		return ret
	}
	p.epochCoord = tupleVar(
		"epoch",
		func() cgen.Gen {
			if p.epochCnt == 1 {
				return il(p.epochFirst)
			}
			return nil
		}(),
	)
	p.fieldCoord = tupleVar(
		"field",
		func() cgen.Gen {
			if len(p.fields) == 1 {
				return il(0)
			}
			return nil
		}(),
	)
	p.nodeFirst = tupleVar(
		"nodeFirst",
		func() cgen.Gen {
			step := p.fields[0].nodeStep
			if step == len(p.nodes) {
				return il(0)
			}
			return nil
		}(),
	)
	var (
		ptIdx  = 3
		ptUsed = false
	)
	ptVar := func(nm string, expr cgen.Gen) cgen.Gen {
		ret := vb(p.name(nm))
		if expr == nil {
			expr = cgen.Elem{
				Arr: callee.Pt, Idx: il(ptIdx),
			}
			ptUsed = true
		}
		ptIdx--
		stmt(cgen.Var{
			Type: cgen.PtrdiffT, What: ret,
			Init: expr,
		})
		return ret
	}
	p.groupCoord = ptVar(
		"group",
		func() cgen.Gen {
			if p.Groups == 1 {
				return il(0)
			}
			return nil
		}(),
	)
	p.toCoord = ptVar(
		"to",
		func() cgen.Gen {
			if p.sumCores == 1 {
				return il(0)
			}
			return nil
		}(),
	)
	p.nodeOff = ptVar(
		"nodeOff",
		func() cgen.Gen {
			for _, fld := range p.fields {
				if fld.nodeStep > 1 {
					return nil
				}
			}
			return il(0)
		}(),
	)
	p.wtCoord = ptVar(
		"w",
		func() cgen.Gen {
			if p.wtHull == 1 {
				return il(0)
			}
			return nil
		}(),
	)
	if !ptUsed {
		stmt(cgen.Cast{
			Type: cgen.Void,
			Expr: callee.Pt,
		})
	}
	stmt(p.kernel1())
	return callee.Func(stmts)
}

func (p *produceSums) kernel1() cgen.Gen {
	var stmts cgen.Stmts
	stmt := func(st cgen.Gen) {
		stmts = append(stmts, st)
	}
	decl := func(nm string, expr cgen.Gen) cgen.Gen {
		ret := vb(p.name(nm))
		stmt(cgen.Var{
			Type: cgen.PtrdiffT, What: ret,
			Init: expr,
		})
		return ret
	}
	p.nodeCoord = decl(
		"node",
		cgen.Add{
			Expr1: p.nodeFirst,
			Expr2: p.nodeOff,
		},
	)
	var (
		tblOff  = 0
		tblUsed = false
	)
	tblVar := func(nm string, expr cgen.Gen) cgen.Gen {
		if expr == nil {
			expr = cgen.Elem{
				Arr: p.nodeTbl,
				Idx: addMul(
					il(tblOff),
					il(3),
					p.nodeCoord,
				),
			}
			tblUsed = true
		}
		tblOff++
		return decl(nm, expr)
	}
	p.lift = tblVar(
		"lift",
		func() cgen.Gen {
			if len(p.lifts) == 1 {
				return il(0)
			}
			return nil
		}(),
	)
	p.pileCoord = tblVar(
		"pile",
		func() cgen.Gen {
			if len(p.shifts) == 1 {
				return il(0)
			}
			return nil
		}(),
	)
	p.base = tblVar(
		"base",
		func() cgen.Gen {
			for _, nd := range p.nodes {
				if !nd.base {
					return nil
				}
			}
			return il(1)
		}(),
	)
	if !tblUsed {
		stmt(cgen.Cast{
			Type: cgen.Void,
			Expr: p.nodeTbl,
		})
	}
	p.fromCoord = decl(
		"from",
		addMul(
			p.toCoord,
			cgen.Quo{
				Expr1: cgen.Cast{
					Type: cgen.SizeT,
					Expr: p.lift,
				},
				Expr2: il(p.datSliceVecs),
			},
			il(p.blkStep),
		),
	)
	stmt(cgen.If1{
		Cond: cgen.CmpGE{
			Expr1: p.fromCoord,
			Expr2: il(p.datCores),
		},
		Then: cgen.Return{},
	})
	stmt(p.kernel2())
	return stmts
}

func (p *produceSums) kernel2() cgen.Gen {
	var stmts cgen.Stmts
	stmt := func(st cgen.Gen) {
		stmts = append(stmts, st)
	}
	decl := func(nm string, expr cgen.Gen) cgen.Gen {
		ret := vb(p.name(nm))
		stmt(cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ret, Init: expr,
		})
		return ret
	}
	tensor := func(x int) cgen.Gen {
		return cgen.Elem{
			Arr: p.tensors, Idx: il(x),
		}
	}
	p.biasPtr = decl(
		"biasPtr",
		func() (ae cgen.Gen) {
			ae = tensor(0)
			ae = addMul(ae, il(p.biasEpochBytes), p.epochCoord)
			ae = addMul(ae, il(p.biasGroupBytes), p.groupCoord)
			return
		}(),
	)
	p.wtPtr = decl(
		"wtPtr",
		func() (ae cgen.Gen) {
			ae = cgen.Add{
				Expr1: tensor(0),
				Expr2: il(p.biasTotalBytes),
			}
			ae = addMul(ae, il(p.wtEpochBytes1), p.epochCoord)
			ae = addMul(ae, il(p.wtGroupBytes), p.groupCoord)
			ae = addMul(ae, il(p.wtNodeBytes), p.nodeCoord)
			return
		}(),
	)
	p.datPtr = decl(
		"datPtr",
		func() (ae cgen.Gen) {
			ae = tensor(1)
			ae = addMul(ae, il(p.datEpochBytes1), p.epochCoord)
			ae = addMul(ae, il(p.datFieldBytes), p.fieldCoord)
			ae = addMul(ae, il(p.datGroupBytes), p.groupCoord)
			ae = addMul(ae, il(p.datCoreBytes), p.fromCoord)
			return
		}(),
	)
	p.sumPtr = decl(
		"sumPtr",
		func() (ae cgen.Gen) {
			ae = tensor(2)
			ae = addMul(ae, il(p.sumGroupBytes), p.groupCoord)
			ae = addMul(ae, il(p.sumCoreBytes), p.toCoord)
			ae = addMul(ae, il(p.sumPileBytes), p.pileCoord)
			return
		}(),
	)
	stmt(p.kernel3())
	return stmts
}

func (p *produceSums) kernel3() cgen.Gen {
	if len(p.lifts) == 1 {
		p.vecs1 = 0
		p.vecs2 = p.datSliceVecs
		return p.kernel4()
	}
	var (
		cases cgen.Stmts
		need  = 0
		pair  = 1
	)
	if p.sumCores > p.blkStep {
		pair = 3
	}
	for _, lift := range p.lifts {
		vecs := lift % p.datSliceVecs
		need |= pair << uint(vecs*2)
	}
	for x := 0; need != 0; x, need = x+1, need>>1 {
		if x == 1 || need&1 == 0 {
			continue
		}
		p.vecs1 = x >> 1
		p.vecs2 = p.datSliceVecs - p.vecs1
		if x&1 == 0 {
			p.vecs1 = 0
		}
		cases = append(
			cases,
			cgen.Case{
				Expr: func() cgen.Gen {
					if x == 0 {
						return nil
					}
					return il(x)
				}(),
				Body: cgen.Stmts{
					p.kernel4(),
					cgen.Break,
				},
			},
		)
	}
	return cgen.Switch{
		Expr: cgen.Add{
			Expr1: cgen.Mul{
				Expr1: cgen.Rem{
					Expr1: cgen.Cast{
						Type: cgen.SizeT,
						Expr: p.lift,
					},
					Expr2: il(p.datSliceVecs),
				},
				Expr2: il(2),
			},
			Expr2: cgen.Paren{
				Inner: cgen.CmpGE{
					Expr1: p.toCoord,
					Expr2: il(p.blkStep),
				},
			},
		},
		Cases: cases,
	}
}

func (p *produceSums) kernel4() cgen.Gen {
	p.bnPre = false
	for x := range p.Filts {
		if p.Filts[x].BnPre > 0 {
			p.bnPre = true
			break
		}
	}
	do := func(b bool) cgen.Gen {
		p.bias = b
		return p.kernel5()
	}
	cond := p.nodeCoord
	switch {
	case p.bnPre:
		if len(p.nodes) == 1 {
			return do(true)
		}
	default:
		if p.epochFirst > 0 {
			return do(false)
		}
		if p.epochCnt == 1 && len(p.nodes) == 1 {
			return do(true)
		}
		cond = cgen.Or{
			Expr1: p.epochCoord,
			Expr2: cond,
		}
	}
	return cgen.Stmts{
		cgen.If{
			Cond: cond,
			Then: cgen.Stmts{
				do(false),
				cgen.Return{},
			},
		},
		do(true),
	}
}

func (p *produceSums) kernel5() cgen.Gen {
	used := false
	do := func(rw bool) cgen.Gen {
		p.rdwr = rw
		var cast cgen.Gen
		if !used {
			cast = cgen.Cast{
				Type: cgen.Void,
				Expr: p.base,
			}
		}
		return cgen.Stmts{
			cast,
			p.kernel6(),
		}
	}
	if p.epochFirst > 0 {
		return do(true)
	}
	if p.epochCnt == 1 {
		if p.bias {
			return do(false)
		}
		all := true
		for _, nd := range p.nodes {
			if !nd.base {
				all = false
				break
			}
		}
		if all {
			return do(false)
		}
	}
	if !p.bnPre && p.bias {
		return do(false)
	}
	used = true
	return cgen.Stmts{
		cgen.If{
			Cond: cgen.Land{
				Expr1: cgen.IsZero{
					Expr: p.epochCoord,
				},
				Expr2: p.base,
			},
			Then: cgen.Stmts{
				do(false),
				cgen.Return{},
			},
		},
		do(true),
	}
}

func (p *produceSums) kernel6() cgen.Gen {
	p.wtIdx = vb(p.name("i"))
	var (
		stmts = make(cgen.Stmts, 4)
		retIf cgen.Gen
	)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: p.wtIdx,
		Init: cgen.Mul{
			Expr1: il(p.wtTile),
			Expr2: p.wtCoord,
		},
	}
	if p.wtHull > 1 {
		var (
			last = vb(p.name("ii"))
			expr cgen.Gen
		)
		switch p.wtTiles {
		case p.wtHull:
			expr = il(p.wtTile - 1)
		case 0:
			expr = il(p.wtScrap - 1)
		default:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: p.wtCoord,
						Expr2: il(p.wtTiles),
					},
					Then: il(p.wtTile - 1),
					Else: il(p.wtScrap - 1),
				},
			}
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: p.wtIdx,
				Expr2: expr,
			},
		}
		retIf = cgen.If1{
			Cond: cgen.CmpGE{
				Expr1: p.wtIdx,
				Expr2: last,
			},
			Then: cgen.Return{},
		}
	}
	if p.wtCores1 > 0 {
		p.wtShort = false
		stmts[2] = cgen.For{
			Cond: cgen.CmpNE{
				Expr1: p.wtIdx,
				Expr2: il(p.wtCores1),
			},
			Post: cgen.IncPre{
				Expr: p.wtIdx,
			},
			Body: cgen.Stmts{
				p.kernel7(),
				retIf,
			},
		}
	}
	if p.wtCores1 < p.wtCores2 {
		p.wtShort = true
		stmts[3] = p.kernel7()
	}
	return stmts
}

func (p *produceSums) kernel7() cgen.Gen {
	switch p.platform {
	case raw.AVX512Float32:
		return p.m512()
	default:
		panic("bug")
	}
}

func (p *produceSums) m512() cgen.Gen {
	var (
		rows     int
		cols     int
		sums     [][]cgen.Gen
		sliceIdx cgen.Gen
		dats     []cgen.Gen
	)
	layer8 := func() cgen.Gen {
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		slicePitch := p.wtSliceBytes1
		if p.wtShort {
			slicePitch = p.wtSliceBytes2
		}
		for r, sums := range sums {
			var (
				ae = p.wtPtr
				wt = vb(p.name("wt"))
			)
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(r * p.wtBytes),
			}
			ae = addMul(ae, il(p.wtCoreBytes), p.wtIdx)
			ae = addMul(ae, il(slicePitch), sliceIdx)
			stmt(cgen.Var{
				Type: avx.M512, What: wt,
				Init: avx.Mm512Set1Ps{
					cgen.At{
						Expr: cgen.Cast{
							Type: cgen.PtrFloat,
							Expr: cgen.Paren{
								Inner: ae,
							},
						},
					},
				},
			})
			for c, sum := range sums {
				stmt(cgen.Assign{
					Expr1: sum,
					Expr2: avx.Mm512FmaddPs{
						wt, dats[c], sum,
					},
				})
			}
		}
		return stmts
	}
	layer7 := func() cgen.Gen {
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		dats = make([]cgen.Gen, cols)
		for c := range dats {
			dat := vb(p.name("dat"))
			dats[c] = dat
		}
		for c, dat := range dats {
			ae := addMul(
				cgen.Add{
					Expr1: p.datPtr,
					Expr2: il(
						p.datSliceBytes +
							(c-cols)*p.datVecBytes,
					),
				},
				il(p.datSliceBytes),
				sliceIdx,
			)
			stmt(cgen.Var{
				Type: avx.M512, What: dat,
				Init: avx.Mm512LoaduPs{ae},
			})
		}
		stmt(layer8())
		return stmts
	}
	layer6 := func() cgen.Gen {
		sliceIdx = vb(p.name("j"))
		return cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT,
				What: sliceIdx,
				Init: il(0),
			},
			Cond: cgen.CmpL{
				Expr1: sliceIdx,
				Expr2: il(p.slices),
			},
			Post: cgen.IncPre{
				Expr: sliceIdx,
			},
			Body: layer7(),
		}
	}
	layer5 := func() cgen.Gen {
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		stmt(layer6())
		for r, sums := range sums {
			for c, sum := range sums {
				off := r * p.datSliceBytes
				off += c * p.datVecBytes
				switch {
				case c < p.vecs1:
					off -= p.blkStep * p.sumCoreBytes
					off += p.vecs2 * p.datVecBytes
				default:
					off -= p.vecs1 * p.datVecBytes
				}
				ae := addMul(
					cgen.Add{
						Expr1: p.sumPtr,
						Expr2: il(off),
					},
					il(p.sumSiteBytes1),
					p.wtIdx,
				)
				if p.rdwr {
					sum = avx.Mm512AddPs{
						sum,
						avx.Mm512LoaduPs{ae},
					}
				}
				stmt(avx.Mm512StoreuPs{
					ae, sum,
				})
			}
		}
		return stmts
	}
	layer4 := func() cgen.Gen {
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		for _, sums := range sums {
			for c := 1; c < cols; c++ {
				stmt(cgen.Var{
					Type: avx.M512,
					What: sums[c],
					Init: sums[0],
				})
			}
		}
		stmt(layer5())
		return stmts
	}
	layer3 := func() cgen.Gen {
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		for r, sums := range sums {
			var expr cgen.Gen
			switch {
			case p.bias:
				expr = addMul(
					cgen.Add{
						Expr1: p.biasPtr,
						Expr2: il(r * p.biasBytes),
					},
					il(p.wtSliceWts1*p.biasBytes),
					p.wtIdx,
				)
				expr = avx.Mm512Set1Ps{
					cgen.At{
						Expr: cgen.Cast{
							Type: cgen.PtrFloat,
							Expr: cgen.Paren{
								Inner: expr,
							},
						},
					},
				}
			default:
				expr = avx.Mm512SetzeroPs
			}
			stmt(cgen.Var{
				Type: avx.M512,
				What: sums[0],
				Init: expr,
			})
		}
		if !p.bias {
			stmt(cgen.Cast{
				Type: cgen.Void,
				Expr: p.biasPtr,
			})
		}
		stmt(layer4())
		return stmts
	}
	layer2 := func() cgen.Gen {
		sums = make([][]cgen.Gen, rows)
		for r := range sums {
			sums[r] = make([]cgen.Gen, cols)
			for c := range sums[r] {
				sum := vb(p.name("sum"))
				sums[r][c] = sum
			}
		}
		return layer3()
	}
	layer1 := func() cgen.Gen {
		rows = p.wtSliceWts1
		if p.wtShort {
			rows = p.wtSliceWts2
		}
		cols = p.vecs1 + p.vecs2
		return layer2()
	}
	return layer1()
}

type ConsumeSums struct {
	*Ctx
	*Spec
	Team       cgen.Gen
	Tensors    []cgen.Gen
	callerName string
}

func (c *ConsumeSums) Prep() cgen.Gen {
	const affix = "ConsumeSums"
	sig := fmt.Sprint(affix, " ", c.Spec)
	if prior, ok := c.dedup[sig]; ok {
		c.callerName = prior.(string)
		return nil
	}
	c.callerName = c.name(c.prefix + affix)
	c.dedup[sig] = c.callerName
	return cgen.Gens{
		&consumeSums{ConsumeSums: c},
		cgen.Newline,
	}
}

func (c *ConsumeSums) Append(to []byte) []byte {
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
			Func: vb(c.callerName),
			Args: cgen.CommaSpaced{
				c.Team, tensors,
			},
		},
	}.Append(to)
}

type consumeSums struct {
	*ConsumeSums
	*layout
	remH       int
	remW       int
	strips1    int
	strips2    int
	cells1     int
	cells2     int
	cellTile   int
	cellTiles  int
	cellScrap  int
	cellHull   int
	stripTile  int
	stripTiles int
	stripScrap int
	stripHull  int
	chanTile   int
	chanTiles  int
	chanScrap  int
	chanHull   int
	groupTile  int
	groupTiles int
	groupScrap int
	groupHull  int
	calleeName string
	tensors    cgen.Gen
	cellCoord  cgen.Gen
	stripCoord cgen.Gen
	chanCoord  cgen.Gen
	groupCoord cgen.Gen
	sumPtr     cgen.Gen
	datPtrs    []cgen.Gen
	bnPtrs     []cgen.Gen
	groupIdx   cgen.Gen
	chanIdx    cgen.Gen
	bnMuls     []cgen.Gen
	bnAdds     []cgen.Gen
	stripIdx   cgen.Gen
	shortH     bool
	cellIdx    cgen.Gen
	shortW     bool
}

func (c *consumeSums) Append(to []byte) []byte {
	c.layout = newLayout(c.Ctx, c.Spec)
	var (
		yield = func(from, pad, filt, dila, str int) int {
			var (
				n1 = from + 2*pad
				n2 = 1 + (filt-1)*dila
			)
			return (n1-n2)/str + 1
		}
		yieldH = yield(
			c.From.Height, c.PaddingH,
			c.FilterH, c.DilationH,
			c.StrideH,
		)
		yieldW = yield(
			c.From.Width, c.PaddingW,
			c.FilterW, c.DilationW,
			c.StrideW,
		)
	)
	c.remH = yieldH % c.datSliceVecs
	c.remW = yieldW % c.datVecDats
	c.strips1 = yieldH / c.datSliceVecs
	c.strips2 = c.strips1 + btoi(c.remH > 0)
	c.cells1 = yieldW / c.datVecDats
	c.cells2 = c.cells1 + btoi(c.remW > 0)
	var (
		cellWork   = len(c.shifts)
		stripWork  = c.cells2 * cellWork
		chanWork   = c.strips2 * stripWork
		groupWork  = c.toChans * chanWork
		threadWork int
	)
	switch c.platform {
	case raw.AVX512Float32:
		threadWork = 512
	default:
		panic("bug")
	}
	c.cellTile = c.cells2
	c.cellTiles = 1
	c.cellScrap = 0
	c.cellHull = 1
	c.stripTile = c.strips2
	c.stripTiles = 1
	c.stripScrap = 0
	c.stripHull = 1
	c.chanTile = 1
	c.chanTiles = c.toChans
	c.chanScrap = 0
	c.chanHull = c.toChans
	c.groupTile = 1
	c.groupTiles = c.Groups
	c.groupScrap = 0
	c.groupHull = c.Groups
	switch {
	case threadWork <= stripWork:
		var (
			tile  = ceilQuo(threadWork, cellWork)
			tiles = max(c.cells2/tile, 1)
		)
		c.cellTile = c.cells2 / tiles
		c.cellTiles = tiles
		c.cellScrap = c.cells2 - tiles*c.cellTile
		c.cellHull = tiles
		if c.cellScrap > 0 {
			c.cellTiles--
			c.cellScrap += c.cellTile
		}
		c.stripTile = 1
		c.stripTiles = c.strips2
		c.stripScrap = 0
		c.stripHull = c.strips2
	case threadWork <= chanWork:
		var (
			tile  = ceilQuo(threadWork, stripWork)
			tiles = max(c.strips2/tile, 1)
		)
		c.stripTile = c.strips2 / tiles
		c.stripTiles = tiles
		c.stripScrap = c.strips2 - tiles*c.stripTile
		c.stripHull = tiles
		if c.stripScrap > 0 {
			c.stripTiles--
			c.stripScrap += c.stripTile
		}
	case threadWork <= groupWork:
		var (
			tile  = ceilQuo(threadWork, chanWork)
			tiles = max(c.toChans/tile, 1)
		)
		c.chanTile = c.toChans / tiles
		c.chanTiles = tiles
		c.chanScrap = c.toChans - tiles*c.chanTile
		c.chanHull = tiles
		if c.chanScrap > 0 {
			c.chanTiles--
			c.chanScrap += c.chanTile
		}
	default:
		c.chanTile = c.toChans
		c.chanTiles = 1
		c.chanScrap = 0
		c.chanHull = 1
		var (
			tile  = ceilQuo(threadWork, groupWork)
			tiles = max(c.Groups/tile, 1)
		)
		c.groupTile = c.Groups / tiles
		c.groupTiles = tiles
		c.groupScrap = c.Groups - tiles*c.groupTile
		c.groupHull = tiles
		if c.groupScrap > 0 {
			c.groupTiles--
			c.groupScrap += c.groupTile
		}
	}
	c.calleeName = c.name(c.callerName + "Callee")
	var (
		team    = vb(c.name("team"))
		tensors = vb(c.name("tensors"))
	)
	return cgen.Gens{
		c.calleeFunc(),
		cgen.Newline,
		cgen.StaticFuncDef{
			ReturnType: cgen.Void,
			Name:       c.callerName,
			Params: cgen.CommaSpaced{
				cgen.Param{
					Type: c.tc.PtrTeam,
					What: team,
				},
				cgen.Param{
					Type: cgen.PtrPtrChar,
					What: tensors,
				},
			},
			Body: &threader.Do{
				Ctx:    c.tc,
				Callee: vb(c.calleeName),
				Any:    tensors,
				Hull: []cgen.Gen{
					il(c.cellHull),
					il(c.stripHull),
					il(c.chanHull),
					il(c.groupHull),
				},
				Team: team,
			},
		},
	}.Append(to)
}

func (c *consumeSums) calleeFunc() cgen.Gen {
	callee := &threader.Callee{
		Ctx:  c.tc,
		Name: c.calleeName,
		Task: vb(c.name("task")),
		Pt:   vb(c.name("pt")),
	}
	var stmts cgen.Stmts
	stmt := func(st cgen.Gen) {
		stmts = append(stmts, st)
	}
	c.tensors = vb(c.name("tensors"))
	stmt(cgen.Var{
		Type: cgen.PtrPtrChar, What: c.tensors,
		Init: callee.Any(),
	})
	var (
		ptIdx  = 0
		ptUsed = false
	)
	ptVar := func(nm string, hull int) cgen.Gen {
		var (
			ret  = vb(c.name(nm))
			expr cgen.Gen
		)
		switch hull {
		case 1:
			expr = il(0)
		default:
			expr = cgen.Elem{
				Arr: callee.Pt, Idx: il(ptIdx),
			}
			ptUsed = true
		}
		ptIdx++
		stmt(cgen.Var{
			Type: cgen.PtrdiffT, What: ret,
			Init: expr,
		})
		return ret
	}
	c.cellCoord = ptVar("cell", c.cellHull)
	c.stripCoord = ptVar("strip", c.stripHull)
	c.chanCoord = ptVar("chan", c.chanHull)
	c.groupCoord = ptVar("group", c.groupHull)
	if !ptUsed {
		stmt(cgen.Cast{
			Type: cgen.Void,
			Expr: callee.Pt,
		})
	}
	stmt(c.kernel1())
	return callee.Func(stmts)
}

func (c *consumeSums) kernel1() cgen.Gen {
	c.datPtrs = nil
	c.bnPtrs = nil
	var (
		stmts     cgen.Stmts
		tensorIdx = 0
	)
	stmt := func(st cgen.Gen) {
		stmts = append(stmts, st)
	}
	decl := func(ptr cgen.Gen) {
		stmt(cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptr,
			Init: cgen.Elem{
				Arr: c.tensors,
				Idx: il(tensorIdx),
			},
		})
		tensorIdx++
	}
	decls := func(n int) {
		for ; n > 0; n-- {
			datPtr := vb(c.name("datPtr"))
			c.datPtrs = append(c.datPtrs, datPtr)
			decl(datPtr)
		}
	}
	c.sumPtr = vb(c.name("sumPtr"))
	decl(c.sumPtr)
	for op := range c.To.Ops {
		op := &c.To.Ops[op]
		switch op.Kind {
		case mod.Add:
			decls(op.Int)
		case mod.Bn:
			bnPtr := vb(c.name("bnPtr"))
			c.bnPtrs = append(c.bnPtrs, bnPtr)
			decl(bnPtr)
		case mod.ReLU:
		default:
			panic("bug")
		}
	}
	var (
		need = len(c.To.Pitch1Bytes)
		have = len(c.datPtrs)
	)
	decls(need - have)
	stmt(c.kernel2())
	return stmts
}

func (c *consumeSums) kernel2() cgen.Gen {
	c.groupIdx = vb(c.name("i"))
	var (
		stmts = make(cgen.Stmts, 3)
		iters = 0
	)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: c.groupIdx,
		Init: cgen.Mul{
			Expr1: il(c.groupTile),
			Expr2: c.groupCoord,
		},
	}
	switch c.groupTiles {
	case c.groupHull:
		iters = c.groupTile
	case 0:
		iters = c.groupScrap
	}
	switch iters {
	case 1:
		stmts[2] = c.kernel3()
	default:
		var (
			last = vb(c.name("ii"))
			expr cgen.Gen
		)
		switch iters {
		case 0:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: c.groupCoord,
						Expr2: il(c.groupTiles),
					},
					Then: il(c.groupTile - 1),
					Else: il(c.groupScrap - 1),
				},
			}
		default:
			expr = il(iters - 1)
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: c.groupIdx,
				Expr2: expr,
			},
		}
		stmts[2] = cgen.For{
			Cond: cgen.CmpLE{
				Expr1: c.groupIdx,
				Expr2: last,
			},
			Post: cgen.IncPre{
				Expr: c.groupIdx,
			},
			Body: c.kernel3(),
		}
	}
	return stmts
}

func (c *consumeSums) kernel3() cgen.Gen {
	c.chanIdx = vb(c.name("j"))
	var (
		stmts = make(cgen.Stmts, 3)
		iters = 0
	)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: c.chanIdx,
		Init: cgen.Mul{
			Expr1: il(c.chanTile),
			Expr2: c.chanCoord,
		},
	}
	switch c.chanTiles {
	case c.chanHull:
		iters = c.chanTile
	case 0:
		iters = c.chanScrap
	}
	switch iters {
	case 1:
		stmts[2] = c.kernel4()
	default:
		var (
			last = vb(c.name("jj"))
			expr cgen.Gen
		)
		switch iters {
		case 0:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: c.chanCoord,
						Expr2: il(c.chanTiles),
					},
					Then: il(c.chanTile - 1),
					Else: il(c.chanScrap - 1),
				},
			}
		default:
			expr = il(iters - 1)
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: c.chanIdx,
				Expr2: expr,
			},
		}
		stmts[2] = cgen.For{
			Cond: cgen.CmpLE{
				Expr1: c.chanIdx,
				Expr2: last,
			},
			Post: cgen.IncPre{
				Expr: c.chanIdx,
			},
			Body: c.kernel4(),
		}
	}
	return stmts
}

func (c *consumeSums) kernel4() cgen.Gen {
	c.bnMuls = nil
	c.bnAdds = nil
	var (
		last = len(c.bnPtrs)
		gens = make(cgen.Gens, last+1)
	)
	ch := cgen.Paren{
		Inner: addMul(
			c.chanIdx,
			il(c.toChans),
			c.groupIdx,
		),
	}
	for x, bnPtr := range c.bnPtrs {
		var (
			bnMul = vb(c.name("bnMul"))
			bnAdd = vb(c.name("bnAdd"))
		)
		c.bnMuls = append(c.bnMuls, bnMul)
		c.bnAdds = append(c.bnAdds, bnAdd)
		gens[x] = &bn.Load{
			Ctx:     c.bc,
			Mas:     bnPtr,
			Channel: ch,
			Mul:     bnMul,
			Add:     bnAdd,
		}
	}
	gens[last] = c.kernel5()
	return gens
}

func (c *consumeSums) kernel5() cgen.Gen {
	c.stripIdx = vb(c.name("k"))
	var (
		stmts = make(cgen.Stmts, 4)
		retIf cgen.Gen
	)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: c.stripIdx,
		Init: cgen.Mul{
			Expr1: il(c.stripTile),
			Expr2: c.stripCoord,
		},
	}
	if c.stripHull > 1 {
		var (
			last = vb(c.name("kk"))
			expr cgen.Gen
		)
		switch c.stripTiles {
		case c.stripHull:
			expr = il(c.stripTile - 1)
		case 0:
			expr = il(c.stripScrap - 1)
		default:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: c.stripCoord,
						Expr2: il(c.stripTiles),
					},
					Then: il(c.stripTile - 1),
					Else: il(c.stripScrap - 1),
				},
			}
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: c.stripIdx,
				Expr2: expr,
			},
		}
		retIf = cgen.If1{
			Cond: cgen.CmpGE{
				Expr1: c.stripIdx,
				Expr2: last,
			},
			Then: cgen.Return{},
		}
	}
	if c.strips1 > 0 {
		c.shortH = false
		stmts[2] = cgen.For{
			Cond: cgen.CmpNE{
				Expr1: c.stripIdx,
				Expr2: il(c.strips1),
			},
			Post: cgen.IncPre{
				Expr: c.stripIdx,
			},
			Body: cgen.Stmts{
				c.kernel6(),
				retIf,
			},
		}
	}
	if c.strips1 < c.strips2 {
		c.shortH = true
		stmts[3] = c.kernel6()
	}
	return stmts
}

func (c *consumeSums) kernel6() cgen.Gen {
	c.cellIdx = vb(c.name("l"))
	var (
		stmts = make(cgen.Stmts, 4)
		retIf cgen.Gen
	)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: c.cellIdx,
		Init: cgen.Mul{
			Expr1: il(c.cellTile),
			Expr2: c.cellCoord,
		},
	}
	if c.cellHull > 1 {
		var (
			last = vb(c.name("ll"))
			expr cgen.Gen
		)
		switch c.cellTiles {
		case c.cellHull:
			expr = il(c.cellTile - 1)
		case 0:
			expr = il(c.cellScrap - 1)
		default:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: c.cellCoord,
						Expr2: il(c.cellTiles),
					},
					Then: il(c.cellTile - 1),
					Else: il(c.cellScrap - 1),
				},
			}
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: c.cellIdx,
				Expr2: expr,
			},
		}
		retIf = cgen.If1{
			Cond: cgen.CmpGE{
				Expr1: c.cellIdx,
				Expr2: last,
			},
			Then: cgen.Return{},
		}
	}
	if c.cells1 > 0 {
		c.shortW = false
		stmts[2] = cgen.For{
			Cond: cgen.CmpNE{
				Expr1: c.cellIdx,
				Expr2: il(c.cells1),
			},
			Post: cgen.IncPre{
				Expr: c.cellIdx,
			},
			Body: cgen.Stmts{
				c.kernel7(),
				retIf,
			},
		}
	}
	if c.cells1 < c.cells2 {
		c.shortW = true
		stmts[3] = c.kernel7()
	}
	return stmts
}

func (c *consumeSums) kernel7() cgen.Gen {
	switch c.platform {
	case raw.AVX512Float32:
		return c.m512()
	default:
		panic("bug")
	}
}

func (c *consumeSums) m512() cgen.Gen {
	var (
		rows   int
		cols   int
		rowIdx int
		stmts  cgen.Stmts
		out    cgen.Gen
	)
	stmt := func(st cgen.Gen) {
		stmts = append(stmts, st)
	}
	layer4 := func() {
		var (
			datPtr = 0
			mask   = loMask(cols)
			bnPtr  = 0
		)
		ae := func() cgen.Gen {
			var (
				ret        = c.datPtrs[datPtr]
				pitch1     = c.To.Pitch1Bytes[datPtr]
				pitch2     = c.To.Pitch2Bytes[datPtr]
				groupPitch = c.toChans * pitch2
				stripPitch = c.datSliceVecs * pitch1
			)
			ret = cgen.Add{
				Expr1: ret,
				Expr2: il(rowIdx * pitch1),
			}
			ret = addMul(ret, il(groupPitch), c.groupIdx)
			ret = addMul(ret, il(pitch2), c.chanIdx)
			ret = addMul(ret, il(stripPitch), c.stripIdx)
			ret = addMul(ret, il(c.datVecBytes), c.cellIdx)
			return ret
		}
		for op := range c.To.Ops {
			op := &c.To.Ops[op]
			switch op.Kind {
			case mod.Add:
				for n := op.Int; n > 0; n-- {
					stmt(cgen.Assign{
						Expr1: out,
						Expr2: avx.Mm512AddPs{
							out,
							avx.Mm512MaskzLoaduPs{
								mask, ae(),
							},
						},
					})
					datPtr++
				}
			case mod.Bn:
				stmt(&bn.Apply{
					Ctx: c.bc,
					Mul: c.bnMuls[bnPtr],
					Add: c.bnAdds[bnPtr],
					To:  out,
				})
				bnPtr++
			case mod.ReLU:
				stmt(&act.ReLU{
					Ctx:      c.ac,
					NegSlope: op.Float,
					Var:      out,
				})
			default:
				panic("bug")
			}
		}
		for datPtr < len(c.datPtrs) {
			stmt(avx.Mm512MaskStoreuPs{
				ae(), mask, out,
			})
			datPtr++
		}
	}
	layer3 := func() {
		var (
			trees []cgen.Gen
		)
		load := func(coreOff, pileIdx int) cgen.Gen {
			var (
				ae         = c.sumPtr
				stripPitch = c.blkStep * c.sumCoreBytes
				ret        = vb(c.name("load"))
			)
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(
					coreOff*c.sumCoreBytes +
						pileIdx*c.sumPileBytes +
						rowIdx*c.datVecBytes,
				),
			}
			ae = addMul(ae, il(c.sumGroupBytes), c.groupIdx)
			ae = addMul(ae, il(stripPitch), c.stripIdx)
			ae = addMul(ae, il(c.sumCoreBytes), c.cellIdx)
			ae = addMul(ae, il(c.datSliceBytes), c.chanIdx)
			stmt(cgen.Var{
				Type: avx.M512, What: ret,
				Init: avx.Mm512LoaduPs{ae},
			})
			return ret
		}
		cast := func(ps cgen.Gen) cgen.Gen {
			ret := vb(c.name("cast"))
			stmt(cgen.Var{
				Type: avx.M512i, What: ret,
				Init: avx.Mm512CastpsSi512{ps},
			})
			return ret
		}
		join := func(hi, lo cgen.Gen, drop int) cgen.Gen {
			var (
				ret    = vb(c.name("join"))
				castLo = cast(lo)
				castHi = castLo
			)
			if hi != nil {
				castHi = cast(hi)
			}
			stmt(cgen.Var{
				Type: avx.M512, What: ret,
				Init: avx.Mm512Castsi512Ps{
					avx.Mm512AlignrEpi32{
						castHi, castLo,
						il(drop),
					},
				},
			})
			return ret
		}
		add := func(older, newer cgen.Gen) cgen.Gen {
			ret := vb(c.name("add"))
			stmt(cgen.Var{
				Type: avx.M512, What: ret,
				Init: avx.Mm512AddPs{
					older, newer,
				},
			})
			return ret
		}
		sublayer3 := func() {
			out = nil
			for _, tree := range trees {
				if tree != nil {
					switch out {
					case nil:
						out = tree
					default:
						out = add(tree, out)
					}
				}
			}
		}
		sublayer2 := func() {
			coreOff := 0
			for pileIdx, shift := range c.shifts {
				at := coreOff*16 - shift
				for ; at <= -16; at += 16 {
					coreOff++
				}
				tree1 := load(coreOff, pileIdx)
				if at < 0 {
					var hi cgen.Gen
					if at+16 < cols {
						hi = load(coreOff+1, pileIdx)
					}
					tree1 = join(hi, tree1, -at)
				}
				for x, tree2 := range trees {
					if tree2 == nil {
						trees[x] = tree1
						break
					}
					tree1 = add(tree2, tree1)
					trees[x] = nil
				}
			}
			sublayer3()
		}
		sublayer1 := func() {
			var (
				n1 = len(c.shifts)
				n2 = 0
			)
			for ; n1 > 0; n1 >>= 1 {
				n2++
			}
			trees = make([]cgen.Gen, n2)
			sublayer2()
		}
		sublayer1()
		layer4()
	}
	layer2 := func() cgen.Gen {
		toMix := make([]cgen.Stmts, rows)
		for x := range toMix {
			rowIdx = x
			stmts = nil
			layer3()
			toMix[x] = stmts
		}
		return mix(toMix)
	}
	layer1 := func() cgen.Gen {
		rows = c.datSliceVecs
		if c.shortH {
			rows = c.remH
		}
		cols = c.datVecDats
		if c.shortW {
			cols = c.remW
		}
		return layer2()
	}
	return layer1()
}
