package strider

import (
	"NN-512/internal/compile/author/act"
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/bn"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/mod"
	"NN-512/internal/compile/author/quadfft"
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
		prefix:      pl.Config.Prefix + "Strider",
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

type form struct {
	padH   int
	padW   int
	datH   int
	datW   int
	yieldH int
	yieldW int
}

type loopB struct {
	fromH    int
	fromW    int
	fromStep int
	blkFirst int
	blkPast  int
	form
}

type loopW struct {
	fromH    int
	fromW    int
	fromStep int
	segFirst int
	segPast  int
	lbs      []*loopB
}

type loopH struct {
	fromH    int
	fromStep int
	segFirst int
	segStep  int
	segPast  int
	lws      []*loopW
}

type segments struct {
	cnt int
	lhs []*loopH
}

func newSegments(ctx *Ctx, spec *Spec, segBlks int) *segments {
	var (
		segs segments
		lb1  loopB
		lb2  loopB
		lw1  loopW
		lw2  loopW
		lh1  loopH
		idx  map[int]int
		tie  int
		at   int
	)
	layer7 := func() {
		lh := func(lws []*loopW) *loopH {
			var (
				fromH    = lws[0].fromH
				segFirst = lws[0].segFirst
				segPast  = lws[len(lws)-1].segPast
			)
			for _, lw := range lws {
				lw.fromH -= fromH
				lw.segFirst -= segFirst
				lw.segPast -= segFirst
			}
			return &loopH{
				fromH:    fromH,
				fromStep: 0,
				segFirst: segFirst,
				segStep:  0,
				segPast:  segPast,
				lws:      lws,
			}
		}
		var (
			i = tie
			n = len(lh1.lws)
		)
		if i == -1 {
			i = n
		}
		if i > 0 {
			pre := lh(lh1.lws[:i])
			segs.lhs = append(
				segs.lhs, pre,
			)
		}
		if i < n {
			cyc := lh(lh1.lws[i:])
			cyc.fromStep = lh1.fromStep
			cyc.segStep = lh1.segStep
			cyc.segPast = lh1.segPast
			segs.lhs = append(
				segs.lhs, cyc,
			)
		}
	}
	layer6 := func(flush bool) {
		match := func(i int) bool {
			var (
				lw = lh1.lws[i]
				n1 = lw.segPast - lw.segFirst
				n2 = lw2.segPast - lw2.segFirst
			)
			if n1 != n2 {
				return false
			}
			if len(lw.lbs) != len(lw2.lbs) {
				return false
			}
			for i, lb := range lw.lbs {
				if *lb != *lw2.lbs[i] {
					return false
				}
			}
			return true
		}
		var (
			cut = false
			pre = false
			cyc = false
		)
		switch {
		case lh1.lws == nil:
			cut = true
		case tie == -1:
			i, ok := idx[lw2.fromW]
			if ok && match(i) {
				lw := lh1.lws[i]
				lh1.fromStep = lw2.fromH - lw.fromH
				lh1.segStep = lw2.segFirst - lw.segFirst
				tie = i
				at = i
				cyc = true
			} else {
				pre = true
			}
		case match(at):
			cyc = true
		default:
			cut = true
		}
		switch {
		case cut:
			if lh1.lws != nil {
				layer7()
				lh1.lws = nil
			}
			idx = make(map[int]int)
			tie = -1
			fallthrough
		case pre:
			lw := lw2
			lw.lbs = make([]*loopB, len(lw2.lbs))
			for i, lb := range lw2.lbs {
				lb := *lb
				lw.lbs[i] = &lb
			}
			idx[lw.fromW] = len(lh1.lws)
			lh1.lws = append(
				lh1.lws, &lw,
			)
		case cyc:
			lh1.segPast = lw2.segPast
			if at++; at == len(lh1.lws) {
				at = tie
			}
		}
		if flush {
			layer7()
		}
	}
	layer5 := func(flush bool) {
		split := true
		switch {
		case lw2.fromH != lw1.fromH:
		case len(lw2.lbs) != len(lw1.lbs):
		default:
			split = false
			for i, lb := range lw2.lbs {
				if *lb != *lw1.lbs[i] {
					split = true
					break
				}
			}
		}
		switch {
		case split:
			if lw2.segFirst < lw2.segPast {
				layer6(false)
			}
			swap := lw2.lbs
			lw2 = lw1
			lw1.lbs = swap
		default:
			if lw2.fromStep == 0 {
				lw2.fromStep = lw1.fromW - lw2.fromW
			}
			lw2.segPast = lw1.segPast
		}
		if flush {
			layer6(true)
		}
	}
	layer4 := func(flush bool) {
		n := len(lw1.lbs)
		if lb2.blkFirst == 0 {
			if n > 0 {
				layer5(false)
			}
			lw1.fromH = lb2.fromH
			lw1.fromW = lb2.fromW
			lw1.segFirst = segs.cnt
			lw1.segPast = segs.cnt + 1
			segs.cnt++
			if lw1.lbs == nil {
				lw1.lbs = make([]*loopB, segBlks)
				for i := range lw1.lbs {
					lw1.lbs[i] = new(loopB)
				}
			}
			n = 0
		}
		lw1.lbs = lw1.lbs[:n+1]
		lb := lw1.lbs[n]
		*lb = lb2
		lb.fromH -= lw1.fromH
		lb.fromW -= lw1.fromW
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
		case lb1.blkFirst == 0:
		case lb2.fromH != lb1.fromH:
		case lb2.form != lb1.form:
		default:
			if lb2.fromStep == 0 {
				lb2.fromStep = lb1.fromW - lb2.fromW
			}
			lb2.blkPast = lb1.blkPast
			return
		}
		if lb2.blkFirst < lb2.blkPast {
			layer4(false)
		}
		lb2 = lb1
	}
	layer2 := func() {
		var (
			h1     = spec.PaddingH
			h2     = h1 + spec.From.Height
			h3     = h2 + spec.PaddingH
			w1     = spec.PaddingW
			w2     = w1 + spec.From.Width
			w3     = w2 + spec.PaddingW
			filtH  = 1 + (spec.FilterH-1)*spec.DilationH
			filtW  = 1 + (spec.FilterW-1)*spec.DilationW
			yieldH = 1 + (16-filtH)/2
			yieldW = 1 + (16-filtW)/2
			blk    = 0
		)
		if filtH > 16 || filtW > 16 {
			panic("bug")
		}
		for h := 0; h+filtH <= h3; h += yieldH * 2 {
			for w := 0; w+filtW <= w3; w += yieldW * 2 {
				lb1.fromH = h
				lb1.fromW = w
				lb1.blkFirst = blk
				lb1.blkPast = blk + 1
				if blk++; blk == segBlks {
					blk = 0
				}
				lb1.padH = min(max(h1-h, 0), 16)
				lb1.padW = min(max(w1-w, 0), 16)
				lb1.datH = min(max(h2-h, 0), 16) - lb1.padH
				lb1.datW = min(max(w2-w, 0), 16) - lb1.padW
				if lb1.datH == 0 || lb1.datW == 0 {
					lb1.padH = 16
					lb1.padW = 16
					lb1.datH = 0
					lb1.datW = 0
				}
				lb1.yieldH = min(1+(h3-h-filtH)/2, yieldH)
				lb1.yieldW = min(1+(w3-w-filtW)/2, yieldW)
				layer3(false)
			}
		}
		layer3(true)
	}
	layer1 := func() *segments {
		sig := fmt.Sprint(
			"newSegments",
			" ",
			spec.From.Height,
			spec.From.Width,
			spec.FilterH,
			spec.FilterW,
			spec.PaddingH,
			spec.PaddingW,
			spec.DilationH,
			spec.DilationW,
			segBlks,
		)
		if prior, ok := ctx.dedup[sig]; ok {
			return prior.(*segments)
		}
		ctx.dedup[sig] = &segs
		layer2()
		return &segs
	}
	return layer1()
}

type layout struct {
	segs          *segments
	blkZones      int
	zoneFrags     int
	fromChans     int
	toChans       int
	slices1       int
	slices2       int
	epochs1       int
	epochs2       int
	alignment     int
	biasBytes     int
	bfFragBytes   int
	bfMeldBytes   int
	bfGroupBytes  int
	bfEpochBytes  int
	bfTotalBytes  int
	wtBytes       int
	wfFragBytes   int
	wfMeldFrags   int
	wfMeldBytes   int
	wfSliceFrags1 int
	wfSliceFrags2 int
	wfSliceMelds1 int
	wfSliceMelds2 int
	wfSliceBytes1 int
	wfSliceBytes2 int
	wfCores1      int
	wfCores2      int
	wfCoreBytes11 int
	wfCoreBytes12 int
	wfCoreBytes21 int
	wfCoreBytes22 int
	wfPileBytes1  int
	wfPileBytes2  int
	wfGroupBytes1 int
	wfGroupBytes2 int
	wfZoneBytes1  int
	wfZoneBytes2  int
	wfEpochBytes1 int
	wfEpochBytes2 int
	wfTotalBytes  int
	datBytes      int
	dfFragBytes   int
	dfMeldFrags   int
	dfMeldBytes   int
	dfSliceFrags1 int
	dfSliceFrags2 int
	dfSliceMelds1 int
	dfSliceMelds2 int
	dfSliceBytes1 int
	dfSliceBytes2 int
	dfCores1      int
	dfCores2      int
	dfCoreBytes11 int
	dfCoreBytes12 int
	dfCoreBytes21 int
	dfCoreBytes22 int
	dfPileBytes1  int
	dfPileBytes2  int
	dfGroupBytes1 int
	dfGroupBytes2 int
	dfZoneBytes1  int
	dfZoneBytes2  int
	dfEpochBytes1 int
	dfEpochBytes2 int
	dfTotalBytes  int
	sfFragBytes   int
	sfMeldBytes11 int
	sfMeldBytes12 int
	sfMeldBytes21 int
	sfMeldBytes22 int
	sfRowBytes11  int
	sfRowBytes12  int
	sfRowBytes21  int
	sfRowBytes22  int
	sfSiteBytes11 int
	sfSiteBytes12 int
	sfSiteBytes21 int
	sfSiteBytes22 int
	sfCoreBytes1  int
	sfCoreBytes2  int
	sfPileBytes   int
	sfGroupBytes  int
	sfTotalBytes  int
}

func newLayout(ctx *Ctx, spec *Spec) *layout {
	var (
		y layout
	)
	layer9 := func() {
		y.dfCoreBytes11 = y.slices1 * y.dfSliceBytes1
		y.dfCoreBytes12 = y.slices1 * y.dfSliceBytes2
		y.dfCoreBytes21 = y.slices2 * y.dfSliceBytes1
		y.dfCoreBytes22 = y.slices2 * y.dfSliceBytes2
		y.dfPileBytes1 = y.dfCores1*y.dfCoreBytes11 + y.dfCoreBytes12
		y.dfPileBytes2 = y.dfCores1*y.dfCoreBytes21 + y.dfCoreBytes22
		y.dfGroupBytes1 = y.zoneFrags * y.dfPileBytes1
		y.dfGroupBytes2 = y.zoneFrags * y.dfPileBytes2
		y.dfZoneBytes1 = spec.Groups * y.dfGroupBytes1
		y.dfZoneBytes2 = spec.Groups * y.dfGroupBytes2
		y.dfEpochBytes1 = y.blkZones * y.dfZoneBytes1
		y.dfEpochBytes2 = y.blkZones * y.dfZoneBytes2
		y.dfTotalBytes = y.epochs1*y.dfEpochBytes1 + y.dfEpochBytes2
	}
	layer8 := func() {
		y.wfCoreBytes11 = y.slices1 * y.wfSliceBytes1
		y.wfCoreBytes12 = y.slices1 * y.wfSliceBytes2
		y.wfCoreBytes21 = y.slices2 * y.wfSliceBytes1
		y.wfCoreBytes22 = y.slices2 * y.wfSliceBytes2
		y.wfPileBytes1 = y.wfCores1*y.wfCoreBytes11 + y.wfCoreBytes12
		y.wfPileBytes2 = y.wfCores1*y.wfCoreBytes21 + y.wfCoreBytes22
		y.wfGroupBytes1 = y.zoneFrags * y.wfPileBytes1
		y.wfGroupBytes2 = y.zoneFrags * y.wfPileBytes2
		y.wfZoneBytes1 = spec.Groups * y.wfGroupBytes1
		y.wfZoneBytes2 = spec.Groups * y.wfGroupBytes2
		y.wfEpochBytes1 = y.blkZones * y.wfZoneBytes1
		y.wfEpochBytes2 = y.blkZones * y.wfZoneBytes2
		y.wfTotalBytes = y.epochs1*y.wfEpochBytes1 + y.wfEpochBytes2
		layer9()
	}
	layer7 := func() {
		y.bfMeldBytes = y.wfMeldFrags * y.bfFragBytes
		y.bfGroupBytes = ceilQuo(y.toChans, y.wfMeldFrags) * y.bfMeldBytes
		y.bfEpochBytes = spec.Groups * y.bfGroupBytes
		y.bfTotalBytes = y.epochs2 * y.bfEpochBytes
		y.bfTotalBytes += y.alignment - 1
		y.bfTotalBytes &= -y.alignment
		layer8()
	}
	layer6 := func() {
		wfSliceBytes := y.wfSliceBytes1
		if y.wfCores1 == 0 {
			wfSliceBytes = y.wfSliceBytes2
		}
		dfSliceBytes := y.dfSliceBytes1
		if y.dfCores1 == 0 {
			dfSliceBytes = y.dfSliceBytes2
		}
		switch ctx.platform {
		case raw.AVX512Float32:
			var (
				sliceBytes = 2*wfSliceBytes + dfSliceBytes
				cacheBytes = ctx.cacheBytes1 + ctx.cacheBytes2
			)
			const (
				empirical1 = 4
				empirical2 = 256
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
		layer7()
	}
	layer5 := func() {
		var (
			wfDiv = y.wfMeldFrags
			wfQuo = y.wfSliceFrags2 / wfDiv
			wfRem = y.wfSliceFrags2 % wfDiv
			dfDiv = y.dfMeldFrags
			dfQuo = y.dfSliceFrags2 / dfDiv
			dfRem = y.dfSliceFrags2 % dfDiv
		)
		y.sfMeldBytes11 = wfDiv * dfDiv * y.sfFragBytes
		y.sfMeldBytes12 = wfDiv * dfRem * y.sfFragBytes
		y.sfMeldBytes21 = wfRem * dfDiv * y.sfFragBytes
		y.sfMeldBytes22 = wfRem * dfRem * y.sfFragBytes
		y.sfRowBytes11 = y.dfSliceMelds1 * y.sfMeldBytes11
		y.sfRowBytes12 = dfQuo*y.sfMeldBytes11 + y.sfMeldBytes12
		y.sfRowBytes21 = y.dfSliceMelds1 * y.sfMeldBytes21
		y.sfRowBytes22 = dfQuo*y.sfMeldBytes21 + y.sfMeldBytes22
		y.sfSiteBytes11 = y.wfSliceMelds1 * y.sfRowBytes11
		y.sfSiteBytes12 = y.wfSliceMelds1 * y.sfRowBytes12
		y.sfSiteBytes21 = wfQuo*y.sfRowBytes11 + y.sfRowBytes21
		y.sfSiteBytes22 = wfQuo*y.sfRowBytes12 + y.sfRowBytes22
		y.sfCoreBytes1 = y.wfCores1*y.sfSiteBytes11 + y.sfSiteBytes21
		y.sfCoreBytes2 = y.wfCores1*y.sfSiteBytes12 + y.sfSiteBytes22
		y.sfPileBytes = y.dfCores1*y.sfCoreBytes1 + y.sfCoreBytes2
		y.sfGroupBytes = y.zoneFrags * y.sfPileBytes
		y.sfTotalBytes = spec.Groups * y.sfGroupBytes
		layer6()
	}
	layer4 := func() {
		y.dfMeldBytes = y.dfMeldFrags * y.dfFragBytes
		y.dfSliceFrags1 = y.dfSliceMelds1 * y.dfMeldFrags
		y.segs = newSegments(ctx, spec, y.dfSliceFrags1)
		var (
			lh = y.segs.lhs[len(y.segs.lhs)-1]
			lw = lh.lws[len(lh.lws)-1]
			lb = lw.lbs[len(lw.lbs)-1]
		)
		y.dfSliceFrags2 = lb.blkPast
		if y.dfSliceFrags2 == y.dfSliceFrags1 {
			y.dfSliceFrags2 = 0
		}
		y.dfSliceMelds2 = ceilQuo(y.dfSliceFrags2, y.dfMeldFrags)
		y.dfSliceBytes1 = y.dfSliceMelds1 * y.dfMeldBytes
		y.dfSliceBytes2 = y.dfSliceMelds2 * y.dfMeldBytes
		y.dfCores1 = y.segs.cnt - btoi(y.dfSliceFrags2 > 0)
		y.dfCores2 = y.segs.cnt
		layer5()
	}
	layer3 := func() {
		y.wfMeldBytes = y.wfMeldFrags * y.wfFragBytes
		y.wfSliceFrags1 = y.wfSliceMelds1 * y.wfMeldFrags
		y.wfSliceFrags2 = y.toChans % y.wfSliceFrags1
		y.wfSliceMelds2 = ceilQuo(y.wfSliceFrags2, y.wfMeldFrags)
		y.wfSliceBytes1 = y.wfSliceMelds1 * y.wfMeldBytes
		y.wfSliceBytes2 = y.wfSliceMelds2 * y.wfMeldBytes
		y.wfCores1 = y.toChans / y.wfSliceFrags1
		y.wfCores2 = y.wfCores1 + btoi(y.wfSliceFrags2 > 0)
		layer4()
	}
	layer2 := func() {
		if len(spec.Filts) > 1 && spec.Groups > 1 {
			panic("bug")
		}
		filts := 0
		for i := range spec.Filts {
			filts += spec.Filts[i].Cnt
		}
		y.fromChans = spec.From.Chans / spec.Groups
		y.toChans = filts / spec.Groups
		layer3()
	}
	layer1 := func() *layout {
		switch ctx.platform {
		case raw.AVX512Float32:
			y.blkZones = 4
			y.zoneFrags = 4
			y.alignment = 64
			y.biasBytes = 4
			y.bfFragBytes = 4
			y.wtBytes = 4
			y.wfFragBytes = 32
			y.wfMeldFrags = 2
			y.wfSliceMelds1 = 2
			y.datBytes = 4
			y.dfFragBytes = 64
			y.dfMeldFrags = 2
			y.dfSliceMelds1 = 3
			y.sfFragBytes = 64
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
	return a.bfTotalBytes + a.wfTotalBytes
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
	pileBytes   int
	groupBytes  int
	zoneBytes   int
	epochFirst  int
	epochCnt    int
	bfPtr       cgen.Gen
	wfPtr       cgen.Gen
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
	repeat      bool
}

func (a *arrangeFilts) Append(to []byte) []byte {
	var (
		threadBlks   int
		groupBundles int
	)
	switch a.platform {
	case raw.AVX512Float32:
		a.bundleFilts = a.wfMeldFrags
		threadBlks = 128
	default:
		panic("bug")
	}
	switch len(a.Filts) {
	case 1:
		groupBundles = ceilQuo(a.toChans, a.bundleFilts)
	default:
		for i := range a.Filts {
			filts := a.Filts[i].Cnt
			groupBundles += ceilQuo(filts, a.bundleFilts)
		}
	}
	var (
		filtBlks   = ceilQuo(a.fromChans, a.epochs2)
		bundleBlks = a.bundleFilts * filtBlks
		groupBlks  = a.toChans * filtBlks
	)
	switch {
	case threadBlks <= bundleBlks:
		a.bundleTile = 1
		a.bundleTiles = groupBundles
		a.bundleScrap = 0
		a.bundleHull = groupBundles
		a.groupTile = 1
		a.groupTiles = a.Groups
		a.groupScrap = 0
		a.groupHull = a.Groups
	case threadBlks <= groupBlks:
		var (
			tile  = ceilQuo(threadBlks, bundleBlks)
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
			tile  = ceilQuo(threadBlks, groupBlks)
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
	impl := func() cgen.Gen {
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
		a.coreBytes = a.wfCoreBytes11
		a.pileBytes = a.wfPileBytes1
		a.groupBytes = a.wfGroupBytes1
		a.zoneBytes = a.wfZoneBytes1
		a.epochFirst = 0
		a.epochCnt = a.epochs1
		put := impl()
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
		a.coreBytes = a.wfCoreBytes21
		a.pileBytes = a.wfPileBytes2
		a.groupBytes = a.wfGroupBytes2
		a.zoneBytes = a.wfZoneBytes2
		a.epochFirst = a.epochs1
		a.epochCnt = 1
		body[6] = impl()
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
		a.bfPtr = vb(a.name("bfPtr"))
		a.wfPtr = vb(a.name("wfPtr"))
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.RestrictPtrChar,
				What: a.bfPtr,
				Init: addMul(
					tensor(n, 0),
					il(a.bfEpochBytes),
					a.epochCoord,
				),
			},
			cgen.Var{
				Type: cgen.RestrictPtrChar,
				What: a.wfPtr,
				Init: addMul(
					cgen.Add{
						Expr1: tensor(n, 0),
						Expr2: il(a.bfTotalBytes),
					},
					il(a.wfEpochBytes1),
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
		repeat bool
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
			past += bundles
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
			a.repeat = false
			stmts[0] = do(quo1)
		}
		if rem1 > 0 {
			a.filts1 = rem1
			a.filts2 = min(tail, a.bundleFilts)
			tail -= a.filts2
			a.repeat = repeat && tail == 0
			stmts[1] = do(1)
		}
		if tail > 0 {
			var (
				head = tail - btoi(repeat)
				quo2 = head / a.bundleFilts
				rem2 = tail - a.bundleFilts*quo2
			)
			if quo2 > 0 {
				a.filts1 = 0
				a.filts2 = a.bundleFilts
				a.repeat = false
				stmts[2] = do(quo2)
			}
			if rem2 > 0 {
				a.filts1 = 0
				a.filts2 = rem2
				a.repeat = repeat
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
			split  = a.toChans - a.wfSliceFrags2
			clamp1 = max(past-split, 0)
			clamp2 = min(clamp1, filts2)
		)
		filts1 = filts2 - clamp2
		repeat = past == a.toChans &&
			past%a.wfMeldFrags > 0
		return layer2()
	}
	return layer1()
}

func (a *arrangeFilts) m512() cgen.Gen {
	var (
		bfs      []cgen.Gen
		preCnt   int
		postMuls []cgen.Gen
		sliceIdx cgen.Gen
		preMul1  cgen.Gen
		preAdd1  cgen.Gen
		filtIdx  int
		wts      []cgen.Gen
		fwd      *quadfft.Fwd
		coreIdx  cgen.Gen
		meldIdx  cgen.Gen
		fragIdx  cgen.Gen
		eo       cgen.Gen
		pileIdx  int
		zoneIdx  int
		wfs      cgen.Gen
	)
	layer17 := func() cgen.Gen {
		emit := func(side int) cgen.Gen {
			var (
				stmts      = make(cgen.Stmts, 2)
				to         = a.wfPtr
				slicePitch = a.wfSliceBytes1
				fragPitch  = a.wfFragBytes / 2
				back       = side * fragPitch
				mask       = 0x0f0f << uint(side*4)
				from       = wfs
			)
			if filtIdx >= a.filts1 {
				slicePitch = a.wfSliceBytes2
			}
			if filtIdx == a.filts2-1 && a.repeat {
				back = 0
				mask = 0xffff
				from = vb(a.name("rep"))
				ctrl := (side+2)<<4 | side
				ctrl |= ctrl << 2
				stmts[0] = cgen.Var{
					Type: avx.M512i, What: from,
					Init: avx.Mm512ShuffleI32x4{
						wfs, wfs, il(ctrl),
					},
				}
			}
			to = cgen.Add{
				Expr1: to,
				Expr2: il(
					(zoneIdx+side)*a.zoneBytes +
						pileIdx*a.pileBytes -
						back,
				),
			}
			to = addMul(to, il(a.groupBytes), a.groupIdx)
			to = addMul(to, il(a.coreBytes), coreIdx)
			to = addMul(to, il(slicePitch), sliceIdx)
			to = addMul(to, il(a.wfMeldBytes), meldIdx)
			to = addMul(to, il(fragPitch), fragIdx)
			stmts[1] = avx.Mm512MaskStoreuEpi32{
				to, il(mask), from,
			}
			return stmts
		}
		return cgen.Gens{
			emit(0),
			emit(1),
		}
	}
	layer16 := func() cgen.Gen {
		wfs = vb(a.name("wfs"))
		var (
			x   = pileIdx*2 + zoneIdx/2*8
			wf1 = fwd.Out[x]
			wf2 = fwd.Out[x+1]
		)
		perm := func(wf cgen.Gen) cgen.Gen {
			if pileIdx == 0 {
				return nil
			}
			return cgen.Assign{
				Expr1: wf,
				Expr2: avx.Mm512PermutexvarPs{
					eo, wf,
				},
			}
		}
		cvt := func(wf cgen.Gen) cgen.Gen {
			return avx.Mm512CvtpsPh{
				wf, avx.FroundToNearestIntNoExc,
			}
		}
		return cgen.Stmts{
			perm(wf1),
			perm(wf2),
			cgen.Var{
				Type: avx.M512i, What: wfs,
				Init: avx.Mm512Castsi256Si512{
					cvt(wf1),
				},
			},
			cgen.Assign{
				Expr1: wfs,
				Expr2: avx.Mm512Inserti64x4{
					wfs, cvt(wf2), il(1),
				},
			},
			layer17(),
		}
	}
	layer15 := func() cgen.Gen {
		var (
			n1   = a.zoneFrags
			n2   = a.blkZones / 2
			gens = make(cgen.Gens, n1*n2)
		)
		for p := 0; p < n1; p++ {
			pileIdx = (p + 1) % n1
			for z := 0; z < n2; z++ {
				zoneIdx = z * 2
				gens[p*n2+z] = layer16()
			}
		}
		return gens
	}
	layer14 := func() cgen.Gen {
		eo = vb(a.name("eo"))
		set := make(avx.Mm512SetEpi32, 16)
		for x := 0; x < 16; x++ {
			set[15-x] = il(x%8*2 + x/8)
		}
		return cgen.Stmts{
			cgen.Var{
				Type: avx.M512i, What: eo,
				Init: set,
			},
			layer15(),
		}
	}
	layer13 := func() cgen.Gen {
		coreIdx = vb(a.name("c"))
		meldIdx = vb(a.name("m"))
		fragIdx = vb(a.name("f"))
		var (
			add  = a.baseFilt + filtIdx
			sub  = a.baseBundle * a.bundleFilts
			expr = cgen.Cast{
				Type: cgen.SizeT,
				Expr: cgen.Paren{
					Inner: addMul(
						il(add-sub),
						il(a.bundleFilts),
						a.bundleIdx,
					),
				},
			}
		)
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: coreIdx,
				Init: cgen.Quo{
					Expr1: expr,
					Expr2: il(a.wfSliceFrags1),
				},
			},
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: meldIdx,
				Init: cgen.Quo{
					Expr1: cgen.Rem{
						Expr1: expr,
						Expr2: il(a.wfSliceFrags1),
					},
					Expr2: il(a.wfMeldFrags),
				},
			},
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: fragIdx,
				Init: cgen.Rem{
					Expr1: expr,
					Expr2: il(a.wfMeldFrags),
				},
			},
			layer14(),
		}
	}
	layer12 := func() cgen.Gen {
		fwd = &quadfft.Fwd{
			Platform: a.platform,
			Nms:      a.nms,
		}
		for x, wt := range wts {
			fwd.In[x*a.DilationH] = wt
		}
		for x := range &fwd.Out {
			wf := vb(a.name("wf"))
			fwd.Out[x] = wf
		}
		return cgen.Gens{
			fwd,
			layer13(),
		}
	}
	layer11 := func() cgen.Gen {
		if a.DilationW == 1 {
			return layer12()
		}
		var (
			last  = 1 + len(wts)
			stmts = make(cgen.Stmts, last+1)
			dw    = vb(a.name("dw"))
			set   = make(avx.Mm512SetEpi32, 16)
			gap   = il(15)
		)
		for x := 0; x < 16; x++ {
			put := gap
			if x%a.DilationW == 0 {
				put = il(x / a.DilationW)
			}
			set[15-x] = put
		}
		stmts[0] = cgen.Var{
			Type: avx.M512i, What: dw,
			Init: set,
		}
		for x, wt := range wts {
			stmts[1+x] = cgen.Assign{
				Expr1: wt,
				Expr2: avx.Mm512PermutexvarPs{
					dw, wt,
				},
			}
		}
		stmts[last] = layer12()
		return stmts
	}
	layer10 := func() cgen.Gen {
		if preCnt == 0 {
			return layer11()
		}
		var (
			last  = len(wts) * 2
			stmts = make(cgen.Stmts, last+1)
			bf    = bfs[filtIdx]
		)
		for x, wt := range wts {
			stmts[x*2] = cgen.Assign{
				Expr1: bf,
				Expr2: avx.Mm512FmaddPs{
					preAdd1, wt, bf,
				},
			}
			stmts[x*2+1] = cgen.Assign{
				Expr1: wt,
				Expr2: avx.Mm512MulPs{
					preMul1, wt,
				},
			}
		}
		stmts[last] = layer11()
		return stmts
	}
	layer9 := func() cgen.Gen {
		if postMuls == nil {
			return layer10()
		}
		var (
			last  = len(wts)
			stmts = make(cgen.Stmts, last+1)
		)
		for x, wt := range wts {
			stmts[x] = cgen.Assign{
				Expr1: wt,
				Expr2: avx.Mm512MulPs{
					postMuls[filtIdx],
					wt,
				},
			}
		}
		stmts[last] = layer10()
		return stmts
	}
	layer8 := func() cgen.Gen {
		wts = make([]cgen.Gen, a.FilterH)
		var (
			last  = len(wts)
			stmts = make(cgen.Stmts, last+1)
		)
		for h := range wts {
			var (
				wt          = vb(a.name("wt"))
				mask        = loMask(a.FilterW)
				ae          = a.wtPtr
				hPitch      = a.FilterW * a.wtBytes
				slicePitch  = a.FilterH * hPitch
				filtPitch   = a.fromChans * slicePitch
				bundlePitch = a.bundleFilts * filtPitch
				groupPitch  = a.toChans * filtPitch
			)
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(
					-a.baseBundle*bundlePitch +
						filtIdx*filtPitch +
						h*hPitch,
				),
			}
			ae = addMul(ae, il(groupPitch), a.groupIdx)
			ae = addMul(ae, il(bundlePitch), a.bundleIdx)
			ae = addMul(ae, il(slicePitch), sliceIdx)
			wts[h] = wt
			stmts[h] = cgen.Var{
				Type: avx.M512, What: wt,
				Init: avx.Mm512MaskzLoaduPs{
					mask, ae,
				},
			}
		}
		stmts[last] = layer9()
		return stmts
	}
	layer7 := func() cgen.Gen {
		gens := make(cgen.Gens, a.filts2)
		for x := range gens {
			filtIdx = x
			gens[x] = layer8()
		}
		return gens
	}
	layer6 := func() cgen.Gen {
		if preCnt == 0 {
			preMul1 = nil
			preAdd1 = nil
			return layer7()
		}
		var (
			last  = preCnt * 3
			stmts = make(cgen.Stmts, last+1)
		)
		preCh := cgen.Paren{
			Inner: addMul(
				sliceIdx,
				il(a.fromChans),
				a.groupIdx,
			),
		}
		for x, prePtr := range a.bnPtrs[:preCnt] {
			var (
				preMul2 = vb(a.name("preMul"))
				preAdd2 = vb(a.name("preAdd"))
			)
			stmts[x*3] = &bn.Load{
				Ctx:     a.bc,
				Mas:     prePtr,
				Channel: preCh,
				Mul:     preMul2,
				Add:     preAdd2,
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
			stmts[x*3+2] = cgen.Assign{
				Expr1: preAdd1,
				Expr2: avx.Mm512FmaddPs{
					preAdd1, preMul2,
					preAdd2,
				},
			}
		}
		stmts[last] = layer7()
		return stmts
	}
	layer5 := func() cgen.Gen {
		sliceIdx = vb(a.name("k"))
		return cgen.Stmts{
			cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT,
					What: sliceIdx,
					Init: il(0),
				},
				Cond: cgen.CmpL{
					Expr1: sliceIdx,
					Expr2: il(a.slices),
				},
				Post: cgen.IncPre{
					Expr: sliceIdx,
				},
				Body: layer6(),
			},
		}
	}
	layer4 := func() cgen.Gen {
		var (
			postPtrs = a.bnPtrs[preCnt:]
			postCnt  = len(postPtrs)
		)
		switch postCnt {
		case 0:
			postMuls = nil
			return layer5()
		default:
			postMuls = make([]cgen.Gen, a.filts2)
		}
		toMix := make([]cgen.Stmts, a.filts2)
		for f := range toMix {
			stmts := make(cgen.Stmts, postCnt*2)
			postCh := cgen.Paren{
				Inner: addMul(
					addMul(
						il(f-a.baseBundle*a.bundleFilts),
						il(a.toChans),
						a.groupIdx,
					),
					il(a.bundleFilts),
					a.bundleIdx,
				),
			}
			for x, postPtr := range postPtrs {
				postMul := vb(a.name("postMul"))
				stmts[x*2] = &bn.Load{
					Ctx:     a.bc,
					Mas:     postPtr,
					Channel: postCh,
					Mul:     postMul,
				}
				if x == 0 {
					postMuls[f] = postMul
					continue
				}
				stmts[x*2+1] = cgen.Assign{
					Expr1: postMuls[f],
					Expr2: avx.Mm512MulPs{
						postMuls[f],
						postMul,
					},
				}
			}
			toMix[f] = stmts
		}
		return cgen.Gens{
			mix(toMix),
			layer5(),
		}
	}
	layer3 := func() cgen.Gen {
		var (
			bias cgen.Gen
		)
		scale := func() cgen.Gen {
			return cgen.Assign{
				Expr1: bfs[0],
				Expr2: avx.Mm512MulPs{
					bfs[0],
					avx.Mm512Set1PsLit(64),
				},
			}
		}
		sublayer5 := func() cgen.Gen {
			var (
				postPtrs = a.bnPtrs[preCnt:]
				postCnt  = len(postPtrs)
			)
			if postCnt == 0 {
				return nil
			}
			stmts := make(cgen.Stmts, postCnt*2)
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
			for x, postPtr := range postPtrs {
				var (
					postMul = vb(a.name("postMul"))
					postAdd = vb(a.name("postAdd"))
				)
				stmts[x*2] = &bn.Load{
					Ctx:     a.bc,
					Mas:     postPtr,
					Channel: postCh,
					Mul:     postMul,
					Add:     postAdd,
					Cnt:     a.filts2,
				}
				stmts[x*2+1] = cgen.Assign{
					Expr1: bias,
					Expr2: avx.Mm512FmaddPs{
						bias, postMul,
						postAdd,
					},
				}
			}
			return stmts
		}
		sublayer4 := func() cgen.Gen {
			var (
				ae          = a.biasPtr
				groupPitch  = a.toChans * a.biasBytes
				bundlePitch = a.bundleFilts * a.biasBytes
				mask        = loMask(a.filts2)
			)
			ae = cgen.Sub{
				Expr1: ae,
				Expr2: il(a.baseBundle * bundlePitch),
			}
			ae = addMul(ae, il(groupPitch), a.groupIdx)
			ae = addMul(ae, il(bundlePitch), a.bundleIdx)
			return cgen.Stmts{
				cgen.Assign{
					Expr1: bias,
					Expr2: avx.Mm512MaskzLoaduPs{
						mask, ae,
					},
				},
				sublayer5(),
			}
		}
		sublayer3 := func() cgen.Gen {
			bias = vb(a.name("bias"))
			var stmt cgen.Gen
			switch preCnt {
			case 0:
				bfs[0] = bias
				stmt = scale()
			default:
				stmt = cgen.Assign{
					Expr1: bfs[0],
					Expr2: avx.Mm512AddPs{
						bfs[0], bias,
					},
				}
			}
			return cgen.Stmts{
				cgen.Var{
					Type: avx.M512, What: bias,
					Init: avx.Mm512SetzeroPs,
				},
				cgen.If{
					Cond: cgen.IsZero{
						Expr: a.epochCoord,
					},
					Then: cgen.Stmts{
						sublayer4(),
						stmt,
					},
				},
			}
		}
		sublayer2 := func() cgen.Gen {
			if a.epochFirst == 0 {
				return sublayer3()
			}
			return nil
		}
		sublayer1 := func() cgen.Gen {
			if preCnt == 0 {
				return sublayer2()
			}
			return cgen.Stmts{
				&sumr.Pack{
					Platform: a.platform,
					Nms:      a.nms,
					Vars:     bfs,
				},
				sublayer2(),
				scale(),
			}
		}
		return cgen.Gens{
			layer4(),
			sublayer1(),
		}
	}
	layer2 := func() cgen.Gen {
		var (
			stmts       = layer3()
			ae          = a.bfPtr
			bundlePitch = a.bundleFilts * a.bfFragBytes
			mask        = loMask(a.filts2)
			bf          = bfs[0]
		)
		ae = cgen.Sub{
			Expr1: ae,
			Expr2: il(
				a.baseBundle*bundlePitch -
					a.baseFilt*a.bfFragBytes,
			),
		}
		ae = addMul(ae, il(a.bfGroupBytes), a.groupIdx)
		ae = addMul(ae, il(bundlePitch), a.bundleIdx)
		if bf == nil {
			bf = avx.Mm512SetzeroPs
		}
		return cgen.Stmts{
			stmts,
			avx.Mm512MaskStoreuPs{
				ae, mask, bf,
			},
		}
	}
	layer1 := func() cgen.Gen {
		bfs = make([]cgen.Gen, a.filts2)
		preCnt = a.Filts[a.filtsIdx].BnPre
		if preCnt == 0 {
			return layer2()
		}
		var (
			last  = len(bfs)
			stmts = make(cgen.Stmts, last+1)
		)
		for x := range bfs {
			bf := vb(a.name("bf"))
			bfs[x] = bf
			stmts[x] = cgen.Var{
				Type: avx.M512, What: bf,
				Init: avx.Mm512SetzeroPs,
			}
		}
		stmts[last] = layer2()
		return stmts
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
	return a.dfTotalBytes
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
	pileBytes   int
	groupBytes  int
	zoneBytes   int
	datPtrs     []cgen.Gen
	bnPtrs      []cgen.Gen
	dfPtr       cgen.Gen
	groupIdx    cgen.Gen
	coreIdx     cgen.Gen
	coreLast    cgen.Gen
	coreH       cgen.Gen
	coreW       cgen.Gen
	lbs         []*loopB
	sliceIdx    cgen.Gen
	bnMuls      []cgen.Gen
	bnAdds      []cgen.Gen
	lb          *loopB
	blkIdx      cgen.Gen
	repeat      bool
	meldIdx     cgen.Gen
	fragIdx     cgen.Gen
}

func (a *arrangeDats) Append(to []byte) []byte {
	var threadBlks int
	switch a.platform {
	case raw.AVX512Float32:
		threadBlks = 128
	default:
		panic("bug")
	}
	var (
		chanBlks1  = a.dfCores1 * a.dfSliceFrags1
		chanBlks2  = chanBlks1 + a.dfSliceFrags2
		groupBlks1 = a.fromChans * chanBlks2
		groupBlks2 = ceilQuo(groupBlks1, a.epochs2)
		coreBlks   = ceilQuo(groupBlks2, a.dfCores2)
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
	case threadBlks <= coreBlks:
		var (
			minSlices = a.slices1
			sliceBlks = ceilQuo(chanBlks2, a.dfCores2)
		)
		switch {
		case a.epochs1 == a.epochs2:
		case a.epochs1 == 0 || a.slices1 > a.slices2:
			minSlices = a.slices2
		}
		var (
			tile  = ceilQuo(threadBlks, sliceBlks)
			tiles = max(minSlices/tile, 1)
		)
		a.sliceTile1 = a.slices1 / tiles
		a.sliceTile2 = a.slices2 / tiles
		a.sliceTiles = tiles
		a.sliceScrap1 = a.slices1 - tiles*a.sliceTile1
		a.sliceScrap2 = a.slices2 - tiles*a.sliceTile2
		a.sliceHull = tiles
		if a.sliceScrap1 > 0 || a.sliceScrap2 > 0 {
			a.sliceTiles--
			a.sliceScrap1 += a.sliceTile1
			a.sliceScrap2 += a.sliceTile2
		}
		a.coreTile = 1
		a.coreTiles = a.dfCores2
		a.coreScrap = 0
		a.coreHull = a.dfCores2
	case threadBlks <= groupBlks2:
		var (
			tile  = ceilQuo(threadBlks, coreBlks)
			tiles = max(a.dfCores2/tile, 1)
		)
		a.coreTile = a.dfCores2 / tiles
		a.coreTiles = tiles
		a.coreScrap = a.dfCores2 - tiles*a.coreTile
		a.coreHull = tiles
		if a.coreScrap > 0 {
			a.coreTiles--
			a.coreScrap += a.coreTile
		}
	default:
		a.coreTile = a.dfCores2
		a.coreTiles = 1
		a.coreScrap = 0
		a.coreHull = 1
		var (
			tile  = ceilQuo(threadBlks, groupBlks2)
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
		a.coreBytes = a.dfCoreBytes11
		a.pileBytes = a.dfPileBytes1
		a.groupBytes = a.dfGroupBytes1
		a.zoneBytes = a.dfZoneBytes1
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
		a.coreBytes = a.dfCoreBytes21
		a.pileBytes = a.dfPileBytes2
		a.groupBytes = a.dfGroupBytes2
		a.zoneBytes = a.dfZoneBytes2
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
	a.dfPtr = vb(a.name("dfPtr"))
	decl(
		a.dfPtr, addMul(
			tensor(),
			il(a.dfEpochBytes1),
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
		lh       *loopH
		rel      cgen.Gen
		base     cgen.Gen
		relBreak int
		lw       *loopW
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
						Expr1: il(lw.segPast - 1),
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
		a.coreH = vb(a.name("h"))
		a.coreW = vb(a.name("w"))
		a.lbs = lw.lbs
		var (
			exprW   cgen.Gen
			breakIf cgen.Gen
		)
		switch lw.fromStep {
		case 0:
			exprW = il(lw.fromW)
		default:
			exprW = addMul(
				il(lw.fromW-lw.fromStep*lw.segFirst),
				il(lw.fromStep),
				rel,
			)
		}
		if lw.segPast == relBreak {
			breakIf = cgen.If1{
				Cond: cgen.CmpGE{
					Expr1: a.coreIdx,
					Expr2: il(lh.segPast),
				},
				Then: cgen.Break,
			}
		}
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: a.coreH,
				Init: cgen.Add{
					Expr1: base,
					Expr2: il(lw.fromH),
				},
			},
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: a.coreW,
				Init: exprW,
			},
			layer6(),
			breakIf,
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
					Expr2: il(lw.segPast),
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
				start = lws[first].segFirst
				stop  = lws[last].segPast
				split = start + (stop-start)/2
				x     = first + 1
			)
			for lws[x].segPast <= split {
				x++
			}
			return cgen.Stmts{
				cgen.If{
					Cond: cgen.CmpL{
						Expr1: rel,
						Expr2: il(lws[x].segFirst),
					},
					Then: tree(first, x-1),
				},
				tree(x, last),
			}
		}
		return tree(0, len(lws)-1)
	}
	layer3 := func() cgen.Gen {
		if lh.segStep == 0 {
			relBreak = -1
			return layer4()
		}
		x := lh.segPast - lh.segFirst
		relBreak = (x-1)%lh.segStep + 1
		return cgen.For{
			Post: cgen.CommaSpaced{
				cgen.Assign{
					Expr1: rel,
					Expr2: il(0),
				},
				cgen.AddAssign{
					Expr1: base,
					Expr2: il(lh.fromStep),
				},
			},
			Body: layer4(),
		}
	}
	layer2 := func() cgen.Gen {
		rel = vb(a.name("rel"))
		base = vb(a.name("base"))
		var (
			relExpr cgen.Gen = cgen.Sub{
				Expr1: a.coreIdx,
				Expr2: il(lh.segFirst),
			}
			baseExpr = il(lh.fromH)
		)
		if lh.segStep != 0 {
			var (
				numer cgen.Gen = cgen.Cast{
					Type: cgen.SizeT,
					Expr: cgen.Paren{
						Inner: relExpr,
					},
				}
				denom = il(lh.segStep)
			)
			relExpr = cgen.Rem{
				Expr1: numer,
				Expr2: denom,
			}
			baseExpr = addMul(
				baseExpr,
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
				What: base,
				Init: baseExpr,
			},
			layer3(),
		}
	}
	layer1 := func() cgen.Gen {
		var (
			lhs  = a.segs.lhs
			tree func(int, int) cgen.Stmts
		)
		leaf := func(x int) cgen.Stmts {
			lh = lhs[x]
			var assn cgen.Gen
			if x+1 < len(lhs) {
				assn = cgen.Assign{
					Expr1: a.coreIdx,
					Expr2: il(lh.segPast),
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
				start = lhs[first].segFirst
				stop  = lhs[last].segPast
				split = start + (stop-start)/2
				x     = first + 1
			)
			for lhs[x].segPast <= split {
				x++
			}
			return cgen.Stmts{
				cgen.If{
					Cond: cgen.CmpL{
						Expr1: a.coreIdx,
						Expr2: il(lhs[x].segFirst),
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
	layer5 := func() cgen.Gen {
		switch a.platform {
		case raw.AVX512Float32:
			return a.m512()
		default:
			panic("bug")
		}
	}
	layer4 := func() cgen.Gen {
		a.meldIdx = vb(a.name("m"))
		a.fragIdx = vb(a.name("f"))
		var (
			numer cgen.Gen = cgen.Cast{
				Type: cgen.SizeT,
				Expr: a.blkIdx,
			}
			denom = il(a.dfMeldFrags)
		)
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: a.meldIdx,
				Init: cgen.Quo{
					Expr1: numer,
					Expr2: denom,
				},
			},
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: a.fragIdx,
				Init: cgen.Rem{
					Expr1: numer,
					Expr2: denom,
				},
			},
			layer5(),
		}
	}
	layer3 := func(repeat int) cgen.Gen {
		var (
			stmts cgen.Stmts
			first = a.lb.blkFirst
			past1 = a.lb.blkPast - repeat
			past2 = a.lb.blkPast
		)
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		do := func(start, stop int) {
			if start == stop {
				return
			}
			a.blkIdx = vb(a.name("b"))
			a.repeat = start == past1
			decl := cgen.Var{
				Type: cgen.PtrdiffT,
				What: a.blkIdx,
				Init: il(start),
			}
			switch stop - start {
			case 1:
				stmt(decl)
				stmt(layer4())
			default:
				stmt(cgen.For{
					Init: decl,
					Cond: cgen.CmpL{
						Expr1: a.blkIdx,
						Expr2: il(stop),
					},
					Post: cgen.IncPre{
						Expr: a.blkIdx,
					},
					Body: layer4(),
				})
			}
		}
		do(first, past1)
		do(past1, past2)
		return stmts
	}
	layer2 := func() cgen.Gen {
		var (
			n    = len(a.lbs)
			gens = make(cgen.Gens, n)
		)
		for x, lb := range a.lbs {
			a.lb = lb
			repeat := 0
			if x == n-1 &&
				lb.blkPast%a.dfMeldFrags > 0 {
				repeat = 1
			}
			gens[x] = layer3(repeat)
		}
		return gens
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
		fwd     *quadfft.Fwd
		eo      cgen.Gen
		pileIdx int
		zoneIdx int
		dfs     []cgen.Gen
	)
	layer6 := func() cgen.Gen {
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		emit := func(side, part int) {
			var (
				to         = a.dfPtr
				slicePitch = a.dfSliceBytes1
				partPitch  = a.dfMeldBytes / 2
				fragPitch  = a.dfFragBytes / 2
				back       = side * fragPitch
				mask       = 0x00ff << uint(side*8)
				from       = dfs[part]
			)
			switch a.lbs[len(a.lbs)-1].blkPast {
			case a.dfSliceFrags2:
				slicePitch = a.dfSliceBytes2
			}
			if a.repeat {
				back = 0
				mask = 0xffff
				var (
					rep  = vb(a.name("rep"))
					ctrl = side * 2
				)
				ctrl |= (ctrl + 1) << 2
				ctrl |= ctrl << 4
				stmt(cgen.Var{
					Type: avx.M512, What: rep,
					Init: avx.Mm512ShuffleF32x4{
						from, from, il(ctrl),
					},
				})
				from = rep
			}
			to = cgen.Add{
				Expr1: to,
				Expr2: il(
					(zoneIdx+side)*a.zoneBytes +
						pileIdx*a.pileBytes +
						part*partPitch -
						back,
				),
			}
			to = addMul(to, il(a.groupBytes), a.groupIdx)
			to = addMul(to, il(a.coreBytes), a.coreIdx)
			to = addMul(to, il(slicePitch), a.sliceIdx)
			to = addMul(to, il(a.dfMeldBytes), a.meldIdx)
			to = addMul(to, il(fragPitch), a.fragIdx)
			stmt(avx.Mm512MaskStoreuPs{
				to, il(mask), from,
			})
		}
		for side := 0; side < 2; side++ {
			for part := 0; part < 2; part++ {
				emit(side, part)
			}
		}
		return stmts
	}
	layer5 := func() cgen.Gen {
		at := pileIdx*2 + zoneIdx/2*8
		dfs = fwd.Out[at : at+2]
		if pileIdx == 0 {
			return layer6()
		}
		stmts := make(cgen.Stmts, 3)
		for x, df := range dfs {
			stmts[x] = cgen.Assign{
				Expr1: df,
				Expr2: avx.Mm512PermutexvarPs{
					eo, df,
				},
			}
		}
		stmts[2] = layer6()
		return stmts
	}
	layer4 := func() cgen.Gen {
		var (
			n1   = a.zoneFrags
			n2   = a.blkZones / 2
			gens = make(cgen.Gens, n1*n2)
		)
		for p := 0; p < n1; p++ {
			pileIdx = (p + 1) % n1
			for z := 0; z < n2; z++ {
				zoneIdx = z * 2
				gens[p*n2+z] = layer5()
			}
		}
		return gens
	}
	layer3 := func() cgen.Gen {
		eo = vb(a.name("eo"))
		set := make(avx.Mm512SetEpi32, 16)
		for x := 0; x < 16; x++ {
			set[15-x] = il(x%8*2 + x/8)
		}
		return cgen.Stmts{
			cgen.Var{
				Type: avx.M512i, What: eo,
				Init: set,
			},
			layer4(),
		}
	}
	layer2 := func() cgen.Gen {
		var (
			stmts cgen.Stmts
			mask1 = 1<<uint(a.lb.datW) - 1
			mask2 = il(mask1 << uint(a.lb.padW))
		)
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		load := func(h, x int) cgen.Gen {
			var (
				ae         = a.datPtrs[x]
				pitch1     = a.From.Pitch1Bytes[x]
				pitch2     = a.From.Pitch2Bytes[x]
				groupPitch = a.fromChans * pitch2
				blkPitch   = a.lb.fromStep * a.datBytes
			)
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(
					(a.lb.fromH+h)*pitch1 +
						a.lb.fromW*a.datBytes -
						a.lb.blkFirst*blkPitch,
				),
			}
			ae = addMul(ae, il(groupPitch), a.groupIdx)
			ae = addMul(ae, il(pitch2), a.sliceIdx)
			ae = addMul(ae, il(pitch1), a.coreH)
			ae = addMul(ae, il(a.datBytes), a.coreW)
			ae = addMul(ae, il(blkPitch), a.blkIdx)
			return avx.Mm512MaskzLoaduPs{
				mask2, ae,
			}
		}
		for h, dat := range &fwd.In {
			if dat == nil {
				continue
			}
			var (
				datPtrIdx = 0
				bnPtrIdx  = 0
			)
			stmt(cgen.Var{
				Type: avx.M512, What: dat,
				Init: load(h, datPtrIdx),
			})
			for op := range a.From.Ops {
				op := &a.From.Ops[op]
				switch op.Kind {
				case mod.Add:
					for n := op.Int; n > 0; n-- {
						datPtrIdx++
						stmt(cgen.Assign{
							Expr1: dat,
							Expr2: avx.Mm512AddPs{
								dat,
								load(h, datPtrIdx),
							},
						})
					}
				case mod.Bn:
					stmt(&bn.Apply{
						Ctx:  a.bc,
						Mul:  a.bnMuls[bnPtrIdx],
						Add:  a.bnAdds[bnPtrIdx],
						To:   dat,
						Mask: mask2,
					})
					bnPtrIdx++
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
		stmt(fwd)
		stmt(layer3())
		return stmts
	}
	layer1 := func() cgen.Gen {
		fwd = &quadfft.Fwd{
			Platform: a.platform,
			Nms:      a.nms,
		}
		var (
			first = a.lb.padH
			past  = first + a.lb.datH
		)
		for x := first; x < past; x++ {
			dat := vb(a.name("dat"))
			fwd.In[x] = dat
		}
		for x := range &fwd.Out {
			df := vb(a.name("df"))
			fwd.Out[x] = df
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
	return p.sfTotalBytes
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
	epochFirst   int
	epochCnt     int
	slices       int
	wfCoreBytes  int
	wfPileBytes  int
	wfGroupBytes int
	wfZoneBytes  int
	dfCoreBytes  int
	dfPileBytes  int
	dfGroupBytes int
	dfZoneBytes  int
	wfTile       int
	wfTiles      int
	wfScrap      int
	wfHull       int
	dfTile       int
	dfTiles      int
	dfScrap      int
	dfHull       int
	pileTile     int
	pileTiles    int
	pileScrap    int
	pileHull     int
	groupTile    int
	groupTiles   int
	groupScrap   int
	groupHull    int
	calleeName   string
	tensors      cgen.Gen
	epochCoord   cgen.Gen
	zoneCoord    cgen.Gen
	groupCoord   cgen.Gen
	pileCoord    cgen.Gen
	dfCoord      cgen.Gen
	wfCoord      cgen.Gen
	epoch0zone0  bool
	bfPtr        cgen.Gen
	wfPtr        cgen.Gen
	dfPtr        cgen.Gen
	sfPtr        cgen.Gen
	groupIdx     cgen.Gen
	pileIdx      cgen.Gen
	pile0        bool
	dfIdx        cgen.Gen
	dfShort      bool
	wfIdx        cgen.Gen
	wfShort      bool
}

func (p *produceSums) Append(to []byte) []byte {
	var threadWork int
	switch p.platform {
	case raw.AVX512Float32:
		threadWork = 256
	default:
		panic("bug")
	}
	callee := func(first, cnt int) cgen.Gen {
		p.epochFirst = first
		p.epochCnt = cnt
		switch {
		case first < p.epochs1:
			p.slices = p.slices1
			p.wfCoreBytes = p.wfCoreBytes11
			p.wfPileBytes = p.wfPileBytes1
			p.wfGroupBytes = p.wfGroupBytes1
			p.wfZoneBytes = p.wfZoneBytes1
			p.dfCoreBytes = p.dfCoreBytes11
			p.dfPileBytes = p.dfPileBytes1
			p.dfGroupBytes = p.dfGroupBytes1
			p.dfZoneBytes = p.dfZoneBytes1
		default:
			p.slices = p.slices2
			p.wfCoreBytes = p.wfCoreBytes21
			p.wfPileBytes = p.wfPileBytes2
			p.wfGroupBytes = p.wfGroupBytes2
			p.wfZoneBytes = p.wfZoneBytes2
			p.dfCoreBytes = p.dfCoreBytes21
			p.dfPileBytes = p.dfPileBytes2
			p.dfGroupBytes = p.dfGroupBytes2
			p.dfZoneBytes = p.dfZoneBytes2
		}
		var (
			wfWork    = p.slices
			dfWork    = p.wfCores2 * wfWork
			pileWork  = p.dfCores2 * dfWork
			groupWork = p.zoneFrags * pileWork
		)
		p.wfTile = 1
		p.wfTiles = p.wfCores2
		p.wfScrap = 0
		p.wfHull = p.wfCores2
		p.dfTile = 1
		p.dfTiles = p.dfCores2
		p.dfScrap = 0
		p.dfHull = p.dfCores2
		p.pileTile = 1
		p.pileTiles = p.zoneFrags
		p.pileScrap = 0
		p.pileHull = p.zoneFrags
		p.groupTile = 1
		p.groupTiles = p.Groups
		p.groupScrap = 0
		p.groupHull = p.Groups
		switch {
		case threadWork <= wfWork:
		case threadWork <= dfWork:
			var (
				tile  = ceilQuo(threadWork, wfWork)
				tiles = max(p.wfCores2/tile, 1)
			)
			p.wfTile = p.wfCores2 / tiles
			p.wfTiles = tiles
			p.wfScrap = p.wfCores2 - tiles*p.wfTile
			p.wfHull = tiles
			if p.wfScrap > 0 {
				p.wfTiles--
				p.wfScrap += p.wfTile
			}
		case threadWork <= pileWork:
			p.wfTile = p.wfCores2
			p.wfTiles = 1
			p.wfScrap = 0
			p.wfHull = 1
			var (
				tile  = ceilQuo(threadWork, dfWork)
				tiles = max(p.dfCores2/tile, 1)
			)
			p.dfTile = p.dfCores2 / tiles
			p.dfTiles = tiles
			p.dfScrap = p.dfCores2 - tiles*p.dfTile
			p.dfHull = tiles
			if p.dfScrap > 0 {
				p.dfTiles--
				p.dfScrap += p.dfTile
			}
		case threadWork <= groupWork:
			p.wfTile = p.wfCores2
			p.wfTiles = 1
			p.wfScrap = 0
			p.wfHull = 1
			p.dfTile = p.dfCores2
			p.dfTiles = 1
			p.dfScrap = 0
			p.dfHull = 1
			var (
				tile  = ceilQuo(threadWork, pileWork)
				tiles = max(p.zoneFrags/tile, 1)
			)
			p.pileTile = p.zoneFrags / tiles
			p.pileTiles = tiles
			p.pileScrap = p.zoneFrags - tiles*p.pileTile
			p.pileHull = tiles
			if p.pileScrap > 0 {
				p.pileTiles--
				p.pileScrap += p.pileTile
			}
		default:
			p.wfTile = p.wfCores2
			p.wfTiles = 1
			p.wfScrap = 0
			p.wfHull = 1
			p.dfTile = p.dfCores2
			p.dfTiles = 1
			p.dfScrap = 0
			p.dfHull = 1
			p.pileTile = p.zoneFrags
			p.pileTiles = 1
			p.pileScrap = 0
			p.pileHull = 1
			var (
				tile  = ceilQuo(threadWork, groupWork)
				tiles = max(p.Groups/tile, 1)
			)
			p.groupTile = p.Groups / tiles
			p.groupTiles = tiles
			p.groupScrap = p.Groups - tiles*p.groupTile
			p.groupHull = tiles
			if p.groupScrap > 0 {
				p.groupTiles--
				p.groupScrap += p.groupTile
			}
		}
		p.calleeName = p.name(
			p.callerName + "Callee",
		)
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
	inner := func() cgen.Gen {
		zone := vb(p.name("z"))
		return cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT,
				What: zone,
				Init: il(0),
			},
			Cond: cgen.CmpL{
				Expr1: zone,
				Expr2: il(p.blkZones),
			},
			Post: cgen.IncPre{
				Expr: zone,
			},
			Body: cgen.Stmts{
				cgen.Assign{
					Expr1: cgen.Elem{
						Arr: tuple, Idx: il(2),
					},
					Expr2: cgen.Cast{
						Type: cgen.PtrVoid,
						Expr: zone,
					},
				},
				&threader.Do{
					Ctx:    p.tc,
					Callee: vb(p.calleeName),
					Any:    tuple,
					Hull: []cgen.Gen{
						il(p.wfHull),
						il(p.dfHull),
						il(p.pileHull),
						il(p.groupHull),
					},
					Team: team,
				},
			},
		}
	}
	outer := func() cgen.Gen {
		epoch := vb(p.name("e"))
		return cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT,
				What: epoch,
				Init: il(p.epochFirst),
			},
			Cond: cgen.CmpL{
				Expr1: epoch,
				Expr2: il(
					p.epochFirst + p.epochCnt,
				),
			},
			Post: cgen.IncPre{
				Expr: epoch,
			},
			Body: cgen.Stmts{
				cgen.Assign{
					Expr1: cgen.Elem{
						Arr: tuple, Idx: il(1),
					},
					Expr2: cgen.Cast{
						Type: cgen.PtrVoid,
						Expr: epoch,
					},
				},
				inner(),
			},
		}
	}
	var (
		prep = make(cgen.Gens, 2)
		body = make(cgen.Stmts, 4)
	)
	body[0] = cgen.Var{
		Type: cgen.PtrVoid,
		What: cgen.Elem{
			Arr: tuple, Idx: il(3),
		},
	}
	body[1] = cgen.Assign{
		Expr1: cgen.Elem{
			Arr: tuple, Idx: il(0),
		},
		Expr2: tensors,
	}
	if p.epochs1 > 0 {
		prep[0] = callee(0, p.epochs1)
		body[2] = outer()
	}
	if p.epochs1 < p.epochs2 {
		prep[1] = callee(p.epochs1, 1)
		body[3] = outer()
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
	var (
		body   = make(cgen.Stmts, 10)
		tuple  = vb(p.name("tuple"))
		usedPt = false
	)
	body[0] = cgen.Var{
		Type: cgen.PtrPtrVoid, What: tuple,
		Init: callee.Any(),
	}
	p.tensors = vb(p.name("tensors"))
	body[1] = cgen.Var{
		Type: cgen.PtrPtrChar, What: p.tensors,
		Init: cgen.Elem{
			Arr: tuple, Idx: il(0),
		},
	}
	p.epochCoord = vb(p.name("e"))
	body[2] = cgen.Var{
		Type: cgen.PtrdiffT, What: p.epochCoord,
		Init: func() cgen.Gen {
			if p.epochCnt == 1 {
				return il(p.epochFirst)
			}
			return cgen.Cast{
				Type: cgen.PtrdiffT,
				Expr: cgen.Elem{
					Arr: tuple, Idx: il(1),
				},
			}
		}(),
	}
	p.zoneCoord = vb(p.name("z"))
	body[3] = cgen.Var{
		Type: cgen.PtrdiffT, What: p.zoneCoord,
		Init: cgen.Cast{
			Type: cgen.PtrdiffT,
			Expr: cgen.Elem{
				Arr: tuple, Idx: il(2),
			},
		},
	}
	coord := func(nm string, hull, i int) cgen.Gen {
		var (
			ret  = vb(p.name(nm))
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
		body[7-i] = cgen.Var{
			Type: cgen.PtrdiffT, What: ret,
			Init: expr,
		}
		return ret
	}
	p.groupCoord = coord("g", p.groupHull, 3)
	p.pileCoord = coord("p", p.pileHull, 2)
	p.dfCoord = coord("d", p.dfHull, 1)
	p.wfCoord = coord("w", p.wfHull, 0)
	if !usedPt {
		body[8] = cgen.Cast{
			Type: cgen.Void,
			Expr: callee.Pt,
		}
	}
	body[9] = p.kernel1()
	return callee.Func(body)
}

func (p *produceSums) kernel1() cgen.Gen {
	layer2 := func(e0z0 bool) cgen.Gen {
		p.epoch0zone0 = e0z0
		return p.kernel2()
	}
	layer1 := func() cgen.Gen {
		return cgen.Stmts{
			func() cgen.Gen {
				if p.epochFirst > 0 {
					return nil
				}
				both := cgen.Paren{
					Inner: cgen.Or{
						Expr1: p.epochCoord,
						Expr2: p.zoneCoord,
					},
				}
				then := cgen.Stmts{
					nil,
					cgen.Assign{
						Expr1: p.zoneCoord,
						Expr2: il(0),
					},
					layer2(true),
					cgen.Return{},
				}
				if p.epochCnt > 1 {
					then[0] = cgen.Assign{
						Expr1: p.epochCoord,
						Expr2: il(0),
					}
				}
				return cgen.If{
					Cond: cgen.Unlikely{
						Cond: cgen.IsZero{
							Expr: both,
						},
					},
					Then: then,
				}
			}(),
			layer2(false),
		}
	}
	return layer1()
}

func (p *produceSums) kernel2() cgen.Gen {
	p.bfPtr = vb(p.name("bfPtr"))
	p.wfPtr = vb(p.name("wfPtr"))
	p.dfPtr = vb(p.name("dfPtr"))
	p.sfPtr = vb(p.name("sfPtr"))
	return cgen.Stmts{
		cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: p.bfPtr,
			Init: addMul(
				cgen.Elem{
					Arr: p.tensors,
					Idx: il(0),
				},
				il(p.bfEpochBytes),
				p.epochCoord,
			),
		},
		cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: p.wfPtr,
			Init: addMul(
				addMul(
					cgen.Add{
						Expr1: cgen.Elem{
							Arr: p.tensors,
							Idx: il(0),
						},
						Expr2: il(p.bfTotalBytes),
					},
					il(p.wfEpochBytes1),
					p.epochCoord,
				),
				il(p.wfZoneBytes),
				p.zoneCoord,
			),
		},
		cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: p.dfPtr,
			Init: addMul(
				addMul(
					cgen.Elem{
						Arr: p.tensors,
						Idx: il(1),
					},
					il(p.dfEpochBytes1),
					p.epochCoord,
				),
				il(p.dfZoneBytes),
				p.zoneCoord,
			),
		},
		cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: p.sfPtr,
			Init: cgen.Elem{
				Arr: p.tensors,
				Idx: il(2),
			},
		},
		p.kernel3(),
	}
}

func (p *produceSums) kernel3() cgen.Gen {
	p.groupIdx = vb(p.name("i"))
	var (
		stmts = make(cgen.Stmts, 3)
		iters = 0
	)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: p.groupIdx,
		Init: cgen.Mul{
			Expr1: il(p.groupTile),
			Expr2: p.groupCoord,
		},
	}
	switch p.groupTiles {
	case p.groupHull:
		iters = p.groupTile
	case 0:
		iters = p.groupScrap
	}
	switch iters {
	case 1:
		stmts[2] = p.kernel4()
	default:
		var (
			last = vb(p.name("ii"))
			expr cgen.Gen
		)
		switch iters {
		case 0:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: p.groupCoord,
						Expr2: il(p.groupTiles),
					},
					Then: il(p.groupTile - 1),
					Else: il(p.groupScrap - 1),
				},
			}
		default:
			expr = il(iters - 1)
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: p.groupIdx,
				Expr2: expr,
			},
		}
		stmts[2] = cgen.For{
			Cond: cgen.CmpLE{
				Expr1: p.groupIdx,
				Expr2: last,
			},
			Post: cgen.IncPre{
				Expr: p.groupIdx,
			},
			Body: p.kernel4(),
		}
	}
	return stmts
}

func (p *produceSums) kernel4() cgen.Gen {
	layer2 := func(p0 bool) cgen.Gen {
		p.pile0 = p0
		return p.kernel5()
	}
	layer1 := func() cgen.Gen {
		p.pileIdx = vb(p.name("j"))
		var (
			stmts = make(cgen.Stmts, 4)
			last  = vb(p.name("jj"))
			expr  cgen.Gen
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: p.pileIdx,
			Init: cgen.Mul{
				Expr1: il(p.pileTile),
				Expr2: p.pileCoord,
			},
		}
		switch p.pileTiles {
		case p.pileHull:
			expr = il(p.pileTile - 1)
		case 0:
			expr = il(p.pileScrap - 1)
		default:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: p.pileCoord,
						Expr2: il(p.pileTiles),
					},
					Then: il(p.pileTile - 1),
					Else: il(p.pileScrap - 1),
				},
			}
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: p.pileIdx,
				Expr2: expr,
			},
		}
		stmts[2] = cgen.If{
			Cond: cgen.Unlikely{
				Cond: cgen.IsZero{
					Expr: p.pileIdx,
				},
			},
			Then: cgen.Stmts{
				layer2(true),
				cgen.Assign{
					Expr1: p.pileIdx,
					Expr2: il(1),
				},
			},
		}
		stmts[3] = cgen.For{
			Cond: cgen.CmpLE{
				Expr1: p.pileIdx,
				Expr2: last,
			},
			Post: cgen.IncPre{
				Expr: p.pileIdx,
			},
			Body: layer2(false),
		}
		return stmts
	}
	return layer1()
}

func (p *produceSums) kernel5() cgen.Gen {
	p.dfIdx = vb(p.name("k"))
	var (
		stmts = make(cgen.Stmts, 4)
		retIf cgen.Gen
	)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: p.dfIdx,
		Init: cgen.Mul{
			Expr1: il(p.dfTile),
			Expr2: p.dfCoord,
		},
	}
	if p.dfHull > 1 {
		var (
			last = vb(p.name("kk"))
			expr cgen.Gen
		)
		switch p.dfTiles {
		case p.dfHull:
			expr = il(p.dfTile - 1)
		case 0:
			expr = il(p.dfScrap - 1)
		default:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: p.dfCoord,
						Expr2: il(p.dfTiles),
					},
					Then: il(p.dfTile - 1),
					Else: il(p.dfScrap - 1),
				},
			}
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: p.dfIdx,
				Expr2: expr,
			},
		}
		retIf = cgen.If1{
			Cond: cgen.CmpGE{
				Expr1: p.dfIdx,
				Expr2: last,
			},
			Then: cgen.Return{},
		}
	}
	if p.dfCores1 > 0 {
		p.dfShort = false
		stmts[2] = cgen.For{
			Cond: cgen.CmpNE{
				Expr1: p.dfIdx,
				Expr2: il(p.dfCores1),
			},
			Post: cgen.IncPre{
				Expr: p.dfIdx,
			},
			Body: cgen.Stmts{
				p.kernel6(),
				retIf,
			},
		}
	}
	if p.dfCores1 < p.dfCores2 {
		p.dfShort = true
		stmts[3] = p.kernel6()
	}
	return stmts
}

func (p *produceSums) kernel6() cgen.Gen {
	p.wfIdx = vb(p.name("l"))
	var (
		stmts = make(cgen.Stmts, 4)
		retIf cgen.Gen
	)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: p.wfIdx,
		Init: cgen.Mul{
			Expr1: il(p.wfTile),
			Expr2: p.wfCoord,
		},
	}
	if p.wfHull > 1 {
		var (
			last = vb(p.name("ll"))
			expr cgen.Gen
		)
		switch p.wfTiles {
		case p.wfHull:
			expr = il(p.wfTile - 1)
		case 0:
			expr = il(p.wfScrap - 1)
		default:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: p.wfCoord,
						Expr2: il(p.wfTiles),
					},
					Then: il(p.wfTile - 1),
					Else: il(p.wfScrap - 1),
				},
			}
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: p.wfIdx,
				Expr2: expr,
			},
		}
		retIf = cgen.If1{
			Cond: cgen.CmpGE{
				Expr1: p.wfIdx,
				Expr2: last,
			},
			Then: cgen.Return{},
		}
	}
	if p.wfCores1 > 0 {
		p.wfShort = false
		stmts[2] = cgen.For{
			Cond: cgen.CmpNE{
				Expr1: p.wfIdx,
				Expr2: il(p.wfCores1),
			},
			Post: cgen.IncPre{
				Expr: p.wfIdx,
			},
			Body: cgen.Stmts{
				p.kernel7(),
				retIf,
			},
		}
	}
	if p.wfCores1 < p.wfCores2 {
		p.wfShort = true
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
		rows1    int
		rows2    int
		cols1    int
		cols2    int
		sfs      [][][][2]cgen.Gen
		sliceIdx cgen.Gen
		wfs      [][3]cgen.Gen
	)
	layer10 := func() cgen.Gen {
		var (
			stmts cgen.Stmts
			col   int
			dfs   [2]cgen.Gen
		)
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		load := func(part int, nm string) {
			var (
				vec        = vb(p.name(nm))
				ae         = p.dfPtr
				slicePitch = p.dfSliceBytes1
			)
			if p.dfShort {
				slicePitch = p.dfSliceBytes2
			}
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(
					col*p.dfMeldBytes +
						part*(p.dfMeldBytes/2),
				),
			}
			ae = addMul(ae, il(p.dfGroupBytes), p.groupIdx)
			ae = addMul(ae, il(p.dfPileBytes), p.pileIdx)
			ae = addMul(ae, il(p.dfCoreBytes), p.dfIdx)
			ae = addMul(ae, il(slicePitch), sliceIdx)
			stmt(cgen.Var{
				Type: avx.M512, What: vec,
				Init: avx.Mm512LoaduPs{ae},
			})
			dfs[part] = vec
		}
		madd := func(rows, acc int) {
			var (
				dfRe = dfs[0]
				dfIm = dfs[1]
			)
			for row := 0; row < rows; row++ {
				var (
					wfRe = wfs[row][0]
					wfIm = wfs[row][1]
					sfRe = sfs[row][col][acc][0]
					sfIm = sfs[row][col][acc][1]
				)
				stmt(cgen.Assign{
					Expr1: sfRe,
					Expr2: avx.Mm512FmaddPs{
						wfRe, dfRe, sfRe,
					},
				})
				switch {
				case p.pile0:
					var (
						mask = il(0xfcfc)
						wfMx = wfs[row][2]
					)
					stmt(cgen.Assign{
						Expr1: sfRe,
						Expr2: avx.Mm512Mask3FmaddPs{
							wfIm, dfIm, sfRe, mask,
						},
					})
					stmt(cgen.Assign{
						Expr1: sfIm,
						Expr2: avx.Mm512FmaddPs{
							wfMx, dfIm, sfIm,
						},
					})
					stmt(cgen.Assign{
						Expr1: sfIm,
						Expr2: avx.Mm512Mask3FnmaddPs{
							wfIm, dfRe, sfIm, mask,
						},
					})
				default:
					stmt(cgen.Assign{
						Expr1: sfRe,
						Expr2: avx.Mm512FmaddPs{
							wfIm, dfIm, sfRe,
						},
					})
					stmt(cgen.Assign{
						Expr1: sfIm,
						Expr2: avx.Mm512FmaddPs{
							wfRe, dfIm, sfIm,
						},
					})
					stmt(cgen.Assign{
						Expr1: sfIm,
						Expr2: avx.Mm512FnmaddPs{
							wfIm, dfRe, sfIm,
						},
					})
				}
			}
		}
		exch := func(part int) {
			var (
				vec  = dfs[part]
				ctrl = 1<<6 | 0<<4 | 3<<2 | 2<<0
			)
			stmt(cgen.Assign{
				Expr1: vec,
				Expr2: avx.Mm512ShuffleF32x4{
					vec, vec, il(ctrl),
				},
			})
		}
		for col = 0; col < cols2; col++ {
			load(0, "dfRe")
			load(1, "dfIm")
			madd(rows2, 0)
			if col == cols1 {
				break
			}
			if rows1 == 0 {
				continue
			}
			exch(0)
			exch(1)
			madd(rows1, 1)
		}
		return stmts
	}
	layer9 := func() cgen.Gen {
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		wfs = make([][3]cgen.Gen, rows2)
		for row := range wfs {
			var (
				ae         = p.wfPtr
				slicePitch = p.wfSliceBytes1
			)
			if p.wfShort {
				slicePitch = p.wfSliceBytes2
			}
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(row * p.wfMeldBytes),
			}
			ae = addMul(ae, il(p.wfGroupBytes), p.groupIdx)
			ae = addMul(ae, il(p.wfPileBytes), p.pileIdx)
			ae = addMul(ae, il(p.wfCoreBytes), p.wfIdx)
			ae = addMul(ae, il(slicePitch), sliceIdx)
			var (
				wfLd = vb(p.name("wfLd"))
				wfRe = vb(p.name("wfRe"))
				wfIm = vb(p.name("wfIm"))
			)
			stmt(cgen.Var{
				Type: avx.M512i, What: wfLd,
				Init: avx.Mm512LoaduSi512{ae},
			})
			wfs[row][0] = wfRe
			stmt(cgen.Var{
				Type: avx.M512, What: wfRe,
				Init: avx.Mm512CvtphPs{
					avx.Mm512Castsi512Si256{
						wfLd,
					},
				},
			})
			wfs[row][1] = wfIm
			stmt(cgen.Var{
				Type: avx.M512, What: wfIm,
				Init: avx.Mm512CvtphPs{
					avx.Mm512Extracti64x4Epi64{
						wfLd, il(1),
					},
				},
			})
			if p.pile0 {
				wfMx := vb(p.name("wfMx"))
				wfs[row][2] = wfMx
				stmt(cgen.Var{
					Type: avx.M512, What: wfMx,
					Init: avx.Mm512MaskMovPs{
						wfIm, il(0xfcfc),
						wfRe,
					},
				})
			}
		}
		stmt(layer10())
		return stmts
	}
	layer8 := func() cgen.Gen {
		sliceIdx = vb(p.name("s"))
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
			Body: layer9(),
		}
	}
	layer7 := func() cgen.Gen {
		var stmts [2]cgen.Stmts
		stmt := func(x int, st cgen.Gen) {
			stmts[x] = append(stmts[x], st)
		}
		do := func(row, col, acc, part int) {
			var (
				vec       = sfs[row][col][acc][part]
				ae        = p.sfPtr
				sitePitch = p.sfSiteBytes11
				rowPitch  = p.sfRowBytes11
				colPitch  = p.sfMeldBytes11
				accPitch  = p.wfMeldFrags * p.sfFragBytes
				partPitch = accPitch / 2
			)
			if p.dfShort {
				sitePitch = p.sfSiteBytes12
				rowPitch = p.sfRowBytes12
			}
			if row == rows1 {
				colPitch = p.sfMeldBytes21
			}
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(
					row*rowPitch +
						col*colPitch +
						acc*accPitch +
						part*partPitch,
				),
			}
			ae = addMul(ae, il(p.sfGroupBytes), p.groupIdx)
			ae = addMul(ae, il(p.sfPileBytes), p.pileIdx)
			ae = addMul(ae, il(p.sfCoreBytes1), p.dfIdx)
			ae = addMul(ae, il(sitePitch), p.wfIdx)
			if !p.epoch0zone0 {
				stmt(0, cgen.Assign{
					Expr1: vec,
					Expr2: avx.Mm512AddPs{
						vec,
						avx.Mm512LoaduPs{ae},
					},
				})
			}
			stmt(1, avx.Mm512StoreuPs{
				ae, vec,
			})
		}
		stmt(0, layer8())
		for row := range sfs {
			for col := range sfs[row] {
				for acc := range sfs[row][col] {
					switch {
					case row == rows1 && col == cols1:
						var (
							sfRe = sfs[row][col][acc][0]
							sfIm = sfs[row][col][acc][1]
							ctrl = 1<<6 | 0<<4 | 1<<2 | 0<<0
						)
						stmt(0, cgen.Assign{
							Expr1: sfRe,
							Expr2: avx.Mm512ShuffleF32x4{
								sfRe, sfIm, il(ctrl),
							},
						})
						do(row, col, acc, 0)
					default:
						for part := range &sfs[row][col][acc] {
							do(row, col, acc, part)
						}
					}
				}
			}
		}
		return cgen.Gens{
			stmts[0],
			stmts[1],
		}
	}
	layer6 := func() cgen.Gen {
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		for row := range sfs {
			for col := range sfs[row] {
				for acc := range sfs[row][col] {
					if col == 0 && acc == 0 {
						continue
					}
					for part, vec := range &sfs[row][col][acc] {
						stmt(cgen.Var{
							Type: avx.M512, What: vec,
							Init: sfs[row][0][0][part],
						})
					}
				}
			}
		}
		stmt(layer7())
		return stmts
	}
	layer5 := func() cgen.Gen {
		bias := func() cgen.Stmts {
			var stmts cgen.Stmts
			stmt := func(st cgen.Gen) {
				stmts = append(stmts, st)
			}
			for row := range sfs {
				var (
					sfRe  = sfs[row][0][0][0]
					frags = p.wfMeldFrags
				)
				if row == rows1 {
					frags = 1
				}
				for frag := 0; frag < frags; frag++ {
					var (
						mask      = 1 << uint(frag*8)
						bf        = p.bfPtr
						corePitch = p.wfSliceMelds1 * p.bfMeldBytes
					)
					if row == rows1 && cols1 > 0 {
						mask |= mask << 8
					}
					bf = cgen.Add{
						Expr1: bf,
						Expr2: il(
							row*p.bfMeldBytes +
								frag*p.bfFragBytes,
						),
					}
					bf = addMul(bf, il(p.bfGroupBytes), p.groupIdx)
					bf = addMul(bf, il(corePitch), p.wfIdx)
					bf = cgen.At{
						Expr: cgen.Cast{
							Type: cgen.PtrFloat,
							Expr: cgen.Paren{
								Inner: bf,
							},
						},
					}
					stmt(cgen.Assign{
						Expr1: sfRe,
						Expr2: avx.Mm512MaskMovPs{
							sfRe, il(mask),
							avx.Mm512Set1Ps{bf},
						},
					})
				}
			}
			return stmts
		}
		return cgen.Stmts{
			func() cgen.Gen {
				if p.pile0 {
					if p.epoch0zone0 {
						return bias()
					}
					if p.epochFirst+p.epochCnt > 1 {
						for x := range p.Filts {
							if p.Filts[x].BnPre == 0 {
								continue
							}
							return cgen.If{
								Cond: cgen.Unlikely{
									Cond: cgen.IsZero{
										Expr: p.zoneCoord,
									},
								},
								Then: bias(),
							}
						}
					}
				}
				return cgen.Cast{
					Type: cgen.Void,
					Expr: p.bfPtr,
				}
			}(),
			layer6(),
		}
	}
	layer4 := func() cgen.Gen {
		var (
			last  = len(sfs) * 2
			stmts = make(cgen.Stmts, last+1)
		)
		for row := range sfs {
			for part, vec := range &sfs[row][0][0] {
				stmts[row*2+part] = cgen.Var{
					Type: avx.M512, What: vec,
					Init: avx.Mm512SetzeroPs,
				}
			}
		}
		stmts[last] = layer5()
		return stmts
	}
	layer3 := func() cgen.Gen {
		sfs = make([][][][2]cgen.Gen, rows2)
		for row := range sfs {
			sfs[row] = make([][][2]cgen.Gen, cols2)
			for col := range sfs[row] {
				accs := p.dfMeldFrags
				if row == rows1 || col == cols1 {
					accs = 1
				}
				sfs[row][col] = make([][2]cgen.Gen, accs)
				for acc := range sfs[row][col] {
					var (
						sfRe = vb(p.name("sfRe"))
						sfIm = vb(p.name("sfIm"))
					)
					sfs[row][col][acc][0] = sfRe
					sfs[row][col][acc][1] = sfIm
				}
			}
		}
		return layer4()
	}
	layer2 := func() cgen.Gen {
		switch {
		case p.dfShort:
			cols2 = p.dfSliceMelds2
			cols1 = cols2 - p.dfSliceFrags2%p.dfMeldFrags
		default:
			cols2 = p.dfSliceMelds1
			cols1 = cols2
		}
		return layer3()
	}
	layer1 := func() cgen.Gen {
		switch {
		case p.wfShort:
			rows2 = p.wfSliceMelds2
			rows1 = rows2 - p.wfSliceFrags2%p.wfMeldFrags
		default:
			rows2 = p.wfSliceMelds1
			rows1 = rows2
		}
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
	wfTile     int
	wfTiles    int
	wfScrap    int
	wfHull     int
	dfTile     int
	dfTiles    int
	dfScrap    int
	dfHull     int
	groupTile  int
	groupTiles int
	groupScrap int
	groupHull  int
	calleeName string
	tensors    cgen.Gen
	wfCoord    cgen.Gen
	dfCoord    cgen.Gen
	groupCoord cgen.Gen
	sfPtr      cgen.Gen
	datSplit   int
	datPtrs    []cgen.Gen
	bnPtrs     []cgen.Gen
	groupIdx   cgen.Gen
	dfIdx      cgen.Gen
	dfLast     cgen.Gen
	toH        cgen.Gen
	toW        cgen.Gen
	lbs        []*loopB
	wfIdx      cgen.Gen
	wfShort    bool
}

func (c *consumeSums) Append(to []byte) []byte {
	c.layout = newLayout(c.Ctx, c.Spec)
	var (
		n1         = c.dfCores1 * c.dfSliceFrags1
		n2         = c.toChans * (n1 + c.dfSliceFrags2)
		wfBlks     = ceilQuo(n2, c.dfCores2*c.wfCores2)
		dfBlks     = ceilQuo(n2, c.dfCores2)
		groupBlks  = n2
		threadBlks int
	)
	switch c.platform {
	case raw.AVX512Float32:
		threadBlks = 512
	default:
		panic("bug")
	}
	c.wfTile = c.wfCores2
	c.wfTiles = 1
	c.wfScrap = 0
	c.wfHull = 1
	c.groupTile = 1
	c.groupTiles = c.Groups
	c.groupScrap = 0
	c.groupHull = c.Groups
	switch {
	case threadBlks <= dfBlks:
		var (
			tile  = ceilQuo(threadBlks, wfBlks)
			tiles = max(c.wfCores2/tile, 1)
		)
		c.wfTile = c.wfCores2 / tiles
		c.wfTiles = tiles
		c.wfScrap = c.wfCores2 - tiles*c.wfTile
		c.wfHull = tiles
		if c.wfScrap > 0 {
			c.wfTiles--
			c.wfScrap += c.wfTile
		}
		c.dfTile = 1
		c.dfTiles = c.dfCores2
		c.dfScrap = 0
		c.dfHull = c.dfCores2
	case threadBlks <= groupBlks:
		var (
			tile  = ceilQuo(threadBlks, dfBlks)
			tiles = max(c.dfCores2/tile, 1)
		)
		c.dfTile = c.dfCores2 / tiles
		c.dfTiles = tiles
		c.dfScrap = c.dfCores2 - tiles*c.dfTile
		c.dfHull = tiles
		if c.dfScrap > 0 {
			c.dfTiles--
			c.dfScrap += c.dfTile
		}
	default:
		c.dfTile = c.dfCores2
		c.dfTiles = 1
		c.dfScrap = 0
		c.dfHull = 1
		var (
			tile  = ceilQuo(threadBlks, groupBlks)
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
					il(c.wfHull),
					il(c.dfHull),
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
	var (
		body   = make(cgen.Stmts, 6)
		usedPt = false
	)
	c.tensors = vb(c.name("tensors"))
	body[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: c.tensors,
		Init: callee.Any(),
	}
	coord := func(nm string, hull, i int) cgen.Gen {
		var (
			ret  = vb(c.name(nm))
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
	c.wfCoord = coord("w", c.wfHull, 0)
	c.dfCoord = coord("d", c.dfHull, 1)
	c.groupCoord = coord("g", c.groupHull, 2)
	if !usedPt {
		body[4] = cgen.Cast{
			Type: cgen.Void,
			Expr: callee.Pt,
		}
	}
	body[5] = c.kernel1()
	return callee.Func(body)
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
	c.sfPtr = vb(c.name("sfPtr"))
	decl(c.sfPtr)
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
	c.datSplit = have
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
	c.dfIdx = vb(c.name("j"))
	switch c.dfHull {
	case 1:
		c.dfLast = nil
	default:
		c.dfLast = vb(c.name("last"))
	}
	stmts := make(cgen.Stmts, 3)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: c.dfIdx,
		Init: cgen.Mul{
			Expr1: il(c.dfTile),
			Expr2: c.dfCoord,
		},
	}
	if c.dfLast != nil {
		var expr cgen.Gen
		switch c.dfTiles {
		case c.dfHull:
			expr = il(c.dfTile - 1)
		case 0:
			expr = il(c.dfScrap - 1)
		default:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: c.dfCoord,
						Expr2: il(c.dfTiles),
					},
					Then: il(c.dfTile - 1),
					Else: il(c.dfScrap - 1),
				},
			}
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: c.dfLast,
			Init: cgen.Add{
				Expr1: c.dfIdx,
				Expr2: expr,
			},
		}
	}
	stmts[2] = c.kernel4()
	return stmts
}

func (c *consumeSums) kernel4() cgen.Gen {
	var (
		lh       *loopH
		lhToH    int
		lhToStep int
		rel      cgen.Gen
		base     cgen.Gen
		relBreak int
		lw       *loopW
		lwToH    int
		lwToW    int
		lwToStep int
	)
	layer7 := func() cgen.Gen {
		var retIf cgen.Gen
		if c.dfLast != nil {
			retIf = cgen.If1{
				Cond: cgen.CmpGE{
					Expr1: c.dfIdx,
					Expr2: c.dfLast,
				},
				Then: cgen.Return{},
			}
		}
		return cgen.Stmts{
			c.kernel5(),
			retIf,
			cgen.IncPre{
				Expr: c.dfIdx,
			},
		}
	}
	layer6 := func() cgen.Gen {
		if lwToStep == 0 {
			return layer7()
		}
		last := vb(c.name("jj"))
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: last,
				Init: cgen.Add{
					Expr1: cgen.Sub{
						Expr1: il(lw.segPast - 1),
						Expr2: rel,
					},
					Expr2: c.dfIdx,
				},
			},
			cgen.For{
				Cond: cgen.CmpLE{
					Expr1: c.dfIdx,
					Expr2: last,
				},
				Post: cgen.AddAssign{
					Expr1: c.toW,
					Expr2: il(lwToStep),
				},
				Body: layer7(),
			},
		}
	}
	layer5 := func() cgen.Gen {
		c.toH = vb(c.name("toH"))
		c.toW = vb(c.name("toW"))
		c.lbs = lw.lbs
		var (
			exprW   cgen.Gen
			breakIf cgen.Gen
		)
		switch lwToStep {
		case 0:
			exprW = il(lwToW)
		default:
			exprW = addMul(
				il(lwToW-lwToStep*lw.segFirst),
				il(lwToStep),
				rel,
			)
		}
		if lw.segPast == relBreak {
			breakIf = cgen.If1{
				Cond: cgen.CmpGE{
					Expr1: c.dfIdx,
					Expr2: il(lh.segPast),
				},
				Then: cgen.Break,
			}
		}
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: c.toH,
				Init: cgen.Add{
					Expr1: base,
					Expr2: il(lwToH),
				},
			},
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: c.toW,
				Init: exprW,
			},
			layer6(),
			breakIf,
		}
	}
	layer4 := func() cgen.Gen {
		var (
			lws  = lh.lws
			tree func(int, int) cgen.Stmts
		)
		leaf := func(x int) cgen.Stmts {
			lw = lws[x]
			lwToH = lw.fromH / 2
			lwToW = lw.fromW / 2
			lwToStep = lw.fromStep / 2
			var assn cgen.Gen
			if x+1 < len(lws) {
				assn = cgen.Assign{
					Expr1: rel,
					Expr2: il(lw.segPast),
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
				start = lws[first].segFirst
				stop  = lws[last].segPast
				split = start + (stop-start)/2
				x     = first + 1
			)
			for lws[x].segPast <= split {
				x++
			}
			return cgen.Stmts{
				cgen.If{
					Cond: cgen.CmpL{
						Expr1: rel,
						Expr2: il(lws[x].segFirst),
					},
					Then: tree(first, x-1),
				},
				tree(x, last),
			}
		}
		return tree(0, len(lws)-1)
	}
	layer3 := func() cgen.Gen {
		if lh.segStep == 0 {
			relBreak = -1
			return layer4()
		}
		var (
			last1 = lh.segPast - lh.segFirst - 1
			last2 = last1 % lh.segStep
		)
		relBreak = last2 + 1
		return cgen.For{
			Post: cgen.CommaSpaced{
				cgen.Assign{
					Expr1: rel,
					Expr2: il(0),
				},
				cgen.AddAssign{
					Expr1: base,
					Expr2: il(lhToStep),
				},
			},
			Body: layer4(),
		}
	}
	layer2 := func() cgen.Gen {
		rel = vb(c.name("rel"))
		base = vb(c.name("base"))
		var (
			relExpr cgen.Gen = cgen.Sub{
				Expr1: c.dfIdx,
				Expr2: il(lh.segFirst),
			}
			baseExpr = il(lhToH)
		)
		if lh.segStep != 0 {
			var (
				numer cgen.Gen = cgen.Cast{
					Type: cgen.SizeT,
					Expr: cgen.Paren{
						Inner: relExpr,
					},
				}
				denom = il(lh.segStep)
			)
			relExpr = cgen.Rem{
				Expr1: numer,
				Expr2: denom,
			}
			baseExpr = addMul(
				baseExpr,
				cgen.Quo{
					Expr1: numer,
					Expr2: denom,
				},
				il(lhToStep),
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
				What: base,
				Init: baseExpr,
			},
			layer3(),
		}
	}
	layer1 := func() cgen.Gen {
		var (
			lhs  = c.segs.lhs
			tree func(int, int) cgen.Stmts
		)
		leaf := func(x int) cgen.Stmts {
			lh = lhs[x]
			lhToH = lh.fromH / 2
			lhToStep = lh.fromStep / 2
			var assn cgen.Gen
			if x+1 < len(lhs) {
				assn = cgen.Assign{
					Expr1: c.dfIdx,
					Expr2: il(lh.segPast),
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
				start = lhs[first].segFirst
				stop  = lhs[last].segPast
				split = start + (stop-start)/2
				x     = first + 1
			)
			for lhs[x].segPast <= split {
				x++
			}
			return cgen.Stmts{
				cgen.If{
					Cond: cgen.CmpL{
						Expr1: c.dfIdx,
						Expr2: il(lhs[x].segFirst),
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

func (c *consumeSums) kernel5() cgen.Gen {
	c.wfIdx = vb(c.name("k"))
	var (
		stmts = make(cgen.Stmts, 4)
		retIf cgen.Gen
	)
	stmts[0] = cgen.Var{
		Type: cgen.PtrdiffT,
		What: c.wfIdx,
		Init: cgen.Mul{
			Expr1: il(c.wfTile),
			Expr2: c.wfCoord,
		},
	}
	if c.wfHull > 1 {
		var (
			last = vb(c.name("kk"))
			expr cgen.Gen
		)
		switch c.wfTiles {
		case c.wfHull:
			expr = il(c.wfTile - 1)
		case 0:
			expr = il(c.wfScrap - 1)
		default:
			expr = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: c.wfCoord,
						Expr2: il(c.wfTiles),
					},
					Then: il(c.wfTile - 1),
					Else: il(c.wfScrap - 1),
				},
			}
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: last,
			Init: cgen.Add{
				Expr1: c.wfIdx,
				Expr2: expr,
			},
		}
		retIf = cgen.If1{
			Cond: cgen.CmpGE{
				Expr1: c.wfIdx,
				Expr2: last,
			},
			Then: cgen.Return{},
		}
	}
	if c.wfCores1 > 0 {
		c.wfShort = false
		stmts[2] = cgen.For{
			Cond: cgen.CmpNE{
				Expr1: c.wfIdx,
				Expr2: il(c.wfCores1),
			},
			Post: cgen.IncPre{
				Expr: c.wfIdx,
			},
			Body: cgen.Stmts{
				c.kernel6(),
				retIf,
			},
		}
	}
	if c.wfCores1 < c.wfCores2 {
		c.wfShort = true
		stmts[3] = c.kernel6()
	}
	return stmts
}

func (c *consumeSums) kernel6() cgen.Gen {
	switch c.platform {
	case raw.AVX512Float32:
		return c.m512()
	default:
		panic("bug")
	}
}

func (c *consumeSums) m512() cgen.Gen {
	type Span struct {
		vec   cgen.Gen
		lane  int
		lanes int
		relC  int
		relH  int
		relW  int
		prior bool
	}
	var (
		lbs      []*loopB
		rowIdx   cgen.Gen
		rowChans int
		bnMuls   [][]cgen.Gen
		bnAdds   [][]cgen.Gen
		blkFirst int
		blkCnt   int
		iters    int
		iterIdx  cgen.Gen
		accBlks  [][]int
		bwd      *quadfft.Bwd
		spans    []*Span
	)
	addr := func(span *Span, ptr int) cgen.Gen {
		var (
			ae         = c.datPtrs[ptr]
			pitch1     = c.To.Pitch1Bytes[ptr]
			pitch2     = c.To.Pitch2Bytes[ptr]
			groupPitch = c.toChans * pitch2
			corePitch  = c.wfSliceFrags1 * pitch2
			rowPitch   = c.wfMeldFrags * pitch2
			toStep     = lbs[blkFirst].fromStep / 2
			blkPitch   = toStep * c.datBytes
			iterPitch  = blkCnt * blkPitch
		)
		ae = cgen.Add{
			Expr1: ae,
			Expr2: il(
				span.relC*pitch2 +
					span.relH*pitch1 +
					span.relW*c.datBytes -
					span.lane*c.datBytes,
			),
		}
		ae = addMul(ae, il(groupPitch), c.groupIdx)
		ae = addMul(ae, il(corePitch), c.wfIdx)
		ae = addMul(ae, il(rowPitch), rowIdx)
		ae = addMul(ae, il(pitch1), c.toH)
		ae = addMul(ae, il(c.datBytes), c.toW)
		ae = addMul(ae, il(iterPitch), iterIdx)
		return ae
	}
	mask := func(span *Span) cgen.Gen {
		var (
			mask1 = 1<<uint(span.lanes) - 1
			mask2 = mask1 << uint(span.lane)
		)
		return il(mask2)
	}
	layer13 := func() cgen.Gen {
		var (
			stmts cgen.Stmts
			ptr   = c.datSplit
			stop  = len(c.datPtrs)
		)
		for ; ptr < stop; ptr++ {
			for _, span := range spans {
				if span == nil {
					continue
				}
				stmts = append(
					stmts,
					avx.Mm512MaskStoreuPs{
						addr(span, ptr),
						mask(span),
						span.vec,
					},
				)
			}
		}
		return stmts
	}
	layer12 := func() cgen.Gen {
		var (
			stmts cgen.Stmts
			co    = make([]*Span, 1, 2)
		)
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		ops := func() {
			var (
				vec    = co[0].vec
				datPtr = 0
				bnPtr  = 0
			)
			for op := range c.To.Ops {
				op := &c.To.Ops[op]
				switch op.Kind {
				case mod.Add:
					for n := op.Int; n > 0; n-- {
						for _, span := range co {
							stmt(cgen.Assign{
								Expr1: vec,
								Expr2: avx.Mm512AddPs{
									vec,
									avx.Mm512MaskzLoaduPs{
										mask(span),
										addr(span, datPtr),
									},
								},
							})
						}
						datPtr++
					}
				case mod.Bn:
					switch {
					case len(co) == 1:
						fallthrough
					case co[0].relC == co[1].relC:
						ch := co[0].relC
						stmt(&bn.Apply{
							Ctx: c.bc,
							Mul: bnMuls[ch][bnPtr],
							Add: bnAdds[ch][bnPtr],
							To:  vec,
						})
					default:
						for _, span := range co {
							ch := span.relC
							stmt(&bn.Apply{
								Ctx:  c.bc,
								Mul:  bnMuls[ch][bnPtr],
								Add:  bnAdds[ch][bnPtr],
								To:   vec,
								Mask: mask(span),
							})
						}
					}
					bnPtr++
				case mod.ReLU:
					stmt(&act.ReLU{
						Ctx:      c.ac,
						NegSlope: op.Float,
						Var:      vec,
					})
				default:
					panic("bug")
				}
			}
		}
		for x1, span1 := range spans {
			if span1 == nil ||
				span1.prior {
				continue
			}
			co[0] = span1
			for x2 := x1 + 1; x2 < len(spans); x2++ {
				span2 := spans[x2]
				if span2 != nil &&
					span2.vec == span1.vec {
					co = co[:2]
					co[1] = span2
					span2.prior = true
					break
				}
			}
			ops()
			co = co[:1]
		}
		stmt(layer13())
		return stmts
	}
	layer11 := func() cgen.Gen {
		if rowChans != c.wfMeldFrags {
			return layer12()
		}
		var (
			stmts cgen.Stmts
			pms   [2]cgen.Gen
		)
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		ctrl := func(side, lanes int) cgen.Gen {
			pm := pms[side]
			if pm == nil {
				pm = vb(c.name("pm"))
				var (
					set  = make(avx.Mm512SetEpi32, 16)
					base = side * (16 + 8)
					off  = 0
				)
				for x := 15; x >= 0; x-- {
					set[x] = il(base + off)
					if off++; off == lanes {
						base ^= 16
						off = 0
					}
				}
				stmt(cgen.Var{
					Type: avx.M512i, What: pm,
					Init: set,
				})
				pms[side] = pm
			}
			return pm
		}
		for x1, span1 := range spans {
			if span1 == nil {
				continue
			}
			var (
				x2    = x1 + 1
				span2 *Span
				pm    cgen.Gen
			)
			for ; x2 < len(spans); x2++ {
				span2 = spans[x2]
				if span2 == nil ||
					span2.relC != span1.relC ||
					span2.relH != span1.relH {
					continue
				}
				switch {
				case span1.relW+span1.lanes == span2.relW:
					pm = ctrl(0, span1.lanes)
				case span2.relW+span2.lanes == span1.relW:
					pm = ctrl(1, span2.lanes)
					span1.relW = span2.relW
				default:
					continue
				}
				break
			}
			if x2 == len(spans) {
				continue
			}
			pack := vb(c.name("pack"))
			stmt(cgen.Var{
				Type: avx.M512, What: pack,
				Init: avx.Mm512Permutex2varPs{
					span1.vec, pm,
					span2.vec,
				},
			})
			span1.vec = pack
			span1.lane = 0
			span1.lanes += span2.lanes
			spans[x2] = nil
		}
		stmt(layer12())
		return stmts
	}
	layer10 := func() cgen.Gen {
		if rowChans != 1 {
			return layer11()
		}
		var (
			stmts  cgen.Stmts
			pms    [2]cgen.Gen
			attach []int
		)
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		ctrl := func(side, lanes int) cgen.Gen {
			pm := pms[side]
			if pm == nil {
				pm = vb(c.name("pm"))
				var (
					set  = make(avx.Mm512SetEpi32, 16)
					base = side * 8
					off  = 0
				)
				for x := 15; x >= 0; x-- {
					set[x] = il(base + off)
					if off++; off == lanes {
						base += 8
						base %= 32
						off = 0
					}
				}
				stmt(cgen.Var{
					Type: avx.M512i, What: pm,
					Init: set,
				})
				pms[side] = pm
			}
			return pm
		}
		for x1, span1 := range spans {
			if span1 == nil {
				continue
			}
			var (
				lanes = span1.lanes
				nextW = span1.relW + lanes
				avail = 16 - lanes
			)
			for x2 := x1 + 1; x2 < len(spans); x2++ {
				span2 := spans[x2]
				if span2 != nil &&
					span2.relH == span1.relH &&
					span2.relW == nextW &&
					span2.lanes <= avail {
					attach = append(attach, x2)
					nextW += span2.lanes
					avail -= span2.lanes
				}
			}
			n := len(attach)
			if n == 0 {
				continue
			}
			var (
				pack = vb(c.name("pack"))
				vec1 = span1.vec
				vec2 = spans[attach[n-1]].vec
				side = btoi(span1.lane != 0)
				pm   = ctrl(side, lanes)
				expr cgen.Gen
			)
			switch vec1 {
			case vec2:
				expr = avx.Mm512PermutexvarPs{
					pm, vec1,
				}
			default:
				expr = avx.Mm512Permutex2varPs{
					vec1, pm, vec2,
				}
			}
			stmt(cgen.Var{
				Type: avx.M512, What: pack,
				Init: expr,
			})
			span1.vec = pack
			span1.lane = 0
			span1.lanes = 16 - avail
			for _, x2 := range attach {
				spans[x2] = nil
			}
			attach = attach[:0]
		}
		stmt(layer11())
		return stmts
	}
	layer9 := func() cgen.Gen {
		spans = spans[:0]
		for h := 0; h < 8; h++ {
			for acc, blks := range accBlks {
				dat := bwd.Out[acc*8+h]
				if dat == nil {
					continue
				}
				for side, blk := range blks {
					lb := lbs[blk]
					if lb.yieldH <= h {
						continue
					}
					var (
						relBlk = blk - lb.blkFirst
						toStep = lb.fromStep / 2
						w      = relBlk * toStep
					)
					spans = append(
						spans, &Span{
							vec:   dat,
							lane:  side * 8,
							lanes: lb.yieldW,
							relC:  side % rowChans,
							relH:  lb.fromH/2 + h,
							relW:  lb.fromW/2 + w,
							prior: false,
						},
					)
				}
			}
		}
		return layer10()
	}
	layer8 := func() cgen.Gen {
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		load := func(pile, acc, part int) cgen.Gen {
			var (
				ae        = c.sfPtr
				sitePitch = c.sfSiteBytes11
				rowPitch  = c.sfRowBytes11
				meldPitch = c.sfMeldBytes11
			)
			if len(lbs) == c.dfSliceFrags2 {
				sitePitch = c.sfSiteBytes12
				rowPitch = c.sfRowBytes12
			}
			if rowChans == 1 {
				meldPitch = c.sfMeldBytes21
			}
			var (
				meldFirst = blkFirst / c.dfMeldFrags
				iterMelds = blkCnt / c.dfMeldFrags
				iterPitch = iterMelds * meldPitch
				accPitch  = c.wfMeldFrags * c.sfFragBytes
				partPitch = accPitch / 2
			)
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(
					pile*c.sfPileBytes +
						meldFirst*meldPitch +
						acc*accPitch +
						part*partPitch,
				),
			}
			ae = addMul(ae, il(c.sfGroupBytes), c.groupIdx)
			ae = addMul(ae, il(c.sfCoreBytes1), c.dfIdx)
			ae = addMul(ae, il(sitePitch), c.wfIdx)
			ae = addMul(ae, il(rowPitch), rowIdx)
			ae = addMul(ae, il(iterPitch), iterIdx)
			return avx.Mm512LoaduPs{ae}
		}
		for pile := 0; pile < c.zoneFrags; pile++ {
			for acc, blks := range accBlks {
				for part := 0; part < 2; part++ {
					var (
						x1   = acc * c.zoneFrags * 2
						x2   = x1 + pile*2 + part
						sf1  = bwd.In[x2]
						expr cgen.Gen
					)
					switch {
					case part == 1 && len(blks) == 1:
						var (
							sf2  = bwd.In[x2-1]
							ctrl = 1<<6 | 0<<4 | 3<<2 | 2<<0
						)
						expr = avx.Mm512ShuffleF32x4{
							sf2, sf2, il(ctrl),
						}
					default:
						expr = load(pile, acc, part)
					}
					stmt(cgen.Var{
						Type: avx.M512, What: sf1,
						Init: expr,
					})
				}
			}
		}
		stmt(bwd)
		stmt(layer9())
		return stmts
	}
	layer7 := func() cgen.Gen {
		bwd = &quadfft.Bwd{
			Platform: c.platform,
			Nms:      c.nms,
		}
		var (
			accs = len(accBlks)
			each = c.zoneFrags * 2
		)
		for x := 0; x < accs*each; x++ {
			var sf cgen.Gen
			switch x % 2 {
			case 0:
				sf = vb(c.name("sfRe"))
			default:
				sf = vb(c.name("sfIm"))
			}
			bwd.In[x] = sf
		}
		for acc, blks := range accBlks {
			yieldH := 0
			for _, blk := range blks {
				yieldH = max(
					yieldH,
					lbs[blk].yieldH,
				)
			}
			for h := 0; h < yieldH; h++ {
				var (
					x   = acc*each + h
					dat = vb(c.name("dat"))
				)
				bwd.Out[x] = dat
			}
		}
		return layer8()
	}
	layer6 := func() cgen.Gen {
		switch {
		case rowChans == 1:
			var (
				quo = blkCnt / 2
				rem = blkCnt % 2
			)
			accBlks = make([][]int, quo+rem)
			for acc := range accBlks {
				var (
					blk1 = blkFirst + acc*2
					blk2 = blk1 + 1
					blks = []int{blk1, blk2}
				)
				if acc == quo {
					blks = blks[:1]
				}
				accBlks[acc] = blks
			}
		case blkCnt == 1:
			accBlks = [][]int{
				{blkFirst, blkFirst},
			}
		default:
			accBlks = [][]int{
				{blkFirst, blkFirst + 1},
				{blkFirst + 1, blkFirst},
			}
		}
		return layer7()
	}
	layer5 := func() cgen.Gen {
		iterIdx = vb(c.name("t"))
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: iterIdx,
				Init: il(0),
			},
			func() cgen.Gen {
				if iters == 1 {
					return layer6()
				}
				return cgen.For{
					Cond: cgen.CmpL{
						Expr1: iterIdx,
						Expr2: il(iters),
					},
					Post: cgen.IncPre{
						Expr: iterIdx,
					},
					Body: layer6(),
				}
			}(),
		}
	}
	layer4 := func() cgen.Gen {
		var gens cgen.Gens
		gen := func() {
			gens = append(gens, layer5())
		}
		each := c.dfMeldFrags
		if rowChans == 1 {
			each *= 2
		}
		for blk := 0; ; {
			rem := len(lbs) - blk
			if rem <= each {
				if rem > 0 {
					blkFirst = blk
					blkCnt = rem
					iters = 1
					gen()
				}
				break
			}
			blkFirst = blk
			blkCnt = each
			iters = 1
			for lb := lbs[blk]; ; {
				blk += each
				if blk+each > len(lbs) ||
					lbs[blk+each-1] != lb {
					break
				}
				iters++
			}
			gen()
		}
		return gens
	}
	layer3 := func() cgen.Gen {
		n := len(c.bnPtrs)
		if n == 0 {
			return layer4()
		}
		bnMuls = make([][]cgen.Gen, rowChans)
		bnAdds = make([][]cgen.Gen, rowChans)
		var (
			last = n * rowChans
			gens = make(cgen.Gens, last+1)
		)
		for ch1 := 0; ch1 < rowChans; ch1++ {
			var (
				muls = make([]cgen.Gen, n)
				adds = make([]cgen.Gen, n)
				ch2  = il(ch1)
			)
			ch2 = addMul(ch2, il(c.toChans), c.groupIdx)
			ch2 = addMul(ch2, il(c.wfSliceFrags1), c.wfIdx)
			ch2 = addMul(ch2, il(c.wfMeldFrags), rowIdx)
			ch2 = cgen.Paren{
				Inner: ch2,
			}
			for x1, ptr := range c.bnPtrs {
				var (
					bnMul = vb(c.name("bnMul"))
					bnAdd = vb(c.name("bnAdd"))
					x2    = x1*rowChans + ch1
				)
				muls[x1] = bnMul
				adds[x1] = bnAdd
				gens[x2] = &bn.Load{
					Ctx:     c.bc,
					Mas:     ptr,
					Channel: ch2,
					Mul:     bnMul,
					Add:     bnAdd,
				}
			}
			bnMuls[ch1] = muls
			bnAdds[ch1] = adds
		}
		gens[last] = layer4()
		return gens
	}
	layer2 := func() cgen.Gen {
		rowIdx = vb(c.name("r"))
		var (
			stmts = make(cgen.Stmts, 3)
			rows1 int
			rows2 int
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: rowIdx,
			Init: il(0),
		}
		switch {
		case c.wfShort:
			rows2 = c.wfSliceMelds2
			rows1 = rows2 - c.wfSliceFrags2%c.wfMeldFrags
		default:
			rows2 = c.wfSliceMelds1
			rows1 = rows2
		}
		if rows1 > 0 {
			rowChans = c.wfMeldFrags
			stmts[1] = cgen.For{
				Cond: cgen.CmpNE{
					Expr1: rowIdx,
					Expr2: il(rows1),
				},
				Post: cgen.IncPre{
					Expr: rowIdx,
				},
				Body: layer3(),
			}
		}
		if rows1 < rows2 {
			rowChans = 1
			stmts[2] = layer3()
		}
		return stmts
	}
	layer1 := func() cgen.Gen {
		var (
			n1 = len(c.lbs)
			n2 = c.lbs[n1-1].blkPast
		)
		lbs = make([]*loopB, n2)
		for _, lb := range c.lbs {
			blk := lb.blkFirst
			for ; blk < lb.blkPast; blk++ {
				lbs[blk] = lb
			}
		}
		return layer2()
	}
	return layer1()
}
