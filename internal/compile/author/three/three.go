package three

import (
	"NN-512/internal/compile/author/act"
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/bn"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/mod"
	"NN-512/internal/compile/author/sumr"
	"NN-512/internal/compile/author/threader"
	"NN-512/internal/compile/author/wct"
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

func void(a cgen.Gen) cgen.Gen {
	return cgen.Cast{
		Type: cgen.Void,
		Expr: a,
	}
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
		prefix:      pl.Config.Prefix + "Three",
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
	From     SpecFrom
	Filts    []SpecFilts
	To       SpecTo
	StrideH  int
	StrideW  int
	PaddingH int
	PaddingW int
	Groups   int
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

type block struct {
	fromH  int
	fromW  int
	padH   int
	padW   int
	datH   int
	datW   int
	yieldH int
	yieldW int
}

type loopW struct {
	fromH    int
	fromW    int
	fromStep int
	segFirst int
	segPast  int
	blks     []*block
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
		segs     segments
		blks     []*block
		lw1      loopW
		idx      map[int]int
		lws      []*loopW
		tie      int
		fromStep int
		segStep  int
		segPast  int
		at       int
	)
	equal := func(seg1, seg2 []*block) bool {
		if len(seg1) != len(seg2) {
			return false
		}
		for i := range seg1 {
			if *seg1[i] != *seg2[i] {
				return false
			}
		}
		return true
	}
	commit3 := func() {
		n1 := len(lws)
		if n1 == 0 {
			return
		}
		n2 := tie
		if n2 == -1 {
			n2 = n1
		}
		if n2 > 0 {
			lh := &loopH{
				fromH:    lws[0].fromH,
				fromStep: 0,
				segFirst: lws[0].segFirst,
				segStep:  0,
				segPast:  lws[n2-1].segPast,
				lws:      make([]*loopW, n2),
			}
			for i, lw := range lws[:n2] {
				lw.fromH -= lh.fromH
				lw.segFirst -= lh.segFirst
				lw.segPast -= lh.segFirst
				lh.lws[i] = lw
			}
			segs.lhs = append(
				segs.lhs, lh,
			)
		}
		if n2 == n1 {
			return
		}
		lh := &loopH{
			fromH:    lws[n2].fromH,
			fromStep: fromStep,
			segFirst: lws[n2].segFirst,
			segStep:  segStep,
			segPast:  segPast,
			lws:      make([]*loopW, n1-n2),
		}
		for i, lw := range lws[n2:] {
			lw.fromH -= lh.fromH
			lw.segFirst -= lh.segFirst
			lw.segPast -= lh.segFirst
			lh.lws[i] = lw
		}
		segs.lhs = append(
			segs.lhs, lh,
		)
	}
	commit2 := func(flush bool) {
		n := len(lw1.blks)
		if n == 0 {
			if flush {
				commit3()
			}
			return
		}
		match := func(lw2 *loopW) bool {
			var (
				iters1 = lw1.segPast - lw1.segFirst
				iters2 = lw2.segPast - lw2.segFirst
			)
			if iters1 != iters2 {
				return false
			}
			return equal(lw1.blks, lw2.blks)
		}
		if tie == -1 {
			if i, ok := idx[lw1.fromW]; ok {
				lw2 := lws[i]
				if match(lw2) {
					tie = i
					fromStep = lw1.fromH - lw2.fromH
					segStep = lw1.segFirst - lw2.segFirst
					segPast = lw1.segPast
					at = i
					if flush {
						commit3()
					}
					return
				}
			}
		} else {
			if at++; at == len(lws) {
				at = tie
			}
			if match(lws[at]) {
				segPast = lw1.segPast
				if flush {
					commit3()
				}
				return
			}
			commit3()
			idx = make(map[int]int)
			lws = lws[:0]
			tie = -1
		}
		lw2 := lw1
		lw2.blks = make([]*block, n)
		for i, blk1 := range lw1.blks {
			blk2 := *blk1
			lw2.blks[i] = &blk2
		}
		idx[lw2.fromW] = len(lws)
		lws = append(lws, &lw2)
		if flush {
			commit3()
		}
	}
	commit1 := func(flush bool) {
		n := len(blks)
		if n == 0 {
			if flush {
				commit2(true)
			}
			return
		}
		i := segs.cnt
		segs.cnt = i + 1
		var (
			h = blks[0].fromH
			w = blks[0].fromW
		)
		for _, blk := range blks {
			blk.fromH -= h
			blk.fromW -= w
		}
		if len(lw1.blks) > 0 {
			if lw1.fromH == h &&
				equal(lw1.blks, blks) {
				if lw1.segFirst == i-1 {
					lw1.fromStep = w - lw1.fromW
				}
				lw1.segPast = i + 1
				if flush {
					commit2(true)
				}
				return
			}
			commit2(false)
		}
		lw1.fromH = h
		lw1.fromW = w
		lw1.fromStep = 0
		lw1.segFirst = i
		lw1.segPast = i + 1
		lw1.blks = lw1.blks[:n]
		for j, blk := range blks {
			*lw1.blks[j] = *blk
		}
		if flush {
			commit2(true)
		}
	}
	layer5 := func() {
		var (
			h1 = spec.PaddingH
			h2 = h1 + spec.From.Height
			h3 = h2 + spec.PaddingH
			w1 = spec.PaddingW
			w2 = w1 + spec.From.Width
			w3 = w2 + spec.PaddingW
		)
		for h := 0; h+3 <= h3; h += 6 {
			for w := 0; w+3 <= w3; w += 6 {
				i := len(blks)
				if i == cap(blks) {
					commit1(false)
					i = 0
				}
				blks = blks[:i+1]
				blk := blks[i]
				blk.fromH = h
				blk.fromW = w
				blk.padH = min(max(h1-h, 0), 8)
				blk.padW = min(max(w1-w, 0), 8)
				blk.datH = min(max(h2-h, 0), 8) - blk.padH
				blk.datW = min(max(w2-w, 0), 8) - blk.padW
				blk.yieldH = min(h3-h-2, 6)
				blk.yieldW = min(w3-w-2, 6)
				if blk.datH == 0 || blk.datW == 0 {
					blk.padH = 8
					blk.padW = 8
					blk.datH = 0
					blk.datW = 0
				}
			}
		}
		commit1(true)
	}
	layer4 := func() {
		idx = make(map[int]int)
		tie = -1
		layer5()
	}
	layer3 := func() {
		lw1.blks = make([]*block, segBlks)
		for i := range lw1.blks {
			lw1.blks[i] = new(block)
		}
		lw1.blks = lw1.blks[:0]
		layer4()
	}
	layer2 := func() {
		blks = make([]*block, segBlks)
		for i := range blks {
			blks[i] = new(block)
		}
		blks = blks[:0]
		layer3()
	}
	layer1 := func() *segments {
		sig := fmt.Sprint(
			"newSegments",
			" ",
			spec.From.Height,
			spec.From.Width,
			spec.PaddingH,
			spec.PaddingW,
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
	blkFrags      int
	fromChans     int
	toChans       int
	slices1       int
	slices2       int
	epochs1       int
	epochs2       int
	alignment     int
	biasBytes     int
	bfBytes       int
	bfEpochBytes  int
	bfTotalBytes  int
	wtBytes       int
	wfBytes       int
	wfSliceWfs1   int
	wfSliceWfs2   int
	wfSliceBytes1 int
	wfSliceBytes2 int
	wfCores1      int
	wfCores2      int
	wfCoreBytes11 int
	wfCoreBytes12 int
	wfCoreBytes21 int
	wfCoreBytes22 int
	wfFragBytes1  int
	wfFragBytes2  int
	wfGroupBytes1 int
	wfGroupBytes2 int
	wfEpochBytes1 int
	wfEpochBytes2 int
	wfTotalBytes  int
	datBytes      int
	dfBytes       int
	dfSliceDfs1   int
	dfSliceDfs2   int
	dfSliceBytes1 int
	dfSliceBytes2 int
	dfCores1      int
	dfCores2      int
	dfCoreBytes11 int
	dfCoreBytes12 int
	dfCoreBytes21 int
	dfCoreBytes22 int
	dfFragBytes1  int
	dfFragBytes2  int
	dfGroupBytes1 int
	dfGroupBytes2 int
	dfEpochBytes1 int
	dfEpochBytes2 int
	dfTotalBytes  int
	sfBytes       int
	sfSumBytes11  int
	sfSumBytes12  int
	sfSumBytes21  int
	sfSumBytes22  int
	sfCoreBytes1  int
	sfCoreBytes2  int
	sfFragBytes   int
	sfGroupBytes  int
	sfTotalBytes  int
}

func newLayout(ctx *Ctx, spec *Spec) *layout {
	var (
		y layout
	)
	pad := func(n int) int {
		n += y.alignment - 1
		n &= -y.alignment
		return n
	}
	layer9 := func() *layout {
		y.dfCoreBytes11 = y.slices1 * y.dfSliceBytes1
		y.dfCoreBytes12 = y.slices1 * y.dfSliceBytes2
		y.dfCoreBytes21 = y.slices2 * y.dfSliceBytes1
		y.dfCoreBytes22 = y.slices2 * y.dfSliceBytes2
		y.dfFragBytes1 = y.dfCores1 * y.dfCoreBytes11
		y.dfFragBytes2 = y.dfCores1 * y.dfCoreBytes21
		if y.dfCores1 < y.dfCores2 {
			y.dfFragBytes1 += y.dfCoreBytes12
			y.dfFragBytes2 += y.dfCoreBytes22
		}
		y.dfGroupBytes1 = y.blkFrags * y.dfFragBytes1
		y.dfGroupBytes2 = y.blkFrags * y.dfFragBytes2
		y.dfEpochBytes1 = spec.Groups * y.dfGroupBytes1
		y.dfEpochBytes2 = spec.Groups * y.dfGroupBytes2
		y.dfTotalBytes = y.epochs1 * y.dfEpochBytes1
		if y.epochs1 < y.epochs2 {
			y.dfTotalBytes += y.dfEpochBytes2
		}
		return &y
	}
	layer8 := func() *layout {
		y.wfCoreBytes11 = pad(y.slices1 * y.wfSliceBytes1)
		y.wfCoreBytes12 = pad(y.slices1 * y.wfSliceBytes2)
		y.wfCoreBytes21 = pad(y.slices2 * y.wfSliceBytes1)
		y.wfCoreBytes22 = pad(y.slices2 * y.wfSliceBytes2)
		y.wfFragBytes1 = y.wfCores1 * y.wfCoreBytes11
		y.wfFragBytes2 = y.wfCores1 * y.wfCoreBytes21
		if y.wfCores1 < y.wfCores2 {
			y.wfFragBytes1 += y.wfCoreBytes12
			y.wfFragBytes2 += y.wfCoreBytes22
		}
		y.wfGroupBytes1 = y.blkFrags * y.wfFragBytes1
		y.wfGroupBytes2 = y.blkFrags * y.wfFragBytes2
		y.wfEpochBytes1 = spec.Groups * y.wfGroupBytes1
		y.wfEpochBytes2 = spec.Groups * y.wfGroupBytes2
		y.wfTotalBytes = y.epochs1 * y.wfEpochBytes1
		if y.epochs1 < y.epochs2 {
			y.wfTotalBytes += y.wfEpochBytes2
		}
		return layer9()
	}
	layer7 := func() *layout {
		y.bfEpochBytes = spec.Groups * y.toChans * y.bfBytes
		y.bfTotalBytes = pad(y.epochs2 * y.bfEpochBytes)
		return layer8()
	}
	layer6 := func() *layout {
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
		return layer7()
	}
	layer5 := func() *layout {
		var (
			sums11 = y.dfSliceDfs1 * y.wfSliceWfs1
			sums12 = y.dfSliceDfs1 * y.wfSliceWfs2
			sums21 = y.dfSliceDfs2 * y.wfSliceWfs1
			sums22 = y.dfSliceDfs2 * y.wfSliceWfs2
		)
		y.sfSumBytes11 = sums11 * y.sfBytes
		y.sfSumBytes12 = sums12 * y.sfBytes
		y.sfSumBytes21 = sums21 * y.sfBytes
		y.sfSumBytes22 = sums22 * y.sfBytes
		y.sfCoreBytes1 = y.wfCores1 * y.sfSumBytes11
		y.sfCoreBytes2 = y.wfCores1 * y.sfSumBytes21
		if y.wfCores1 < y.wfCores2 {
			y.sfCoreBytes1 += y.sfSumBytes12
			y.sfCoreBytes2 += y.sfSumBytes22
		}
		y.sfFragBytes = y.dfCores1 * y.sfCoreBytes1
		if y.dfCores1 < y.dfCores2 {
			y.sfFragBytes += y.sfCoreBytes2
		}
		y.sfGroupBytes = y.blkFrags * y.sfFragBytes
		y.sfTotalBytes = spec.Groups * y.sfGroupBytes
		return layer6()
	}
	layer4 := func() *layout {
		y.segs = newSegments(ctx, spec, y.dfSliceDfs1)
		var (
			lh = y.segs.lhs[len(y.segs.lhs)-1]
			lw = lh.lws[len(lh.lws)-1]
		)
		y.dfSliceDfs2 = len(lw.blks)
		if y.dfSliceDfs2 == y.dfSliceDfs1 {
			y.dfSliceDfs2 = 0
		}
		y.dfSliceBytes1 = y.dfSliceDfs1 * y.dfBytes
		y.dfSliceBytes2 = y.dfSliceDfs2 * y.dfBytes
		y.dfCores1 = y.segs.cnt - btoi(y.dfSliceDfs2 > 0)
		y.dfCores2 = y.segs.cnt
		return layer5()
	}
	layer3 := func() *layout {
		y.wfSliceWfs2 = y.toChans % y.wfSliceWfs1
		y.wfSliceBytes1 = y.wfSliceWfs1 * y.wfBytes
		y.wfSliceBytes2 = y.wfSliceWfs2 * y.wfBytes
		y.wfCores1 = y.toChans / y.wfSliceWfs1
		y.wfCores2 = y.wfCores1 + btoi(y.wfSliceWfs2 > 0)
		return layer4()
	}
	layer2 := func() *layout {
		if len(spec.Filts) > 1 && spec.Groups > 1 {
			panic("bug")
		}
		filts := 0
		for i := range spec.Filts {
			filts += spec.Filts[i].Cnt
		}
		y.fromChans = spec.From.Chans / spec.Groups
		y.toChans = filts / spec.Groups
		return layer3()
	}
	layer1 := func() *layout {
		switch ctx.platform {
		case raw.AVX512Float32:
			y.blkFrags = 4
			y.alignment = 64
			y.biasBytes = 4
			y.bfBytes = 4
			y.wtBytes = 4
			y.wfBytes = 32
			y.wfSliceWfs1 = 4
			y.datBytes = 4
			y.dfBytes = 64
			y.dfSliceDfs1 = 6
			y.sfBytes = 64
		default:
			panic("bug")
		}
		return layer2()
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
	fragBytes   int
	groupBytes  int
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
	coreIdx     cgen.Gen
	coreCut     cgen.Gen
}

func (a *arrangeFilts) Append(to []byte) []byte {
	var (
		threadBlks   int
		groupBundles int
	)
	switch a.platform {
	case raw.AVX512Float32:
		a.bundleFilts = 4
		threadBlks = 512
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
		if hull == 1 {
			expr = il(0)
		} else {
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
		body[4] = void(callee.Pt)
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
		a.fragBytes = a.wfFragBytes1
		a.groupBytes = a.wfGroupBytes1
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
		a.fragBytes = a.wfFragBytes2
		a.groupBytes = a.wfGroupBytes2
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
	ptrDecl := func(ptr, expr cgen.Gen) cgen.Gen {
		return cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptr, Init: expr,
		}
	}
	ptrDecls := func(filtsIdx int) cgen.Gen {
		wtDecl := func() cgen.Gen {
			a.wtPtr = vb(a.name("wtPtr"))
			return ptrDecl(
				a.wtPtr, addMul(
					tensor(filtsIdx, 0),
					il(a.slices1*3*3*a.wtBytes),
					a.epochCoord,
				),
			)
		}
		biasDecl := func() cgen.Gen {
			if a.epochFirst == 0 {
				a.biasPtr = vb(a.name("biasPtr"))
				return ptrDecl(
					a.biasPtr,
					tensor(filtsIdx, 1),
				)
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
				ret[x] = ptrDecl(bnPtr, expr)
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
	idxDecl := func(idx, expr cgen.Gen) cgen.Gen {
		return cgen.Var{
			Type: cgen.PtrdiffT,
			What: idx, Init: expr,
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
			return cgen.Stmts{
				ptrDecls(x),
				a.kernel2(),
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
				upper = start + (stop-start)/2
				x     = first + 1
			)
			for atBundle[x+1] <= upper {
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
		stmts[0] = idxDecl(
			a.bundleIdx, cgen.Mul{
				Expr1: il(a.bundleTile),
				Expr2: a.bundleCoord,
			},
		)
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
			stmts[1] = idxDecl(
				a.bundleLast, cgen.Add{
					Expr1: a.bundleIdx,
					Expr2: expr,
				},
			)
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
		stmts[0] = idxDecl(
			a.groupIdx, cgen.Mul{
				Expr1: il(a.groupTile),
				Expr2: a.groupCoord,
			},
		)
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
			if iters == 0 {
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
			} else {
				expr = il(iters - 1)
			}
			stmts[1] = idxDecl(
				last, cgen.Add{
					Expr1: a.groupIdx,
					Expr2: expr,
				},
			)
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
			ptrDecl(
				a.bfPtr, addMul(
					tensor(n, 0),
					il(a.bfEpochBytes),
					a.epochCoord,
				),
			),
			ptrDecl(
				a.wfPtr, addMul(
					cgen.Add{
						Expr1: tensor(n, 0),
						Expr2: il(a.bfTotalBytes),
					},
					il(a.wfEpochBytes1),
					a.epochCoord,
				),
			),
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
		a.coreIdx = vb(a.name("k"))
		a.coreCut = vb(a.name("cut"))
		var (
			idx1 = a.baseFilt / a.wfSliceWfs1
			cut1 = a.baseFilt % a.wfSliceWfs1
			idx2 cgen.Gen
			cut2 cgen.Gen
		)
		switch a.bundleFilts % a.wfSliceWfs1 {
		case 0:
			ratio := a.bundleFilts / a.wfSliceWfs1
			idx2 = addMul(
				il(idx1-a.baseBundle*ratio),
				il(ratio),
				a.bundleIdx,
			)
			cut2 = il(cut1)
		default:
			var (
				off = addMul(
					il(cut1-a.baseBundle*a.bundleFilts),
					il(a.bundleFilts),
					a.bundleIdx,
				)
				numer cgen.Gen = cgen.Cast{
					Type: cgen.SizeT,
					Expr: cgen.Paren{Inner: off},
				}
				denom = il(a.wfSliceWfs1)
			)
			idx2 = cgen.Add{
				Expr1: il(idx1),
				Expr2: cgen.Quo{
					Expr1: numer,
					Expr2: denom,
				},
			}
			cut2 = cgen.Rem{
				Expr1: numer,
				Expr2: denom,
			}
		}
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: a.coreIdx,
				Init: idx2,
			},
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: a.coreCut,
				Init: cut2,
			},
			a.kernel3(),
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
			rem1  = filts1 % a.bundleFilts
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
		var (
			quo2 = tail / a.bundleFilts
			rem2 = tail % a.bundleFilts
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
			split  = a.toChans - a.wfSliceWfs2
			clamp1 = max(past-split, 0)
			clamp2 = min(clamp1, filts2)
		)
		filts1 = filts2 - clamp2
		return layer2()
	}
	return layer1()
}

func (a *arrangeFilts) kernel3() cgen.Gen {
	switch a.platform {
	case raw.AVX512Float32:
		return a.m512()
	default:
		panic("bug")
	}
}

func (a *arrangeFilts) m512() cgen.Gen {
	var (
		bfs       []cgen.Gen
		preCnt    int
		postMuls  []cgen.Gen
		stepDepth int
		stepIdx   cgen.Gen
		depth     int
		wts       []cgen.Gen
		ins       [3]cgen.Gen
		outs      []cgen.Gen
		offs      []cgen.Gen
	)
	layer12 := func() cgen.Gen {
		var (
			n     = len(outs)
			stmts = make(cgen.Stmts, 2*n)
			x     = 0
		)
		for r := 0; r < a.blkFrags; r++ {
			for f := 0; f < a.filts2; f++ {
				var (
					off        = offs[f]
					sliceBytes = a.wfSliceBytes1
				)
				if f >= a.filts1 {
					sliceBytes = a.wfSliceBytes2
				}
				for d := 0; d < depth; d++ {
					var (
						ae        = a.wfPtr
						stepPitch = stepDepth * sliceBytes
						wf        = vb(a.name("wf"))
						blk       = f*depth + d
						out       = outs[blk*a.blkFrags+r]
						mask      = loMask(a.wfBytes / 4)
					)
					ae = cgen.Add{
						Expr1: ae,
						Expr2: il(r*a.fragBytes + d*sliceBytes),
					}
					ae = addMul(ae, il(a.groupBytes), a.groupIdx)
					ae = addMul(ae, il(a.coreBytes), a.coreIdx)
					ae = cgen.Add{
						Expr1: ae, Expr2: off,
					}
					ae = addMul(ae, il(stepPitch), stepIdx)
					stmts[x] = cgen.Var{
						Type: avx.M512i, What: wf,
						Init: avx.Mm512Castsi256Si512{
							avx.Mm512CvtpsPh{
								out,
								avx.FroundToNearestIntNoExc,
							},
						},
					}
					stmts[x+n] = avx.Mm512MaskStoreuEpi32{
						ae, mask, wf,
					}
					x++
				}
			}
		}
		return stmts
	}
	layer11 := func() cgen.Gen {
		offs = make([]cgen.Gen, a.filts2)
		for x := range offs {
			offs[x] = vb(a.name("off"))
		}
		var (
			last  = len(offs)
			stmts = make(cgen.Stmts, last+1)
		)
		for x, off := range offs {
			var expr cgen.Gen
			switch x {
			case 0:
				expr = cgen.Mul{
					Expr1: il(a.wfBytes),
					Expr2: a.coreCut,
				}
			default:
				var (
					numer cgen.Gen = cgen.Cast{
						Type: cgen.SizeT,
						Expr: cgen.Paren{
							Inner: cgen.Add{
								Expr1: a.coreCut,
								Expr2: il(x),
							},
						},
					}
					denom = il(a.wfSliceWfs1)
				)
				expr = cgen.Add{
					Expr1: cgen.Mul{
						Expr1: cgen.Quo{
							Expr1: numer,
							Expr2: denom,
						},
						Expr2: il(a.coreBytes),
					},
					Expr2: cgen.Mul{
						Expr1: cgen.Rem{
							Expr1: numer,
							Expr2: denom,
						},
						Expr2: il(a.wfBytes),
					},
				}
			}
			stmts[x] = cgen.Var{
				Type: cgen.PtrdiffT, What: off,
				Init: expr,
			}
		}
		stmts[last] = layer12()
		return stmts
	}
	layer10 := func() cgen.Gen {
		op := &wct.Wts{
			Platform: a.platform,
			Nms:      a.nms,
			Blks:     len(wts),
			In:       ins,
		}
		outs = op.Out[:op.Blks*a.blkFrags]
		for x := range outs {
			outs[x] = vb(a.name("out"))
		}
		return cgen.Gens{
			op,
			layer11(),
		}
	}
	layer9 := func() cgen.Gen {
		var stmts cgen.Stmts
		decl := func(t, id, expr cgen.Gen) {
			stmts = append(
				stmts, cgen.Var{
					Type: t, What: id,
					Init: expr,
				},
			)
		}
		yield := func(x int, expr cgen.Gen) {
			in := vb(a.name("in"))
			ins[x] = in
			decl(avx.M512, in, expr)
		}
		rot := func(id cgen.Gen, cnt int) cgen.Gen {
			return avx.Mm512Castsi512Ps{
				avx.Mm512AlignrEpi32{
					id, id, il(cnt),
				},
			}
		}
		ctrl := func(at int) cgen.Gen {
			var (
				pm  = vb(a.name("pm"))
				set = make(avx.Mm512SetEpi32, 16)
			)
			for x := 0; x < 16; x++ {
				y := at + x/6*3 + x%3 + x/3%2*16
				set[15-x] = il(y % 32)
			}
			decl(avx.M512i, pm, set)
			return pm
		}
		perm := func(lo, pm, hi cgen.Gen) cgen.Gen {
			return avx.Mm512Permutex2varPs{
				lo, pm, hi,
			}
		}
		temp := func(expr cgen.Gen) cgen.Gen {
			tmp := vb(a.name("tmp"))
			decl(avx.M512, tmp, expr)
			return tmp
		}
		switch n := len(wts); n {
		case 1:
			var (
				wt   = wts[0]
				via  = vb(a.name("via"))
				expr = avx.Mm512CastpsSi512{wt}
			)
			decl(avx.M512i, via, expr)
			yield(0, wt)
			yield(1, rot(via, 3))
			yield(2, rot(via, 6))
		case 2:
			var (
				wt1 = wts[0]
				wt2 = wts[1]
				pm1 = ctrl(0)
				pm2 = ctrl(3)
				pm3 = ctrl(6)
			)
			yield(0, perm(wt1, pm1, wt2))
			yield(1, perm(wt1, pm2, wt2))
			yield(2, perm(wt1, pm3, wt2))
		default:
			var (
				wt1 = wts[0]
				wt2 = wts[1]
				wt3 = wts[2]
				wt4 = wt2
			)
			if n == 4 {
				wt4 = wts[3]
			}
			var (
				pm1  = ctrl(0)
				pm2  = ctrl(6)
				tmp1 = temp(perm(wt1, pm1, wt3))
				tmp2 = temp(perm(wt2, pm1, wt4))
				tmp3 = temp(perm(wt1, pm2, wt3))
				tmp4 = temp(perm(wt2, pm2, wt4))
			)
			yield(0, perm(tmp1, pm1, tmp2))
			yield(1, perm(tmp1, pm2, tmp2))
			yield(2, perm(tmp3, pm1, tmp4))
		}
		return cgen.Gens{
			stmts,
			layer10(),
		}
	}
	layer8 := func() cgen.Gen {
		if preCnt == 0 {
			return layer9()
		}
		var (
			preMuls = make([]cgen.Gen, depth)
			preAdds = make([]cgen.Gen, depth)
		)
		sublayer2 := func() cgen.Gen {
			var (
				n     = len(wts)
				stmts = make(cgen.Stmts, 2*n)
			)
			for d := 0; d < depth; d++ {
				var (
					preMul = preMuls[d]
					preAdd = preAdds[d]
				)
				for f := 0; f < a.filts2; f++ {
					var (
						bf = bfs[f]
						wt = wts[f*depth+d]
						x  = d*a.filts2 + f
					)
					stmts[x] = cgen.Assign{
						Expr1: bf,
						Expr2: avx.Mm512FmaddPs{
							wt, preAdd, bf,
						},
					}
					stmts[x+n] = cgen.Assign{
						Expr1: wt,
						Expr2: avx.Mm512MulPs{
							wt, preMul,
						},
					}
				}
			}
			return stmts
		}
		sublayer1 := func() cgen.Gen {
			toMix := make([]cgen.Stmts, depth)
			for d := range toMix {
				stmts := make(cgen.Stmts, preCnt*3)
				preCh := cgen.Paren{
					Inner: addMul(
						addMul(
							il(d),
							il(a.fromChans),
							a.groupIdx,
						),
						il(stepDepth),
						stepIdx,
					),
				}
				for x, prePtr := range a.bnPtrs[:preCnt] {
					var (
						preMul = vb(a.name("preMul"))
						preAdd = vb(a.name("preAdd"))
					)
					stmts[x*3] = &bn.Load{
						Ctx:     a.bc,
						Mas:     prePtr,
						Channel: preCh,
						Mul:     preMul,
						Add:     preAdd,
					}
					if x == 0 {
						preMuls[d] = preMul
						preAdds[d] = preAdd
						continue
					}
					stmts[x*3+1] = cgen.Assign{
						Expr1: preMuls[d],
						Expr2: avx.Mm512MulPs{
							preMuls[d], preMul,
						},
					}
					stmts[x*3+2] = cgen.Assign{
						Expr1: preAdds[d],
						Expr2: avx.Mm512FmaddPs{
							preAdds[d], preMul,
							preAdd,
						},
					}
				}
				toMix[d] = stmts
			}
			return cgen.Gens{
				mix(toMix),
				sublayer2(),
			}
		}
		return cgen.Gens{
			sublayer1(),
			layer9(),
		}
	}
	layer7 := func() cgen.Gen {
		if postMuls == nil {
			return layer8()
		}
		var (
			last  = len(wts)
			stmts = make(cgen.Stmts, last+1)
		)
		for f := 0; f < a.filts2; f++ {
			postMul := postMuls[f]
			for d := 0; d < depth; d++ {
				x := f*depth + d
				stmts[x] = cgen.Assign{
					Expr1: wts[x],
					Expr2: avx.Mm512MulPs{
						wts[x], postMul,
					},
				}
			}
		}
		stmts[last] = layer8()
		return stmts
	}
	layer6 := func() cgen.Gen {
		var (
			wtCnt = a.filts2 * depth
			stmts = make(cgen.Stmts, wtCnt+1)
		)
		wts = make([]cgen.Gen, wtCnt)
		for f := 0; f < a.filts2; f++ {
			for d := 0; d < depth; d++ {
				var (
					x           = f*depth + d
					wt          = vb(a.name("wt"))
					ae          = a.wtPtr
					spatial     = 3 * 3
					depthPitch  = spatial * a.wtBytes
					filtPitch   = a.fromChans * depthPitch
					groupPitch  = a.toChans * filtPitch
					bundlePitch = a.bundleFilts * filtPitch
					stepPitch   = stepDepth * depthPitch
					mask        = loMask(spatial)
				)
				ae = cgen.Add{
					Expr1: ae,
					Expr2: il(
						f*filtPitch + d*depthPitch -
							a.baseBundle*bundlePitch,
					),
				}
				ae = addMul(ae, il(groupPitch), a.groupIdx)
				ae = addMul(ae, il(bundlePitch), a.bundleIdx)
				ae = addMul(ae, il(stepPitch), stepIdx)
				wts[x] = wt
				stmts[x] = cgen.Var{
					Type: avx.M512, What: wt,
					Init: avx.Mm512MaskzLoaduPs{
						mask, ae,
					},
				}
			}
		}
		stmts[wtCnt] = layer7()
		return stmts
	}
	layer5 := func() cgen.Gen {
		const wctBlks = 4
		stepDepth = wctBlks / a.filts2
		stepIdx = vb(a.name("s"))
		var (
			stmts = make(cgen.Stmts, 3)
			quo   = a.slices / stepDepth
			rem   = a.slices % stepDepth
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: stepIdx,
			Init: il(0),
		}
		if quo > 0 {
			depth = stepDepth
			stmts[1] = cgen.For{
				Cond: cgen.CmpNE{
					Expr1: stepIdx,
					Expr2: il(quo),
				},
				Post: cgen.IncPre{
					Expr: stepIdx,
				},
				Body: layer6(),
			}
		}
		if rem > 0 {
			depth = rem
			stmts[2] = layer6()
		}
		return stmts
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
			var merge cgen.Gen
			switch preCnt {
			case 0:
				bfs[0] = bias
			default:
				merge = cgen.Assign{
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
						merge,
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
			return cgen.Gens{
				&sumr.Pack{
					Platform: a.platform,
					Nms:      a.nms,
					Vars:     bfs,
				},
				sublayer2(),
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
			groupPitch  = a.toChans * a.bfBytes
			bundlePitch = a.bundleFilts * a.bfBytes
			mask        = loMask(a.filts2)
			bf          = bfs[0]
		)
		ae = cgen.Sub{
			Expr1: ae,
			Expr2: il(
				a.baseBundle*bundlePitch -
					a.baseFilt*a.bfBytes,
			),
		}
		ae = addMul(ae, il(groupPitch), a.groupIdx)
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
	fragBytes   int
	groupBytes  int
	ptrs        []cgen.Gen
	groupIdx    cgen.Gen
	coreIdx     cgen.Gen
	coreH       cgen.Gen
	coreW       cgen.Gen
	blks        []*block
}

func (a *arrangeDats) Append(to []byte) []byte {
	var threadBlks int
	switch a.platform {
	case raw.AVX512Float32:
		threadBlks = 512
	default:
		panic("bug")
	}
	var (
		n1         = a.dfSliceDfs1 * a.dfCores1
		n2         = a.dfSliceDfs2 * (a.dfCores2 - a.dfCores1)
		chanBlks   = n1 + n2
		sliceBlks  = ceilQuo(chanBlks, a.dfCores2)
		coreSlices = ceilQuo(a.fromChans, a.epochs2)
		coreBlks   = coreSlices * sliceBlks
		groupBlks  = coreSlices * chanBlks
	)
	switch {
	case threadBlks <= coreBlks:
		minSlices := a.slices1
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
		a.groupTile = 1
		a.groupTiles = a.Groups
		a.groupScrap = 0
		a.groupHull = a.Groups
	case threadBlks <= groupBlks:
		a.sliceTile1 = a.slices1
		a.sliceTile2 = a.slices2
		a.sliceTiles = 1
		a.sliceScrap1 = 0
		a.sliceScrap2 = 0
		a.sliceHull = 1
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
		a.groupTile = 1
		a.groupTiles = a.Groups
		a.groupScrap = 0
		a.groupHull = a.Groups
	default:
		a.sliceTile1 = a.slices1
		a.sliceTile2 = a.slices2
		a.sliceTiles = 1
		a.sliceScrap1 = 0
		a.sliceScrap2 = 0
		a.sliceHull = 1
		a.coreTile = a.dfCores2
		a.coreTiles = 1
		a.coreScrap = 0
		a.coreHull = 1
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
		if hull == 1 {
			expr = il(0)
		} else {
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
		body[5] = void(callee.Pt)
	}
	impl := func(first, cnt int) cgen.Gen {
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
		a.fragBytes = a.dfFragBytes1
		a.groupBytes = a.dfGroupBytes1
		put := impl(0, a.epochs1)
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
		a.fragBytes = a.dfFragBytes2
		a.groupBytes = a.dfGroupBytes2
		body[7] = impl(a.epochs1, 1)
	}
	return callee.Func(body)
}

func (a *arrangeDats) kernel1() cgen.Gen {
	a.ptrs = a.ptrs[:0]
	var (
		stmts     cgen.Stmts
		tensorIdx = 0
		datPtrIdx = 0
	)
	decl := func(ptr, expr cgen.Gen) {
		a.ptrs = append(a.ptrs, ptr)
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
		i := datPtrIdx
		datPtrIdx++
		var (
			pitch1 = a.From.Pitch1Bytes[i]
			pitch2 = a.From.Pitch2Bytes[i]
		)
		decl(
			vb(a.name("datPtr")),
			addMul(
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
		decl(
			vb(a.name("bnPtr")),
			&bn.Offset{
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
	decl(
		vb(a.name("dfPtr")),
		addMul(
			tensor(),
			il(a.dfEpochBytes1),
			a.epochCoord,
		),
	)
	return cgen.Gens{
		stmts,
		a.kernel2(),
	}
}

func (a *arrangeDats) kernel2() cgen.Gen {
	var (
		coreLast cgen.Gen
		lh       *loopH
		rel      cgen.Gen
		base     cgen.Gen
		relBreak int
		lw       *loopW
	)
	layer9 := func() cgen.Gen {
		var retIf cgen.Gen
		if coreLast != nil {
			retIf = cgen.If1{
				Cond: cgen.CmpGE{
					Expr1: a.coreIdx,
					Expr2: coreLast,
				},
				Then: cgen.Return{},
			}
		}
		return cgen.Stmts{
			a.kernel3(),
			retIf,
			cgen.IncPre{
				Expr: a.coreIdx,
			},
		}
	}
	layer8 := func() cgen.Gen {
		if lw.fromStep == 0 {
			return layer9()
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
				Body: layer9(),
			},
		}
	}
	layer7 := func() cgen.Gen {
		a.coreH = vb(a.name("h"))
		a.coreW = vb(a.name("w"))
		a.blks = lw.blks
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
			layer8(),
			breakIf,
		}
	}
	layer6 := func() cgen.Gen {
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
				layer7(),
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
	layer5 := func() cgen.Gen {
		if lh.segStep == 0 {
			relBreak = -1
			return layer6()
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
			Body: layer6(),
		}
	}
	layer4 := func() cgen.Gen {
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
			layer5(),
		}
	}
	layer3 := func() cgen.Gen {
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
				layer4(),
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
	layer2 := func() cgen.Gen {
		a.coreIdx = vb(a.name("j"))
		switch a.coreHull {
		case 1:
			coreLast = nil
		default:
			coreLast = vb(a.name("last"))
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
		if coreLast != nil {
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
				What: coreLast,
				Init: cgen.Add{
					Expr1: a.coreIdx,
					Expr2: expr,
				},
			}
		}
		stmts[2] = layer3()
		return stmts
	}
	layer1 := func() cgen.Gen {
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
			stmts[2] = layer2()
		default:
			var (
				last = vb(a.name("ii"))
				expr cgen.Gen
			)
			if iters == 0 {
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
			} else {
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
				Body: layer2(),
			}
		}
		return stmts
	}
	return layer1()
}

func (a *arrangeDats) kernel3() cgen.Gen {
	switch a.platform {
	case raw.AVX512Float32:
		return a.m512()
	default:
		panic("bug")
	}
}

func (a *arrangeDats) m512() cgen.Gen {
	type Sig struct {
		dual int
		pads [2]int
		offs [2]int
		cnts [2]int
	}
	var (
		outerIdx  cgen.Gen
		outerCnt  int
		innerIdx  cgen.Gen
		innerMul  int
		innerCnt  int
		bnMuls    [][]cgen.Gen
		bnAdds    [][]cgen.Gen
		batchBlks []*block
		batchQuos []int
		batchRems []int
		Wct       *wct.Dats
		spanDats  []cgen.Gen
		spanSrcs  []int
		spanOffs  []int
		pms       []cgen.Gen
		sigs      []Sig
	)
	bnPrep := func(x int) cgen.Gen {
		if bnMuls == nil {
			bnMuls = make([][]cgen.Gen, innerCnt)
			bnAdds = make([][]cgen.Gen, innerCnt)
		}
		if bnMuls[x] != nil {
			return nil
		}
		var (
			loads  cgen.Gens
			ptrIdx = 1
			ch     = il(x)
		)
		ch = addMul(ch, il(a.fromChans), a.groupIdx)
		ch = addMul(ch, il(a.sliceTile), outerIdx)
		ch = addMul(ch, il(innerMul), innerIdx)
		ch = cgen.Paren{
			Inner: ch,
		}
		for op := range a.From.Ops {
			op := &a.From.Ops[op]
			switch op.Kind {
			case mod.Add:
				ptrIdx += op.Int
			case mod.Bn:
				var (
					bnMul = vb(a.name("bnMul"))
					bnAdd = vb(a.name("bnAdd"))
				)
				bnMuls[x] = append(bnMuls[x], bnMul)
				bnAdds[x] = append(bnAdds[x], bnAdd)
				loads = append(loads, &bn.Load{
					Ctx:     a.bc,
					Mas:     a.ptrs[ptrIdx],
					Channel: ch,
					Mul:     bnMul,
					Add:     bnAdd,
				})
				ptrIdx++
			case mod.ReLU:
			default:
				panic("bug")
			}
		}
		return loads
	}
	spanDat := func(x, h, w, n int) cgen.Gen {
		dat := vb(a.name("dat"))
		spanDats = append(spanDats, dat)
		var (
			stmts    cgen.Stmts
			pitchIdx = 0
			bnIdx    = 0
			mask     = loMask(n)
		)
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		load := func() cgen.Gen {
			var (
				ae     = a.ptrs[pitchIdx+bnIdx]
				pitch1 = a.From.Pitch1Bytes[pitchIdx]
				pitch2 = a.From.Pitch2Bytes[pitchIdx]
			)
			pitchIdx++
			var (
				groupPitch = a.fromChans * pitch2
				outerPitch = a.sliceTile * pitch2
				innerPitch = innerMul * pitch2
			)
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(
					x*pitch2 + h*pitch1 + w*a.datBytes,
				),
			}
			ae = addMul(ae, il(groupPitch), a.groupIdx)
			ae = addMul(ae, il(pitch1), a.coreH)
			ae = addMul(ae, il(a.datBytes), a.coreW)
			ae = addMul(ae, il(outerPitch), outerIdx)
			ae = addMul(ae, il(innerPitch), innerIdx)
			return avx.Mm512MaskzLoaduPs{
				mask, ae,
			}
		}
		stmt(cgen.Var{
			Type: avx.M512, What: dat,
			Init: load(),
		})
		for op := range a.From.Ops {
			op := &a.From.Ops[op]
			switch op.Kind {
			case mod.Add:
				more := op.Int
				for ; more > 0; more-- {
					stmt(cgen.Assign{
						Expr1: dat,
						Expr2: avx.Mm512AddPs{
							dat, load(),
						},
					})
				}
			case mod.Bn:
				stmt(bnPrep(x))
				stmt(&bn.Apply{
					Ctx:  a.bc,
					Mul:  bnMuls[x][bnIdx],
					Add:  bnAdds[x][bnIdx],
					To:   dat,
					Mask: mask,
				})
				bnIdx++
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
		return stmts
	}
	ctrl := func(sig Sig) (pm, decl cgen.Gen) {
		for x := range sigs {
			if sigs[x] == sig {
				pm = pms[x]
				return
			}
		}
		pm = vb(a.name("pm"))
		pms = append(pms, pm)
		sigs = append(sigs, sig)
		var (
			set  = make(avx.Mm512SetEpi32, 16)
			zero cgen.Gen
		)
		for x := 0; x < 16; x++ {
			var (
				q    = x / 8
				r    = x % 8
				pad  = sig.pads[q]
				elem cgen.Gen
			)
			switch {
			case r < pad:
				fallthrough
			case r >= pad+sig.cnts[q]:
				if zero == nil {
					zero = il(15)
				}
				elem = zero
			default:
				elem = il(
					r - pad +
						sig.offs[q] +
						sig.dual*q*16,
				)
			}
			set[15-x] = elem
		}
		decl = cgen.Var{
			Type: avx.M512i, What: pm,
			Init: set,
		}
		return
	}
	arrange := func(h int) cgen.Gen {
		stmts := make(cgen.Stmts, 0, 4)
		for x1 := 0; x1 < Wct.Blks; x1 += 2 {
			in := Wct.In[x1/2*8+h]
			if in == nil {
				continue
			}
			var (
				blk1 = batchBlks[x1]
				src1 = spanSrcs[x1]
				src2 int
			)
			sig := Sig{
				pads: [2]int{blk1.padW, 8},
				offs: [2]int{spanOffs[x1], 0},
				cnts: [2]int{blk1.datW, 0},
			}
			if x2 := x1 + 1; x2 < Wct.Blks {
				src2 = spanSrcs[x2]
				if src1 != src2 &&
					src1 != -1 &&
					src2 != -1 {
					sig.dual = 1
				}
				if src1 == -1 {
					sig.pads[0] = 8
					sig.offs[0] = 0
					sig.cnts[0] = 0
				}
				if src2 != -1 {
					blk2 := batchBlks[x2]
					sig.pads[1] = blk2.padW
					sig.offs[1] = spanOffs[x2]
					sig.cnts[1] = blk2.datW
				}
			}
			pm, decl := ctrl(sig)
			var perm cgen.Gen
			switch sig.dual {
			case 0:
				src := src1
				if src == -1 {
					src = src2
				}
				perm = avx.Mm512PermutexvarPs{
					pm, spanDats[src],
				}
			default:
				perm = avx.Mm512Permutex2varPs{
					spanDats[src1], pm,
					spanDats[src2],
				}
			}
			stmts = append(stmts,
				decl,
				cgen.Var{
					Type: avx.M512, What: in,
					Init: perm,
				},
			)
		}
		return stmts
	}
	layer7 := func() cgen.Gen {
		var (
			n     = Wct.Blks * a.blkFrags
			stmts = make(cgen.Stmts, 1, 1+n)
		)
		stmts[0] = Wct
		for f := 0; f < a.blkFrags; f++ {
			for x1 := 0; x1 < 2; x1++ {
				for x := x1; x < Wct.Blks; x += 2 {
					var (
						ae         = a.ptrs[len(a.ptrs)-1]
						slicePitch = len(a.blks) * a.dfBytes
						outerPitch = a.sliceTile * slicePitch
						innerPitch = innerMul * slicePitch
						out        = Wct.Out[x*a.blkFrags+f]
					)
					ae = cgen.Add{
						Expr1: ae,
						Expr2: il(
							f*a.fragBytes +
								batchQuos[x]*slicePitch +
								batchRems[x]*a.dfBytes,
						),
					}
					ae = addMul(ae, il(a.groupBytes), a.groupIdx)
					ae = addMul(ae, il(a.coreBytes), a.coreIdx)
					ae = addMul(ae, il(outerPitch), outerIdx)
					ae = addMul(ae, il(innerPitch), innerIdx)
					stmts = append(
						stmts, avx.Mm512StoreuPs{
							ae, out,
						},
					)
				}
			}
		}
		return stmts
	}
	layer6 := func() cgen.Gen {
		var ret cgen.Gens
		for h := 0; h < 8; h++ {
			if Wct.In[h] == nil &&
				Wct.In[h+8] == nil {
				continue
			}
			srcOff := func(src, off int) {
				spanSrcs = append(spanSrcs, src)
				spanOffs = append(spanOffs, off)
			}
			for x1 := 0; x1 < Wct.Blks; {
				blk1 := batchBlks[x1]
				if h < blk1.padH ||
					blk1.padH+blk1.datH <= h {
					srcOff(-1, -1)
					x1++
					continue
				}
				srcOff(len(spanDats), 0)
				var (
					w  = blk1.fromW
					n  = blk1.datW
					x2 = x1 + 1
				)
				for ; x2 < Wct.Blks; x2++ {
					if batchQuos[x2] != batchQuos[x1] {
						break
					}
					blk2 := batchBlks[x2]
					if blk2.fromH != blk1.fromH ||
						blk2.fromW != w+6 {
						break
					}
					if blk2.datW == 0 {
						break
					}
					more := max(blk2.padW+blk2.datW-2, 0)
					if n+more > 15 {
						break
					}
					w = blk2.fromW
					n += more
					srcOff(len(spanDats), n-blk2.datW)
				}
				ret = append(ret, spanDat(
					batchQuos[x1],
					blk1.fromH+h,
					blk1.fromW+blk1.padW,
					n,
				))
				x1 = x2
			}
			ret = append(ret, arrange(h))
			spanDats = spanDats[:0]
			spanSrcs = spanSrcs[:0]
			spanOffs = spanOffs[:0]
		}
		pms = pms[:0]
		sigs = sigs[:0]
		ret = append(ret, layer7())
		return ret
	}
	layer5 := func() cgen.Gen {
		Wct = &wct.Dats{
			Platform: a.platform,
			Nms:      a.nms,
			Blks:     len(batchBlks),
		}
		for x, blk := range batchBlks {
			var (
				first = blk.padW
				past  = first + blk.datW
			)
			Wct.LZCols[x] = first
			Wct.TZCols[x] = 8 - past
		}
		for x1 := 0; x1 < Wct.Blks; x1 += 2 {
			var (
				blk1 = batchBlks[x1]
				blk2 = blk1
			)
			if x2 := x1 + 1; x2 < Wct.Blks {
				blk2 = batchBlks[x2]
			}
			for h := 0; h < 8; h++ {
				switch {
				case blk1.padH <= h &&
					h < blk1.padH+blk1.datH:
				case blk2.padH <= h &&
					h < blk2.padH+blk2.datH:
				default:
					continue
				}
				in := vb(a.name("in"))
				Wct.In[x1/2*8+h] = in
			}
		}
		for x := range batchBlks {
			for f := 0; f < a.blkFrags; f++ {
				out := vb(a.name("out"))
				Wct.Out[x*a.blkFrags+f] = out
			}
		}
		return layer6()
	}
	layer4 := func() cgen.Gen {
		const n1 = 4
		var (
			n2  = len(a.blks)
			n3  = innerCnt * n2
			n4  = ceilQuo(n3, n1)
			ret = make(cgen.Gens, 0, n4)
		)
		flush := func() {
			if len(batchBlks) > 0 {
				ret = append(ret, layer5())
				batchBlks = batchBlks[:0]
				batchQuos = batchQuos[:0]
				batchRems = batchRems[:0]
			}
		}
		for x := 0; x < n3; x++ {
			var (
				quo = x / n2
				rem = x % n2
				blk = a.blks[rem]
			)
			batchBlks = append(batchBlks, blk)
			batchQuos = append(batchQuos, quo)
			batchRems = append(batchRems, rem)
			if len(batchBlks) == n1 {
				flush()
			}
		}
		flush()
		return ret
	}
	layer3 := func() cgen.Gen {
		bnMuls = nil
		bnAdds = nil
		return layer4()
	}
	layer2 := func() cgen.Gen {
		innerIdx = vb(a.name("k"))
		innerMul = 1
		for innerMul*len(a.blks)%4 != 0 {
			innerMul *= 2
		}
		var (
			stmts = make(cgen.Stmts, 3)
			quo   = outerCnt / innerMul
			rem   = outerCnt % innerMul
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: innerIdx,
			Init: il(0),
		}
		if quo > 0 {
			innerCnt = innerMul
			stmts[1] = cgen.For{
				Cond: cgen.CmpNE{
					Expr1: innerIdx,
					Expr2: il(quo),
				},
				Post: cgen.IncPre{
					Expr: innerIdx,
				},
				Body: layer3(),
			}
		}
		if rem > 0 {
			innerCnt = rem
			stmts[2] = layer3()
		}
		return stmts
	}
	layer1 := func() cgen.Gen {
		slices := 0
		switch {
		case a.sliceTile == a.sliceScrap:
			fallthrough
		case a.sliceTiles == a.sliceHull:
			slices = a.sliceTile
		case a.sliceTiles == 0:
			slices = a.sliceScrap
		}
		if slices != 0 {
			outerIdx = a.sliceCoord
			outerCnt = slices
			return layer2()
		}
		impl := func(x, n int) cgen.Stmts {
			var decl cgen.Gen
			switch n {
			case 1:
				outerIdx = vb(a.name("ss"))
				decl = cgen.Var{
					Type: cgen.PtrdiffT,
					What: outerIdx,
					Init: il(x),
				}
			default:
				outerIdx = a.sliceCoord
			}
			return cgen.Stmts{
				decl,
				layer2(),
			}
		}
		outerCnt = a.sliceTile
		tiles := impl(0, a.sliceTiles)
		outerCnt = a.sliceScrap
		scrap := impl(a.sliceTiles, 1)
		return cgen.If{
			Cond: cgen.CmpL{
				Expr1: a.sliceCoord,
				Expr2: il(a.sliceTiles),
			},
			Then: tiles,
			Else: scrap,
		}
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
	wfFragBytes  int
	wfGroupBytes int
	dfCoreBytes  int
	dfFragBytes  int
	dfGroupBytes int
	wfTile       int
	wfTiles      int
	wfScrap      int
	wfHull       int
	dfTile       int
	dfTiles      int
	dfScrap      int
	dfHull       int
	fragTile     int
	fragTiles    int
	fragScrap    int
	fragHull     int
	groupTile    int
	groupTiles   int
	groupScrap   int
	groupHull    int
	calleeName   string
	tensors      cgen.Gen
	epochCoord   cgen.Gen
	groupCoord   cgen.Gen
	fragCoord    cgen.Gen
	dfCoord      cgen.Gen
	wfCoord      cgen.Gen
	bfPtr        cgen.Gen
	wfPtr        cgen.Gen
	dfPtr        cgen.Gen
	sfPtr        cgen.Gen
	groupIdx     cgen.Gen
	fragIdx      cgen.Gen
	dfIdx        cgen.Gen
	dfShort      bool
	wfIdx        cgen.Gen
	wfShort      bool
}

func (p *produceSums) Append(to []byte) []byte {
	callee := func(first, cnt int) cgen.Gen {
		p.epochFirst = first
		p.epochCnt = cnt
		switch {
		case first < p.epochs1:
			p.slices = p.slices1
			p.wfCoreBytes = p.wfCoreBytes11
			p.wfFragBytes = p.wfFragBytes1
			p.wfGroupBytes = p.wfGroupBytes1
			p.dfCoreBytes = p.dfCoreBytes11
			p.dfFragBytes = p.dfFragBytes1
			p.dfGroupBytes = p.dfGroupBytes1
		default:
			p.slices = p.slices2
			p.wfCoreBytes = p.wfCoreBytes21
			p.wfFragBytes = p.wfFragBytes2
			p.wfGroupBytes = p.wfGroupBytes2
			p.dfCoreBytes = p.dfCoreBytes21
			p.dfFragBytes = p.dfFragBytes2
			p.dfGroupBytes = p.dfGroupBytes2
		}
		var (
			wfWork     = p.slices
			dfWork     = p.wfCores2 * wfWork
			fragWork   = p.dfCores2 * dfWork
			groupWork  = p.blkFrags * fragWork
			threadWork int
		)
		switch p.platform {
		case raw.AVX512Float32:
			threadWork = 512
		default:
			panic("bug")
		}
		p.wfTile = 1
		p.wfTiles = p.wfCores2
		p.wfScrap = 0
		p.wfHull = p.wfCores2
		p.dfTile = 1
		p.dfTiles = p.dfCores2
		p.dfScrap = 0
		p.dfHull = p.dfCores2
		p.fragTile = 1
		p.fragTiles = p.blkFrags
		p.fragScrap = 0
		p.fragHull = p.blkFrags
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
		case threadWork <= fragWork:
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
				tile  = ceilQuo(threadWork, fragWork)
				tiles = max(p.blkFrags/tile, 1)
			)
			p.fragTile = p.blkFrags / tiles
			p.fragTiles = tiles
			p.fragScrap = p.blkFrags - tiles*p.fragTile
			p.fragHull = tiles
			if p.fragScrap > 0 {
				p.fragTiles--
				p.fragScrap += p.fragTile
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
			p.fragTile = p.blkFrags
			p.fragTiles = 1
			p.fragScrap = 0
			p.fragHull = 1
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
		pair    = vb(p.name("pair"))
	)
	do := func(epoch cgen.Gen) cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		switch {
		case p.epochFirst == 0:
			stmts[0] = cgen.Var{
				Type: cgen.PtrVoid,
				What: cgen.Elem{Arr: pair},
				Init: cgen.Brace{
					Inner: cgen.CommaSpaced{
						tensors, epoch,
					},
				},
			}
		case p.epochCnt > 1:
			stmts[0] = cgen.Assign{
				Expr1: cgen.Elem{
					Arr: pair, Idx: il(1),
				},
				Expr2: cgen.Cast{
					Type: cgen.PtrVoid,
					Expr: epoch,
				},
			}
		}
		stmts[1] = &threader.Do{
			Ctx:    p.tc,
			Callee: vb(p.calleeName),
			Any:    pair,
			Hull: []cgen.Gen{
				il(p.wfHull),
				il(p.dfHull),
				il(p.fragHull),
				il(p.groupHull),
			},
			Team: team,
		}
		return stmts
	}
	var (
		prep = make(cgen.Gens, 3)
		body = make(cgen.Gens, 3)
	)
	prep[0] = callee(0, 1)
	body[0] = do(il(0))
	if n2 := p.epochs2; n2 > 1 {
		n1 := p.epochs1
		if n1 > 1 {
			prep[1] = callee(1, n1-1)
			epoch := vb(p.name("e"))
			body[1] = cgen.Stmts{
				cgen.For{
					Init: cgen.Var{
						Type: cgen.PtrdiffT,
						What: epoch,
						Init: il(1),
					},
					Cond: cgen.CmpL{
						Expr1: epoch,
						Expr2: il(n1),
					},
					Post: cgen.IncPre{
						Expr: epoch,
					},
					Body: do(epoch),
				},
			}
		}
		if n1 < n2 {
			prep[2] = callee(n1, 1)
			body[2] = do(il(n1))
		}
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
		body      = make(cgen.Stmts, 9)
		pair      = vb(p.name("pair"))
		epochExpr cgen.Gen
		usedPt    = false
	)
	body[0] = cgen.Var{
		Type: cgen.PtrPtrVoid, What: pair,
		Init: callee.Any(),
	}
	p.tensors = vb(p.name("tensors"))
	body[1] = cgen.Var{
		Type: cgen.PtrPtrChar, What: p.tensors,
		Init: cgen.Elem{
			Arr: pair, Idx: il(0),
		},
	}
	switch p.epochCnt {
	case 1:
		epochExpr = il(p.epochFirst)
	default:
		epochExpr = cgen.Cast{
			Type: cgen.PtrdiffT,
			Expr: cgen.Elem{
				Arr: pair, Idx: il(1),
			},
		}
	}
	p.epochCoord = vb(p.name("e"))
	body[2] = cgen.Var{
		Type: cgen.PtrdiffT, What: p.epochCoord,
		Init: epochExpr,
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
		body[6-i] = cgen.Var{
			Type: cgen.PtrdiffT, What: ret,
			Init: expr,
		}
		return ret
	}
	p.groupCoord = coord("g", p.groupHull, 3)
	p.fragCoord = coord("f", p.fragHull, 2)
	p.dfCoord = coord("d", p.dfHull, 1)
	p.wfCoord = coord("w", p.wfHull, 0)
	if !usedPt {
		body[7] = void(callee.Pt)
	}
	body[8] = p.kernel1()
	return callee.Func(body)
}

func (p *produceSums) kernel1() cgen.Gen {
	p.bfPtr = vb(p.name("bfPtr"))
	p.wfPtr = vb(p.name("wfPtr"))
	p.dfPtr = vb(p.name("dfPtr"))
	p.sfPtr = vb(p.name("sfPtr"))
	tensor := func(x int) cgen.Gen {
		return cgen.Elem{
			Arr: p.tensors,
			Idx: il(x),
		}
	}
	decl := func(ptr, expr cgen.Gen) cgen.Gen {
		return cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptr, Init: expr,
		}
	}
	return cgen.Stmts{
		decl(
			p.bfPtr, addMul(
				tensor(0),
				il(p.bfEpochBytes),
				p.epochCoord,
			),
		),
		decl(
			p.wfPtr, addMul(
				cgen.Add{
					Expr1: tensor(0),
					Expr2: il(p.bfTotalBytes),
				},
				il(p.wfEpochBytes1),
				p.epochCoord,
			),
		),
		decl(
			p.dfPtr, addMul(
				tensor(1),
				il(p.dfEpochBytes1),
				p.epochCoord,
			),
		),
		decl(
			p.sfPtr, tensor(2),
		),
		p.kernel2(),
	}
}

func (p *produceSums) kernel2() cgen.Gen {
	layer5 := func() cgen.Gen {
		switch p.platform {
		case raw.AVX512Float32:
			return p.m512()
		default:
			panic("bug")
		}
	}
	layer4 := func() cgen.Gen {
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
					layer5(),
					retIf,
				},
			}
		}
		if p.wfCores1 < p.wfCores2 {
			p.wfShort = true
			stmts[3] = layer5()
		}
		return stmts
	}
	layer3 := func() cgen.Gen {
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
					layer4(),
					retIf,
				},
			}
		}
		if p.dfCores1 < p.dfCores2 {
			p.dfShort = true
			stmts[3] = layer4()
		}
		return stmts
	}
	layer2 := func() cgen.Gen {
		p.fragIdx = vb(p.name("j"))
		var (
			stmts = make(cgen.Stmts, 3)
			iters = 0
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: p.fragIdx,
			Init: cgen.Mul{
				Expr1: il(p.fragTile),
				Expr2: p.fragCoord,
			},
		}
		switch p.fragTiles {
		case p.fragHull:
			iters = p.fragTile
		case 0:
			iters = p.fragScrap
		}
		switch iters {
		case 1:
			stmts[2] = layer3()
		default:
			var (
				last = vb(p.name("jj"))
				expr cgen.Gen
			)
			if iters == 0 {
				expr = cgen.Paren{
					Inner: cgen.Ternary{
						Cond: cgen.CmpL{
							Expr1: p.fragCoord,
							Expr2: il(p.fragTiles),
						},
						Then: il(p.fragTile - 1),
						Else: il(p.fragScrap - 1),
					},
				}
			} else {
				expr = il(iters - 1)
			}
			stmts[1] = cgen.Var{
				Type: cgen.PtrdiffT,
				What: last,
				Init: cgen.Add{
					Expr1: p.fragIdx,
					Expr2: expr,
				},
			}
			stmts[2] = cgen.For{
				Cond: cgen.CmpLE{
					Expr1: p.fragIdx,
					Expr2: last,
				},
				Post: cgen.IncPre{
					Expr: p.fragIdx,
				},
				Body: layer3(),
			}
		}
		return stmts
	}
	layer1 := func() cgen.Gen {
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
			stmts[2] = layer2()
		default:
			var (
				last = vb(p.name("ii"))
				expr cgen.Gen
			)
			if iters == 0 {
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
			} else {
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
				Body: layer2(),
			}
		}
		return stmts
	}
	return layer1()
}

func (p *produceSums) m512() cgen.Gen {
	var (
		bfAdd     bool
		sfAdd     bool
		rows      int
		cols      int
		sums      []cgen.Gen
		bundleIdx cgen.Gen
		bundleMax int
		bundleCnt int
	)
	layer9 := func() cgen.Gen {
		var (
			stmts cgen.Stmts
			wfs   cgen.Gen
			dfs   = make([]cgen.Gen, cols)
		)
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		Wf := func(s, r int) cgen.Gen {
			var (
				wf   = vb(p.name("wf"))
				half cgen.Gen
			)
			switch wfs {
			case nil:
				wfs = vb(p.name("wfs"))
				var (
					lanes       = 2 * p.wfBytes / 4
					ae          = p.wfPtr
					slicePitch  = rows * p.wfBytes
					bundlePitch = bundleMax * slicePitch
				)
				if s == bundleCnt-1 && r == rows-1 {
					lanes /= 2
				}
				ae = cgen.Add{
					Expr1: ae,
					Expr2: il(s*slicePitch + r*p.wfBytes),
				}
				ae = addMul(ae, il(p.wfGroupBytes), p.groupIdx)
				ae = addMul(ae, il(p.wfFragBytes), p.fragIdx)
				ae = addMul(ae, il(p.wfCoreBytes), p.wfIdx)
				ae = addMul(ae, il(bundlePitch), bundleIdx)
				stmt(cgen.Var{
					Type: avx.M512i, What: wfs,
					Init: avx.Mm512MaskzLoaduEpi32{
						loMask(lanes), ae,
					},
				})
				half = avx.Mm512Castsi512Si256{wfs}
			default:
				half = avx.Mm512Extracti64x4Epi64{
					wfs, il(1),
				}
				wfs = nil
			}
			stmt(cgen.Var{
				Type: avx.M512, What: wf,
				Init: avx.Mm512CvtphPs{half},
			})
			return wf
		}
		Df := func(s, r, c int) cgen.Gen {
			if r == 0 {
				var (
					df          = vb(p.name("df"))
					ae          = p.dfPtr
					slicePitch  = cols * p.dfBytes
					bundlePitch = bundleMax * slicePitch
				)
				ae = cgen.Add{
					Expr1: ae,
					Expr2: il(s*slicePitch + c*p.dfBytes),
				}
				ae = addMul(ae, il(p.dfGroupBytes), p.groupIdx)
				ae = addMul(ae, il(p.dfFragBytes), p.fragIdx)
				ae = addMul(ae, il(p.dfCoreBytes), p.dfIdx)
				ae = addMul(ae, il(bundlePitch), bundleIdx)
				stmt(cgen.Var{
					Type: avx.M512, What: df,
					Init: avx.Mm512LoaduPs{ae},
				})
				dfs[c] = df
			}
			return dfs[c]
		}
		for s := 0; s < bundleCnt; s++ {
			for r := 0; r < rows; r++ {
				wf := Wf(s, r)
				for c := 0; c < cols; c++ {
					var (
						df  = Df(s, r, c)
						sum = sums[r*cols+c]
					)
					stmt(cgen.Assign{
						Expr1: sum,
						Expr2: avx.Mm512FmaddPs{
							wf, df, sum,
						},
					})
				}
			}
		}
		return stmts
	}
	layer8 := func() cgen.Gen {
		bundleIdx = vb(p.name("b"))
		bundleMax = 1 + rows%2
		var (
			stmts = make(cgen.Stmts, 3)
			quo   = p.slices / bundleMax
			rem   = p.slices % bundleMax
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: bundleIdx,
			Init: il(0),
		}
		if quo > 0 {
			bundleCnt = bundleMax
			stmts[1] = cgen.For{
				Cond: cgen.CmpNE{
					Expr1: bundleIdx,
					Expr2: il(quo),
				},
				Post: cgen.IncPre{
					Expr: bundleIdx,
				},
				Body: layer9(),
			}
		}
		if rem > 0 {
			bundleCnt = rem
			stmts[2] = layer9()
		}
		return stmts
	}
	layer7 := func() cgen.Gen {
		var (
			n     = len(sums)
			stmts = make(cgen.Stmts, 1+n+n)
		)
		stmts[0] = layer8()
		for x, sum := range sums {
			var (
				ae      = p.sfPtr
				wfPitch = p.sfSumBytes11
			)
			if p.dfShort {
				wfPitch = p.sfSumBytes21
			}
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(x * p.sfBytes),
			}
			ae = addMul(ae, il(p.sfGroupBytes), p.groupIdx)
			ae = addMul(ae, il(p.sfFragBytes), p.fragIdx)
			ae = addMul(ae, il(p.sfCoreBytes1), p.dfIdx)
			ae = addMul(ae, il(wfPitch), p.wfIdx)
			if sfAdd {
				stmts[1+x] = cgen.Assign{
					Expr1: sum,
					Expr2: avx.Mm512AddPs{
						sum,
						avx.Mm512LoaduPs{ae},
					},
				}
			}
			stmts[1+x+n] = avx.Mm512StoreuPs{
				ae, sum,
			}
		}
		return stmts
	}
	layer6 := func() cgen.Gen {
		var (
			n1    = cols - 1
			n2    = rows * n1
			stmts = make(cgen.Stmts, n2+1)
		)
		for r := 0; r < rows; r++ {
			for c := 1; c < cols; c++ {
				stmts[r*n1+c-1] = cgen.Var{
					Type: avx.M512,
					What: sums[r*cols+c],
					Init: sums[r*cols],
				}
			}
		}
		stmts[n2] = layer7()
		return stmts
	}
	layer5 := func() cgen.Gen {
		if !bfAdd {
			return layer6()
		}
		bias := func() cgen.Stmts {
			stmts := make(cgen.Stmts, rows)
			for r := 0; r < rows; r++ {
				var (
					bf         = p.bfPtr
					groupPitch = p.toChans * p.bfBytes
					wfPitch    = p.wfSliceWfs1 * p.bfBytes
				)
				bf = cgen.Add{
					Expr1: bf,
					Expr2: il(r * p.bfBytes),
				}
				bf = addMul(bf, il(groupPitch), p.groupIdx)
				bf = addMul(bf, il(wfPitch), p.wfIdx)
				bf = cgen.At{
					Expr: cgen.Cast{
						Type: cgen.PtrFloat,
						Expr: cgen.Paren{
							Inner: bf,
						},
					},
				}
				stmts[r] = cgen.Assign{
					Expr1: sums[r*cols],
					Expr2: avx.Mm512MaskMovPs{
						avx.Mm512SetzeroPs,
						il(1 << 9),
						avx.Mm512Set1Ps{bf},
					},
				}
			}
			return stmts
		}
		zero := func() cgen.Stmts {
			stmts := make(cgen.Stmts, rows)
			for r := 0; r < rows; r++ {
				stmts[r] = cgen.Assign{
					Expr1: sums[r*cols],
					Expr2: avx.Mm512SetzeroPs,
				}
			}
			return stmts
		}
		return cgen.Stmts{
			cgen.If{
				Cond: cgen.Unlikely{
					Cond: cgen.IsZero{
						Expr: p.fragIdx,
					},
				},
				Then: bias(),
				Else: zero(),
			},
			layer6(),
		}
	}
	layer4 := func() cgen.Gen {
		var (
			stmts = make(cgen.Stmts, rows+1)
			expr  cgen.Gen
		)
		if !bfAdd {
			expr = avx.Mm512SetzeroPs
		}
		for r := 0; r < rows; r++ {
			stmts[r] = cgen.Var{
				Type: avx.M512,
				What: sums[r*cols],
				Init: expr,
			}
		}
		stmts[rows] = layer5()
		return stmts
	}
	layer3 := func() cgen.Gen {
		sums = make([]cgen.Gen, rows*cols)
		for x := range sums {
			sums[x] = vb(p.name("sum"))
		}
		return layer4()
	}
	layer2 := func() cgen.Gen {
		rows = p.wfSliceWfs1
		if p.wfShort {
			rows = p.wfSliceWfs2
		}
		cols = p.dfSliceDfs1
		if p.dfShort {
			cols = p.dfSliceDfs2
		}
		return layer3()
	}
	layer1 := func() cgen.Gen {
		switch p.epochFirst {
		case 0:
			bfAdd = true
			sfAdd = false
		default:
			bfAdd = false
			sfAdd = true
			for x := range p.Filts {
				if p.Filts[x].BnPre > 0 {
					bfAdd = true
					break
				}
			}
		}
		var unused cgen.Gen
		if !bfAdd {
			unused = void(p.bfPtr)
		}
		return cgen.Stmts{
			unused,
			layer2(),
		}
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
	toH        cgen.Gen
	toW        cgen.Gen
	blks       []*block
	wfIdx      cgen.Gen
	wfShort    bool
}

func (c *consumeSums) Append(to []byte) []byte {
	c.layout = newLayout(c.Ctx, c.Spec)
	var (
		n1         = c.dfCores1 * c.dfSliceDfs1
		n2         = c.toChans * (n1 + c.dfSliceDfs2)
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
	c.dfTile = 1
	c.dfTiles = c.dfCores2
	c.dfScrap = 0
	c.dfHull = c.dfCores2
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
	case threadBlks <= groupBlks:
		c.wfTile = c.wfCores2
		c.wfTiles = 1
		c.wfScrap = 0
		c.wfHull = 1
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
		c.wfTile = c.wfCores2
		c.wfTiles = 1
		c.wfScrap = 0
		c.wfHull = 1
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
		body[4] = void(callee.Pt)
	}
	body[5] = c.kernel1()
	return callee.Func(body)
}

func (c *consumeSums) kernel1() cgen.Gen {
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
	dps := func(n int) {
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
			dps(op.Int)
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
	dps(need - have)
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
		if iters == 0 {
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
		} else {
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
	var (
		retIf    cgen.Gen
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
	layer9 := func() cgen.Gen {
		return cgen.Stmts{
			c.kernel4(),
			retIf,
			cgen.IncPre{
				Expr: c.dfIdx,
			},
		}
	}
	layer8 := func() cgen.Gen {
		if lwToStep == 0 {
			return layer9()
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
				Body: layer9(),
			},
		}
	}
	layer7 := func() cgen.Gen {
		c.toH = vb(c.name("toH"))
		c.toW = vb(c.name("toW"))
		c.blks = lw.blks
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
			layer8(),
			breakIf,
		}
	}
	layer6 := func() cgen.Gen {
		var (
			lws  = lh.lws
			tree func(int, int) cgen.Stmts
		)
		leaf := func(x int) cgen.Stmts {
			lw = lws[x]
			lwToH = lw.fromH / c.StrideH
			lwToW = lw.fromW / c.StrideW
			lwToStep = lw.fromStep / c.StrideW
			var assn cgen.Gen
			if x+1 < len(lws) {
				assn = cgen.Assign{
					Expr1: rel,
					Expr2: il(lw.segPast),
				}
			}
			return cgen.Stmts{
				layer7(),
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
	layer5 := func() cgen.Gen {
		if lh.segStep == 0 {
			relBreak = -1
			return layer6()
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
			Body: layer6(),
		}
	}
	layer4 := func() cgen.Gen {
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
			layer5(),
		}
	}
	layer3 := func() cgen.Gen {
		var (
			lhs  = c.segs.lhs
			tree func(int, int) cgen.Stmts
		)
		leaf := func(x int) cgen.Stmts {
			lh = lhs[x]
			lhToH = lh.fromH / c.StrideH
			lhToStep = lh.fromStep / c.StrideH
			var assn cgen.Gen
			if x+1 < len(lhs) {
				assn = cgen.Assign{
					Expr1: c.dfIdx,
					Expr2: il(lh.segPast),
				}
			}
			return cgen.Stmts{
				layer4(),
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
	layer2 := func() cgen.Gen {
		c.dfIdx = vb(c.name("j"))
		stmts := make(cgen.Stmts, 3)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: c.dfIdx,
			Init: cgen.Mul{
				Expr1: il(c.dfTile),
				Expr2: c.dfCoord,
			},
		}
		if c.dfHull == 1 {
			retIf = nil
		} else {
			var (
				last = vb(c.name("last"))
				expr cgen.Gen
			)
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
				What: last,
				Init: cgen.Add{
					Expr1: c.dfIdx,
					Expr2: expr,
				},
			}
			retIf = cgen.If1{
				Cond: cgen.CmpGE{
					Expr1: c.dfIdx,
					Expr2: last,
				},
				Then: cgen.Return{},
			}
		}
		stmts[2] = layer3()
		return stmts
	}
	layer1 := func() cgen.Gen {
		if 6%c.StrideH != 0 ||
			6%c.StrideW != 0 {
			panic("bug")
		}
		return layer2()
	}
	return layer1()
}

func (c *consumeSums) kernel4() cgen.Gen {
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
				c.kernel5(),
				retIf,
			},
		}
	}
	if c.wfCores1 < c.wfCores2 {
		c.wfShort = true
		stmts[3] = c.kernel5()
	}
	return stmts
}

func (c *consumeSums) kernel5() cgen.Gen {
	switch c.platform {
	case raw.AVX512Float32:
		return c.m512()
	default:
		panic("bug")
	}
}

func (c *consumeSums) m512() cgen.Gen {
	type Part struct {
		off   int
		lanes int
		id    int
		h     int
	}
	var (
		sfRows    int
		sfCols    int
		outerIdx  cgen.Gen
		outerFull int
		outerRows int
		bnMuls    [][]cgen.Gen
		bnAdds    [][]cgen.Gen
		innerRows []int
		innerCols []int
		innerBlks []*block
		innerLocs []int
		Wct       *wct.Sums
		pm        cgen.Gen
		vecs      []cgen.Gen
		cnts      []int
		parts     []Part
	)
	bnPrep := func(row int) cgen.Gen {
		if bnMuls[row] != nil {
			return nil
		}
		var (
			n     = len(c.bnPtrs)
			muls  = make([]cgen.Gen, n)
			adds  = make([]cgen.Gen, n)
			loads = make(cgen.Gens, n)
			ch    = il(row)
		)
		ch = addMul(ch, il(c.toChans), c.groupIdx)
		ch = addMul(ch, il(c.wfSliceWfs1), c.wfIdx)
		ch = addMul(ch, il(outerFull), outerIdx)
		ch = cgen.Paren{
			Inner: ch,
		}
		for x, ptr := range c.bnPtrs {
			var (
				bnMul = vb(c.name("bnMul"))
				bnAdd = vb(c.name("bnAdd"))
			)
			muls[x] = bnMul
			adds[x] = bnAdd
			loads[x] = &bn.Load{
				Ctx:     c.bc,
				Mas:     ptr,
				Channel: ch,
				Mul:     bnMul,
				Add:     bnAdd,
			}
		}
		bnMuls[row] = muls
		bnAdds[row] = adds
		return loads
	}
	adj := func(x1, x2 int) bool {
		var (
			row1 = innerRows[x1]
			row2 = innerRows[x2]
		)
		if row1 != row2 {
			return false
		}
		var (
			blk1 = innerBlks[x1]
			blk2 = innerBlks[x2]
		)
		if blk1.fromH != blk2.fromH {
			return false
		}
		var (
			w1 = blk1.fromW
			w2 = blk2.fromW
		)
		return w1+6 == w2
	}
	mask := func(part int) int {
		var (
			off   = parts[part].off
			lanes = parts[part].lanes
			run   = 1<<uint(lanes) - 1
		)
		return run << uint(off)
	}
	addr := func(part, ptr int) cgen.Gen {
		var (
			id         = parts[part].id
			blk        = innerBlks[id]
			ae         = c.datPtrs[ptr]
			pitch1     = c.To.Pitch1Bytes[ptr]
			pitch2     = c.To.Pitch2Bytes[ptr]
			groupPitch = c.toChans * pitch2
			wfIdxPitch = c.wfSliceWfs1 * pitch2
			outerPitch = outerFull * pitch2
		)
		ae = cgen.Add{
			Expr1: ae,
			Expr2: il(
				innerRows[id]*pitch2 +
					blk.fromH/c.StrideH*pitch1 +
					blk.fromW/c.StrideW*c.datBytes +
					parts[part].h*pitch1 -
					parts[part].off*c.datBytes,
			),
		}
		ae = addMul(ae, il(groupPitch), c.groupIdx)
		ae = addMul(ae, il(pitch1), c.toH)
		ae = addMul(ae, il(c.datBytes), c.toW)
		ae = addMul(ae, il(wfIdxPitch), c.wfIdx)
		ae = addMul(ae, il(outerPitch), outerIdx)
		return ae
	}
	layer12 := func() cgen.Gen {
		var (
			stmts cgen.Stmts
			ptr   = c.datSplit
			n     = len(c.datPtrs)
		)
		for ; ptr < n; ptr++ {
			part := 0
			for x, vec := range vecs {
				stop := part + cnts[x]
				for ; part < stop; part++ {
					stmts = append(
						stmts,
						avx.Mm512MaskStoreuPs{
							addr(part, ptr),
							il(mask(part)),
							vec,
						},
					)
				}
			}
		}
		return stmts
	}
	layer11 := func() cgen.Gen {
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		var (
			dpIdx = 0
			bnIdx = 0
		)
		for op := range c.To.Ops {
			op := &c.To.Ops[op]
			switch op.Kind {
			case mod.Add:
				ptr := dpIdx
				dpIdx += op.Int
				for ; ptr < dpIdx; ptr++ {
					part := 0
					for x, vec := range vecs {
						stop := part + cnts[x]
						for ; part < stop; part++ {
							stmt(cgen.Assign{
								Expr1: vec,
								Expr2: avx.Mm512AddPs{
									vec,
									avx.Mm512MaskzLoaduPs{
										il(mask(part)),
										addr(part, ptr),
									},
								},
							})
						}
					}
				}
			case mod.Bn:
				part := 0
				for x, vec := range vecs {
					var (
						stop     = part + cnts[x]
						tagMasks [4][2]int
					)
					for ; part < stop; part++ {
						var (
							id   = parts[part].id
							row  = innerRows[id]
							tag  = ^row
							Mask = mask(part)
						)
						for y := range &tagMasks {
							tm := &tagMasks[y]
							switch tm[0] {
							case 0:
								tm[0] = tag
								tm[1] = Mask
							case tag:
								tm[1] |= Mask
							default:
								continue
							}
							break
						}
					}
					one := tagMasks[1][0] == 0
					for y := range &tagMasks {
						var (
							tm  = &tagMasks[y]
							tag = tm[0]
						)
						if tag == 0 {
							break
						}
						var (
							row  = ^tag
							prep = bnPrep(row)
							Mask cgen.Gen
						)
						if prep != nil {
							stmt(prep)
						}
						if !one {
							Mask = il(tm[1])
						}
						stmt(&bn.Apply{
							Ctx:  c.bc,
							Mul:  bnMuls[row][bnIdx],
							Add:  bnAdds[row][bnIdx],
							To:   vec,
							Mask: Mask,
						})
					}
				}
				bnIdx++
			case mod.ReLU:
				for _, vec := range vecs {
					stmt(&act.ReLU{
						Ctx:      c.ac,
						NegSlope: op.Float,
						Var:      vec,
					})
				}
			default:
				panic("bug")
			}
		}
		stmt(layer12())
		return stmts
	}
	layer10 := func() cgen.Gen {
		var (
			to   = 0
			from = 0
		)
		for x, cnt := range cnts {
			stop := from + cnt
			for from < stop {
				y := from + 1
				for ; y < stop; y++ {
					var (
						id1 = parts[y-1].id
						id2 = parts[y].id
					)
					if !adj(id1, id2) {
						break
					}
					var (
						off   = parts[from].off
						lanes = &parts[from].lanes
					)
					if off+*lanes != parts[y].off {
						break
					}
					*lanes += parts[y].lanes
					cnts[x]--
				}
				parts[to] = parts[from]
				to++
				from = y
			}
		}
		parts = parts[:to]
		return layer11()
	}
	layer9 := func() cgen.Gen {
		vecs = vecs[:0]
		cnts = cnts[:0]
		parts = parts[:0]
		part := func(vec cgen.Gen, slot, loc, h int) {
			if vec == nil {
				return
			}
			id := -1
			for x := range innerLocs {
				if innerLocs[x] == loc {
					id = x
					break
				}
			}
			if id == -1 {
				return
			}
			blk := innerBlks[id]
			if h >= blk.yieldH {
				return
			}
			last := len(vecs) - 1
			switch {
			case last == -1:
				fallthrough
			case vecs[last] != vec:
				vecs = append(vecs, vec)
				cnts = append(cnts, 1)
			default:
				cnts[last]++
			}
			parts = append(parts, Part{
				off:   slot * 6 / c.StrideW,
				lanes: (blk.yieldW-1)/c.StrideW + 1,
				id:    id,
				h:     h / c.StrideH,
			})
		}
		var stmts cgen.Stmts
		for h := 0; h < 6; h++ {
			var (
				vec1 = Wct.Out[h]
				vec2 = Wct.Out[h+6]
			)
			if vec1 == nil && vec2 == nil {
				continue
			}
			if c.StrideW == 1 {
				part(vec1, 0, 0, h)
				part(vec1, 1, 2, h)
				part(vec2, 0, 1, h)
				part(vec2, 1, 3, h)
				continue
			}
			var expr cgen.Gen
			switch {
			case vec1 == nil || vec2 == nil:
				loc := 0
				if vec1 == nil {
					vec1 = vec2
					loc = 1
				}
				expr = avx.Mm512PermutexvarPs{
					pm, vec1,
				}
				part(vec1, 0, loc, h)
				part(vec1, 1, loc+2, h)
			default:
				expr = avx.Mm512Permutex2varPs{
					vec1, pm, vec2,
				}
				if Wct.Blks == 2 {
					part(vec1, 0, 0, h)
					part(vec1, 1, 1, h)
					break
				}
				part(vec1, 0, 0, h)
				part(vec1, 1, 2, h)
				part(vec1, 2, 1, h)
				part(vec1, 3, 3, h)
			}
			stmts = append(
				stmts, cgen.Assign{
					Expr1: vec1,
					Expr2: expr,
				},
			)
		}
		return append(
			stmts,
			layer10(),
		)
	}
	layer8 := func() cgen.Gen {
		if c.StrideW == 1 {
			pm = nil
			return layer9()
		}
		pm = vb(c.name("pm"))
		var (
			set  = make(avx.Mm512SetEpi32, 16)
			each = 6
		)
		if Wct.Blks != 2 {
			each *= 2
		}
		for x := 0; x < 16; x++ {
			take := x * c.StrideW % (2 * each)
			if take >= each {
				take += 16 - each
			}
			set[15-x] = il(take)
		}
		return cgen.Stmts{
			cgen.Var{
				Type: avx.M512i, What: pm,
				Init: set,
			},
			layer9(),
		}
	}
	layer7 := func() cgen.Gen {
		var stmts cgen.Stmts
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		load := func(x, frag int) cgen.Gen {
			var (
				sf         = vb(c.name("sf"))
				ae         = c.sfPtr
				rowPitch   = sfCols * c.sfBytes
				wfIdxPitch = c.wfSliceWfs1 * rowPitch
				outerPitch = outerFull * rowPitch
			)
			ae = cgen.Add{
				Expr1: ae,
				Expr2: il(
					frag*c.sfFragBytes +
						innerRows[x]*rowPitch +
						innerCols[x]*c.sfBytes,
				),
			}
			ae = addMul(ae, il(c.sfGroupBytes), c.groupIdx)
			ae = addMul(ae, il(c.sfCoreBytes1), c.dfIdx)
			ae = addMul(ae, il(wfIdxPitch), c.wfIdx)
			ae = addMul(ae, il(outerPitch), outerIdx)
			stmt(cgen.Var{
				Type: avx.M512, What: sf,
				Init: avx.Mm512LoaduPs{ae},
			})
			return sf
		}
		for frag := 0; frag < c.blkFrags; frag++ {
			for loc := 0; loc < Wct.Blks; loc += 2 {
				var (
					pair  = loc/2*8 + frag*2
					in1   = Wct.In[pair]
					in2   = Wct.In[pair+1]
					sf1   cgen.Gen
					sf2   cgen.Gen
					expr1 cgen.Gen
					expr2 cgen.Gen
				)
				for x := range innerLocs {
					switch innerLocs[x] {
					case loc:
						sf1 = load(x, frag)
					case loc + 1:
						sf2 = load(x, frag)
					}
				}
				switch sf2 {
				case nil:
					expr1, sf2 = sf1, sf1
				default:
					expr1 = avx.Mm512ShuffleF32x4{
						sf1, sf2,
						il(1<<6 | 0<<4 | 1<<2 | 0<<0),
					}
				}
				expr2 = avx.Mm512ShuffleF32x4{
					sf1, sf2,
					il(3<<6 | 2<<4 | 3<<2 | 2<<0),
				}
				stmt(cgen.Var{
					Type: avx.M512, What: in1,
					Init: expr1,
				})
				stmt(cgen.Var{
					Type: avx.M512, What: in2,
					Init: expr2,
				})
			}
		}
		stmt(Wct)
		stmt(layer8())
		return stmts
	}
	layer6 := func() cgen.Gen {
		Wct = &wct.Sums{
			Platform: c.platform,
			Nms:      c.nms,
			Blks:     len(innerBlks),
		}
		for x, blk := range innerBlks {
			var (
				cols = 0
				past = blk.yieldW
				step = c.StrideW
				loc  = innerLocs[x]
			)
			for y := 0; y < past; y += step {
				cols |= 1 << uint(y)
			}
			Wct.Cols[loc] = cols
		}
		for loc := 0; loc < Wct.Blks; loc += 2 {
			for x := 0; x < 8; x++ {
				in := vb(c.name("in"))
				Wct.In[loc/2*8+x] = in
			}
		}
		for x, blk := range innerBlks {
			var (
				past = blk.yieldH
				step = c.StrideH
				loc  = innerLocs[x]
			)
			for y := 0; y < past; y += step {
				to := &Wct.Out[loc%2*6+y]
				if *to == nil {
					*to = vb(c.name("out"))
				}
			}
		}
		return layer7()
	}
	layer5 := func() cgen.Gen {
		var (
			n    = len(innerBlks)
			used [4]bool
		)
		for loc := 0; loc+2 < n; loc++ {
			for x := 0; x+1 < n; x++ {
				if innerLocs[x] == -1 &&
					innerLocs[x+1] == -1 &&
					adj(x, x+1) {
					innerLocs[x] = loc
					innerLocs[x+1] = loc + 2
					used[loc] = true
					used[loc+2] = true
					break
				}
			}
		}
		for loc := 0; loc < n; loc++ {
			if used[loc] {
				continue
			}
			for x := range innerLocs {
				if innerLocs[x] == -1 {
					innerLocs[x] = loc
					break
				}
			}
		}
		return layer6()
	}
	layer4 := func() cgen.Gen {
		const n1 = 4
		var (
			n2  = outerRows * sfCols
			n3  = ceilQuo(n2, n1)
			ret = make(cgen.Gens, 0, n3)
		)
		flush := func() {
			if len(innerRows) == 0 {
				return
			}
			ret = append(ret, layer5())
			innerRows = innerRows[:0]
			innerCols = innerCols[:0]
			innerBlks = innerBlks[:0]
			innerLocs = innerLocs[:0]
		}
		for x := 0; x < n2; x++ {
			var (
				row = x / sfCols
				col = x % sfCols
				blk = c.blks[col]
				loc = -1
			)
			innerRows = append(innerRows, row)
			innerCols = append(innerCols, col)
			innerBlks = append(innerBlks, blk)
			innerLocs = append(innerLocs, loc)
			if len(innerRows) == n1 {
				flush()
			}
		}
		flush()
		return ret
	}
	layer3 := func() cgen.Gen {
		if len(c.bnPtrs) != 0 {
			bnMuls = make([][]cgen.Gen, outerRows)
			bnAdds = make([][]cgen.Gen, outerRows)
		}
		return layer4()
	}
	layer2 := func() cgen.Gen {
		outerIdx = vb(c.name("l"))
		outerFull = 1
		for outerFull*sfCols%4 != 0 {
			outerFull *= 2
		}
		var (
			stmts = make(cgen.Stmts, 3)
			quo   = sfRows / outerFull
			rem   = sfRows % outerFull
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: outerIdx,
			Init: il(0),
		}
		if quo > 0 {
			outerRows = outerFull
			stmts[1] = cgen.For{
				Cond: cgen.CmpNE{
					Expr1: outerIdx,
					Expr2: il(quo),
				},
				Post: cgen.IncPre{
					Expr: outerIdx,
				},
				Body: layer3(),
			}
		}
		if rem > 0 {
			outerRows = rem
			stmts[2] = layer3()
		}
		return stmts
	}
	layer1 := func() cgen.Gen {
		sfRows = c.wfSliceWfs1
		if c.wfShort {
			sfRows = c.wfSliceWfs2
		}
		sfCols = len(c.blks)
		return layer2()
	}
	return layer1()
}
