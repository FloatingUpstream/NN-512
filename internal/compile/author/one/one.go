package one

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

func cast(i int) cgen.Gen {
	return cgen.Cast{
		Type: cgen.PtrdiffT,
		Expr: il(i),
	}
}

func addMul(x, y, z cgen.Gen) cgen.Gen {
	return cgen.Add{
		Expr1: x,
		Expr2: cgen.Mul{
			Expr1: y,
			Expr2: z,
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
		prefix:      pl.Config.Prefix + "One",
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

type ctxSpec struct {
	*Ctx
	*Spec
}

type tokens struct {
	IdxPast int
	Sects   []*section
}

type section struct {
	IdxFirst int
	IdxPast  int
	FromBase int
	FromWrap int
	ToBase   int
	ToWrap   int
	Uniqs    []*token
}

type token struct {
	From  tokenFrom
	To    tokenTo
	Slots int
}

type tokenFrom struct {
	FirstH int
	FirstW int
	LastH  int
	Cmds   []interface{}
}

type tokenTo struct {
	FirstH int
	Cmds   []interface{}
}

func tokenize(cs ctxSpec, slots int) *tokens {
	var (
		ret  tokens
		toks []*token
	)
	switch cs.platform {
	case raw.AVX512Float32:
		toks = m512Toks(cs, slots)
	default:
		panic("bug")
	}
	ret.IdxPast = len(toks)
	put := func(s *section, i, j int) {
		s.Uniqs = make([]*token, j-i)
		copy(s.Uniqs, toks[i:j])
		for _, tok := range s.Uniqs {
			tok.From.FirstH -= s.FromBase
			tok.From.LastH -= s.FromBase
			tok.To.FirstH -= s.ToBase
		}
		ret.Sects = append(
			ret.Sects, s,
		)
	}
	var (
		start = 0
		loop  = -1
		tie   = -1
		stop  = 0
	)
	encode := func() {
		if loop == -1 {
			loop = stop
		}
		if start < loop {
			sect := &section{
				IdxFirst: start,
				IdxPast:  loop,
				FromBase: toks[start].From.FirstH,
				ToBase:   toks[start].To.FirstH,
			}
			put(sect, start, loop)
		}
		if loop < stop {
			var (
				loopTok  = toks[loop]
				tieTok   = toks[tie]
				fromBase = loopTok.From.FirstH
				toBase   = loopTok.To.FirstH
			)
			sect := &section{
				IdxFirst: loop,
				IdxPast:  stop,
				FromBase: fromBase,
				FromWrap: tieTok.From.FirstH - fromBase,
				ToBase:   toBase,
				ToWrap:   tieTok.To.FirstH - toBase,
			}
			put(sect, loop, tie)
		}
		start = stop
		loop = -1
		tie = -1
	}
	zone := func(h int) int {
		switch {
		case h < cs.PaddingH:
			return 0
		case h < cs.PaddingH+cs.From.Height:
			return 1
		}
		return 2
	}
	type Sig struct {
		FirstW int
		Zones  int
		Span   int
	}
	idx := make(map[Sig]int)
	for i, tok := range toks {
		var (
			firstH = tok.From.FirstH
			lastH  = tok.From.LastH
		)
		sig := Sig{
			FirstW: tok.From.FirstW,
			Zones:  zone(lastH)<<2 | zone(firstH),
			Span:   lastH - firstH,
		}
		if at, ok := idx[sig]; ok {
			if loop == -1 {
				loop = at
				tie = i
			}
		} else {
			idx[sig] = i
			if loop != -1 {
				encode()
			}
		}
		stop++
	}
	encode()
	return &ret
}

type m512CmdZero struct {
	Id int
}

type m512CmdCopy struct {
	DstId int
	SrcId int
}

type m512CmdRotate struct {
	DstId int
	SrcId int
	Cnt   int
}

type m512CmdBlend struct {
	DstId int
	SrcId int
	Off   int
	Cnt   int
}

type m512CmdPermute1 struct {
	DstId int
	SrcId int
	Off   int
	Inc   int
}

type m512CmdPermute2 struct {
	DstId  int
	SrcId1 int
	SrcId2 int
	Off    int
	Inc    int
}

type m512CmdLoad struct {
	Id   int
	RelH int
	W    int
	Cnt  int
}

type m512CmdFromModAddPre struct {
	Id   int
	RelH int
	W    int
	Cnt  int
}

type m512CmdFromModPostAdd struct {
	Id   int
	Mask int
}

type m512CmdSlotPut struct {
	Slot int
	Id   int
}

type m512CmdSlotGet struct {
	Id   int
	Slot int
}

type m512CmdToModPreAdd struct {
	Id int
}

type m512CmdToModAddPost struct {
	Id   int
	RelH int
	W    int
	Cnt  int
}

type m512CmdStore struct {
	Id   int
	RelH int
	W    int
	Cnt  int
}

func m512Toks(cs ctxSpec, slots int) (ret []*token) {
	const lanes = 16
	var (
		strideH      = cs.StrideH
		strideW      = cs.StrideW
		padH         = cs.PaddingH
		padW         = cs.PaddingW
		edgeH        = padH + cs.From.Height
		edgeW        = padW + cs.From.Width
		fromH        = edgeH + padH
		fromW        = edgeW + padW
		toH          = 1 + (fromH-1)/strideH
		toW          = 1 + (fromW-1)/strideW
		h            = 0
		w            = 0
		n            int
		hh           int
		ww           int
		tok          *token
		fromNextId   int
		fromPileId   int
		fromPileFree int
		fromPileMask int
		fromPileSlot int
		toNextId     int
		toPileId     int
		loPad        int
		hiPad        int
		nonPad       int
	)
	fromCmd := func(cmd interface{}) {
		tok.From.Cmds = append(
			tok.From.Cmds, cmd,
		)
	}
	toCmd := func(cmd interface{}) {
		tok.To.Cmds = append(
			tok.To.Cmds, cmd,
		)
	}
	newSlot := func() int {
		slot := tok.Slots
		tok.Slots++
		return slot
	}
	fromNewId := func() int {
		id := fromNextId
		fromNextId++
		return id
	}
	toNewId := func() int {
		id := toNextId
		toNextId++
		return id
	}
	zero := func() int {
		id := fromNewId()
		fromCmd(&m512CmdZero{
			Id: id,
		})
		return id
	}
	load := func(at, cnt int) int {
		var (
			id   = fromNewId()
			relH = hh - tok.From.FirstH
		)
		fromCmd(&m512CmdLoad{
			Id:   id,
			RelH: relH,
			W:    at,
			Cnt:  cnt,
		})
		fromCmd(&m512CmdFromModAddPre{
			Id:   id,
			RelH: relH,
			W:    at,
			Cnt:  cnt,
		})
		return id
	}
	broadcast := func(at int) int {
		return load(at, 0)
	}
	pilePut := func() {
		if fromPileId == -1 {
			return
		}
		if fromPileMask != 0 {
			fromCmd(&m512CmdFromModPostAdd{
				Id:   fromPileId,
				Mask: fromPileMask,
			})
		}
		fromCmd(&m512CmdSlotPut{
			Slot: fromPileSlot,
			Id:   fromPileId,
		})
	}
	slotGet := func(slot int) int {
		id := toNewId()
		toCmd(&m512CmdSlotGet{
			Id:   id,
			Slot: slot,
		})
		toCmd(&m512CmdToModPreAdd{
			Id: id,
		})
		return id
	}
	store := func(id int) {
		relH := h - tok.To.FirstH
		toCmd(&m512CmdToModAddPost{
			Id:   id,
			RelH: relH,
			W:    w,
			Cnt:  n,
		})
		toCmd(&m512CmdStore{
			Id:   id,
			RelH: relH,
			W:    w,
			Cnt:  n,
		})
	}
	build := func() int {
		switch {
		case nonPad == 0:
			return zero()
		case n == 1:
			return broadcast(ww)
		}
		var (
			lane = loPad
			at   = ww + loPad*strideW
		)
		if n <= fromPileFree {
			lane += lanes - fromPileFree
		}
		if strideW == 1 || nonPad == 1 {
			id := load(at, nonPad)
			if lane > 0 {
				fromCmd(&m512CmdRotate{
					DstId: id,
					SrcId: id,
					Cnt:   lanes - lane,
				})
			}
			return id
		}
		var (
			id   = -1
			each = 1 + (lanes-1)/strideW
			take int
		)
		for have := 0; have < nonPad; have += take {
			take = min(nonPad-have, 2*each)
			var (
				off   = lane + have
				lower = at + have*strideW
				tight int
			)
			switch {
			case take == 1:
				tight = broadcast(lower)
			case take <= each:
				var (
					span  = 1 + (take-1)*strideW
					loose = load(lower, span)
				)
				tight = fromNewId()
				fromCmd(&m512CmdPermute1{
					DstId: tight,
					SrcId: loose,
					Off:   off,
					Inc:   strideW,
				})
			default:
				var (
					upper  = lower + each*strideW
					span1  = 1 + (each-1)*strideW
					span2  = 1 + (take-each-1)*strideW
					loose1 = load(lower, span1)
					loose2 = load(upper, span2)
				)
				tight = fromNewId()
				fromCmd(&m512CmdPermute2{
					DstId:  tight,
					SrcId1: loose1,
					SrcId2: loose2,
					Off:    off,
					Inc:    strideW,
				})
			}
			if id == -1 {
				if loPad+hiPad == 0 {
					id = tight
					continue
				}
				id = zero()
			}
			fromCmd(&m512CmdBlend{
				DstId: id,
				SrcId: tight,
				Off:   off,
				Cnt:   take,
			})
		}
		return id
	}
	encode := func() {
		loPad = n
		hiPad = 0
		if hh >= padH && hh < edgeH {
			var (
				lo = padW - ww
				nn = 1 + (n-1)*strideW
				hi = ww + nn - edgeW
			)
			loPad = 0
			if lo > 0 {
				loPad = 1 + (lo-1)/strideW
				loPad = min(loPad, n)
			}
			if hi > 0 {
				hiPad = 1 + (hi-1)/strideW
				hiPad = min(hiPad, n)
			}
		}
		nonPad = n - loPad - hiPad
		var (
			fromId = build()
			mask1  = 1<<uint(nonPad) - 1
			mask2  = mask1 << uint(loPad)
		)
		switch {
		case n == lanes:
			if mask2 != 0 {
				fromCmd(&m512CmdFromModPostAdd{
					Id:   fromId,
					Mask: mask2,
				})
			}
			slot := newSlot()
			fromCmd(&m512CmdSlotPut{
				Slot: slot,
				Id:   fromId,
			})
			toId := slotGet(slot)
			store(toId)
		case n <= fromPileFree:
			off := lanes - fromPileFree
			fromCmd(&m512CmdBlend{
				DstId: fromPileId,
				SrcId: fromId,
				Off:   off,
				Cnt:   n,
			})
			fromPileFree -= n
			fromPileMask |= mask2 << uint(off)
			toId := toNewId()
			toCmd(&m512CmdRotate{
				DstId: toId,
				SrcId: toPileId,
				Cnt:   off,
			})
			store(toId)
		default:
			pilePut()
			fromPileId = fromId
			fromPileFree = lanes - n
			fromPileMask = mask2
			fromPileSlot = newSlot()
			toPileId = slotGet(fromPileSlot)
			toId := toNewId()
			toCmd(&m512CmdCopy{
				DstId: toId,
				SrcId: toPileId,
			})
			store(toId)
		}
	}
	for {
		for {
			n = toW - w
			if n == 0 {
				if h++; h == toH {
					break
				}
				w = 0
				n = toW
			}
			n = min(n, lanes)
			hh = h * strideH
			ww = w * strideW
			if tok == nil {
				tok = &token{
					From: tokenFrom{
						FirstH: hh,
						FirstW: ww,
					},
					To: tokenTo{
						FirstH: h,
					},
				}
				fromNextId = 0
				fromPileId = -1
				fromPileFree = 0
				toNextId = 0
			}
			if tok.Slots == slots {
				if n > fromPileFree {
					break
				}
			}
			tok.From.LastH = hh
			encode()
			w += n
		}
		if tok != nil {
			pilePut()
			ret = append(ret, tok)
			tok = nil
		}
		if h == toH {
			break
		}
	}
	return
}

type layout struct {
	fromChans      int
	toChans        int
	slices1        int
	slices2        int
	epochs1        int
	epochs2        int
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
	wtGroupBytes1  int
	wtGroupBytes2  int
	wtEpochBytes1  int
	wtEpochBytes2  int
	wtTotalBytes   int
	datBytes       int
	slotDats       int
	slotBytes      int
	datSliceSlots1 int
	datSliceSlots2 int
	datSliceDats1  int
	datSliceDats2  int
	datSliceBytes1 int
	datSliceBytes2 int
	datCores1      int
	datCores2      int
	datCoreBytes11 int
	datCoreBytes12 int
	datCoreBytes21 int
	datCoreBytes22 int
	datGroupBytes1 int
	datGroupBytes2 int
	datEpochBytes1 int
	datEpochBytes2 int
	datTotalBytes  int
	toks           *tokens
}

func newLayout(cs ctxSpec) *layout {
	var y layout
	special := func() bool {
		if cs.StrideH != 1 ||
			cs.StrideW != 1 ||
			cs.PaddingH != 0 ||
			cs.PaddingW != 0 {
			return false
		}
		tight := cs.From.Width * y.datBytes
		for _, pitch := range cs.From.Pitch1Bytes {
			if pitch != tight {
				return false
			}
		}
		for _, pitch := range cs.To.Pitch1Bytes {
			if pitch != tight {
				return false
			}
		}
		return true
	}
	stage7 := func() {
		y.datCoreBytes11 = y.slices1 * y.datSliceBytes1
		y.datCoreBytes12 = y.slices1 * y.datSliceBytes2
		y.datCoreBytes21 = y.slices2 * y.datSliceBytes1
		y.datCoreBytes22 = y.slices2 * y.datSliceBytes2
		y.datGroupBytes1 = y.datCores1 * y.datCoreBytes11
		y.datGroupBytes2 = y.datCores1 * y.datCoreBytes21
		if y.datCores1 < y.datCores2 {
			y.datGroupBytes1 += y.datCoreBytes12
			y.datGroupBytes2 += y.datCoreBytes22
		}
		y.datEpochBytes1 = cs.Groups * y.datGroupBytes1
		y.datEpochBytes2 = cs.Groups * y.datGroupBytes2
		y.datTotalBytes = y.epochs1 * y.datEpochBytes1
		if y.epochs1 < y.epochs2 {
			y.datTotalBytes += y.datEpochBytes2
		}
	}
	stage6 := func() {
		var (
			withBias1 = 1 + y.slices1
			withBias2 = 1 + y.slices2
		)
		y.wtCoreBytes11 = withBias1 * y.wtSliceBytes1
		y.wtCoreBytes12 = withBias1 * y.wtSliceBytes2
		y.wtCoreBytes21 = withBias2 * y.wtSliceBytes1
		y.wtCoreBytes22 = withBias2 * y.wtSliceBytes2
		y.wtGroupBytes1 = y.wtCores1 * y.wtCoreBytes11
		y.wtGroupBytes2 = y.wtCores1 * y.wtCoreBytes21
		if y.wtCores1 < y.wtCores2 {
			y.wtGroupBytes1 += y.wtCoreBytes12
			y.wtGroupBytes2 += y.wtCoreBytes22
		}
		y.wtEpochBytes1 = cs.Groups * y.wtGroupBytes1
		y.wtEpochBytes2 = cs.Groups * y.wtGroupBytes2
		y.wtTotalBytes = y.epochs1 * y.wtEpochBytes1
		if y.epochs1 < y.epochs2 {
			y.wtTotalBytes += y.wtEpochBytes2
		}
		stage7()
	}
	stage5 := func() {
		wtSliceBytes := y.wtSliceBytes1
		if y.wtCores1 == 0 {
			wtSliceBytes = y.wtSliceBytes2
		}
		datSliceBytes := y.datSliceBytes1
		if y.datCores1 == 0 {
			datSliceBytes = y.datSliceBytes2
		}
		switch cs.platform {
		case raw.AVX512Float32:
			var (
				sliceBytes = 2*wtSliceBytes + datSliceBytes
				cacheBytes = cs.cacheBytes1 + cs.cacheBytes2
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
		stage6()
	}
	stage4 := func() {
		if special() {
			chanDats := cs.From.Height * cs.From.Width
			y.datSliceDats2 = chanDats % y.datSliceDats1
			y.datSliceSlots2 = ceilQuo(y.datSliceDats2, y.slotDats)
			y.datSliceBytes2 = y.datSliceSlots2 * y.slotBytes
			y.datCores1 = chanDats / y.datSliceDats1
			y.datCores2 = y.datCores1 + btoi(y.datSliceDats2 > 0)
		} else {
			sig := fmt.Sprint(
				"tokenize", " ",
				cs.From.Height, cs.From.Width,
				cs.StrideH, cs.StrideW,
				cs.PaddingH, cs.PaddingW,
			)
			if prior, ok := cs.dedup[sig]; ok {
				y.toks = prior.(*tokens)
			} else {
				y.toks = tokenize(cs, y.datSliceSlots1)
				cs.dedup[sig] = y.toks
			}
			y.datCores1 = y.toks.IdxPast
			y.datCores2 = y.datCores1
			var (
				sect = y.toks.Sects[len(y.toks.Sects)-1]
				tok  = sect.Uniqs[len(sect.Uniqs)-1]
			)
			if tok.Slots != y.datSliceSlots1 {
				y.datSliceSlots2 = tok.Slots
				y.datSliceDats2 = y.datSliceSlots2 * y.slotDats
				y.datSliceBytes2 = y.datSliceSlots2 * y.slotBytes
				y.datCores1--
			}
		}
		stage5()
	}
	stage3 := func() {
		y.wtSliceWts2 = y.toChans % y.wtSliceWts1
		y.wtSliceBytes2 = y.wtSliceWts2 * y.wtBytes
		y.wtCores1 = y.toChans / y.wtSliceWts1
		y.wtCores2 = y.wtCores1 + btoi(y.wtSliceWts2 > 0)
		stage4()
	}
	stage2 := func() {
		y.fromChans = cs.From.Chans / cs.Groups
		for i := range cs.Filts {
			y.toChans += cs.Filts[i].Cnt
		}
		y.toChans /= cs.Groups
		stage3()
	}
	stage1 := func() {
		switch cs.platform {
		case raw.AVX512Float32:
			y.wtBytes = 4
			y.wtSliceWts1 = 6
			y.datBytes = 4
			y.slotDats = 16
			y.datSliceSlots1 = 4
		default:
			panic("bug")
		}
		y.wtSliceBytes1 = y.wtSliceWts1 * y.wtBytes
		y.slotBytes = y.slotDats * y.datBytes
		y.datSliceDats1 = y.datSliceSlots1 * y.slotDats
		y.datSliceBytes1 = y.datSliceSlots1 * y.slotBytes
		stage2()
	}
	stage1()
	return &y
}

type ArrangeWts struct {
	*Ctx
	*Spec
	Team    cgen.Gen
	Tensors []cgen.Gen
	*layout
	callerName string
}

func (a *ArrangeWts) Prep() cgen.Gen {
	a.layout = newLayout(ctxSpec{
		Ctx:  a.Ctx,
		Spec: a.Spec,
	})
	const affix = "ArrangeWts"
	sig := fmt.Sprint(affix, " ", a.Spec)
	if prior, ok := a.dedup[sig]; ok {
		a.callerName = prior.(string)
		return nil
	}
	a.callerName = a.name(a.prefix + affix)
	a.dedup[sig] = a.callerName
	return cgen.Gens{
		&arrangeWts{ArrangeWts: a},
		cgen.Newline,
	}
}

func (a *ArrangeWts) Bytes() int {
	return a.wtTotalBytes
}

func (a *ArrangeWts) Append(to []byte) []byte {
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

type arrangeWts struct {
	*ArrangeWts
	bundleChans int
	bundleTile  int
	bundleTiles int
	bundleScrap int
	bundleHull  int
	groupTile   int
	groupTiles  int
	groupScrap  int
	groupHull   int
	calleeName  string
	bundleCoord cgen.Gen
	groupCoord  cgen.Gen
	epochCoord  cgen.Gen
	slices      int
	coreBytes   int
	groupBytes  int
	epochFirst  int
	epochCnt    int
	wtPtrs      []cgen.Gen
	biasPtrs    []cgen.Gen
	bnPtrs      [][]cgen.Gen
	arranged    cgen.Gen
	groupIdx    cgen.Gen
	filtsIdx    int
	shortCore   bool
	workChan    cgen.Gen
	workChans   int
	workCore    cgen.Gen
	workCut     cgen.Gen
	workCores   int
}

func (a *arrangeWts) Append(to []byte) []byte {
	var (
		threadWts    int
		groupBundles int
	)
	switch a.platform {
	case raw.AVX512Float32:
		a.bundleChans = 16
		threadWts = a.bundleChans * 512
	default:
		panic("bug")
	}
	if len(a.Filts) == 1 {
		groupBundles = ceilQuo(a.toChans, a.bundleChans)
	} else {
		if a.Groups != 1 {
			panic("bug")
		}
		for i := range a.Filts {
			chans := a.Filts[i].Cnt
			groupBundles += ceilQuo(chans, a.bundleChans)
		}
	}
	var (
		n1        = a.slices1 * a.epochs1
		n2        = a.slices2 * (a.epochs2 - a.epochs1)
		chanWts   = ceilQuo(n1+n2, a.epochs2)
		bundleWts = a.bundleChans * chanWts
		groupWts  = a.toChans * chanWts
	)
	switch {
	case threadWts <= bundleWts:
		a.bundleTile = 1
		a.bundleTiles = groupBundles
		a.bundleScrap = 0
		a.bundleHull = groupBundles
		a.groupTile = 1
		a.groupTiles = a.Groups
		a.groupScrap = 0
		a.groupHull = a.Groups
	case threadWts <= groupWts:
		var (
			threadBundles = ceilQuo(threadWts, bundleWts)
			fit           = max(groupBundles/threadBundles, 1)
		)
		a.bundleTile = groupBundles / fit
		a.bundleTiles = fit
		a.bundleScrap = groupBundles - fit*a.bundleTile
		a.bundleHull = fit
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
			threadGroups = ceilQuo(threadWts, groupWts)
			fit          = max(a.Groups/threadGroups, 1)
		)
		a.groupTile = a.Groups / fit
		a.groupTiles = fit
		a.groupScrap = a.Groups - fit*a.groupTile
		a.groupHull = fit
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

func (a *arrangeWts) calleeFunc() cgen.Gen {
	callee := &threader.Callee{
		Ctx:  a.tc,
		Name: a.calleeName,
		Task: vb(a.name("task")),
		Pt:   vb(a.name("pt")),
	}
	var (
		body    = make(cgen.Stmts, 6)
		tensors = vb(a.name("tensors"))
		usedPt  = false
	)
	body[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: tensors,
		Init: callee.Any(),
	}
	coord := func(hull, i int, nm string) cgen.Gen {
		if hull == 1 {
			return nil
		}
		ret := vb(a.name(nm))
		body[1+i] = cgen.Var{
			Type: cgen.PtrdiffT, What: ret,
			Init: cgen.Elem{
				Arr: callee.Pt, Idx: il(i),
			},
		}
		usedPt = true
		return ret
	}
	a.bundleCoord = coord(a.bundleHull, 0, "b")
	a.groupCoord = coord(a.groupHull, 1, "g")
	a.epochCoord = coord(a.epochs2, 2, "e")
	if !usedPt {
		body[1] = cgen.Cast{
			Type: cgen.Void,
			Expr: callee.Pt,
		}
	}
	impl := func() cgen.Gen {
		return cgen.Gens{
			a.ptrs(tensors),
			a.kernel(),
		}
	}
	if a.epochs1 > 0 {
		a.slices = a.slices1
		a.coreBytes = a.wtCoreBytes11
		a.groupBytes = a.wtGroupBytes1
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
		body[4] = put
	}
	if a.epochs1 < a.epochs2 {
		a.slices = a.slices2
		a.coreBytes = a.wtCoreBytes21
		a.groupBytes = a.wtGroupBytes2
		a.epochFirst = a.epochs1
		a.epochCnt = 1
		body[5] = impl()
	}
	return callee.Func(body)
}

func (a *arrangeWts) ptrs(tensors cgen.Gen) cgen.Gen {
	var (
		parts   = len(a.Filts)
		epoch   = a.epochCoord
		group   = a.groupCoord
		wtOff   cgen.Gen
		biasOff cgen.Gen
		preCh   cgen.Gen
		postCh  cgen.Gen
		arOff   cgen.Gen
	)
	stage6 := func() cgen.Gen {
		var (
			stmts cgen.Stmts
			next  = 0
		)
		stmt := func(s cgen.Gen) {
			stmts = append(stmts, s)
		}
		tensor := func() cgen.Gen {
			i := next
			next++
			return cgen.Elem{
				Arr: tensors,
				Idx: il(i),
			}
		}
		decl := func(what, off cgen.Gen) {
			stmt(cgen.Var{
				Type: cgen.RestrictPtrChar,
				What: what,
				Init: cgen.Add{
					Expr1: tensor(),
					Expr2: off,
				},
			})
		}
		for i := 0; i < parts; i++ {
			decl(a.wtPtrs[i], wtOff)
			if a.epochFirst == 0 {
				decl(a.biasPtrs[i], biasOff)
			} else {
				next++
			}
			split := a.Filts[i].BnPre
			for j, ptr := range a.bnPtrs[i] {
				ch := preCh
				if j >= split {
					ch = postCh
				}
				stmt(cgen.Var{
					Type: cgen.RestrictPtrChar,
					What: ptr,
					Init: &bn.Offset{
						Ctx:     a.bc,
						Mas:     tensor(),
						Channel: ch,
					},
				})
			}
		}
		decl(a.arranged, arOff)
		return stmts
	}
	stage5 := func() cgen.Gen {
		a.arranged = vb(a.name("arranged"))
		arOff = cgen.Add{
			Expr1: cgen.Mul{
				Expr1: cast(a.wtEpochBytes1),
				Expr2: epoch,
			},
			Expr2: cgen.Mul{
				Expr1: cast(a.groupBytes),
				Expr2: group,
			},
		}
		return stage6()
	}
	stage4 := func() cgen.Gen {
		a.bnPtrs = make([][]cgen.Gen, parts)
		for i := range a.bnPtrs {
			var (
				pre  = a.Filts[i].BnPre
				post = a.Filts[i].BnPost
				put  = make([]cgen.Gen, pre+post)
			)
			for j := range put {
				put[j] = vb(a.name("bnPtr"))
			}
			a.bnPtrs[i] = put
		}
		preCh = cgen.Paren{
			Inner: cgen.Add{
				Expr1: cgen.Mul{
					Expr1: cast(a.slices1),
					Expr2: epoch,
				},
				Expr2: cgen.Mul{
					Expr1: cast(a.fromChans),
					Expr2: group,
				},
			},
		}
		postCh = cgen.Mul{
			Expr1: il(a.toChans),
			Expr2: group,
		}
		return stage5()
	}
	stage3 := func() cgen.Gen {
		if a.epochFirst == 0 {
			a.biasPtrs = make([]cgen.Gen, parts)
			for i := range a.biasPtrs {
				a.biasPtrs[i] = vb(a.name("biasPtr"))
			}
			biasOff = cgen.Mul{
				Expr1: cast(a.toChans * a.wtBytes),
				Expr2: group,
			}
		} else {
			a.biasPtrs = nil
		}
		return stage4()
	}
	stage2 := func() cgen.Gen {
		a.wtPtrs = make([]cgen.Gen, parts)
		for i := range a.wtPtrs {
			a.wtPtrs[i] = vb(a.name("wtPtr"))
		}
		filtBytes := a.fromChans * a.wtBytes
		wtOff = cgen.Add{
			Expr1: cgen.Mul{
				Expr1: cast(a.slices1 * a.wtBytes),
				Expr2: epoch,
			},
			Expr2: cgen.Mul{
				Expr1: cast(a.toChans * filtBytes),
				Expr2: group,
			},
		}
		return stage3()
	}
	stage1 := func() cgen.Gen {
		if a.epochCnt == 1 {
			epoch = il(a.epochFirst)
		}
		if group == nil {
			group = il(0)
		} else {
			group = cgen.Mul{
				Expr1: il(a.groupTile),
				Expr2: group,
			}
		}
		return stage2()
	}
	return stage1()
}

func (a *arrangeWts) kernel() cgen.Gen {
	var (
		bundleIdx   cgen.Gen
		outerChan   int
		outerChans  int
		outerBundle int
		innerChan   int
		innerChans  int
		innerBundle int
	)
	layer8 := func() cgen.Gen {
		switch a.platform {
		case raw.AVX512Float32:
			return a.m512()
		default:
			panic("bug")
		}
	}
	layer7 := func() cgen.Gen {
		var (
			n     = a.wtSliceWts1
			spans = make([]int, n)
			stop  = innerChan + innerChans
			each  = a.workChans
		)
		for ch := innerChan; ch != stop; ch += each {
			cut := ch % n
			if spans[cut] != 0 {
				break
			}
			span := 1
			if fill := n - cut; each > fill {
				span += ceilQuo(each-fill, n)
			}
			spans[cut] = span
		}
		only := 0
		for _, span := range spans {
			switch {
			case span == 0:
			case only == 0:
				only = span
			case only != span:
				only = -1
			}
		}
		if only != -1 {
			a.workCores = only
			return layer8()
		}
		var (
			cases = make(cgen.Stmts, 0, n)
			cuts  = make([]int, 0, n-1)
		)
		for {
			var (
				take = 0
				last = true
			)
			for cut, span := range spans {
				switch {
				case span == 0:
				case take == 0:
					take = span
					fallthrough
				case take == span:
					spans[cut] = 0
					cuts = append(cuts, cut)
				default:
					last = false
				}
			}
			a.workCores = take
			if last {
				var assn cgen.Gen
				if len(cuts) == 1 {
					assn = cgen.Assign{
						Expr1: a.workCut,
						Expr2: il(cuts[0]),
					}
				}
				cases = append(
					cases, cgen.Case{
						Body: cgen.Stmts{
							assn,
							layer8(),
						},
					},
				)
				break
			}
			for x, cut := range cuts {
				var body cgen.Gen
				if x == len(cuts)-1 {
					body = cgen.Stmts{
						layer8(),
						cgen.Break,
					}
				}
				cases = append(
					cases, cgen.Case{
						Expr: il(cut),
						Body: body,
					},
				)
			}
			cuts = cuts[:0]
		}
		return cgen.Switch{
			Expr:  a.workCut,
			Cases: cases,
		}
	}
	layer6 := func() cgen.Gen {
		a.workCore = vb(a.name("l"))
		a.workCut = vb(a.name("cut"))
		var (
			stmts = make(cgen.Stmts, 3)
			numer = cgen.Cast{
				Type: cgen.SizeT,
				Expr: cgen.Paren{
					Inner: cgen.Add{
						Expr1: il(outerChan),
						Expr2: a.workChan,
					},
				},
			}
			denom = il(a.wtSliceWts1)
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: a.workCore,
			Init: cgen.Quo{
				Expr1: numer,
				Expr2: denom,
			},
		}
		var cut cgen.Gen
		if a.bundleChans%a.wtSliceWts1 == 0 {
			cut = il(outerChan % a.wtSliceWts1)
		} else {
			cut = cgen.Rem{
				Expr1: numer,
				Expr2: denom,
			}
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: a.workCut,
			Init: cut,
		}
		stmts[2] = layer7()
		return stmts
	}
	layer5 := func() cgen.Stmts {
		a.workChan = vb(a.name("k"))
		a.workChans = min(innerChans, a.bundleChans)
		var (
			stmts = make(cgen.Stmts, 2)
			ch    = il(innerChan - outerChan)
		)
		if a.workChans < innerChans {
			ch = addMul(
				ch,
				il(a.bundleChans),
				cgen.Paren{
					Inner: cgen.Sub{
						Expr1: bundleIdx,
						Expr2: il(innerBundle),
					},
				},
			)
		}
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: a.workChan,
			Init: ch,
		}
		stmts[1] = layer6()
		return stmts
	}
	layer4 := func() cgen.Stmts {
		var stmts cgen.Stmts
		ite := func(upper int) {
			if stmts == nil {
				stmts = layer5()
				return
			}
			stmts = cgen.Stmts{
				cgen.If{
					Cond: cgen.CmpL{
						Expr1: bundleIdx,
						Expr2: il(upper),
					},
					Then: layer5(),
					Else: stmts,
				},
			}
		}
		var (
			first  = a.toChans - a.wtSliceWts2
			past   = outerChan + outerChans
			short  = min(max(past-first, 0), outerChans)
			bunds  = (outerChans - short) / a.bundleChans
			chans1 = bunds * a.bundleChans
			chans2 = outerChans - chans1
			quo    = chans2 / a.bundleChans
			rem    = chans2 % a.bundleChans
			split  = outerBundle + bunds
		)
		if rem > 0 {
			a.shortCore = short > 0
			innerChan = past - rem
			innerChans = rem
			innerBundle = split + quo
			stmts = layer5()
		}
		if quo > 0 {
			a.shortCore = true
			innerChan = past - chans2
			innerChans = chans2 - rem
			innerBundle = split
			ite(split + quo)
		}
		if chans1 > 0 {
			a.shortCore = false
			innerChan = outerChan
			innerChans = chans1
			innerBundle = outerBundle
			ite(split)
		}
		return stmts
	}
	layer3 := func() cgen.Gen {
		parts := len(a.Filts)
		if parts == 1 {
			a.filtsIdx = 0
			outerChan = 0
			outerChans = a.toChans
			outerBundle = 0
			return layer4()
		}
		var (
			atChan = make([]int, parts+1)
			atBund = make([]int, parts+1)
		)
		for part := 0; part < parts; part++ {
			var (
				chans = a.Filts[part].Cnt
				bunds = ceilQuo(chans, a.bundleChans)
			)
			atChan[part+1] = atChan[part] + chans
			atBund[part+1] = atBund[part] + bunds
		}
		leaf := func(part int) cgen.Stmts {
			a.filtsIdx = part
			outerChan = atChan[part]
			outerChans = a.Filts[part].Cnt
			outerBundle = atBund[part]
			return layer4()
		}
		var tree func(int, int) cgen.Stmts
		tree = func(first, last int) cgen.Stmts {
			if first == last {
				return leaf(first)
			}
			var (
				start = atBund[first]
				stop  = atBund[last+1]
				upper = start + (stop-start)/2
				split = first + 1
			)
			for atBund[split+1] <= upper {
				split++
			}
			return cgen.Stmts{
				cgen.If{
					Cond: cgen.CmpL{
						Expr1: bundleIdx,
						Expr2: il(atBund[split]),
					},
					Then: tree(first, split-1),
					Else: tree(split, last),
				},
			}
		}
		return tree(0, parts-1)
	}
	layer2 := func() cgen.Gen {
		bundleIdx = vb(a.name("j"))
		var (
			past  = vb(a.name("jj"))
			first cgen.Gen
			iters cgen.Gen
		)
		if a.bundleCoord == nil {
			first = il(0)
		} else {
			first = cgen.Mul{
				Expr1: il(a.bundleTile),
				Expr2: a.bundleCoord,
			}
		}
		switch a.bundleTiles {
		case a.bundleHull:
			iters = il(a.bundleTile)
		case 0:
			iters = il(a.bundleScrap)
		default:
			iters = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: a.bundleCoord,
						Expr2: il(a.bundleTiles),
					},
					Then: il(a.bundleTile),
					Else: il(a.bundleScrap),
				},
			}
		}
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: bundleIdx,
				Init: first,
			},
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: past,
				Init: cgen.Add{
					Expr1: bundleIdx,
					Expr2: iters,
				},
			},
			cgen.For{
				Cond: cgen.CmpL{
					Expr1: bundleIdx,
					Expr2: past,
				},
				Post: cgen.IncPre{
					Expr: bundleIdx,
				},
				Body: layer3(),
			},
		}
	}
	layer1 := func() cgen.Gen {
		a.groupIdx = vb(a.name("i"))
		var (
			past  = vb(a.name("ii"))
			iters cgen.Gen
		)
		switch a.groupTiles {
		case a.groupHull:
			iters = il(a.groupTile)
		case 0:
			iters = il(a.groupScrap)
		default:
			iters = cgen.Ternary{
				Cond: cgen.CmpL{
					Expr1: a.groupCoord,
					Expr2: il(a.groupTiles),
				},
				Then: il(a.groupTile),
				Else: il(a.groupScrap),
			}
		}
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: past, Init: iters,
			},
			cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT,
					What: a.groupIdx,
					Init: il(0),
				},
				Cond: cgen.CmpL{
					Expr1: a.groupIdx,
					Expr2: past,
				},
				Post: cgen.IncPre{
					Expr: a.groupIdx,
				},
				Body: layer2(),
			},
		}
	}
	return layer1()
}

func (a *arrangeWts) m512() cgen.Gen {
	const lanes = 16
	var (
		sum        cgen.Gen
		postMul1   cgen.Gen
		cellIdx    cgen.Gen
		cellSlices int
		wts        []cgen.Gen
	)
	emit := func(what, sliceIdx cgen.Gen) cgen.Stmts {
		var (
			cores       = a.workCores
			stmts       = make(cgen.Stmts, cores)
			ae          = a.arranged
			slicePitch1 = il(a.wtSliceBytes1)
			slicePitch2 = slicePitch1
			n           = a.wtSliceWts1
		)
		ae = addMul(ae, il(a.groupBytes), a.groupIdx)
		ae = addMul(ae, il(a.coreBytes), a.workCore)
		ae = addMul(ae, il(a.wtBytes), a.workCut)
		if a.shortCore {
			slicePitch2 = il(a.wtSliceBytes2)
		}
		if cores == 1 {
			stmts[0] = avx.Mm512MaskStoreuPs{
				addMul(ae, slicePitch2, sliceIdx),
				loMask(a.workChans),
				what,
			}
			return stmts
		}
		for x := 0; x < cores-1; x++ {
			var (
				fwd = x * a.coreBytes
				bwd = x * a.wtSliceBytes1
			)
			stmts[x] = avx.Mm512MaskStoreuPs{
				cgen.Add{
					Expr1: addMul(ae, slicePitch1, sliceIdx),
					Expr2: cast(fwd - bwd),
				},
				cgen.ShiftLow{
					Expr1: il((1<<uint(n) - 1) << uint(x*n)),
					Expr2: a.workCut,
				},
				what,
			}
		}
		var (
			x   = cores - 1
			fwd = x * a.coreBytes
			bwd = x * a.wtSliceBytes1
		)
		stmts[x] = avx.Mm512MaskStoreuPs{
			cgen.Add{
				Expr1: addMul(ae, slicePitch2, sliceIdx),
				Expr2: cast(fwd - bwd),
			},
			cgen.Sub{
				Expr1: loMask(a.workChans),
				Expr2: cgen.Paren{
					Inner: cgen.ShiftLow{
						Expr1: loMask(x * n),
						Expr2: a.workCut,
					},
				},
			},
			what,
		}
		return stmts
	}
	layer7 := func() cgen.Gen {
		toMix := make([]cgen.Stmts, cellSlices)
		for x := range toMix {
			toMix[x] = emit(
				wts[x],
				cgen.Paren{
					Inner: addMul(
						il(1+x), il(lanes), cellIdx,
					),
				},
			)
		}
		return mix(toMix)
	}
	layer6 := func() cgen.Gen {
		preCnt := a.Filts[a.filtsIdx].BnPre
		if preCnt == 0 {
			return layer7()
		}
		var (
			n1      = cellSlices
			outer   = make(cgen.Gens, n1+1)
			prePtrs = a.bnPtrs[a.filtsIdx][:preCnt]
		)
		for x1 := 0; x1 < n1; x1++ {
			var (
				n2      = preCnt * 3
				inner   = make(cgen.Stmts, n2+2)
				wt      = wts[x1]
				preMul1 cgen.Gen
				preAdd1 cgen.Gen
			)
			preCh := cgen.Paren{
				Inner: addMul(
					addMul(il(x1), il(lanes), cellIdx),
					il(a.fromChans),
					a.groupIdx,
				),
			}
			for x2, prePtr := range prePtrs {
				var (
					preMul2 = vb(a.name("preMul"))
					preAdd2 = vb(a.name("preAdd"))
				)
				inner[x2*3] = &bn.Load{
					Ctx:     a.bc,
					Mas:     prePtr,
					Channel: preCh,
					Mul:     preMul2,
					Add:     preAdd2,
				}
				if x2 == 0 {
					preMul1 = preMul2
					preAdd1 = preAdd2
					continue
				}
				inner[x2*3+1] = cgen.Assign{
					Expr1: preMul1,
					Expr2: avx.Mm512MulPs{
						preMul1, preMul2,
					},
				}
				inner[x2*3+2] = cgen.Assign{
					Expr1: preAdd1,
					Expr2: avx.Mm512FmaddPs{
						preAdd1, preMul2, preAdd2,
					},
				}
			}
			inner[n2] = cgen.Assign{
				Expr1: sum,
				Expr2: avx.Mm512FmaddPs{
					wt, preAdd1, sum,
				},
			}
			inner[n2+1] = cgen.Assign{
				Expr1: wt,
				Expr2: avx.Mm512MulPs{
					wt, preMul1,
				},
			}
			outer[x1] = inner
		}
		outer[n1] = layer7()
		return outer
	}
	layer5 := func() cgen.Gen {
		if postMul1 == nil {
			return layer6()
		}
		var (
			n     = cellSlices
			stmts = make(cgen.Stmts, n+1)
		)
		for x := 0; x < n; x++ {
			wt := wts[x]
			stmts[x] = cgen.Assign{
				Expr1: wt,
				Expr2: avx.Mm512MulPs{
					wt, postMul1,
				},
			}
		}
		stmts[n] = layer6()
		return stmts
	}
	layer4 := func() cgen.Gen {
		var (
			rows = a.workChans
			cols = cellSlices
		)
		wts = make([]cgen.Gen, max(rows, cols))
		for x := range wts {
			wts[x] = vb(a.name("wt"))
		}
		var (
			stmts      = make(cgen.Stmts, rows+2)
			mask       = loMask(cols)
			ae         = a.wtPtrs[a.filtsIdx]
			filtBytes  = a.fromChans * a.wtBytes
			groupPitch = il(a.toChans * filtBytes)
			chanPitch  = il(filtBytes)
			cellPitch  = il(lanes * a.wtBytes)
		)
		ae = addMul(ae, groupPitch, a.groupIdx)
		ae = addMul(ae, chanPitch, a.workChan)
		ae = addMul(ae, cellPitch, cellIdx)
		for x := 0; x < rows; x++ {
			stmts[x] = cgen.Var{
				Type: avx.M512, What: wts[x],
				Init: avx.Mm512MaskzLoaduPs{
					mask,
					cgen.Add{
						Expr1: ae,
						Expr2: cast(x * filtBytes),
					},
				},
			}
		}
		stmts[rows] = &trans.Pose{
			Platform: a.platform,
			Nms:      a.nms,
			Rows:     rows,
			Cols:     cols,
			Vars:     wts,
		}
		stmts[rows+1] = layer5()
		return stmts
	}
	layer3 := func() cgen.Gen {
		cellIdx = vb(a.name("c"))
		var (
			stmts = make(cgen.Stmts, 3)
			quo   = a.slices / lanes
			rem   = a.slices % lanes
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: cellIdx,
			Init: il(0),
		}
		if quo > 0 {
			cellSlices = lanes
			stmts[1] = cgen.For{
				Cond: cgen.CmpNE{
					Expr1: cellIdx,
					Expr2: il(quo),
				},
				Post: cgen.IncPre{
					Expr: cellIdx,
				},
				Body: layer4(),
			}
		}
		if rem > 0 {
			cellSlices = rem
			stmts[2] = layer4()
		}
		return stmts
	}
	layer2 := func() cgen.Gen {
		var (
			outer   = make(cgen.Gens, 3)
			filts   = &a.Filts[a.filtsIdx]
			preCnt  = filts.BnPre
			postCnt = filts.BnPost
		)
		if postCnt > 0 {
			var (
				inner    = make(cgen.Stmts, postCnt*3)
				postPtrs = a.bnPtrs[a.filtsIdx][preCnt:]
			)
			postCh := cgen.Paren{
				Inner: addMul(
					a.workChan,
					il(a.toChans),
					a.groupIdx,
				),
			}
			for x, postPtr := range postPtrs {
				var (
					postMul2 = vb(a.name("postMul"))
					postAdd  = vb(a.name("postAdd"))
				)
				inner[x*3] = &bn.Load{
					Ctx:     a.bc,
					Mas:     postPtr,
					Channel: postCh,
					Mul:     postMul2,
					Add:     postAdd,
					Cnt:     a.workChans,
				}
				if x == 0 {
					postMul1 = postMul2
				} else {
					inner[x*3+1] = cgen.Assign{
						Expr1: postMul1,
						Expr2: avx.Mm512MulPs{
							postMul1, postMul2,
						},
					}
				}
				var stmt cgen.Gen
				if a.epochFirst == 0 {
					stmt = cgen.Assign{
						Expr1: sum,
						Expr2: avx.Mm512FmaddPs{
							sum, postMul2, postAdd,
						},
					}
					if a.epochCnt > 1 {
						stmt = cgen.If1{
							Cond: cgen.IsZero{
								Expr: a.epochCoord,
							},
							Then: stmt,
						}
					}
				} else {
					stmt = cgen.Cast{
						Type: cgen.Void,
						Expr: postAdd,
					}
				}
				inner[x*3+2] = stmt
			}
			outer[0] = inner
		}
		if preCnt > 0 {
			outer[1] = layer3()
			outer[2] = emit(sum, il(0))
		} else {
			outer[1] = emit(sum, il(0))
			outer[2] = layer3()
		}
		return outer
	}
	layer1 := func() cgen.Gen {
		sum = vb(a.name("sum"))
		var (
			decl = cgen.Var{
				Type: avx.M512, What: sum,
			}
			bias cgen.Gen
			assn cgen.Gen
		)
		if a.epochFirst == 0 {
			var (
				ae         = a.biasPtrs[a.filtsIdx]
				groupPitch = il(a.toChans * a.wtBytes)
				chanPitch  = il(a.wtBytes)
				mask       = loMask(a.workChans)
			)
			ae = addMul(ae, groupPitch, a.groupIdx)
			ae = addMul(ae, chanPitch, a.workChan)
			bias = avx.Mm512MaskzLoaduPs{
				mask, ae,
			}
		}
		switch {
		case bias == nil:
			decl.Init = avx.Mm512SetzeroPs
		case a.epochCnt == 1:
			decl.Init = bias
		default:
			assn = cgen.If{
				Cond: cgen.IsZero{
					Expr: a.epochCoord,
				},
				Then: cgen.Stmts{cgen.Assign{
					Expr1: sum,
					Expr2: bias,
				}},
				Else: cgen.Stmts{cgen.Assign{
					Expr1: sum,
					Expr2: avx.Mm512SetzeroPs,
				}},
			}
		}
		return cgen.Stmts{
			decl,
			assn,
			layer2(),
		}
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
	a.layout = newLayout(ctxSpec{
		Ctx:  a.Ctx,
		Spec: a.Spec,
	})
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
	sliceCoord  cgen.Gen
	coreCoord   cgen.Gen
	groupCoord  cgen.Gen
	epochCoord  cgen.Gen
	sliceTile   int
	sliceScrap  int
	coreBytes   int
	groupBytes  int
	epochFirst  int
	epochCnt    int
	datPtrs     []cgen.Gen
	bnPtrs      []cgen.Gen
	arranged    cgen.Gen
	groupIdx    cgen.Gen
	coreIdx     cgen.Gen
	short       bool
	sectH       cgen.Gen
	tok         *token
	sliceIdx    cgen.Gen
}

func (a *arrangeDats) Append(to []byte) []byte {
	var (
		threadSlots int
		sliceSlots  = a.datSliceSlots1
	)
	switch a.platform {
	case raw.AVX512Float32:
		threadSlots = 512
	default:
		panic("bug")
	}
	if a.datCores1 == 0 {
		sliceSlots = a.datSliceSlots2
	}
	var (
		threadSlices = ceilQuo(threadSlots, sliceSlots)
		coreSlices   = a.slices1
	)
	switch {
	case a.epochs1 == a.epochs2:
	case a.epochs1 == 0 || a.slices1 > a.slices2:
		coreSlices = a.slices2
	}
	a.coreTile = 1
	a.coreTiles = a.datCores2
	a.coreScrap = 0
	a.coreHull = a.coreTiles
	a.groupTile = 1
	a.groupTiles = a.Groups
	a.groupScrap = 0
	a.groupHull = a.groupTiles
	if threadSlices < coreSlices {
		var (
			fit   = coreSlices / threadSlices
			tiles = fit - 1
		)
		a.sliceTile1 = a.slices1 / fit
		a.sliceTile2 = a.slices2 / fit
		a.sliceTiles = tiles
		a.sliceScrap1 = a.slices1 - tiles*a.sliceTile1
		a.sliceScrap2 = a.slices2 - tiles*a.sliceTile2
		a.sliceHull = fit
	} else {
		a.sliceTile1 = a.slices1
		a.sliceTile2 = a.slices2
		a.sliceTiles = 1
		a.sliceScrap1 = 0
		a.sliceScrap2 = 0
		a.sliceHull = 1
		var (
			threadCores = ceilQuo(threadSlices, coreSlices)
			groupCores  = a.datCores2
		)
		if threadCores < groupCores {
			fit := groupCores / threadCores
			a.coreTile = groupCores / fit
			a.coreTiles = fit
			a.coreScrap = groupCores - fit*a.coreTile
			a.coreHull = fit
			if a.coreScrap > 0 {
				a.coreTiles--
				a.coreScrap += a.coreTile
			}
		} else {
			a.coreTile = groupCores
			a.coreTiles = 1
			a.coreScrap = 0
			a.coreHull = 1
			var (
				threadGroups = ceilQuo(threadCores, groupCores)
				epochGroups  = a.Groups
			)
			if threadGroups < epochGroups {
				fit := epochGroups / threadGroups
				a.groupTile = epochGroups / fit
				a.groupTiles = fit
				a.groupScrap = epochGroups - fit*a.groupTile
				a.groupHull = fit
				if a.groupScrap > 0 {
					a.groupTiles--
					a.groupScrap += a.groupTile
				}
			} else {
				a.groupTile = epochGroups
				a.groupTiles = 1
				a.groupScrap = 0
				a.groupHull = 1
			}
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
		body    = make(cgen.Stmts, 7)
		tensors = vb(a.name("tensors"))
		usedPt  = false
	)
	body[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: tensors,
		Init: callee.Any(),
	}
	coord := func(hull, i int, nm string) cgen.Gen {
		if hull == 1 {
			return nil
		}
		ret := vb(a.name(nm))
		body[1+i] = cgen.Var{
			Type: cgen.PtrdiffT, What: ret,
			Init: cgen.Elem{
				Arr: callee.Pt, Idx: il(i),
			},
		}
		usedPt = true
		return ret
	}
	a.sliceCoord = coord(a.sliceHull, 0, "s")
	a.coreCoord = coord(a.coreHull, 1, "c")
	a.groupCoord = coord(a.groupHull, 2, "g")
	a.epochCoord = coord(a.epochs2, 3, "e")
	if !usedPt {
		body[1] = cgen.Cast{
			Type: cgen.Void,
			Expr: callee.Pt,
		}
	}
	impl := func() cgen.Gen {
		return cgen.Gens{
			a.ptrs(tensors),
			a.kernel(),
		}
	}
	if a.epochs1 > 0 {
		a.sliceTile = a.sliceTile1
		a.sliceScrap = a.sliceScrap1
		a.coreBytes = a.datCoreBytes11
		a.groupBytes = a.datGroupBytes1
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
		a.sliceTile = a.sliceTile2
		a.sliceScrap = a.sliceScrap2
		a.coreBytes = a.datCoreBytes21
		a.groupBytes = a.datGroupBytes2
		a.epochFirst = a.epochs1
		a.epochCnt = 1
		body[6] = impl()
	}
	return callee.Func(body)
}

func (a *arrangeDats) ptrs(tensors cgen.Gen) cgen.Gen {
	var (
		epoch    = a.epochCoord
		group    = a.groupCoord
		datCnt   = len(a.From.Pitch1Bytes)
		bnCnt    = 0
		datExprs []cgen.Gen
		bnExpr   cgen.Gen
		arExpr   cgen.Gen
	)
	stage5 := func() cgen.Gen {
		var (
			stmtCnt   = datCnt + bnCnt + 1
			stmts     = make(cgen.Stmts, stmtCnt)
			stmtIdx   = 0
			tensorIdx = 0
			datIdx    = 0
			bnIdx     = 0
		)
		stmt := func(s cgen.Gen) {
			stmts[stmtIdx] = s
			stmtIdx++
		}
		tensor := func() cgen.Gen {
			i := tensorIdx
			tensorIdx++
			return cgen.Elem{
				Arr: tensors,
				Idx: il(i),
			}
		}
		dp := func() {
			i := datIdx
			datIdx++
			stmt(cgen.Var{
				Type: cgen.RestrictPtrChar,
				What: a.datPtrs[i],
				Init: cgen.Add{
					Expr1: tensor(),
					Expr2: datExprs[i],
				},
			})
		}
		ndp := func(n int) {
			for ; n > 0; n-- {
				dp()
			}
		}
		bp := func() {
			i := bnIdx
			bnIdx++
			stmt(cgen.Var{
				Type: cgen.RestrictPtrChar,
				What: a.bnPtrs[i],
				Init: &bn.Offset{
					Ctx:     a.bc,
					Mas:     tensor(),
					Channel: bnExpr,
				},
			})
		}
		dp()
		for i := range a.From.Ops {
			op := &a.From.Ops[i]
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
		stmt(cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: a.arranged,
			Init: cgen.Add{
				Expr1: tensor(),
				Expr2: arExpr,
			},
		})
		return stmts
	}
	stage4 := func() cgen.Gen {
		a.arranged = vb(a.name("arranged"))
		arExpr = cgen.Add{
			Expr1: cgen.Mul{
				Expr1: cast(a.datEpochBytes1),
				Expr2: epoch,
			},
			Expr2: cgen.Mul{
				Expr1: cast(a.groupBytes),
				Expr2: group,
			},
		}
		return stage5()
	}
	stage3 := func() cgen.Gen {
		a.bnPtrs = make([]cgen.Gen, bnCnt)
		for i := range a.bnPtrs {
			a.bnPtrs[i] = vb(a.name("bnPtr"))
		}
		bnExpr = cgen.Paren{
			Inner: cgen.Add{
				Expr1: cgen.Mul{
					Expr1: cast(a.slices1),
					Expr2: epoch,
				},
				Expr2: cgen.Mul{
					Expr1: cast(a.fromChans),
					Expr2: group,
				},
			},
		}
		return stage4()
	}
	stage2 := func() cgen.Gen {
		a.datPtrs = make([]cgen.Gen, datCnt)
		for i := range a.datPtrs {
			a.datPtrs[i] = vb(a.name("datPtr"))
		}
		datExprs = make([]cgen.Gen, datCnt)
		for i := range datExprs {
			var (
				pitch1     = a.From.Pitch1Bytes[i]
				pitch2     = a.From.Pitch2Bytes[i]
				padH       = a.PaddingH * pitch1
				padW       = a.PaddingW * a.datBytes
				expr       = cast(-padH + -padW)
				epochPitch = cast(a.slices1 * pitch2)
				groupPitch = cast(a.fromChans * pitch2)
			)
			expr = addMul(expr, epochPitch, epoch)
			expr = addMul(expr, groupPitch, group)
			datExprs[i] = expr
		}
		return stage3()
	}
	stage1 := func() cgen.Gen {
		if a.epochCnt == 1 {
			epoch = il(a.epochFirst)
		}
		if group == nil {
			group = il(0)
		} else {
			group = cgen.Mul{
				Expr1: il(a.groupTile),
				Expr2: group,
			}
		}
		for i := range a.From.Ops {
			if a.From.Ops[i].Kind == mod.Bn {
				bnCnt++
			}
		}
		return stage2()
	}
	return stage1()
}

func (a *arrangeDats) kernel() cgen.Gen {
	var (
		gotoNext cgen.Gen
	)
	layer6 := func() cgen.Gen {
		if a.toks == nil {
			return a.special()
		}
		return a.general()
	}
	layer5 := func() cgen.Gen {
		a.sliceIdx = vb(a.name("k"))
		var (
			stmts = make(cgen.Stmts, 3)
			first cgen.Gen
			iters cgen.Gen
			past  = vb(a.name("kk"))
		)
		if a.sliceCoord == nil {
			first = il(0)
		} else {
			first = cgen.Mul{
				Expr1: il(a.sliceTile),
				Expr2: a.sliceCoord,
			}
		}
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: a.sliceIdx,
			Init: first,
		}
		switch {
		case a.sliceTiles == a.sliceHull:
			iters = il(a.sliceTile)
		case a.sliceTiles == 0:
			fallthrough
		case a.sliceTile == a.sliceScrap:
			iters = il(a.sliceScrap)
		default:
			iters = cgen.Paren{
				Inner: cgen.Ternary{
					Cond: cgen.CmpL{
						Expr1: a.sliceCoord,
						Expr2: il(a.sliceTiles),
					},
					Then: il(a.sliceTile),
					Else: il(a.sliceScrap),
				},
			}
		}
		stmts[1] = cgen.Var{
			Type: cgen.PtrdiffT, What: past,
			Init: cgen.Add{
				Expr1: a.sliceIdx,
				Expr2: iters,
			},
		}
		stmts[2] = cgen.For{
			Cond: cgen.CmpL{
				Expr1: a.sliceIdx,
				Expr2: past,
			},
			Post: cgen.IncPre{
				Expr: a.sliceIdx,
			},
			Body: layer6(),
		}
		return stmts
	}
	layer4Special := func() cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if a.datCores1 > 0 {
			a.short = false
			stmts[0] = cgen.For{
				Cond: cgen.CmpNE{
					Expr1: a.coreIdx,
					Expr2: il(a.datCores1),
				},
				Post: cgen.IncPre{
					Expr: a.coreIdx,
				},
				Body: cgen.Stmts{
					layer5(),
					gotoNext,
				},
			}
		}
		if a.datCores1 < a.datCores2 {
			a.short = true
			stmts[1] = layer5()
		}
		return stmts
	}
	layer4General := func() cgen.Gen {
		leaf := func(sect *section) cgen.Stmts {
			var (
				decl  cgen.Gen
				which cgen.Gen
				n     = len(sect.Uniqs)
				cases = make(cgen.Stmts, n)
			)
			which = cgen.Sub{
				Expr1: cgen.Cast{
					Type: cgen.SizeT,
					Expr: a.coreIdx,
				},
				Expr2: il(sect.IdxFirst),
			}
			if sect.FromWrap == 0 {
				a.sectH = cast(sect.FromBase)
				for x, tok := range sect.Uniqs {
					a.tok = tok
					var (
						expr cgen.Gen
						body = make(cgen.Stmts, 3)
					)
					if x < n-1 {
						expr = il(x)
					}
					body[0] = cgen.Assign{
						Expr1: a.coreIdx,
						Expr2: il(sect.IdxFirst + x),
					}
					body[1] = layer5()
					body[2] = gotoNext
					cases[x] = cgen.Case{
						Expr: expr,
						Body: body,
					}
				}
			} else {
				which = cgen.Paren{
					Inner: which,
				}
				a.sectH = vb(a.name("h"))
				decl = cgen.Var{
					Type: cgen.PtrdiffT, What: a.sectH,
					Init: addMul(
						il(sect.FromBase),
						cgen.Quo{
							Expr1: which,
							Expr2: il(n),
						},
						il(sect.FromWrap),
					),
				}
				which = cgen.Rem{
					Expr1: which,
					Expr2: il(n),
				}
				var (
					wrap = cgen.Label(a.name("wrap"))
					last = sect.IdxPast - 1
					at   = (last - sect.IdxFirst) % n
				)
				for x, tok := range sect.Uniqs {
					a.tok = tok
					var (
						expr cgen.Gen
						body = make(cgen.Stmts, 7)
					)
					if x < n-1 {
						expr = il(x)
					}
					if x == 0 {
						body[0] = wrap
					}
					body[1] = layer5()
					body[2] = gotoNext
					if x == at {
						body[3] = cgen.If1{
							Cond: cgen.CmpGE{
								Expr1: a.coreIdx,
								Expr2: il(last),
							},
							Then: cgen.Break,
						}
					}
					body[4] = cgen.IncPre{
						Expr: a.coreIdx,
					}
					if x == n-1 {
						body[5] = cgen.AddAssign{
							Expr1: a.sectH,
							Expr2: il(sect.FromWrap),
						}
						body[6] = cgen.Goto(wrap)
					}
					cases[x] = cgen.Case{
						Expr: expr,
						Body: body,
					}
				}
			}
			return cgen.Stmts{
				decl,
				cgen.Switch{
					Expr:  which,
					Cases: cases,
				},
				cgen.Assign{
					Expr1: a.coreIdx,
					Expr2: il(sect.IdxPast),
				},
			}
		}
		var (
			sects = a.toks.Sects
			tree  func(int, int) cgen.Stmts
		)
		tree = func(first, last int) cgen.Stmts {
			if first == last {
				return leaf(sects[first])
			}
			var (
				start = sects[first].IdxFirst
				stop  = sects[last].IdxPast
				split = start + (stop-start)/2
				x     = first + 1
			)
			for sects[x].IdxPast <= split {
				x++
			}
			return cgen.Stmts{
				cgen.If{
					Cond: cgen.CmpL{
						Expr1: a.coreIdx,
						Expr2: il(sects[x].IdxFirst),
					},
					Then: tree(first, x-1),
				},
				tree(x, last),
			}
		}
		return tree(0, len(sects)-1)
	}
	layer3 := func() cgen.Gen {
		if a.toks == nil {
			return layer4Special()
		}
		return layer4General()
	}
	layer2 := func() cgen.Gen {
		a.coreIdx = vb(a.name("j"))
		var (
			stmts = make(cgen.Stmts, 4)
			first cgen.Gen
		)
		if a.coreCoord == nil {
			first = il(0)
		} else {
			first = cgen.Mul{
				Expr1: il(a.coreTile),
				Expr2: a.coreCoord,
			}
		}
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: a.coreIdx,
			Init: first,
		}
		if a.coreCoord != nil {
			var (
				last = vb(a.name("jj"))
				expr cgen.Gen
			)
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
				Type: cgen.PtrdiffT, What: last,
				Init: cgen.Add{
					Expr1: a.coreIdx,
					Expr2: expr,
				},
			}
			next := cgen.Label(a.name("next"))
			gotoNext = cgen.If1{
				Cond: cgen.CmpGE{
					Expr1: a.coreIdx,
					Expr2: last,
				},
				Then: cgen.Goto(next),
			}
			stmts[3] = next
		}
		stmts[2] = layer3()
		return stmts
	}
	layer1 := func() cgen.Gen {
		a.groupIdx = vb(a.name("i"))
		var (
			past  = vb(a.name("ii"))
			iters cgen.Gen
		)
		switch a.groupTiles {
		case a.groupHull:
			iters = il(a.groupTile)
		case 0:
			iters = il(a.groupScrap)
		default:
			iters = cgen.Ternary{
				Cond: cgen.CmpL{
					Expr1: a.groupCoord,
					Expr2: il(a.groupTiles),
				},
				Then: il(a.groupTile),
				Else: il(a.groupScrap),
			}
		}
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: past, Init: iters,
			},
			cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT,
					What: a.groupIdx,
					Init: il(0),
				},
				Cond: cgen.CmpL{
					Expr1: a.groupIdx,
					Expr2: past,
				},
				Post: cgen.IncPre{
					Expr: a.groupIdx,
				},
				Body: layer2(),
			},
		}
	}
	return layer1()
}

func (a *arrangeDats) special() cgen.Gen {
	switch a.platform {
	case raw.AVX512Float32:
		return a.m512Special()
	default:
		panic("bug")
	}
}

func (a *arrangeDats) general() cgen.Gen {
	switch a.platform {
	case raw.AVX512Float32:
		return a.m512General()
	default:
		panic("bug")
	}
}

func (a *arrangeDats) m512Special() cgen.Gen {
	var (
		bnMuls  []cgen.Gen
		bnAdds  []cgen.Gen
		slotIdx int
		mask    cgen.Gen
		stmts   cgen.Stmts
		slot    cgen.Gen
	)
	stmt := func(s cgen.Gen) {
		stmts = append(stmts, s)
	}
	datLoad := func(x int) cgen.Gen {
		var (
			dat        = vb(a.name("dat"))
			ae         = a.datPtrs[x]
			pitch2     = a.From.Pitch2Bytes[x]
			groupPitch = il(a.fromChans * pitch2)
			corePitch  = il(a.datSliceBytes1)
			slicePitch = il(pitch2)
		)
		ae = addMul(ae, groupPitch, a.groupIdx)
		ae = addMul(ae, corePitch, a.coreIdx)
		ae = addMul(ae, slicePitch, a.sliceIdx)
		ae = cgen.Add{
			Expr1: ae,
			Expr2: cast(slotIdx * a.slotBytes),
		}
		stmt(cgen.Var{
			Type: avx.M512, What: dat,
			Init: avx.Mm512MaskzLoaduPs{
				mask, ae,
			},
		})
		return dat
	}
	inner3 := func() {
		var (
			datIdx = 1
			bnIdx  = 0
		)
		for op := range a.From.Ops {
			op := &a.From.Ops[op]
			switch op.Kind {
			case mod.Add:
				var (
					n  = 1 + op.Int
					ds = make([]cgen.Gen, n)
				)
				ds[0] = slot
				for x := 1; x < n; x++ {
					ds[x] = datLoad(datIdx)
					datIdx++
				}
				for n > 1 {
					fold := n >> 1
					n -= fold
					for x := 0; x < fold; x++ {
						keep := ds[x]
						stmt(cgen.Assign{
							Expr1: keep,
							Expr2: avx.Mm512AddPs{
								keep, ds[n+x],
							},
						})
					}
				}
			case mod.Bn:
				x := bnIdx
				bnIdx++
				var (
					bnMul = bnMuls[x]
					bnAdd = bnAdds[x]
				)
				if bnMul == nil {
					bnMul = vb(a.name("bnMul"))
					bnAdd = vb(a.name("bnAdd"))
					bnMuls[x] = bnMul
					bnAdds[x] = bnAdd
					stmt(&bn.Load{
						Ctx: a.bc,
						Mas: a.bnPtrs[x],
						Channel: cgen.Paren{
							Inner: addMul(
								a.sliceIdx,
								il(a.fromChans),
								a.groupIdx,
							),
						},
						Mul: bnMul,
						Add: bnAdd,
					})
				} else {
					stmt(nil)
				}
				stmt(&bn.Apply{
					Ctx: a.bc,
					Mul: bnMul,
					Add: bnAdd,
					To:  slot,
				})
			case mod.ReLU:
				stmt(&act.ReLU{
					Ctx:      a.ac,
					NegSlope: op.Float,
					Var:      slot,
				})
			default:
				panic("bug")
			}
		}
	}
	inner2 := func() {
		inner3()
		var (
			ae         = a.arranged
			groupPitch = il(a.groupBytes)
			corePitch  = il(a.coreBytes)
			slicePitch = il(a.datSliceBytes1)
		)
		if a.short {
			slicePitch = il(a.datSliceBytes2)
		}
		ae = addMul(ae, groupPitch, a.groupIdx)
		ae = addMul(ae, corePitch, a.coreIdx)
		ae = addMul(ae, slicePitch, a.sliceIdx)
		ae = cgen.Add{
			Expr1: ae,
			Expr2: cast(slotIdx * a.slotBytes),
		}
		stmt(avx.Mm512MaskStoreuPs{
			ae, mask, slot,
		})
	}
	inner1 := func() cgen.Stmts {
		stmts = nil
		slot = datLoad(0)
		inner2()
		return stmts
	}
	outer2 := func() cgen.Gen {
		var (
			ns = a.datSliceSlots1
			nd = a.datSliceDats1
		)
		if a.short {
			ns = a.datSliceSlots2
			nd = a.datSliceDats2
		}
		toMix := make([]cgen.Stmts, ns)
		for x := range toMix {
			switch slotIdx = x; x {
			case ns - 1:
				rem := nd - x*a.slotDats
				mask = loMask(rem)
			case 0:
				mask = loMask(a.slotDats)
			}
			toMix[x] = inner1()
		}
		return mix(toMix)
	}
	outer1 := func() cgen.Gen {
		n := len(a.bnPtrs)
		bnMuls = make([]cgen.Gen, n)
		bnAdds = make([]cgen.Gen, n)
		return outer2()
	}
	return outer1()
}

func (a *arrangeDats) m512General() cgen.Gen {
	const lanes = 16
	var (
		stmts   cgen.Stmts
		bnMuls  []cgen.Gen
		bnAdds  []cgen.Gen
		bnSplit = 0
		opSplit = 0
		slots   []cgen.Gen
		dats    []cgen.Gen
	)
	stmt := func(s cgen.Gen) {
		stmts = append(stmts, s)
	}
	eval := func(id int, expr cgen.Gen) {
		for id >= len(dats) {
			dats = append(dats, nil)
		}
		dat := dats[id]
		if dat == nil {
			dat = vb(a.name("dat"))
			dats[id] = dat
			stmt(cgen.Var{
				Type: avx.M512, What: dat,
				Init: expr,
			})
		} else {
			stmt(cgen.Assign{
				Expr1: dat,
				Expr2: expr,
			})
		}
	}
	rot := func(cmd *m512CmdRotate) {
		var (
			dst = cmd.DstId
			src = dats[cmd.SrcId]
			cnt = il(cmd.Cnt)
			via = vb(a.name("via"))
		)
		stmt(cgen.Var{
			Type: avx.M512i, What: via,
			Init: avx.Mm512CastpsSi512{src},
		})
		stmt(cgen.Assign{
			Expr1: via,
			Expr2: avx.Mm512AlignrEpi32{
				via, via, cnt,
			},
		})
		eval(dst, avx.Mm512Castsi512Ps{via})
	}
	blend := func(cmd *m512CmdBlend) {
		var (
			dst   = dats[cmd.DstId]
			src   = dats[cmd.SrcId]
			mask1 = 1<<uint(cmd.Cnt) - 1
			mask2 = mask1 << uint(cmd.Off)
		)
		stmt(cgen.Assign{
			Expr1: dst,
			Expr2: avx.Mm512MaskMovPs{
				dst, il(mask2), src,
			},
		})
	}
	ctrl := func(off, inc int) cgen.Gen {
		var (
			pm  = vb(a.name("pm"))
			set = make(avx.Mm512SetEpi32, lanes)
			x   = 0
		)
		for lane := 0; lane < lanes; lane++ {
			if lane > off {
				was := x
				x += inc
				if was < lanes && x > lanes {
					x = lanes
				}
			}
			var entry cgen.Gen
			if x == 0 || x >= 2*lanes {
				entry = cgen.Zero
			} else {
				entry = il(x)
			}
			set[lanes-1-lane] = entry
		}
		stmt(cgen.Var{
			Type: avx.M512i, What: pm,
			Init: set,
		})
		return pm
	}
	perm1 := func(cmd *m512CmdPermute1) {
		var (
			dst = cmd.DstId
			src = dats[cmd.SrcId]
			pm  = ctrl(cmd.Off, cmd.Inc)
		)
		eval(dst, avx.Mm512PermutexvarPs{
			pm, src,
		})
	}
	perm2 := func(cmd *m512CmdPermute2) {
		var (
			dst  = cmd.DstId
			src1 = dats[cmd.SrcId1]
			src2 = dats[cmd.SrcId2]
			pm   = ctrl(cmd.Off, cmd.Inc)
		)
		eval(dst, avx.Mm512Permutex2varPs{
			src1, pm, src2,
		})
	}
	datLoad := func(x, relH, w, cnt int) cgen.Gen {
		var (
			ae         = a.datPtrs[x]
			pitch1     = a.From.Pitch1Bytes[x]
			pitch2     = a.From.Pitch2Bytes[x]
			groupPitch = il(a.fromChans * pitch2)
			h          = a.tok.From.FirstH + relH
		)
		ae = addMul(ae, groupPitch, a.groupIdx)
		ae = addMul(ae, il(pitch1), a.sectH)
		ae = addMul(ae, il(pitch2), a.sliceIdx)
		ae = cgen.Add{
			Expr1: ae,
			Expr2: cast(h*pitch1 + w*a.datBytes),
		}
		if cnt == 0 {
			return avx.Mm512Set1Ps{
				cgen.At{Expr: cgen.Cast{
					Type: cgen.PtrFloat,
					Expr: cgen.Paren{Inner: ae},
				}},
			}
		}
		return avx.Mm512MaskzLoaduPs{
			loMask(cnt), ae,
		}
	}
	load := func(cmd *m512CmdLoad) {
		var (
			dst  = cmd.Id
			relH = cmd.RelH
			w    = cmd.W
			cnt  = cmd.Cnt
		)
		eval(dst, datLoad(0, relH, w, cnt))
	}
	bnLoad := func(x int) {
		if bnMuls[x] != nil {
			return
		}
		var (
			bnMul = vb(a.name("bnMul"))
			bnAdd = vb(a.name("bnAdd"))
		)
		stmt(&bn.Load{
			Ctx: a.bc,
			Mas: a.bnPtrs[x],
			Channel: cgen.Paren{
				Inner: addMul(
					a.sliceIdx,
					il(a.fromChans),
					a.groupIdx,
				),
			},
			Mul: bnMul,
			Add: bnAdd,
		})
		bnMuls[x] = bnMul
		bnAdds[x] = bnAdd
	}
	modAddPre := func(cmd *m512CmdFromModAddPre) {
		var (
			dst    = dats[cmd.Id]
			relH   = cmd.RelH
			w      = cmd.W
			cnt    = cmd.Cnt
			datIdx = 1
			bnIdx  = 0
		)
		for op := 0; op < opSplit; op++ {
			op := &a.From.Ops[op]
			switch op.Kind {
			case mod.Add:
				var (
					n  = 1 + op.Int
					ds = make([]cgen.Gen, n)
				)
				ds[0] = dst
				for x := 1; x < n; x++ {
					dat := vb(a.name("dat"))
					ds[x] = dat
					stmt(cgen.Var{
						Type: avx.M512, What: dat,
						Init: datLoad(
							datIdx, relH, w, cnt,
						),
					})
					datIdx++
				}
				for n > 1 {
					fold := n >> 1
					n -= fold
					for x := 0; x < fold; x++ {
						keep := ds[x]
						stmt(cgen.Assign{
							Expr1: keep,
							Expr2: avx.Mm512AddPs{
								keep, ds[n+x],
							},
						})
					}
				}
			case mod.Bn:
				n := cnt
				if n == 0 {
					n = lanes
				}
				bnLoad(bnIdx)
				stmt(&bn.Apply{
					Ctx:  a.bc,
					Mul:  bnMuls[bnIdx],
					Add:  bnAdds[bnIdx],
					To:   dst,
					Mask: loMask(n),
				})
				bnIdx++
			case mod.ReLU:
				stmt(&act.ReLU{
					Ctx:      a.ac,
					NegSlope: op.Float,
					Var:      dst,
				})
			default:
				panic("bug")
			}
		}
	}
	modPostAdd := func(cmd *m512CmdFromModPostAdd) {
		var (
			dst   = dats[cmd.Id]
			mask  = il(cmd.Mask)
			bnIdx = bnSplit
			ops   = a.From.Ops[opSplit:]
		)
		for op := range ops {
			op := &ops[op]
			switch op.Kind {
			case mod.Bn:
				bnLoad(bnIdx)
				stmt(&bn.Apply{
					Ctx:  a.bc,
					Mul:  bnMuls[bnIdx],
					Add:  bnAdds[bnIdx],
					To:   dst,
					Mask: mask,
				})
				bnIdx++
			case mod.ReLU:
				stmt(&act.ReLU{
					Ctx:      a.ac,
					NegSlope: op.Float,
					Var:      dst,
				})
			default:
				panic("bug")
			}
		}
	}
	stage3 := func() {
		for _, cmd := range a.tok.From.Cmds {
			switch cmd := cmd.(type) {
			case *m512CmdZero:
				eval(cmd.Id, avx.Mm512SetzeroPs)
			case *m512CmdRotate:
				rot(cmd)
			case *m512CmdBlend:
				blend(cmd)
			case *m512CmdPermute1:
				perm1(cmd)
			case *m512CmdPermute2:
				perm2(cmd)
			case *m512CmdLoad:
				load(cmd)
			case *m512CmdFromModAddPre:
				modAddPre(cmd)
			case *m512CmdFromModPostAdd:
				modPostAdd(cmd)
			case *m512CmdSlotPut:
				slots[cmd.Slot] = dats[cmd.Id]
			default:
				panic("bug")
			}
		}
	}
	stage2 := func() {
		n := a.tok.Slots
		slots = make([]cgen.Gen, n)
		stage3()
		ae := a.arranged
		ae = addMul(ae, il(a.groupBytes), a.groupIdx)
		ae = addMul(ae, il(a.coreBytes), a.coreIdx)
		ae = addMul(ae, il(n*a.slotBytes), a.sliceIdx)
		for x, slot := range slots {
			stmt(avx.Mm512StoreuPs{
				cgen.Add{
					Expr1: ae,
					Expr2: cast(x * a.slotBytes),
				},
				slot,
			})
		}
	}
	stage1 := func() {
		bnCnt := len(a.bnPtrs)
		bnMuls = make([]cgen.Gen, bnCnt)
		bnAdds = make([]cgen.Gen, bnCnt)
		bnPre := 0
		for x := range a.From.Ops {
			switch a.From.Ops[x].Kind {
			case mod.Add:
				bnSplit = bnPre
				opSplit = x + 1
			case mod.Bn:
				bnPre++
			}
		}
		stage2()
	}
	stage1()
	return stmts
}

type Apply struct {
	*Ctx
	*Spec
	Team       cgen.Gen
	Tensors    []cgen.Gen
	callerName string
}

func (a *Apply) Prep() cgen.Gen {
	const affix = "Apply"
	sig := fmt.Sprint(affix, " ", a.Spec)
	if prior, ok := a.dedup[sig]; ok {
		a.callerName = prior.(string)
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
	*layout
	slices        int
	wtCoreBytes   int
	wtGroupBytes  int
	datCoreBytes  int
	datGroupBytes int
	epochFirst    bool
	epochLast     bool
	wtTile        int
	wtTiles       int
	wtScrap       int
	wtHull        int
	datTile       int
	datTiles      int
	datScrap      int
	datHull       int
	groupTile     int
	groupTiles    int
	groupScrap    int
	groupHull     int
	calleeName    string
	epochCoord    cgen.Gen
	groupCoord    cgen.Gen
	datCoord      cgen.Gen
	wtCoord       cgen.Gen
	arrangedWts   cgen.Gen
	arrangedDats  cgen.Gen
	datSplit      int
	datPtrs       []cgen.Gen
	bnPtrs        []cgen.Gen
	groupIdx      cgen.Gen
	datCore       cgen.Gen
	datShort      bool
	sectH         cgen.Gen
	tok           *token
	wtCore        cgen.Gen
	wtShort       bool
	rows          int
	cols          int
	sums          []cgen.Gen
}

func (a *apply) Append(to []byte) []byte {
	a.layout = newLayout(ctxSpec{
		Ctx:  a.Ctx,
		Spec: a.Spec,
	})
	callee := func(epoch int) cgen.Gen {
		if epoch < a.epochs1 {
			a.slices = a.slices1
			a.wtCoreBytes = a.wtCoreBytes11
			a.wtGroupBytes = a.wtGroupBytes1
			a.datCoreBytes = a.datCoreBytes11
			a.datGroupBytes = a.datGroupBytes1
		} else {
			a.slices = a.slices2
			a.wtCoreBytes = a.wtCoreBytes21
			a.wtGroupBytes = a.wtGroupBytes2
			a.datCoreBytes = a.datCoreBytes21
			a.datGroupBytes = a.datGroupBytes2
		}
		a.epochFirst = epoch == 0
		a.epochLast = epoch == a.epochs2-1
		a.wtTile = 1
		a.wtTiles = a.wtCores2
		a.wtScrap = 0
		a.wtHull = a.wtCores2
		a.datTile = 1
		a.datTiles = a.datCores2
		a.datScrap = 0
		a.datHull = a.datCores2
		a.groupTile = 1
		a.groupTiles = a.Groups
		a.groupScrap = 0
		a.groupHull = a.Groups
		a.calleeName = a.name(
			a.callerName + "Callee",
		)
		var (
			wtWork     = a.slices
			datWork    = a.wtCores2 * wtWork
			groupWork  = a.datCores2 * datWork
			threadWork int
		)
		switch a.platform {
		case raw.AVX512Float32:
			threadWork = 512
		default:
			panic("bug")
		}
		switch {
		case threadWork <= wtWork:
		case threadWork <= datWork:
			var (
				tile  = ceilQuo(threadWork, wtWork)
				tiles = max(a.wtCores2/tile, 1)
			)
			a.wtTile = a.wtCores2 / tiles
			a.wtTiles = tiles
			a.wtScrap = a.wtCores2 - tiles*a.wtTile
			a.wtHull = tiles
			if a.wtScrap > 0 {
				a.wtTiles--
				a.wtScrap += a.wtTile
			}
		case threadWork <= groupWork:
			a.wtTile = a.wtCores2
			a.wtTiles = 1
			a.wtScrap = 0
			a.wtHull = 1
			var (
				tile  = ceilQuo(threadWork, datWork)
				tiles = max(a.datCores2/tile, 1)
			)
			a.datTile = a.datCores2 / tiles
			a.datTiles = tiles
			a.datScrap = a.datCores2 - tiles*a.datTile
			a.datHull = tiles
			if a.datScrap > 0 {
				a.datTiles--
				a.datScrap += a.datTile
			}
		default:
			a.wtTile = a.wtCores2
			a.wtTiles = 1
			a.wtScrap = 0
			a.wtHull = 1
			a.datTile = a.datCores2
			a.datTiles = 1
			a.datScrap = 0
			a.datHull = 1
			var (
				tile  = ceilQuo(threadWork, groupWork)
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
		return cgen.Gens{
			a.calleeFunc(),
			cgen.Newline,
		}
	}
	var (
		team    = vb(a.name("team"))
		tensors = vb(a.name("tensors"))
		pair    = vb(a.name("pair"))
	)
	do := func(epoch cgen.Gen) cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if a.epochFirst {
			stmts[0] = cgen.Var{
				Type: cgen.PtrVoid,
				What: cgen.Elem{Arr: pair},
				Init: cgen.Brace{
					Inner: cgen.CommaSpaced{
						tensors, epoch,
					},
				},
			}
		} else {
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
			Ctx:    a.tc,
			Callee: vb(a.calleeName),
			Any:    pair,
			Hull: []cgen.Gen{
				il(a.wtHull),
				il(a.datHull),
				il(a.groupHull),
			},
			Team: team,
		}
		return stmts
	}
	var (
		prep = make(cgen.Gens, 3)
		body = make(cgen.Gens, 3)
	)
	prep[0] = callee(0)
	body[0] = do(il(0))
	if a.epochs2 > 1 {
		var (
			start = 1
			stop  = a.epochs1
		)
		if stop == a.epochs2 {
			if len(a.To.Ops) > 0 ||
				len(a.To.Pitch1Bytes) > 1 {
				stop--
			}
		}
		if start < stop {
			prep[1] = callee(start)
			epoch := vb(a.name("e"))
			body[1] = cgen.Stmts{
				cgen.For{
					Init: cgen.Var{
						Type: cgen.PtrdiffT,
						What: epoch,
						Init: il(start),
					},
					Cond: cgen.CmpL{
						Expr1: epoch,
						Expr2: il(stop),
					},
					Post: cgen.IncPre{
						Expr: epoch,
					},
					Body: do(epoch),
				},
			}
		}
		if stop < a.epochs2 {
			prep[2] = callee(stop)
			body[2] = do(il(stop))
		}
	}
	return cgen.Gens{
		prep,
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
			Body: body,
		},
	}.Append(to)
}

func (a *apply) calleeFunc() cgen.Gen {
	callee := &threader.Callee{
		Ctx:  a.tc,
		Name: a.calleeName,
		Task: vb(a.name("task")),
		Pt:   vb(a.name("pt")),
	}
	var (
		body    = make(cgen.Stmts, 9)
		pair    = vb(a.name("pair"))
		tensors = vb(a.name("tensors"))
		epoch   cgen.Gen
		usedPt  = false
	)
	body[0] = cgen.Var{
		Type: cgen.PtrPtrVoid, What: pair,
		Init: callee.Any(),
	}
	body[1] = cgen.Var{
		Type: cgen.PtrPtrChar, What: tensors,
		Init: cgen.Elem{
			Arr: pair, Idx: il(0),
		},
	}
	switch {
	case a.epochFirst:
		epoch = il(0)
	case a.epochLast:
		epoch = il(a.epochs2 - 1)
	default:
		epoch = cgen.Cast{
			Type: cgen.PtrdiffT,
			Expr: cgen.Elem{
				Arr: pair, Idx: il(1),
			},
		}
	}
	a.epochCoord = vb(a.name("e"))
	body[2] = cgen.Var{
		Type: cgen.PtrdiffT, What: a.epochCoord,
		Init: epoch,
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
		body[5-i] = cgen.Var{
			Type: cgen.PtrdiffT, What: ret,
			Init: expr,
		}
		return ret
	}
	a.groupCoord = coord("g", a.groupHull, 2)
	a.datCoord = coord("d", a.datHull, 1)
	a.wtCoord = coord("w", a.wtHull, 0)
	if !usedPt {
		body[6] = cgen.Cast{
			Type: cgen.Void,
			Expr: callee.Pt,
		}
	}
	body[7] = a.ptrs(tensors)
	body[8] = a.kernel()
	return callee.Func(body)
}

func (a *apply) ptrs(tensors cgen.Gen) cgen.Gen {
	var (
		group   cgen.Gen
		awOff   cgen.Gen
		adOff   cgen.Gen
		datOffs []cgen.Gen
		bnCh    cgen.Gen
	)
	stage6 := func() cgen.Gen {
		var (
			stmts     cgen.Stmts
			tensorIdx = 0
			datIdx    = 0
			bnIdx     = 0
		)
		stmt := func(s cgen.Gen) {
			stmts = append(stmts, s)
		}
		tensor := func() cgen.Gen {
			i := tensorIdx
			tensorIdx++
			return cgen.Elem{
				Arr: tensors,
				Idx: il(i),
			}
		}
		decl := func(what, off cgen.Gen) {
			stmt(cgen.Var{
				Type: cgen.RestrictPtrChar,
				What: what,
				Init: cgen.Add{
					Expr1: tensor(),
					Expr2: off,
				},
			})
		}
		decl(a.arrangedWts, awOff)
		decl(a.arrangedDats, adOff)
		dps := func(n int) {
			for ; n > 0; n-- {
				i := datIdx
				datIdx++
				datPtr := a.datPtrs[i]
				if datPtr == nil {
					tensorIdx++
				} else {
					decl(datPtr, datOffs[i])
				}
			}
		}
		bp := func() {
			bnPtr := a.bnPtrs[bnIdx]
			bnIdx++
			if bnPtr == nil {
				tensorIdx++
				return
			}
			stmt(cgen.Var{
				Type: cgen.RestrictPtrChar,
				What: bnPtr,
				Init: &bn.Offset{
					Ctx:     a.bc,
					Mas:     tensor(),
					Channel: bnCh,
				},
			})
		}
		for i := range a.To.Ops {
			op := &a.To.Ops[i]
			switch op.Kind {
			case mod.Add:
				dps(op.Int)
			case mod.Bn:
				bp()
			case mod.ReLU:
			default:
				panic("bug")
			}
		}
		dps(len(a.datPtrs) - datIdx)
		return stmts
	}
	stage5 := func() cgen.Gen {
		n := 0
		for i := range a.To.Ops {
			if a.To.Ops[i].Kind == mod.Bn {
				n++
			}
		}
		a.bnPtrs = make([]cgen.Gen, n)
		if a.epochLast {
			for i := range a.bnPtrs {
				a.bnPtrs[i] = vb(a.name("bnPtr"))
			}
			bnCh = cgen.Mul{
				Expr1: il(a.toChans),
				Expr2: group,
			}
		}
		return stage6()
	}
	stage4 := func() cgen.Gen {
		a.datSplit = 0
		for i := range a.To.Ops {
			op := &a.To.Ops[i]
			if op.Kind == mod.Add {
				a.datSplit += op.Int
			}
		}
		n := len(a.To.Pitch1Bytes)
		a.datPtrs = make([]cgen.Gen, n)
		for i := range a.datPtrs {
			if a.epochLast || i == a.datSplit {
				a.datPtrs[i] = vb(a.name("datPtr"))
			}
		}
		datOffs = make([]cgen.Gen, n)
		for i := range datOffs {
			if a.datPtrs[i] == nil {
				continue
			}
			chanPitch := a.To.Pitch2Bytes[i]
			datOffs[i] = cgen.Mul{
				Expr1: cast(a.toChans * chanPitch),
				Expr2: group,
			}
		}
		return stage5()
	}
	stage3 := func() cgen.Gen {
		a.arrangedDats = vb(a.name("arrangedDats"))
		adOff = cgen.Add{
			Expr1: cgen.Mul{
				Expr1: il(a.datEpochBytes1),
				Expr2: a.epochCoord,
			},
			Expr2: cgen.Mul{
				Expr1: cast(a.datGroupBytes),
				Expr2: group,
			},
		}
		return stage4()
	}
	stage2 := func() cgen.Gen {
		a.arrangedWts = vb(a.name("arrangedWts"))
		awOff = cgen.Add{
			Expr1: cgen.Mul{
				Expr1: il(a.wtEpochBytes1),
				Expr2: a.epochCoord,
			},
			Expr2: cgen.Mul{
				Expr1: cast(a.wtGroupBytes),
				Expr2: group,
			},
		}
		return stage3()
	}
	stage1 := func() cgen.Gen {
		group = cgen.Mul{
			Expr1: il(a.groupTile),
			Expr2: a.groupCoord,
		}
		return stage2()
	}
	return stage1()
}

func (a *apply) kernel() cgen.Gen {
	var (
		datRet cgen.Gen
		wtRet  cgen.Gen
	)
	layer7 := func() cgen.Gen {
		if a.toks == nil {
			return a.special()
		}
		return a.general()
	}
	layer6 := func() cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if a.wtCores1 > 0 {
			a.wtShort = false
			stmts[0] = cgen.For{
				Cond: cgen.CmpNE{
					Expr1: a.wtCore,
					Expr2: il(a.wtCores1),
				},
				Post: cgen.IncPre{
					Expr: a.wtCore,
				},
				Body: cgen.Stmts{
					layer7(),
					wtRet,
				},
			}
		}
		if a.wtCores1 < a.wtCores2 {
			a.wtShort = true
			stmts[1] = layer7()
		}
		return stmts
	}
	layer5 := func() cgen.Gen {
		a.wtCore = vb(a.name("k"))
		stmts := make(cgen.Stmts, 3)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: a.wtCore,
			Init: cgen.Mul{
				Expr1: il(a.wtTile),
				Expr2: a.wtCoord,
			},
		}
		if a.wtHull > 1 {
			var (
				last = vb(a.name("kk"))
				expr cgen.Gen
			)
			switch a.wtTiles {
			case a.wtHull:
				expr = il(a.wtTile - 1)
			case 0:
				expr = il(a.wtScrap - 1)
			default:
				expr = cgen.Paren{
					Inner: cgen.Ternary{
						Cond: cgen.CmpL{
							Expr1: a.wtCoord,
							Expr2: il(a.wtTiles),
						},
						Then: il(a.wtTile - 1),
						Else: il(a.wtScrap - 1),
					},
				}
			}
			stmts[1] = cgen.Var{
				Type: cgen.PtrdiffT,
				What: last,
				Init: cgen.Add{
					Expr1: a.wtCore,
					Expr2: expr,
				},
			}
			wtRet = cgen.If1{
				Cond: cgen.CmpGE{
					Expr1: a.wtCore,
					Expr2: last,
				},
				Then: cgen.Return{},
			}
		}
		stmts[2] = layer6()
		return stmts
	}
	layer4Special := func() cgen.Gen {
		stmts := make(cgen.Stmts, 2)
		if a.datCores1 > 0 {
			a.datShort = false
			stmts[0] = cgen.For{
				Cond: cgen.CmpNE{
					Expr1: a.datCore,
					Expr2: il(a.datCores1),
				},
				Post: cgen.IncPre{
					Expr: a.datCore,
				},
				Body: cgen.Stmts{
					layer5(),
					datRet,
				},
			}
		}
		if a.datCores1 < a.datCores2 {
			a.datShort = true
			stmts[1] = layer5()
		}
		return stmts
	}
	layer4General := func() cgen.Gen {
		leaf := func(sect *section) cgen.Stmts {
			a.sectH = vb(a.name("h"))
			var (
				initH cgen.Gen
				which cgen.Gen
				n     = len(sect.Uniqs)
				cases = make(cgen.Stmts, n)
			)
			if sect.ToWrap == 0 {
				initH = il(sect.ToBase)
				which = a.datCore
				for x, tok := range sect.Uniqs {
					a.tok = tok
					var (
						expr = il(sect.IdxFirst + x)
						body = make(cgen.Stmts, 3)
					)
					body[0] = cgen.Assign{
						Expr1: a.datCore,
						Expr2: expr,
					}
					if x == n-1 {
						expr = nil
					}
					body[1] = layer5()
					body[2] = datRet
					cases[x] = cgen.Case{
						Expr: expr,
						Body: body,
					}
				}
			} else {
				var (
					numer = cgen.Paren{
						Inner: cgen.Sub{
							Expr1: cgen.Cast{
								Type: cgen.SizeT,
								Expr: a.datCore,
							},
							Expr2: il(sect.IdxFirst),
						},
					}
					denom = il(n)
				)
				initH = addMul(
					il(sect.ToBase),
					cgen.Quo{
						Expr1: numer,
						Expr2: denom,
					},
					il(sect.ToWrap),
				)
				which = cgen.Rem{
					Expr1: numer,
					Expr2: denom,
				}
				var (
					wrap = cgen.Label(a.name("wrap"))
					last = sect.IdxPast - 1
					at   = (last - sect.IdxFirst) % n
				)
				for x, tok := range sect.Uniqs {
					a.tok = tok
					var (
						expr cgen.Gen
						body = make(cgen.Stmts, 7)
					)
					if x < n-1 {
						expr = il(x)
					}
					if x == 0 {
						body[0] = wrap
					}
					body[1] = layer5()
					body[2] = datRet
					if x == at {
						body[3] = cgen.If1{
							Cond: cgen.CmpGE{
								Expr1: a.datCore,
								Expr2: il(last),
							},
							Then: cgen.Break,
						}
					}
					body[4] = cgen.IncPre{
						Expr: a.datCore,
					}
					if x == n-1 {
						body[5] = cgen.AddAssign{
							Expr1: a.sectH,
							Expr2: il(sect.ToWrap),
						}
						body[6] = cgen.Goto(wrap)
					}
					cases[x] = cgen.Case{
						Expr: expr,
						Body: body,
					}
				}
			}
			return cgen.Stmts{
				cgen.Var{
					Type: cgen.PtrdiffT,
					What: a.sectH, Init: initH,
				},
				cgen.Switch{
					Expr:  which,
					Cases: cases,
				},
				cgen.Assign{
					Expr1: a.datCore,
					Expr2: il(sect.IdxPast),
				},
			}
		}
		var (
			sects = a.toks.Sects
			tree  func(int, int) cgen.Stmts
		)
		tree = func(first, last int) cgen.Stmts {
			if first == last {
				return leaf(sects[first])
			}
			var (
				start = sects[first].IdxFirst
				stop  = sects[last].IdxPast
				upper = start + (stop-start)/2
				x     = first + 1
			)
			for sects[x].IdxPast <= upper {
				x++
			}
			return cgen.Stmts{
				cgen.If{
					Cond: cgen.CmpL{
						Expr1: a.datCore,
						Expr2: il(sects[x].IdxFirst),
					},
					Then: tree(first, x-1),
				},
				tree(x, last),
			}
		}
		return tree(0, len(sects)-1)
	}
	layer3 := func() cgen.Gen {
		if a.toks == nil {
			return layer4Special()
		}
		return layer4General()
	}
	layer2 := func() cgen.Gen {
		a.datCore = vb(a.name("j"))
		stmts := make(cgen.Stmts, 3)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: a.datCore,
			Init: cgen.Mul{
				Expr1: il(a.datTile),
				Expr2: a.datCoord,
			},
		}
		if a.datHull > 1 {
			var (
				last = vb(a.name("jj"))
				expr cgen.Gen
			)
			switch a.datTiles {
			case a.datHull:
				expr = il(a.datTile - 1)
			case 0:
				expr = il(a.datScrap - 1)
			default:
				expr = cgen.Paren{
					Inner: cgen.Ternary{
						Cond: cgen.CmpL{
							Expr1: a.datCoord,
							Expr2: il(a.datTiles),
						},
						Then: il(a.datTile - 1),
						Else: il(a.datScrap - 1),
					},
				}
			}
			stmts[1] = cgen.Var{
				Type: cgen.PtrdiffT,
				What: last,
				Init: cgen.Add{
					Expr1: a.datCore,
					Expr2: expr,
				},
			}
			datRet = cgen.If1{
				Cond: cgen.CmpGE{
					Expr1: a.datCore,
					Expr2: last,
				},
				Then: cgen.Return{},
			}
		}
		stmts[2] = layer3()
		return stmts
	}
	layer1 := func() cgen.Gen {
		a.groupIdx = vb(a.name("i"))
		var (
			past   = vb(a.name("ii"))
			groups cgen.Gen
		)
		switch a.groupTiles {
		case a.groupHull:
			groups = il(a.groupTile)
		case 0:
			groups = il(a.groupScrap)
		default:
			groups = cgen.Ternary{
				Cond: cgen.CmpL{
					Expr1: a.groupCoord,
					Expr2: il(a.groupTiles),
				},
				Then: il(a.groupTile),
				Else: il(a.groupScrap),
			}
		}
		return cgen.Stmts{
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: past, Init: groups,
			},
			cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT,
					What: a.groupIdx,
					Init: il(0),
				},
				Cond: cgen.CmpL{
					Expr1: a.groupIdx,
					Expr2: past,
				},
				Post: cgen.IncPre{
					Expr: a.groupIdx,
				},
				Body: layer2(),
			},
		}
	}
	return layer1()
}

func (a *apply) special() cgen.Gen {
	switch a.platform {
	case raw.AVX512Float32:
		return a.m512Special()
	default:
		panic("bug")
	}
}

func (a *apply) general() cgen.Gen {
	switch a.platform {
	case raw.AVX512Float32:
		return a.m512General()
	default:
		panic("bug")
	}
}

func (a *apply) m512Special() cgen.Gen {
	var (
		row    int
		col    int
		slot   cgen.Gen
		stmts  cgen.Stmts
		bnMuls []cgen.Gen
		bnAdds []cgen.Gen
	)
	stmt := func(st cgen.Gen) {
		stmts = append(stmts, st)
	}
	mask := func() cgen.Gen {
		n := a.slotDats
		if a.datShort &&
			col == a.cols-1 {
			rem := a.datSliceDats2 % n
			if rem > 0 {
				n = rem
			}
		}
		return loMask(n)
	}
	ae := func(datPtrIdx int) cgen.Gen {
		var (
			ret          = a.datPtrs[datPtrIdx]
			chanPitch    = a.To.Pitch2Bytes[datPtrIdx]
			groupPitch   = a.toChans * chanPitch
			datCorePitch = a.datSliceBytes1
			wtCorePitch  = a.wtSliceWts1 * chanPitch
			rowPitch     = chanPitch
			colPitch     = a.slotBytes
		)
		ret = addMul(ret, il(groupPitch), a.groupIdx)
		ret = addMul(ret, il(datCorePitch), a.datCore)
		ret = addMul(ret, il(wtCorePitch), a.wtCore)
		ret = cgen.Add{
			Expr1: ret,
			Expr2: cast(row*rowPitch + col*colPitch),
		}
		return ret
	}
	addLd := func(datPtrIdx int) {
		stmt(cgen.Assign{
			Expr1: slot,
			Expr2: avx.Mm512AddPs{
				slot,
				avx.Mm512MaskzLoaduPs{
					mask(),
					ae(datPtrIdx),
				},
			},
		})
	}
	layer5 := func() {
		var (
			datPtrIdx = 0
			bnPtrIdx  = 0
		)
		bnPrep := func() cgen.Gen {
			if col > 0 {
				return nil
			}
			var (
				bnCnt = len(a.bnPtrs)
				bnLds = make(cgen.Gens, bnCnt)
			)
			bnCh := cgen.Paren{
				Inner: addMul(
					addMul(
						il(row),
						il(a.wtSliceWts1),
						a.wtCore,
					),
					il(a.toChans),
					a.groupIdx,
				),
			}
			bnMuls = make([]cgen.Gen, bnCnt)
			bnAdds = make([]cgen.Gen, bnCnt)
			for x, bnPtr := range a.bnPtrs {
				var (
					bnMul = vb(a.name("bnMul"))
					bnAdd = vb(a.name("bnAdd"))
				)
				bnLds[x] = &bn.Load{
					Ctx:     a.bc,
					Mas:     bnPtr,
					Channel: bnCh,
					Mul:     bnMul,
					Add:     bnAdd,
				}
				bnMuls[x] = bnMul
				bnAdds[x] = bnAdd
			}
			return bnLds
		}
		for op := range a.To.Ops {
			op := &a.To.Ops[op]
			switch op.Kind {
			case mod.Add:
				for n := op.Int; n > 0; n-- {
					addLd(datPtrIdx)
					datPtrIdx++
				}
			case mod.Bn:
				if bnPtrIdx == 0 {
					stmt(bnPrep())
				}
				stmt(&bn.Apply{
					Ctx: a.bc,
					Mul: bnMuls[bnPtrIdx],
					Add: bnAdds[bnPtrIdx],
					To:  slot,
				})
				bnPtrIdx++
			case mod.ReLU:
				stmt(&act.ReLU{
					Ctx:      a.ac,
					NegSlope: op.Float,
					Var:      slot,
				})
			default:
				panic("bug")
			}
		}
	}
	layer4 := func() {
		var (
			to   = a.datSplit
			stop = to + 1
		)
		if !a.epochFirst {
			addLd(to)
		}
		if a.epochLast {
			layer5()
			stop = len(a.datPtrs)
		}
		for ; to < stop; to++ {
			stmt(avx.Mm512MaskStoreuPs{
				ae(to), mask(), slot,
			})
		}
	}
	layer3 := func() cgen.Stmts {
		stmts = nil
		layer4()
		return stmts
	}
	layer2 := func() cgen.Gen {
		var (
			rr    = a.rows
			cc    = a.cols
			ret   = make(cgen.Gens, rr)
			toMix = make([]cgen.Stmts, cc)
		)
		for r := 0; r < rr; r++ {
			row = r
			for c := 0; c < cc; c++ {
				col = c
				slot = a.sums[r*cc+c]
				toMix[c] = layer3()
			}
			ret[r] = mix(toMix)
		}
		return ret
	}
	layer1 := func() cgen.Gen {
		return cgen.Gens{
			a.m512Dot(),
			layer2(),
		}
	}
	return layer1()
}

func (a *apply) m512General() cgen.Gen {
	var (
		opSplit int
		bnSplit int
		row     int
		stmts   [2]cgen.Stmts
		dats    []cgen.Gen
		bnMuls  []cgen.Gen
		bnAdds  []cgen.Gen
	)
	stmt := func(x int, st cgen.Gen) {
		stmts[x] = append(
			stmts[x], st,
		)
	}
	idPrep := func(id int) {
		for id >= len(dats) {
			dats = append(dats, nil)
		}
	}
	eval := func(id int, expr cgen.Gen) {
		idPrep(id)
		dat := dats[id]
		if dat == nil {
			dat = vb(a.name("dat"))
			dats[id] = dat
			stmt(0, cgen.Var{
				Type: avx.M512, What: dat,
				Init: expr,
			})
		} else {
			stmt(0, cgen.Assign{
				Expr1: dat,
				Expr2: expr,
			})
		}
	}
	rot := func(cmd *m512CmdRotate) {
		var (
			dst = cmd.DstId
			src = dats[cmd.SrcId]
			cnt = il(cmd.Cnt)
			via = vb(a.name("via"))
		)
		stmt(0, cgen.Var{
			Type: avx.M512i, What: via,
			Init: avx.Mm512CastpsSi512{src},
		})
		stmt(0, cgen.Assign{
			Expr1: via,
			Expr2: avx.Mm512AlignrEpi32{
				via, via, cnt,
			},
		})
		eval(dst, avx.Mm512Castsi512Ps{via})
	}
	slotGet := func(cmd *m512CmdSlotGet) {
		var (
			id   = cmd.Id
			slot = cmd.Slot
		)
		idPrep(id)
		dats[id] = a.sums[row*a.cols+slot]
	}
	bnPrep := func(x int) {
		if bnMuls[x] != nil {
			return
		}
		var (
			bnMul = vb(a.name("bnMul"))
			bnAdd = vb(a.name("bnAdd"))
		)
		stmt(0, &bn.Load{
			Ctx: a.bc,
			Mas: a.bnPtrs[x],
			Channel: cgen.Paren{
				Inner: addMul(
					addMul(
						il(row),
						il(a.wtSliceWts1),
						a.wtCore,
					),
					il(a.toChans),
					a.groupIdx,
				),
			},
			Mul: bnMul,
			Add: bnAdd,
		})
		bnMuls[x] = bnMul
		bnAdds[x] = bnAdd
	}
	modPreAdd := func(cmd *m512CmdToModPreAdd) {
		if !a.epochLast {
			return
		}
		var (
			dst      = dats[cmd.Id]
			bnPtrIdx = 0
		)
		for op := 0; op < opSplit; op++ {
			op := &a.To.Ops[op]
			switch op.Kind {
			case mod.Bn:
				bnPrep(bnPtrIdx)
				stmt(0, &bn.Apply{
					Ctx: a.bc,
					Mul: bnMuls[bnPtrIdx],
					Add: bnAdds[bnPtrIdx],
					To:  dst,
				})
				bnPtrIdx++
			case mod.ReLU:
				stmt(0, &act.ReLU{
					Ctx:      a.ac,
					NegSlope: op.Float,
					Var:      dst,
				})
			default:
				panic("bug")
			}
		}
	}
	ae := func(x, relH, w int) cgen.Gen {
		var (
			ret         = a.datPtrs[x]
			cPitch      = a.To.Pitch2Bytes[x]
			hPitch      = a.To.Pitch1Bytes[x]
			wPitch      = a.datBytes
			groupPitch  = a.toChans * cPitch
			wtCorePitch = a.wtSliceWts1 * cPitch
			rowPitch    = cPitch
			h           = a.tok.To.FirstH + relH
		)
		ret = addMul(ret, il(groupPitch), a.groupIdx)
		ret = addMul(ret, il(hPitch), a.sectH)
		ret = addMul(ret, il(wtCorePitch), a.wtCore)
		ret = cgen.Add{
			Expr1: ret,
			Expr2: cast(row*rowPitch + h*hPitch + w*wPitch),
		}
		return ret
	}
	addLd := func(dst cgen.Gen, x, relH, w, cnt int) {
		stmt(0, cgen.Assign{
			Expr1: dst,
			Expr2: avx.Mm512AddPs{
				dst,
				avx.Mm512MaskzLoaduPs{
					loMask(cnt),
					ae(x, relH, w),
				},
			},
		})
	}
	modAddPost := func(cmd *m512CmdToModAddPost) {
		var (
			dst  = dats[cmd.Id]
			relH = cmd.RelH
			w    = cmd.W
			cnt  = cmd.Cnt
		)
		if !a.epochFirst {
			addLd(dst, a.datSplit, relH, w, cnt)
		}
		if !a.epochLast {
			return
		}
		var (
			ops       = a.To.Ops[opSplit:]
			datPtrIdx = 0
			bnPtrIdx  = bnSplit
		)
		for op := range ops {
			op := &ops[op]
			switch op.Kind {
			case mod.Add:
				for n := op.Int; n > 0; n-- {
					addLd(dst, datPtrIdx, relH, w, cnt)
					datPtrIdx++
				}
			case mod.Bn:
				bnPrep(bnPtrIdx)
				stmt(0, &bn.Apply{
					Ctx: a.bc,
					Mul: bnMuls[bnPtrIdx],
					Add: bnAdds[bnPtrIdx],
					To:  dst,
				})
				bnPtrIdx++
			case mod.ReLU:
				stmt(0, &act.ReLU{
					Ctx:      a.ac,
					NegSlope: op.Float,
					Var:      dst,
				})
			default:
				panic("bug")
			}
		}
	}
	store := func(cmd *m512CmdStore) {
		var (
			src  = dats[cmd.Id]
			relH = cmd.RelH
			w    = cmd.W
			cnt  = cmd.Cnt
			to   = a.datSplit
			stop = to + 1
		)
		if a.epochLast {
			stop = len(a.datPtrs)
		}
		for ; to < stop; to++ {
			stmt(1, avx.Mm512MaskStoreuPs{
				ae(to, relH, w),
				loMask(cnt),
				src,
			})
		}
	}
	layer6 := func() {
		for _, cmd := range a.tok.To.Cmds {
			switch cmd := cmd.(type) {
			case *m512CmdCopy:
				src := dats[cmd.SrcId]
				eval(cmd.DstId, src)
			case *m512CmdRotate:
				rot(cmd)
			case *m512CmdSlotGet:
				slotGet(cmd)
			case *m512CmdToModPreAdd:
				modPreAdd(cmd)
			case *m512CmdToModAddPost:
				modAddPost(cmd)
			case *m512CmdStore:
				store(cmd)
			default:
				panic("bug")
			}
		}
	}
	layer5 := func() {
		dats = dats[:0]
		if a.epochLast {
			n := len(a.bnPtrs)
			bnMuls = make([]cgen.Gen, n)
			bnAdds = make([]cgen.Gen, n)
		}
		layer6()
	}
	layer4 := func() cgen.Gen {
		stmts[0] = nil
		stmts[1] = nil
		layer5()
		return cgen.Gens{
			stmts[0],
			stmts[1],
		}
	}
	layer3 := func() cgen.Gen {
		var (
			rr  = a.rows
			ret = make(cgen.Gens, rr)
		)
		for r := 0; r < rr; r++ {
			row = r
			ret[r] = layer4()
		}
		return ret
	}
	layer2 := func() cgen.Gen {
		if a.epochLast {
			opSplit = 0
			bnSplit = 0
			if a.epochFirst {
				for op := range a.To.Ops {
					kind := a.To.Ops[op].Kind
					if kind == mod.Add {
						break
					}
					opSplit++
					if kind == mod.Bn {
						bnSplit++
					}
				}
			}
		}
		return layer3()
	}
	layer1 := func() cgen.Gen {
		return cgen.Gens{
			a.m512Dot(),
			layer2(),
		}
	}
	return layer1()
}

func (a *apply) m512Dot() cgen.Gen {
	var (
		sliceIdx cgen.Gen
	)
	ldWt := func(row int) cgen.Gen {
		var (
			ae         = a.arrangedWts
			groupPitch = il(a.wtGroupBytes)
			corePitch  = il(a.wtCoreBytes)
			sliceBytes = a.rows * a.wtBytes
		)
		ae = addMul(ae, groupPitch, a.groupIdx)
		ae = addMul(ae, corePitch, a.wtCore)
		ae = addMul(ae, il(sliceBytes), sliceIdx)
		ae = cgen.Add{
			Expr1: ae,
			Expr2: cast(sliceBytes + row*a.wtBytes),
		}
		return avx.Mm512Set1Ps{
			cgen.At{
				Expr: cgen.Cast{
					Type: cgen.PtrFloat,
					Expr: cgen.Paren{
						Inner: ae,
					},
				},
			},
		}
	}
	ldDat := func(col int) cgen.Gen {
		var (
			ae         = a.arrangedDats
			groupPitch = il(a.datGroupBytes)
			corePitch  = il(a.datCoreBytes)
			slicePitch = il(a.cols * a.slotBytes)
		)
		ae = addMul(ae, groupPitch, a.groupIdx)
		ae = addMul(ae, corePitch, a.datCore)
		ae = addMul(ae, slicePitch, sliceIdx)
		ae = cgen.Add{
			Expr1: ae,
			Expr2: cast(col * a.slotBytes),
		}
		return avx.Mm512LoaduPs{ae}
	}
	layer5 := func() cgen.Gen {
		var (
			rr    = a.rows
			cc    = a.cols
			dats  = make([]cgen.Gen, cc)
			stmts = make(cgen.Stmts, 0, cc+rr*(1+cc))
		)
		stmt := func(st cgen.Gen) {
			stmts = append(stmts, st)
		}
		for c := 0; c < cc; c++ {
			dat := vb(a.name("dat"))
			dats[c] = dat
			stmt(cgen.Var{
				Type: avx.M512, What: dat,
				Init: ldDat(c),
			})
		}
		for r := 0; r < rr; r++ {
			wt := vb(a.name("wt"))
			stmt(cgen.Var{
				Type: avx.M512, What: wt,
				Init: ldWt(r),
			})
			for c := 0; c < cc; c++ {
				var (
					sum = a.sums[r*cc+c]
					dat = dats[c]
				)
				stmt(cgen.Assign{
					Expr1: sum,
					Expr2: avx.Mm512FmaddPs{
						wt, dat, sum,
					},
				})
			}
		}
		return stmts
	}
	layer4 := func() cgen.Gen {
		return cgen.For{
			Init: cgen.Assign{
				Expr1: sliceIdx,
				Expr2: il(0),
			},
			Cond: cgen.CmpL{
				Expr1: sliceIdx,
				Expr2: il(a.slices),
			},
			Post: cgen.IncPre{
				Expr: sliceIdx,
			},
			Body: layer5(),
		}
	}
	layer3 := func() cgen.Gen {
		var (
			rr    = a.rows
			cc    = a.cols
			stmts = make(cgen.Stmts, rr*(cc-1)+1)
			x     = 0
		)
		for r := 0; r < rr; r++ {
			for c := 1; c < cc; c++ {
				stmts[x] = cgen.Var{
					Type: avx.M512,
					What: a.sums[r*cc+c],
					Init: a.sums[r*cc],
				}
				x++
			}
		}
		stmts[x] = layer4()
		return stmts
	}
	layer2 := func() cgen.Gen {
		sliceIdx = vb(a.name("s"))
		var (
			rr    = a.rows
			stmts = make(cgen.Stmts, 1+rr+1)
		)
		stmts[0] = cgen.Var{
			Type: cgen.PtrdiffT,
			What: sliceIdx,
			Init: il(-1),
		}
		for r := 0; r < rr; r++ {
			stmts[1+r] = cgen.Var{
				Type: avx.M512,
				What: a.sums[r*a.cols],
				Init: ldWt(r),
			}
		}
		stmts[1+rr] = layer3()
		return stmts
	}
	layer1 := func() cgen.Gen {
		a.rows = a.wtSliceWts1
		if a.wtShort {
			a.rows = a.wtSliceWts2
		}
		if a.toks == nil {
			a.cols = a.datSliceSlots1
			if a.datShort {
				a.cols = a.datSliceSlots2
			}
		} else {
			a.cols = a.tok.Slots
		}
		a.sums = make([]cgen.Gen, a.rows*a.cols)
		for x := range a.sums {
			a.sums[x] = vb(a.name("sum"))
		}
		return layer2()
	}
	return layer1()
}
