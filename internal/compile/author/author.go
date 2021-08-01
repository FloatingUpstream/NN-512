package author

import (
	"NN-512/internal/compile/author/act"
	"NN-512/internal/compile/author/bn"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/cpu"
	"NN-512/internal/compile/author/elwi"
	"NN-512/internal/compile/author/engine"
	"NN-512/internal/compile/author/eof"
	"NN-512/internal/compile/author/errmsg"
	"NN-512/internal/compile/author/exp"
	"NN-512/internal/compile/author/fc"
	"NN-512/internal/compile/author/glopl"
	"NN-512/internal/compile/author/hc"
	"NN-512/internal/compile/author/include"
	"NN-512/internal/compile/author/license"
	"NN-512/internal/compile/author/loom"
	"NN-512/internal/compile/author/mod"
	"NN-512/internal/compile/author/net"
	"NN-512/internal/compile/author/one"
	"NN-512/internal/compile/author/params"
	"NN-512/internal/compile/author/rsqrt"
	"NN-512/internal/compile/author/softmax"
	"NN-512/internal/compile/author/strider"
	"NN-512/internal/compile/author/threader"
	"NN-512/internal/compile/author/three"
	"NN-512/internal/compile/author/thrpl"
	"NN-512/internal/compile/author/tobuild"
	"NN-512/internal/compile/author/twopl"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
)

func Implement(a *plan.Plan) (h, c []byte) {
	st := state{pl: a, nms: nmsrc.New()}
	st.stages()
	return st.hc.Join()
}

type state struct {
	pl          *plan.Plan
	hc          hc.Sections
	nms         nmsrc.Src
	paramsName  string
	errmsgCtx   *errmsg.Ctx
	threaderCtx *threader.Ctx
	expCtx      *exp.Ctx
	softmaxCtx  *softmax.Ctx
	actCtx      *act.Ctx
	rsqrtCtx    *rsqrt.Ctx
	bnCtx       *bn.Ctx
	elwiCtx     *elwi.Ctx
	gloplCtx    *glopl.Ctx
	twoplCtx    *twopl.Ctx
	thrplCtx    *thrpl.Ctx
	fcCtx       *fc.Ctx
	oneCtx      *one.Ctx
	threeCtx    *three.Ctx
	striderCtx  *strider.Ctx
	loomCtx     *loom.Ctx
	netCtx      *net.Ctx
	engineCtx   *engine.Ctx
}

func (st *state) stages() {
	st.stage1()
	st.stage2()
	st.stage3()
	st.stage4()
	st.stage5()
	st.stage6()
	st.stage7()
	st.stage8()
	st.stage9()
	st.stage10()
	st.stage11()
	st.stage12()
}

func (st *state) stage1() {
	st.hc.Append(hc.HPragmaOnce, cgen.PragmaOnce, cgen.Newline)
	st.hc.Append(hc.HLicense, license.Gen, cgen.Newline)
	st.hc.Append(hc.HInclude, include.H(), cgen.Newline)
	st.hc.Append(hc.HLinkage1, cgen.Linkage1, cgen.Newline)
	st.hc.Append(hc.HLinkage2, cgen.Linkage2, cgen.Newline)
	st.hc.Append(hc.HLast, eof.Gen)
}

func (st *state) stage2() {
	st.hc.Append(hc.CToBuild, tobuild.Gen(st.pl), cgen.Newline)
	st.hc.Append(hc.CLicense, license.Gen, cgen.Newline)
	st.hc.Append(hc.CInclude, include.C(st.pl), cgen.Newline)
	st.hc.Append(hc.CLast, eof.Gen)
}

func (st *state) stage3() {
	name := params.Name(st.pl)
	st.hc.Append(hc.HParams1, params.Fwd(name), cgen.Newline)
	st.hc.Append(hc.HParams2, params.Def(st.pl, name), cgen.Newline)
	st.paramsName = name
}

func (st *state) stage4() {
	ctx := errmsg.NewCtx(st.pl, st.nms)
	prep := &errmsg.Prep{Ctx: ctx}
	st.hc.Append(hc.CErrmsg, prep, cgen.Newline)
	st.errmsgCtx = ctx
}

func (st *state) stage5() {
	ctx := threader.NewCtx(st.pl, st.nms, st.errmsgCtx)
	prep := &threader.Prep{Ctx: ctx}
	st.hc.Append(hc.CThreader, prep, cgen.Newline)
	st.threaderCtx = ctx
}

func (st *state) stage6() {
	ctx := exp.NewCtx(st.pl, st.nms)
	prep := &exp.Prep{Ctx: ctx}
	st.hc.Append(hc.CExp, prep, cgen.Newline)
	st.expCtx = ctx
}

func (st *state) stage7() {
	st.softmaxCtx = softmax.NewCtx(st.pl, st.nms, st.threaderCtx, st.expCtx)
	st.actCtx = act.NewCtx(st.pl, st.nms)
}

func (st *state) stage8() {
	ctx := rsqrt.NewCtx(st.pl, st.nms)
	st.hc.Append(hc.CRsqrt, ctx.Prep(), cgen.Newline)
	st.rsqrtCtx = ctx
}

func (st *state) stage9() {
	st.bnCtx = bn.NewCtx(st.pl, st.nms, st.rsqrtCtx)
	st.elwiCtx = elwi.NewCtx(st.pl, st.nms, st.threaderCtx, st.actCtx, st.bnCtx)
	st.gloplCtx = glopl.NewCtx(st.pl, st.nms, st.threaderCtx, st.actCtx, st.bnCtx)
	st.twoplCtx = twopl.NewCtx(st.pl, st.nms, st.threaderCtx, st.actCtx, st.bnCtx)
	st.thrplCtx = thrpl.NewCtx(st.pl, st.nms, st.threaderCtx, st.actCtx, st.bnCtx)
	st.fcCtx = fc.NewCtx(st.pl, st.nms, st.threaderCtx, st.actCtx, st.bnCtx)
	st.oneCtx = one.NewCtx(st.pl, st.nms, st.threaderCtx, st.actCtx, st.bnCtx)
	st.threeCtx = three.NewCtx(st.pl, st.nms, st.threaderCtx, st.actCtx, st.bnCtx)
	st.striderCtx = strider.NewCtx(st.pl, st.nms, st.threaderCtx, st.actCtx, st.bnCtx)
	st.loomCtx = loom.NewCtx(st.pl, st.nms, st.threaderCtx, st.actCtx, st.bnCtx)
}

func (st *state) stage10() {
	ctx := net.NewCtx(st.pl, st.nms, st.paramsName)
	st.hc.Append(hc.HNet, ctx.Comment(), cgen.Newline)
	st.hc.Append(hc.HNet, ctx.StructFwd(), cgen.Newline)
	st.hc.Append(hc.HNet, ctx.CreateDecl(), cgen.Newline)
	st.hc.Append(hc.HNet, ctx.DestroyDecl(), cgen.Newline)
	st.hc.Append(hc.CNet, ctx.StructDef(), cgen.Newline)
	st.hc.Append(hc.CNet, ctx.DestroyDef(), cgen.Newline)
	st.netCtx = ctx
}

func (st *state) stage11() {
	ctx := engine.NewCtx(st.pl, st.nms, st.errmsgCtx, st.threaderCtx, st.netCtx)
	st.hc.Append(hc.HEngine, ctx.Comment(), cgen.Newline)
	st.hc.Append(hc.HEngine, ctx.StructFwd(), cgen.Newline)
	st.hc.Append(hc.HEngine, ctx.CreateDecl(), cgen.Newline)
	st.hc.Append(hc.HEngine, ctx.PthreadTDecl(), cgen.Newline)
	st.hc.Append(hc.HEngine, ctx.InferenceDecl(), cgen.Newline)
	st.hc.Append(hc.HEngine, ctx.DestroyDecl(), cgen.Newline)
	st.hc.Append(hc.CEngine, ctx.StructDef(), cgen.Newline)
	st.hc.Append(hc.CEngine, ctx.PthreadTDef(), cgen.Newline)
	st.hc.Append(hc.CEngine, ctx.DestroyDef(), cgen.Newline)
	st.engineCtx = ctx
}

func (st *state) stage12() {
	type link struct {
		chans       int
		height      int
		width       int
		elemBytes   int
		pitch1Bytes []int
		pitch2Bytes []int
		addrExprs   []cgen.Gen
		ops         [][]mod.Op
	}
	type bank struct {
		filts     int
		bnPre     int
		bnPost    int
		addrExprs []cgen.Gen
	}
	var (
		usedParams  bool
		netAlloc    cgen.Gen
		netAlign    cgen.Gen
		netBytes    int
		tmpAlloc    cgen.Gen
		tmpAlign    cgen.Gen
		tmpEdge     int
		tmpBytes    int
		netTeam     cgen.Gen
		netStmts    cgen.Stmts
		netBlocks   cgen.Stmts
		engNetAlign cgen.Gen
		engTeam     cgen.Gen
		engAlign    cgen.Gen
		engEdge     int
		engBytes    int
		engStmts    cgen.Stmts
		engBlocks   cgen.Stmts
		bnPersist   map[string]bool
		bnNetEng    map[string][2]cgen.Gen
		planOp      *plan.Op
		linkFrom    *link
		linkTo      *link
		banks       []*bank
	)
	il := func(i int) cgen.Gen {
		return cgen.IntLit(i)
	}
	vb := func(s string) cgen.Gen {
		return cgen.Vb(s)
	}
	nm := func(s string) cgen.Gen {
		return vb(st.nms.Name(s))
	}
	ptr := func(t cgen.Gen) cgen.Gen {
		return cgen.Ptr{Type: t}
	}
	unused := func(what cgen.Gen) cgen.Gen {
		return cgen.Cast{
			Type: cgen.Void,
			Expr: what,
		}
	}
	param := func(name string) cgen.Gen {
		usedParams = true
		return cgen.Arrow{
			Expr: st.netCtx.CreateParams,
			Name: name,
		}
	}
	netAddr := func() cgen.Gen {
		if netAlloc == nil {
			netAlloc = nm("alloc")
			netAlign = nm("align")
		}
		netBytes += st.netCtx.Alignment - 1
		netBytes &= -st.netCtx.Alignment
		return cgen.Add{
			Expr1: netAlign,
			Expr2: il(netBytes),
		}
	}
	netExtend := func(bytes int) {
		netBytes += bytes
	}
	tmpAddr := func() cgen.Gen {
		if tmpAlloc == nil {
			tmpAlloc = nm("tmpAlloc")
			tmpAlign = nm("tmpAlign")
		}
		tmpEdge += st.netCtx.Alignment - 1
		tmpEdge &= -st.netCtx.Alignment
		return cgen.Add{
			Expr1: tmpAlign,
			Expr2: il(tmpEdge),
		}
	}
	tmpExtend := func(bytes int) {
		tmpEdge += bytes
	}
	tmpRewind := func() {
		if tmpBytes < tmpEdge {
			tmpBytes = tmpEdge
		}
		tmpEdge = 0
	}
	NetTeam := func() cgen.Gen {
		if netTeam == nil {
			netTeam = nm("team")
		}
		return netTeam
	}
	netStmt := func(stmt cgen.Gen) {
		netStmts = append(netStmts, stmt)
	}
	netBlock := func() {
		if netStmts == nil {
			return
		}
		netBlocks = append(
			netBlocks, cgen.Block{
				Inner: netStmts,
			},
		)
		netStmts = nil
	}
	engNetAddr := func() cgen.Gen {
		if engNetAlign == nil {
			engNetAlign = nm("netAlign")
		}
		return cgen.Add{
			Expr1: engNetAlign,
			Expr2: il(netBytes),
		}
	}
	EngTeam := func() cgen.Gen {
		if engTeam == nil {
			engTeam = nm("team")
		}
		return engTeam
	}
	engAddr := func(off int) cgen.Gen {
		if engAlign == nil {
			engAlign = nm("align")
		}
		if off == -1 {
			engEdge += st.engineCtx.Alignment - 1
			engEdge &= -st.engineCtx.Alignment
			off = st.engineCtx.Split + engEdge
		}
		return cgen.Add{
			Expr1: engAlign,
			Expr2: il(off),
		}
	}
	engExtend := func(bytes int) {
		engEdge += bytes
	}
	engRewind := func() {
		if engBytes < engEdge {
			engBytes = engEdge
		}
		engEdge = 0
	}
	engStmt := func(stmt cgen.Gen) {
		engStmts = append(engStmts, stmt)
	}
	engBlock := func() {
		if engStmts == nil {
			return
		}
		engBlocks = append(
			engBlocks, cgen.Block{
				Inner: engStmts,
			},
		)
		engStmts = nil
	}
	bnKey := func(node *raw.BatchNorm) string {
		return node.MeansTensor
	}
	bnSimplify := func(
		node *raw.BatchNorm,
		chans int,
	) (netEng [2]cgen.Gen) {
		var (
			key     = bnKey(node)
			persist = bnPersist[key]
		)
		switch {
		case persist:
			var ok bool
			if netEng, ok = bnNetEng[key]; ok {
				return
			}
			netEng[0] = netAddr()
			netEng[1] = engNetAddr()
			bnNetEng[key] = netEng
		default:
			netEng[0] = tmpAddr()
			netEng[1] = nil
		}
		simplify := &bn.Simplify{
			Ctx:       st.bnCtx,
			Channels:  chans,
			Epsilon:   node.Epsilon,
			Means:     param(node.MeansTensor),
			Variances: param(node.VariancesTensor),
			Scales:    param(node.ScalesTensor),
			Shifts:    param(node.ShiftsTensor),
			Mas:       netEng[0],
		}
		st.hc.Append(hc.CBn, simplify.Prep())
		bytes := simplify.MasBytes()
		netStmt(simplify)
		switch {
		case persist:
			netExtend(bytes)
		default:
			tmpExtend(bytes)
		}
		return
	}
	linker := func(
		spans []*plan.Span,
		pmods [][]plan.Mod,
		prior bool,
	) (lnk *link) {
		doOnce := func(span *plan.Span) {
			var (
				chans = 0
				pile  = span.Piles[0]
			)
			for _, n := range span.Counts {
				chans += n
			}
			lnk = &link{
				chans:     chans,
				height:    pile.Height,
				width:     pile.Width,
				elemBytes: pile.ElemBytes,
				ops: make(
					[][]mod.Op, len(pmods),
				),
			}
		}
		ioName := func(pile *plan.Pile) string {
			for _, span := range pile.Writers {
				switch node := span.Op.Nodes[0].(type) {
				case *raw.Input:
					return node.ToTensor
				}
			}
			for _, span := range pile.Readers {
				switch node := span.Op.Nodes[0].(type) {
				case *raw.Output:
					return node.FromTensor
				}
			}
			panic("bug")
		}
		doSpan := func(span *plan.Span) {
			for i, pile := range span.Piles {
				var (
					pitch1 = pile.Pitch1Bytes
					pitch2 = pile.Pitch2Bytes
					off1   = pile.OffsetBytes
					ae     cgen.Gen
				)
				switch off1 {
				case -1:
					ae = cgen.Cast{
						Type: cgen.PtrChar,
						Expr: vb(ioName(pile)),
					}
				default:
					off2 := span.Offsets[i] * pitch2
					ae = engAddr(off1 + off2)
				}
				lnk.pitch1Bytes = append(
					lnk.pitch1Bytes, pitch1,
				)
				lnk.pitch2Bytes = append(
					lnk.pitch2Bytes, pitch2,
				)
				lnk.addrExprs = append(
					lnk.addrExprs, ae,
				)
			}
		}
		doSpans := func(spans []*plan.Span) {
			for _, span := range spans {
				doSpan(span)
			}
		}
		doBn := func(node *raw.BatchNorm) {
			netEng := bnSimplify(
				node, lnk.chans,
			)
			lnk.addrExprs = append(
				lnk.addrExprs,
				netEng[1],
			)
		}
		doMod := func(pmod *plan.Mod) (op mod.Op) {
			switch node := pmod.Nodes[0].(type) {
			case *raw.Activation:
				switch node.Kind {
				case raw.ReLU:
					op.Kind = mod.ReLU
					op.Float = node.Param
				default:
					panic("bug")
				}
			case *raw.Add:
				op.Kind = mod.Add
				op.Int = len(pmod.From)
				doSpans(pmod.From)
			case *raw.BatchNorm:
				op.Kind = mod.Bn
				doBn(node)
			default:
				panic("bug")
			}
			return
		}
		doMods := func(i int) {
			var (
				pms = pmods[i]
				ops = make([]mod.Op, len(pms))
			)
			for j := range pms {
				ops[j] = doMod(&pms[j])
			}
			lnk.ops[i] = ops
		}
		for i, span := range spans {
			if i == 0 {
				doOnce(span)
			}
			if prior {
				doSpan(span)
			}
			doMods(i)
			if !prior {
				doSpan(span)
			}
		}
		return
	}
	layer7 := func() {
		switch node := planOp.Nodes[0].(type) {
		case *raw.Activation, *raw.Add, *raw.BatchNorm:
			switch node := node.(type) {
			case *raw.Activation:
				switch node.Kind {
				case raw.ReLU:
					linkTo.ops[0] = append(
						[]mod.Op{{
							Kind:  mod.ReLU,
							Float: node.Param,
						}},
						linkTo.ops[0]...,
					)
				default:
					panic("bug")
				}
			case *raw.BatchNorm:
				netEng := bnSimplify(
					node, linkTo.chans,
				)
				linkTo.addrExprs = append(
					[]cgen.Gen{netEng[1]},
					linkTo.addrExprs...,
				)
				linkTo.ops[0] = append(
					[]mod.Op{{Kind: mod.Bn}},
					linkTo.ops[0]...,
				)
			}
			spec := &elwi.Spec{
				Channels:  linkFrom.chans,
				Height:    linkFrom.height,
				Width:     linkFrom.width,
				ElemBytes: linkFrom.elemBytes,
				Pitch1Bytes: append(
					linkFrom.pitch1Bytes,
					linkTo.pitch1Bytes...,
				),
				Pitch2Bytes: append(
					linkFrom.pitch2Bytes,
					linkTo.pitch2Bytes...,
				),
				Ops: append(
					linkFrom.ops,
					linkTo.ops[0],
				),
			}
			call := &elwi.Call{
				Ctx:  st.elwiCtx,
				Spec: spec,
				Team: EngTeam(),
				Tensors: append(
					linkFrom.addrExprs,
					linkTo.addrExprs...,
				),
			}
			st.hc.Append(hc.CElwi, call.Prep())
			engStmt(call)
		case *raw.Conv:
			useOne := func() bool {
				return true &&
					node.FilterH == 1 &&
					node.FilterW == 1
			}
			useThree := func() bool {
				return true &&
					node.FilterH == 3 &&
					node.FilterW == 3 &&
					node.StrideH == 1 &&
					node.StrideW == 1 &&
					node.DilationH == 1 &&
					node.DilationW == 1
			}
			useStrider := func() bool {
				return true &&
					node.FilterH <= 14 &&
					node.FilterW <= 14 &&
					node.FilterH*node.FilterW >= 9 &&
					node.StrideH == 2 &&
					node.StrideW == 2 &&
					node.DilationH == 1 &&
					node.DilationW == 1
			}
			switch {
			case useOne():
				spec := &one.Spec{
					From: one.SpecFrom{
						Chans:       linkFrom.chans,
						Height:      linkFrom.height,
						Width:       linkFrom.width,
						Pitch1Bytes: linkFrom.pitch1Bytes,
						Pitch2Bytes: linkFrom.pitch2Bytes,
						Ops:         linkFrom.ops[0],
					},
					Filts: make(
						[]one.SpecFilts, len(banks),
					),
					To: one.SpecTo{
						Pitch1Bytes: linkTo.pitch1Bytes,
						Pitch2Bytes: linkTo.pitch2Bytes,
						Ops:         linkTo.ops[0],
					},
					StrideH:  node.StrideH,
					StrideW:  node.StrideW,
					PaddingH: node.PaddingH,
					PaddingW: node.PaddingW,
					Groups:   node.Groups,
				}
				for i, bnk := range banks {
					spec.Filts[i] = one.SpecFilts{
						Cnt:    bnk.filts,
						BnPre:  bnk.bnPre,
						BnPost: bnk.bnPost,
					}
					if i == 0 {
						continue
					}
					banks[0].addrExprs = append(
						banks[0].addrExprs,
						bnk.addrExprs...,
					)
				}
				arrangeWts := &one.ArrangeWts{
					Ctx:  st.oneCtx,
					Spec: spec,
					Team: NetTeam(),
					Tensors: append(
						banks[0].addrExprs,
						netAddr(),
					),
				}
				addrWts := engNetAddr()
				st.hc.Append(hc.COne, arrangeWts.Prep())
				netExtend(arrangeWts.Bytes())
				netStmt(arrangeWts)
				addrDats := engAddr(-1)
				arrangeDats := &one.ArrangeDats{
					Ctx:  st.oneCtx,
					Spec: spec,
					Team: EngTeam(),
					Tensors: append(
						linkFrom.addrExprs,
						addrDats,
					),
				}
				st.hc.Append(hc.COne, arrangeDats.Prep())
				engExtend(arrangeDats.Bytes())
				engStmt(arrangeDats)
				apply := &one.Apply{
					Ctx:  st.oneCtx,
					Spec: spec,
					Team: EngTeam(),
					Tensors: append(
						[]cgen.Gen{
							addrWts,
							addrDats,
						},
						linkTo.addrExprs...,
					),
				}
				st.hc.Append(hc.COne, apply.Prep())
				engStmt(apply)
			case useThree():
				spec := &three.Spec{
					From: three.SpecFrom{
						Chans:       linkFrom.chans,
						Height:      linkFrom.height,
						Width:       linkFrom.width,
						Pitch1Bytes: linkFrom.pitch1Bytes,
						Pitch2Bytes: linkFrom.pitch2Bytes,
						Ops:         linkFrom.ops[0],
					},
					Filts: make(
						[]three.SpecFilts, len(banks),
					),
					To: three.SpecTo{
						Pitch1Bytes: linkTo.pitch1Bytes,
						Pitch2Bytes: linkTo.pitch2Bytes,
						Ops:         linkTo.ops[0],
					},
					StrideH:  node.StrideH,
					StrideW:  node.StrideW,
					PaddingH: node.PaddingH,
					PaddingW: node.PaddingW,
					Groups:   node.Groups,
				}
				for i, bnk := range banks {
					spec.Filts[i] = three.SpecFilts{
						Cnt:    bnk.filts,
						BnPre:  bnk.bnPre,
						BnPost: bnk.bnPost,
					}
					if i == 0 {
						continue
					}
					banks[0].addrExprs = append(
						banks[0].addrExprs,
						bnk.addrExprs...,
					)
				}
				arrangeFilts := &three.ArrangeFilts{
					Ctx:  st.threeCtx,
					Spec: spec,
					Team: NetTeam(),
					Tensors: append(
						banks[0].addrExprs,
						netAddr(),
					),
				}
				addrFilts := engNetAddr()
				st.hc.Append(hc.CThree, arrangeFilts.Prep())
				netExtend(arrangeFilts.Bytes())
				netStmt(arrangeFilts)
				addrDats := engAddr(-1)
				arrangeDats := &three.ArrangeDats{
					Ctx:  st.threeCtx,
					Spec: spec,
					Team: EngTeam(),
					Tensors: append(
						linkFrom.addrExprs,
						addrDats,
					),
				}
				st.hc.Append(hc.CThree, arrangeDats.Prep())
				engExtend(arrangeDats.Bytes())
				engStmt(arrangeDats)
				addrSums := engAddr(-1)
				produceSums := &three.ProduceSums{
					Ctx:  st.threeCtx,
					Spec: spec,
					Team: EngTeam(),
					Tensors: []cgen.Gen{
						addrFilts,
						addrDats,
						addrSums,
					},
				}
				st.hc.Append(hc.CThree, produceSums.Prep())
				engExtend(produceSums.Bytes())
				engStmt(produceSums)
				consumeSums := &three.ConsumeSums{
					Ctx:  st.threeCtx,
					Spec: spec,
					Team: EngTeam(),
					Tensors: append(
						[]cgen.Gen{addrSums},
						linkTo.addrExprs...,
					),
				}
				st.hc.Append(hc.CThree, consumeSums.Prep())
				engStmt(consumeSums)
			case useStrider():
				spec := &strider.Spec{
					From: strider.SpecFrom{
						Chans:       linkFrom.chans,
						Height:      linkFrom.height,
						Width:       linkFrom.width,
						Pitch1Bytes: linkFrom.pitch1Bytes,
						Pitch2Bytes: linkFrom.pitch2Bytes,
						Ops:         linkFrom.ops[0],
					},
					Filts: make(
						[]strider.SpecFilts, len(banks),
					),
					To: strider.SpecTo{
						Pitch1Bytes: linkTo.pitch1Bytes,
						Pitch2Bytes: linkTo.pitch2Bytes,
						Ops:         linkTo.ops[0],
					},
					FilterH:   node.FilterH,
					FilterW:   node.FilterW,
					PaddingH:  node.PaddingH,
					PaddingW:  node.PaddingW,
					DilationH: node.DilationH,
					DilationW: node.DilationW,
					Groups:    node.Groups,
				}
				for i, bnk := range banks {
					spec.Filts[i] = strider.SpecFilts{
						Cnt:    bnk.filts,
						BnPre:  bnk.bnPre,
						BnPost: bnk.bnPost,
					}
					if i == 0 {
						continue
					}
					banks[0].addrExprs = append(
						banks[0].addrExprs,
						bnk.addrExprs...,
					)
				}
				arrangeFilts := &strider.ArrangeFilts{
					Ctx:  st.striderCtx,
					Spec: spec,
					Team: NetTeam(),
					Tensors: append(
						banks[0].addrExprs,
						netAddr(),
					),
				}
				addrFilts := engNetAddr()
				st.hc.Append(hc.CStrider, arrangeFilts.Prep())
				netExtend(arrangeFilts.Bytes())
				netStmt(arrangeFilts)
				addrDats := engAddr(-1)
				arrangeDats := &strider.ArrangeDats{
					Ctx:  st.striderCtx,
					Spec: spec,
					Team: EngTeam(),
					Tensors: append(
						linkFrom.addrExprs,
						addrDats,
					),
				}
				st.hc.Append(hc.CStrider, arrangeDats.Prep())
				engExtend(arrangeDats.Bytes())
				engStmt(arrangeDats)
				addrSums := engAddr(-1)
				produceSums := &strider.ProduceSums{
					Ctx:  st.striderCtx,
					Spec: spec,
					Team: EngTeam(),
					Tensors: []cgen.Gen{
						addrFilts,
						addrDats,
						addrSums,
					},
				}
				st.hc.Append(hc.CStrider, produceSums.Prep())
				engExtend(produceSums.Bytes())
				engStmt(produceSums)
				consumeSums := &strider.ConsumeSums{
					Ctx:  st.striderCtx,
					Spec: spec,
					Team: EngTeam(),
					Tensors: append(
						[]cgen.Gen{addrSums},
						linkTo.addrExprs...,
					),
				}
				st.hc.Append(hc.CStrider, consumeSums.Prep())
				engStmt(consumeSums)
			default:
				spec := &loom.Spec{
					From: loom.SpecFrom{
						Chans:       linkFrom.chans,
						Height:      linkFrom.height,
						Width:       linkFrom.width,
						Pitch1Bytes: linkFrom.pitch1Bytes,
						Pitch2Bytes: linkFrom.pitch2Bytes,
						Ops:         linkFrom.ops[0],
					},
					Filts: make(
						[]loom.SpecFilts, len(banks),
					),
					To: loom.SpecTo{
						Pitch1Bytes: linkTo.pitch1Bytes,
						Pitch2Bytes: linkTo.pitch2Bytes,
						Ops:         linkTo.ops[0],
					},
					FilterH:   node.FilterH,
					FilterW:   node.FilterW,
					StrideH:   node.StrideH,
					StrideW:   node.StrideW,
					PaddingH:  node.PaddingH,
					PaddingW:  node.PaddingW,
					DilationH: node.DilationH,
					DilationW: node.DilationW,
					Groups:    node.Groups,
				}
				for i, bnk := range banks {
					spec.Filts[i] = loom.SpecFilts{
						Cnt:    bnk.filts,
						BnPre:  bnk.bnPre,
						BnPost: bnk.bnPost,
					}
					if i == 0 {
						continue
					}
					banks[0].addrExprs = append(
						banks[0].addrExprs,
						bnk.addrExprs...,
					)
				}
				arrangeFilts := &loom.ArrangeFilts{
					Ctx:  st.loomCtx,
					Spec: spec,
					Team: NetTeam(),
					Tensors: append(
						banks[0].addrExprs,
						netAddr(),
					),
				}
				addrFilts := engNetAddr()
				st.hc.Append(hc.CLoom, arrangeFilts.Prep())
				netExtend(arrangeFilts.Bytes())
				netStmt(arrangeFilts)
				addrDats := engAddr(-1)
				arrangeDats := &loom.ArrangeDats{
					Ctx:  st.loomCtx,
					Spec: spec,
					Team: EngTeam(),
					Tensors: append(
						linkFrom.addrExprs,
						addrDats,
					),
				}
				st.hc.Append(hc.CLoom, arrangeDats.Prep())
				engExtend(arrangeDats.Bytes())
				engStmt(arrangeDats)
				addrSums := engAddr(-1)
				produceSums := &loom.ProduceSums{
					Ctx:  st.loomCtx,
					Spec: spec,
					Team: EngTeam(),
					Tensors: []cgen.Gen{
						addrFilts,
						addrDats,
						addrSums,
					},
				}
				st.hc.Append(hc.CLoom, produceSums.Prep())
				engExtend(produceSums.Bytes())
				engStmt(produceSums)
				consumeSums := &loom.ConsumeSums{
					Ctx:  st.loomCtx,
					Spec: spec,
					Team: EngTeam(),
					Tensors: append(
						[]cgen.Gen{addrSums},
						linkTo.addrExprs...,
					),
				}
				st.hc.Append(hc.CLoom, consumeSums.Prep())
				engStmt(consumeSums)
			}
		case *raw.FullyConnected:
			netEng := [2]cgen.Gen{
				netAddr(),
				engNetAddr(),
			}
			arrange := &fc.Arrange{
				Ctx:    st.fcCtx,
				ToC:    linkTo.chans,
				FromC:  linkFrom.chans,
				FromH:  linkFrom.height,
				FromW:  linkFrom.width,
				BnPre:  banks[0].bnPre,
				BnPost: banks[0].bnPost,
				Team:   NetTeam(),
				Tensors: append(
					banks[0].addrExprs,
					netEng[0],
				),
			}
			st.hc.Append(hc.CFc, arrange.Prep())
			netExtend(arrange.Bytes())
			netStmt(arrange)
			apply := &fc.Apply{
				Ctx:   st.fcCtx,
				ToC:   linkTo.chans,
				FromC: linkFrom.chans,
				FromH: linkFrom.height,
				FromW: linkFrom.width,
				Ops:   linkTo.ops[0],
				Team:  EngTeam(),
				Tensors: append(
					[]cgen.Gen{
						netEng[1],
						linkFrom.addrExprs[0],
					},
					linkTo.addrExprs...,
				),
			}
			st.hc.Append(hc.CFc, apply.Prep())
			engStmt(apply)
		case *raw.Input:
		case *raw.Output:
		case *raw.Pooling:
			tensors := append(
				linkFrom.addrExprs,
				linkTo.addrExprs...,
			)
			switch node.Kind {
			case raw.Max2x2Stride2, raw.Avg2x2Stride2:
				spec := &twopl.Spec{
					Kind:     node.Kind,
					PaddingH: node.PaddingH,
					PaddingW: node.PaddingW,
					Channels: linkFrom.chans,
					From: twopl.SpecFrom{
						Height:      linkFrom.height,
						Width:       linkFrom.width,
						Pitch1Bytes: linkFrom.pitch1Bytes,
						Pitch2Bytes: linkFrom.pitch2Bytes,
						Ops:         linkFrom.ops[0],
					},
					To: twopl.SpecTo{
						Pitch1Bytes: linkTo.pitch1Bytes,
						Pitch2Bytes: linkTo.pitch2Bytes,
						Ops:         linkTo.ops[0],
					},
				}
				call := &twopl.Call{
					Ctx:     st.twoplCtx,
					Spec:    spec,
					Team:    EngTeam(),
					Tensors: tensors,
				}
				st.hc.Append(hc.CTwopl, call.Prep())
				engStmt(call)
			case raw.Max3x3Stride2, raw.Avg3x3Stride2:
				spec := &thrpl.Spec{
					Kind:     node.Kind,
					PaddingH: node.PaddingH,
					PaddingW: node.PaddingW,
					Channels: linkFrom.chans,
					From: thrpl.SpecFrom{
						Height:      linkFrom.height,
						Width:       linkFrom.width,
						Pitch1Bytes: linkFrom.pitch1Bytes,
						Pitch2Bytes: linkFrom.pitch2Bytes,
						Ops:         linkFrom.ops[0],
					},
					To: thrpl.SpecTo{
						Pitch1Bytes: linkTo.pitch1Bytes,
						Pitch2Bytes: linkTo.pitch2Bytes,
						Ops:         linkTo.ops[0],
					},
				}
				call := &thrpl.Call{
					Ctx:     st.thrplCtx,
					Spec:    spec,
					Team:    EngTeam(),
					Tensors: tensors,
				}
				st.hc.Append(hc.CThrpl, call.Prep())
				engStmt(call)
			case raw.MaxGlobal, raw.AvgGlobal:
				spec := &glopl.Spec{
					Kind:      node.Kind,
					Channels:  linkFrom.chans,
					ElemBytes: linkFrom.elemBytes,
					From: glopl.SpecFrom{
						Height:      linkFrom.height,
						Width:       linkFrom.width,
						Pitch1Bytes: linkFrom.pitch1Bytes,
						Pitch2Bytes: linkFrom.pitch2Bytes,
						Ops:         linkFrom.ops[0],
					},
					To: glopl.SpecTo{
						Ops: linkTo.ops[0],
						Cnt: len(planOp.To[0].Piles),
					},
				}
				call := &glopl.Call{
					Ctx:     st.gloplCtx,
					Spec:    spec,
					Team:    EngTeam(),
					Tensors: tensors,
				}
				st.hc.Append(hc.CGlopl, call.Prep())
				engStmt(call)
			default:
				panic("bug")
			}
		case *raw.Softmax:
			var (
				n       = 1 + len(linkTo.addrExprs)
				shapes  = make([]softmax.Shape, n)
				tensors = make([]cgen.Gen, n)
			)
			for i := 0; i < n; i++ {
				var (
					lnk = linkFrom
					j   = 0
				)
				if i > 0 {
					lnk = linkTo
					j = i - 1
				}
				shapes[i] = softmax.Shape{
					Channels:    lnk.chans,
					Height:      lnk.height,
					Width:       lnk.width,
					ElemBytes:   lnk.elemBytes,
					Pitch1Bytes: lnk.pitch1Bytes[j],
					Pitch2Bytes: lnk.pitch2Bytes[j],
				}
				tensors[i] = lnk.addrExprs[j]
			}
			call := &softmax.Call{
				Ctx:     st.softmaxCtx,
				Team:    EngTeam(),
				Tensors: tensors,
				Shapes:  shapes,
			}
			st.hc.Append(hc.CSoftmax, call.Prep())
			engStmt(call)
		default:
			panic("bug")
		}
	}
	layer6 := func() {
		sublayer2 := func() {
			var (
				span = planOp.To[0]
				n1   = len(span.Counts)
			)
			banks = make([]*bank, n1)
			for i, filts := range span.Counts {
				var (
					ps     = planOp.Params[i]
					pms    = planOp.ParamMods[i]
					bnPre  = len(pms[0])
					bnPost = len(pms[1])
					n2     = 2 + bnPre + bnPost
					aes    = make([]cgen.Gen, n2)
				)
				banks[i] = &bank{
					filts:     filts,
					bnPre:     bnPre,
					bnPost:    bnPost,
					addrExprs: aes,
				}
				for j := 0; j < 2; j++ {
					aes[j] = cgen.Cast{
						Type: cgen.PtrChar,
						Expr: param(ps[j].Tensor),
					}
				}
				for j := 0; j < bnPre; j++ {
					node := pms[0][j].Nodes[0].(*raw.BatchNorm)
					netEng := bnSimplify(node, linkFrom.chans)
					aes[2+j] = netEng[0]
				}
				for j := 0; j < bnPost; j++ {
					node := pms[1][j].Nodes[0].(*raw.BatchNorm)
					netEng := bnSimplify(node, filts)
					aes[2+bnPre+j] = netEng[0]
				}
			}
		}
		sublayer1 := func() {
			switch planOp.Nodes[0].(type) {
			case *raw.Conv, *raw.FullyConnected:
				sublayer2()
			default:
				banks = nil
			}
			layer7()
		}
		sublayer1()
	}
	layer5 := func() {
		linkFrom = linker(
			planOp.From,
			planOp.FromMods,
			true,
		)
		linkTo = linker(
			planOp.To,
			planOp.ToMods,
			false,
		)
		layer6()
	}
	layer4 := func() {
		for _, planOp = range st.pl.Seq {
			layer5()
			tmpRewind()
			engRewind()
			netBlock()
			engBlock()
		}
	}
	layer3 := func() {
		sublayer2 := func() {
			do := func(nodes []raw.Node) {
				for _, node := range nodes {
					switch node := node.(type) {
					case *raw.BatchNorm:
						key := bnKey(node)
						bnPersist[key] = true
					}
				}
			}
			doMods := func(pmods [][]plan.Mod) {
				for _, pms := range pmods {
					for i := range pms {
						pm := &pms[i]
						do(pm.Nodes)
					}
				}
			}
			for _, op := range st.pl.Seq {
				do(op.Nodes)
				doMods(op.FromMods)
				doMods(op.ToMods)
			}
		}
		sublayer1 := func() {
			bnPersist = make(map[string]bool)
			bnNetEng = make(map[string][2]cgen.Gen)
			sublayer2()
			layer4()
		}
		sublayer1()
	}
	layer2 := func() {
		sublayer4 := func() cgen.Gen {
			var (
				eng  = st.engineCtx.InferenceEng
				used = false
			)
			field := func(name string) cgen.Gen {
				used = true
				return cgen.Arrow{
					Expr: eng, Name: name,
				}
			}
			return cgen.Stmts{
				func() cgen.Gen {
					if engNetAlign == nil {
						return nil
					}
					return cgen.Var{
						Type: cgen.PtrChar,
						What: engNetAlign,
						Init: cgen.Arrow{
							Expr: field(st.engineCtx.StructNet),
							Name: st.netCtx.StructAlign,
						},
					}
				}(),
				func() cgen.Gen {
					if engTeam == nil {
						return nil
					}
					return cgen.Var{
						Type: st.threaderCtx.PtrTeam,
						What: engTeam,
						Init: field(st.engineCtx.StructTeam),
					}
				}(),
				func() cgen.Gen {
					if engAlign == nil {
						return nil
					}
					return cgen.Var{
						Type: cgen.PtrChar,
						What: engAlign,
						Init: field(st.engineCtx.StructAlign),
					}
				}(),
				func() cgen.Gen {
					if used {
						return nil
					}
					return unused(eng)
				}(),
				engBlocks,
			}
		}
		sublayer3 := func() {
			var (
				body = sublayer4()
				def  = st.engineCtx.InferenceDef(body)
			)
			st.hc.Append(hc.CEngine, def, cgen.Newline)
		}
		sublayer2 := func() {
			def := st.engineCtx.CreateDef(engBytes)
			st.hc.Append(hc.CEngine, def, cgen.Newline)
			sublayer3()
		}
		sublayer1 := func() {
			engNetAlign = nil
			engTeam = nil
			engAlign = nil
			engEdge = 0
			engBytes = 0
			engStmts = nil
			engBlocks = nil
			layer3()
			sublayer2()
		}
		sublayer1()
	}
	layer1 := func() {
		malloc := func(bytes cgen.Gen) cgen.Gen {
			return cgen.Call{
				Func: cgen.Malloc,
				Args: bytes,
			}
		}
		alloc := func(what, unwind cgen.Gen, bytes int) cgen.Gen {
			return cgen.Stmts{
				cgen.Var{
					Type: cgen.PtrChar,
					What: what,
					Init: malloc(il(
						st.netCtx.Alignment - 1 + bytes,
					)),
				},
				&errmsg.ErrnoIf{
					Ctx:    st.errmsgCtx,
					Cond:   cgen.IsZero{Expr: what},
					Unwind: unwind,
				},
			}
		}
		align := func(dest, src cgen.Gen) cgen.Gen {
			expr := cgen.And{
				Expr1: cgen.Paren{
					Inner: cgen.Add{
						Expr1: cgen.Cast{
							Type: cgen.SizeT,
							Expr: src,
						},
						Expr2: il(st.netCtx.Alignment - 1),
					},
				},
				Expr2: il(-st.netCtx.Alignment),
			}
			return cgen.Var{
				Type: cgen.PtrChar,
				What: dest,
				Init: cgen.Cast{
					Type: cgen.PtrVoid,
					Expr: cgen.Paren{Inner: expr},
				},
			}
		}
		free := func(what cgen.Gen) cgen.Gen {
			if what == nil {
				return nil
			}
			return cgen.Call{
				Func: cgen.Free,
				Args: what,
			}
		}
		sublayer6 := func() cgen.Gen {
			if netTeam == nil {
				return netBlocks
			}
			unwind := func() cgen.Gen {
				var (
					free1 = free(tmpAlloc)
					free2 = free(netAlloc)
				)
				switch {
				case free1 == nil:
					return free2
				case free2 == nil:
					return free1
				}
				return cgen.Stmts{
					free1,
					free2,
				}
			}()
			return cgen.Stmts{
				cgen.Var{
					Type: st.threaderCtx.PtrTeam,
					What: netTeam,
					Init: il(0),
				},
				&threader.Create{
					Ctx:    st.threaderCtx,
					Team:   cgen.Addr{Expr: netTeam},
					Nt:     st.netCtx.CreateThreads,
					Unwind: unwind,
				},
				netBlocks,
				&threader.Destroy{
					Ctx:  st.threaderCtx,
					Team: netTeam,
				},
			}
		}
		sublayer5 := func() cgen.Gen {
			if tmpAlloc == nil {
				return sublayer6()
			}
			return cgen.Stmts{
				alloc(tmpAlloc, free(netAlloc), tmpBytes),
				align(tmpAlign, tmpAlloc),
				sublayer6(),
				free(tmpAlloc),
			}
		}
		sublayer4 := func() cgen.Gen {
			out := nm("net")
			put := func(dest string, src cgen.Gen) cgen.Gen {
				if src == nil {
					src = il(0)
				}
				return cgen.Assign{
					Expr1: cgen.Arrow{
						Expr: out, Name: dest,
					},
					Expr2: src,
				}
			}
			return cgen.Stmts{
				func() cgen.Gen {
					if netAlloc == nil {
						return nil
					}
					return cgen.Stmts{
						alloc(netAlloc, nil, netBytes),
						align(netAlign, netAlloc),
					}
				}(),
				sublayer5(),
				cgen.Var{
					Type: ptr(vb(st.netCtx.StructName)),
					What: out,
					Init: malloc(cgen.Sizeof{
						What: vb(st.netCtx.StructName),
					}),
				},
				&errmsg.ErrnoIf{
					Ctx:    st.errmsgCtx,
					Cond:   cgen.IsZero{Expr: out},
					Unwind: free(netAlloc),
				},
				put(st.netCtx.StructAlloc, netAlloc),
				put(st.netCtx.StructAlign, netAlign),
				cgen.Assign{
					Expr1: cgen.At{
						Expr: st.netCtx.CreateNet,
					},
					Expr2: out,
				},
			}
		}
		sublayer3 := func() cgen.Gen {
			return cgen.Stmts{
				func() cgen.Gen {
					if usedParams {
						return nil
					}
					return unused(st.netCtx.CreateParams)
				}(),
				func() cgen.Gen {
					if netTeam != nil {
						return nil
					}
					return unused(st.netCtx.CreateThreads)
				}(),
				&cpu.Chk{
					Platform: st.pl.Config.Platform,
					Emc:      st.errmsgCtx,
				},
				sublayer4(),
				cgen.Return{
					Expr: il(0),
				},
			}
		}
		sublayer2 := func() {
			var (
				body = sublayer3()
				def  = st.netCtx.CreateDef(body)
			)
			st.hc.Append(hc.CNet, def, cgen.Newline)
		}
		sublayer1 := func() {
			usedParams = false
			netAlloc = nil
			netAlign = nil
			netBytes = 0
			tmpAlloc = nil
			tmpAlign = nil
			tmpEdge = 0
			tmpBytes = 0
			netTeam = nil
			netStmts = nil
			netBlocks = nil
			layer2()
			sublayer2()
		}
		sublayer1()
	}
	layer1()
}
