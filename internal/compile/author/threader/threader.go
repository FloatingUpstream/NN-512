package threader

import (
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/errmsg"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
)

const maxNd = 4

func vb(a string) cgen.Gen {
	return cgen.Vb(a)
}

type Ctx struct {
	prefix     string
	cacheLine  int
	nms        nmsrc.Src
	emc        *errmsg.Ctx
	taskType   string
	taskCallee string
	taskAny    string
	taskNd     string
	taskHull   string
	teamType   string
	destroy    string
	create     string
	pthreadT   string
	do         string
	ptrTask    cgen.Gen
	PtrTeam    cgen.Gen
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src, emc *errmsg.Ctx) *Ctx {
	prefix := pl.Config.Prefix + "Threader"
	var cacheLine int
	switch pl.Config.Platform {
	case raw.AVX512Float32:
		cacheLine = 1 << 6
	default:
		panic("bug")
	}
	ctx := &Ctx{
		prefix:     prefix,
		cacheLine:  cacheLine,
		nms:        nms,
		emc:        emc,
		taskType:   nms.Name(prefix + "Task"),
		taskCallee: nms.Name("callee"),
		taskAny:    nms.Name("any"),
		taskNd:     nms.Name("nd"),
		taskHull:   nms.Name("hull"),
		teamType:   nms.Name(prefix + "Team"),
		destroy:    nms.Name(prefix + "Destroy"),
		create:     nms.Name(prefix + "Create"),
		pthreadT:   nms.Name(prefix + "PthreadT"),
		do:         nms.Name(prefix + "Do"),
	}
	ctx.ptrTask = cgen.Ptr{Type: vb(ctx.taskType)}
	ctx.PtrTeam = cgen.Ptr{Type: vb(ctx.teamType)}
	return ctx
}

func (c *Ctx) name(a string) string {
	return c.nms.Name(a)
}

func (c *Ctx) nameP(a string) string {
	return c.name(c.prefix + a)
}

type must cgen.Call

func (m must) Append(to []byte) []byte {
	return cgen.For{Cond: cgen.Unlikely{
		Cond: cgen.Call(m),
	}}.Append(to)
}

type lock struct {
	ptr cgen.Gen
	mut string
}

func (l lock) Append(to []byte) []byte {
	return must{
		Func: cgen.PthreadMutexLock,
		Args: cgen.AddrArrow{Expr: l.ptr, Name: l.mut},
	}.Append(to)
}

type unlock struct {
	ptr cgen.Gen
	mut string
}

func (u unlock) Append(to []byte) []byte {
	return must{
		Func: cgen.PthreadMutexUnlock,
		Args: cgen.AddrArrow{Expr: u.ptr, Name: u.mut},
	}.Append(to)
}

type wait struct {
	ptr  cgen.Gen
	cond string
	mut  string
}

func (w wait) Append(to []byte) []byte {
	return must{
		Func: cgen.PthreadCondWait,
		Args: cgen.CommaSpaced{
			cgen.AddrArrow{Expr: w.ptr, Name: w.cond},
			cgen.AddrArrow{Expr: w.ptr, Name: w.mut},
		},
	}.Append(to)
}

type signal struct {
	ptr  cgen.Gen
	cond string
}

func (s signal) Append(to []byte) []byte {
	return must{
		Func: cgen.PthreadCondSignal,
		Args: cgen.AddrArrow{Expr: s.ptr, Name: s.cond},
	}.Append(to)
}

type join struct {
	ptr cgen.Gen
	thr string
}

func (j join) Append(to []byte) []byte {
	return must{
		Func: cgen.PthreadJoin,
		Args: cgen.CommaSpaced{
			cgen.Arrow{Expr: j.ptr, Name: j.thr},
			cgen.Zero,
		},
	}.Append(to)
}

type destroyMut struct {
	ptr cgen.Gen
	mut string
}

func (d destroyMut) Append(to []byte) []byte {
	return must{
		Func: cgen.PthreadMutexDestroy,
		Args: cgen.AddrArrow{Expr: d.ptr, Name: d.mut},
	}.Append(to)
}

type destroyCond struct {
	ptr  cgen.Gen
	cond string
}

func (d destroyCond) Append(to []byte) []byte {
	return must{
		Func: cgen.PthreadCondDestroy,
		Args: cgen.AddrArrow{Expr: d.ptr, Name: d.cond},
	}.Append(to)
}

type round struct {
	expr cgen.Gen
	pow2 cgen.IntLit
}

func (r round) Append(to []byte) []byte {
	return cgen.And{
		Expr1: cgen.Paren{Inner: cgen.Add{
			Expr1: r.expr,
			Expr2: r.pow2 - 1,
		}},
		Expr2: -r.pow2,
	}.Append(to)
}

type Prep struct {
	*Ctx
	to              []byte
	maxNd           cgen.Gen
	calleeType      cgen.Gen
	hubType         string
	hubMut          string
	hubCond         string
	hubPending      string
	hubOffset       string
	hubMask         string
	hubStatus       string
	nodeType        string
	nodeMut         string
	nodeNp          string
	nodePt          string
	nodeTask        string
	nodeCond        string
	nodeTeam        string
	nodeThr         string
	unwindType      string
	unwindJoin      string
	unwindNodeConds string
	unwindNodeMuts  string
	unwindHubCond   string
	unwindHubMut    string
	unwindNodes     string
	unwindHub       string
	teamNt          string
	teamHub         string
	teamNodes       string
	teamUnwind      string
	ptrHub          cgen.Gen
	ptrNode         cgen.Gen
	inc             string
	put             string
	add             string
	main            string
}

func (p *Prep) Append(to []byte) []byte {
	p.to = to
	p.stage1()
	p.stage2()
	p.stage3()
	p.stage4()
	p.stage5()
	p.stage6()
	p.stage7()
	p.stage8()
	p.stage9()
	return p.to
}

func (p *Prep) newline() {
	p.to = cgen.Newline.Append(p.to)
}

func (p *Prep) stage1() {
	p.maxNd = cgen.IntLit(maxNd)
	p.calleeType = vb(p.nameP("Callee"))
	p.hubType = p.nameP("Hub")
	p.hubMut = p.name("mut")
	p.hubCond = p.name("cond")
	p.hubPending = p.name("pending")
	p.hubOffset = p.name("offset")
	p.hubMask = p.name("mask")
	p.hubStatus = p.name("status")
	p.nodeType = p.nameP("Node")
	p.nodeMut = p.name("mut")
	p.nodeNp = p.name("np")
	p.nodePt = p.name("pt")
	p.nodeTask = p.name("task")
	p.nodeCond = p.name("cond")
	p.nodeTeam = p.name("team")
	p.nodeThr = p.name("thr")
	p.unwindType = p.nameP("Unwind")
	p.unwindJoin = p.name("join")
	p.unwindNodeConds = p.name("nodeConds")
	p.unwindNodeMuts = p.name("nodeMuts")
	p.unwindHubCond = p.name("hubCond")
	p.unwindHubMut = p.name("hubMut")
	p.unwindNodes = p.name("nodes")
	p.unwindHub = p.name("hub")
	p.teamNt = p.name("nt")
	p.teamHub = p.name("hub")
	p.teamNodes = p.name("nodes")
	p.teamUnwind = p.name("unwind")
	p.ptrHub = cgen.Ptr{Type: vb(p.hubType)}
	p.ptrNode = cgen.Ptr{Type: vb(p.nodeType)}
	p.inc = p.nameP("Inc")
	p.put = p.nameP("Put")
	p.add = p.nameP("Add")
	p.main = p.nameP("Main")
}

func (p *Prep) stage2() {
	p.to = cgen.Gens{
		cgen.StructFwd(p.taskType),
		cgen.TypedefPtrFunc{
			ReturnType: cgen.Void,
			What:       p.calleeType,
			Params:     cgen.CommaSpaced{p.ptrTask, cgen.PtrInt64T},
		},
		cgen.StructFwd(p.hubType),
		cgen.StructFwd(p.nodeType),
		cgen.StructFwd(p.unwindType),
		cgen.StructFwd(p.teamType),
		cgen.Newline,
	}.Append(p.to)
}

func (p *Prep) stage3() {
	p.to = cgen.Gens{
		cgen.StructDef{
			Name: p.taskType,
			Fields: cgen.Stmts{
				cgen.Field{Type: p.calleeType, What: vb(p.taskCallee)},
				cgen.Field{Type: cgen.PtrVoid, What: vb(p.taskAny)},
				cgen.Field{Type: cgen.PtrdiffT, What: vb(p.taskNd)},
				cgen.Field{
					Type: cgen.Int64T,
					What: cgen.Elem{Arr: vb(p.taskHull), Idx: p.maxNd},
				},
			},
		},
		cgen.Newline,
		cgen.StructDef{
			Name: p.hubType,
			Fields: cgen.Stmts{
				cgen.Field{Type: cgen.PthreadMutexT, What: vb(p.hubMut)},
				cgen.Field{Type: cgen.PthreadCondT, What: vb(p.hubCond)},
				cgen.Field{Type: cgen.PtrdiffT, What: vb(p.hubPending)},
				cgen.Field{Type: cgen.PtrdiffT, What: vb(p.hubOffset)},
				cgen.Field{Type: cgen.Long, What: vb(p.hubMask)},
				cgen.Field{
					Type: cgen.Long,
					What: cgen.Elem{Arr: vb(p.hubStatus)},
				},
			},
		},
		cgen.Newline,
		cgen.StructDef{
			Name: p.nodeType,
			Fields: cgen.Stmts{
				cgen.Field{Type: cgen.PthreadMutexT, What: vb(p.nodeMut)},
				cgen.Field{Type: cgen.Int64T, What: vb(p.nodeNp)},
				cgen.Field{
					Type: cgen.Int64T,
					What: cgen.Elem{Arr: vb(p.nodePt), Idx: p.maxNd},
				},
				cgen.Field{Type: p.ptrTask, What: vb(p.nodeTask)},
				cgen.Field{Type: cgen.PthreadCondT, What: vb(p.nodeCond)},
				cgen.Field{Type: p.PtrTeam, What: vb(p.nodeTeam)},
				cgen.Field{Type: cgen.PthreadT, What: vb(p.nodeThr)},
			},
			Attrs: cgen.Aligned(p.cacheLine),
		},
		cgen.Newline,
		cgen.StructDef{
			Name: p.unwindType,
			Fields: cgen.Stmts{
				cgen.Field{Type: cgen.PtrdiffT, What: vb(p.unwindJoin)},
				cgen.Field{Type: cgen.PtrdiffT, What: vb(p.unwindNodeConds)},
				cgen.Field{Type: cgen.PtrdiffT, What: vb(p.unwindNodeMuts)},
				cgen.Field{Type: cgen.PtrdiffT, What: vb(p.unwindHubCond)},
				cgen.Field{Type: cgen.PtrdiffT, What: vb(p.unwindHubMut)},
				cgen.Field{Type: cgen.PtrVoid, What: vb(p.unwindNodes)},
				cgen.Field{Type: cgen.PtrVoid, What: vb(p.unwindHub)},
			},
		},
		cgen.Newline,
		cgen.StructDef{
			Name: p.teamType,
			Fields: cgen.Stmts{
				cgen.Field{Type: cgen.PtrdiffT, What: vb(p.teamNt)},
				cgen.Field{Type: p.ptrHub, What: vb(p.teamHub)},
				cgen.Field{Type: p.ptrNode, What: vb(p.teamNodes)},
				cgen.Field{Type: vb(p.unwindType), What: vb(p.teamUnwind)},
			},
		},
		cgen.Newline,
	}.Append(p.to)
}

func (p *Prep) stage4Inc() {
	var (
		nd   = vb(p.name("nd"))
		hull = vb(p.name("hull"))
		pt   = vb(p.name("pt"))
		i    = vb(p.name("i"))
		elem = vb(p.name("elem"))
		ptI  = cgen.Elem{Arr: pt, Idx: i}
	)
	p.to = cgen.StaticFuncDef{
		ReturnType: cgen.Void,
		Name:       p.inc,
		Params: cgen.CommaLines{
			cgen.Param{Type: cgen.PtrdiffT, What: nd},
			cgen.Param{Type: cgen.RestrictPtrInt64T, What: hull},
			cgen.Param{Type: cgen.RestrictPtrInt64T, What: pt},
		},
		Body: cgen.Stmts{cgen.For{
			Init: cgen.Var{Type: cgen.PtrdiffT, What: i, Init: cgen.Zero},
			Cond: cgen.CmpL{Expr1: i, Expr2: nd},
			Post: cgen.IncPre{Expr: i},
			Body: cgen.Stmts{
				cgen.Var{Type: cgen.Int64T, What: elem, Init: ptI},
				cgen.If{
					Cond: cgen.CmpE{
						Expr1: cgen.IncPre{Expr: elem},
						Expr2: cgen.Elem{Arr: hull, Idx: i},
					},
					Then: cgen.Stmts{cgen.Assign{Expr1: ptI, Expr2: cgen.Zero}},
					Else: cgen.Stmts{
						cgen.Assign{Expr1: ptI, Expr2: elem},
						cgen.Break,
					},
				},
			},
		}},
	}.Append(p.to)
}

func (p *Prep) stage4Put() {
	var (
		nd    = vb(p.name("nd"))
		hull  = vb(p.name("hull"))
		pt    = vb(p.name("pt"))
		val   = vb(p.name("val"))
		i     = vb(p.name("i"))
		iOk   = cgen.CmpL{Expr1: i, Expr2: nd}
		wrap  = vb(p.name("wrap"))
		carry = vb(p.name("carry"))
	)
	ptI := cgen.Elem{
		Arr: pt,
		Idx: cgen.IncPost{Expr: i},
	}
	p.to = cgen.StaticFuncDef{
		ReturnType: cgen.Void,
		Name:       p.put,
		Params: cgen.CommaLines{
			cgen.Param{Type: cgen.PtrdiffT, What: nd},
			cgen.Param{Type: cgen.RestrictPtrInt64T, What: hull},
			cgen.Param{Type: cgen.RestrictPtrInt64T, What: pt},
			cgen.Param{Type: cgen.Int64T, What: val},
		},
		Body: cgen.Stmts{
			cgen.Var{Type: cgen.PtrdiffT, What: i, Init: cgen.Zero},
			cgen.For{
				Cond: cgen.Land{Expr1: iOk, Expr2: val},
				Body: cgen.Stmts{
					cgen.Var{
						Type: cgen.Int64T,
						What: wrap,
						Init: cgen.Elem{Arr: hull, Idx: i},
					},
					cgen.Var{
						Type: cgen.Int64T,
						What: carry,
						Init: cgen.Quo{Expr1: val, Expr2: wrap},
					},
					cgen.Assign{
						Expr1: ptI,
						Expr2: cgen.Sub{
							Expr1: val,
							Expr2: cgen.Mul{Expr1: carry, Expr2: wrap},
						},
					},
					cgen.Assign{Expr1: val, Expr2: carry},
				},
			},
			cgen.For{
				Cond: iOk,
				Post: cgen.Assign{Expr1: ptI, Expr2: cgen.Zero},
			},
		},
	}.Append(p.to)
}

func (p *Prep) stage4Add() {
	var (
		nd    = vb(p.name("nd"))
		hull  = vb(p.name("hull"))
		pt    = vb(p.name("pt"))
		plus  = vb(p.name("plus"))
		carry = vb(p.name("carry"))
		i     = vb(p.name("i"))
		wrap  = vb(p.name("wrap"))
		sum   = vb(p.name("sum"))
		ptI   = cgen.Elem{Arr: pt, Idx: i}
	)
	p.to = cgen.StaticFuncDef{
		ReturnType: cgen.Void,
		Name:       p.add,
		Params: cgen.CommaLines{
			cgen.Param{Type: cgen.PtrdiffT, What: nd},
			cgen.Param{Type: cgen.RestrictPtrInt64T, What: hull},
			cgen.Param{Type: cgen.RestrictPtrInt64T, What: pt},
			cgen.Param{Type: cgen.RestrictPtrInt64T, What: plus},
			cgen.Param{Type: cgen.Int64T, What: carry},
		},
		Body: cgen.Stmts{cgen.For{
			Init: cgen.Var{Type: cgen.PtrdiffT, What: i, Init: cgen.Zero},
			Cond: cgen.CmpL{Expr1: i, Expr2: nd},
			Post: cgen.IncPre{Expr: i},
			Body: cgen.Stmts{
				cgen.Var{
					Type: cgen.Int64T,
					What: wrap,
					Init: cgen.Elem{Arr: hull, Idx: i},
				},
				cgen.Var{
					Type: cgen.Int64T,
					What: sum,
					Init: cgen.Add{
						Expr1: cgen.Add{
							Expr1: ptI,
							Expr2: cgen.Elem{Arr: plus, Idx: i},
						},
						Expr2: carry,
					},
				},
				cgen.If{
					Cond: cgen.CmpL{Expr1: sum, Expr2: wrap},
					Then: cgen.Stmts{
						cgen.Assign{Expr1: ptI, Expr2: sum},
						cgen.Assign{Expr1: carry, Expr2: cgen.Zero},
					},
					Else: cgen.Stmts{
						cgen.Assign{
							Expr1: ptI,
							Expr2: cgen.Sub{Expr1: sum, Expr2: wrap},
						},
						cgen.Assign{Expr1: carry, Expr2: cgen.One},
					},
				},
			},
		}},
	}.Append(p.to)
}

func (p *Prep) stage4() {
	p.stage4Inc()
	p.newline()
	p.stage4Put()
	p.newline()
	p.stage4Add()
	p.newline()
}

type stage5 struct {
	*Prep
	arg         cgen.Gen
	callee      cgen.Gen
	hand        cgen.Gen
	hub         cgen.Gen
	hullField   cgen.Gen
	lockHub     cgen.Gen
	lockNode1   cgen.Gen
	lockNode2   cgen.Gen
	mask        cgen.Gen
	maskField   cgen.Gen
	nd          cgen.Gen
	node1       cgen.Gen
	node2       cgen.Gen
	nodes       cgen.Gen
	np          cgen.Gen
	npField1    cgen.Gen
	npField2    cgen.Gen
	nt          cgen.Gen
	offset      cgen.Gen
	offsetField cgen.Gen
	pending     cgen.Gen
	pt          cgen.Gen
	ptField1    cgen.Gen
	ptField2    cgen.Gen
	role        cgen.Gen
	statusField cgen.Gen
	target      cgen.Gen
	task        cgen.Gen
	taskField   cgen.Gen
	team        cgen.Gen
	unlockHub   cgen.Gen
	unlockNode1 cgen.Gen
	unlockNode2 cgen.Gen
	wrapped     cgen.Gen
}

func (s *stage5) local() cgen.Gen {
	return cgen.Stmts{
		cgen.Var{
			Type: s.calleeType,
			What: s.callee,
			Init: cgen.Arrow{Expr: s.task, Name: s.taskCallee},
		},
		cgen.Var{
			Type: cgen.PtrdiffT,
			What: s.nd,
			Init: cgen.Arrow{Expr: s.task, Name: s.taskNd},
		},
		cgen.Var{
			Type: cgen.Int64T,
			What: cgen.Elem{Arr: s.pt, Idx: s.maxNd},
		},
		cgen.For{
			Cond: s.np,
			Post: cgen.Assign{Expr1: s.np, Expr2: s.npField1},
			Body: cgen.Stmts{
				cgen.Call{
					Func: cgen.Memcpy,
					Args: cgen.CommaSpaced{
						s.pt, s.ptField1, cgen.Sizeof{What: s.pt},
					},
				},
				cgen.Assign{
					Expr1: s.npField1,
					Expr2: cgen.Sub{Expr1: s.np, Expr2: cgen.One},
				},
				cgen.Call{
					Func: vb(s.inc),
					Args: cgen.CommaSpaced{s.nd, s.hullField, s.ptField1},
				},
				s.unlockNode1,
				cgen.Call{
					Func: s.callee,
					Args: cgen.CommaSpaced{s.task, s.pt},
				},
				s.lockNode1,
			},
		},
	}
}

func (s *stage5) steal() cgen.Gen {
	return cgen.Stmts{
		cgen.Var{
			Type: s.ptrNode,
			What: s.node2,
			Init: cgen.Add{Expr1: s.nodes, Expr2: s.target},
		},
		s.lockNode2,
		cgen.For{
			Init: cgen.Assign{Expr1: s.np, Expr2: s.npField2},
			Cond: s.np,
			Post: cgen.Assign{Expr1: s.np, Expr2: s.npField2},
			Body: cgen.Stmts{
				cgen.Call{
					Func: cgen.Memcpy,
					Args: cgen.CommaSpaced{
						s.pt, s.ptField2, cgen.Sizeof{What: s.pt},
					},
				},
				cgen.Assign{
					Expr1: s.npField2,
					Expr2: cgen.Sub{Expr1: s.np, Expr2: cgen.One},
				},
				cgen.Call{
					Func: vb(s.inc),
					Args: cgen.CommaSpaced{s.nd, s.hullField, s.ptField2},
				},
				s.unlockNode2,
				cgen.Call{
					Func: s.callee,
					Args: cgen.CommaSpaced{s.task, s.pt},
				},
				s.lockNode2,
			},
		},
		s.unlockNode2,
	}
}

func (s *stage5) nonlocal() cgen.Gen {
	return cgen.Stmts{
		cgen.AndAssign{
			Expr1: cgen.Elem{
				Arr: s.statusField,
				Idx: cgen.Quo{Expr1: s.role, Expr2: cgen.BitsPerLong},
			},
			Expr2: cgen.Not{Expr: cgen.Paren{
				Inner: cgen.ShiftHigh{
					Expr1: cgen.Cast{Type: cgen.Long, Expr: cgen.One},
					Expr2: cgen.Rem{Expr1: s.role, Expr2: cgen.BitsPerLong},
				},
			}},
		},
		cgen.Var{Type: cgen.PtrdiffT, What: s.offset, Init: s.offsetField},
		cgen.Var{Type: cgen.Long, What: s.mask, Init: s.maskField},
		cgen.Var{Type: cgen.PtrdiffT, What: s.wrapped, Init: cgen.Zero},
		cgen.For{Body: cgen.Stmts{
			cgen.Var{
				Type: cgen.Long,
				What: s.hand,
				Init: cgen.And{
					Expr1: cgen.Elem{Arr: s.statusField, Idx: s.offset},
					Expr2: s.mask,
				},
			},
			cgen.If{
				Cond: cgen.IsZero{Expr: s.hand},
				Then: cgen.Stmts{
					cgen.IncPre{Expr: s.offset},
					cgen.Assign{Expr1: s.mask, Expr2: cgen.NegOne},
					cgen.Continue,
				},
			},
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: s.target,
				Init: cgen.Add{
					Expr1: cgen.Mul{Expr1: s.offset, Expr2: cgen.BitsPerLong},
					Expr2: cgen.Call{Func: cgen.Ctzl, Args: s.hand},
				},
			},
			cgen.If{
				Cond: cgen.CmpE{Expr1: s.target, Expr2: s.nt},
				Then: cgen.Stmts{
					cgen.If1{Cond: s.wrapped, Then: cgen.Break},
					cgen.Assign{Expr1: s.offset, Expr2: cgen.Zero},
					cgen.Assign{Expr1: s.mask, Expr2: cgen.NegOne},
					cgen.Assign{Expr1: s.wrapped, Expr2: cgen.One},
					cgen.Continue,
				},
			},
			cgen.AndAssign{
				Expr1: s.hand,
				Expr2: cgen.Neg{Expr: s.hand},
			},
			cgen.Assign{Expr1: s.offsetField, Expr2: s.offset},
			cgen.Assign{
				Expr1: s.maskField,
				Expr2: cgen.Sub{Expr1: s.mask, Expr2: s.hand},
			},
			s.unlockHub,
			s.steal(),
			s.lockHub,
			cgen.AndAssign{
				Expr1: cgen.Elem{Arr: s.statusField, Idx: s.offset},
				Expr2: cgen.Not{Expr: s.hand},
			},
			cgen.Assign{Expr1: s.offset, Expr2: s.offsetField},
			cgen.Assign{Expr1: s.mask, Expr2: s.maskField},
			cgen.Assign{Expr1: s.wrapped, Expr2: cgen.Zero},
		}},
	}
}

func (s *stage5) fn() {
	s.to = cgen.StaticFuncDef{
		ReturnType: cgen.PtrVoid,
		Name:       s.main,
		Params:     cgen.Param{Type: cgen.PtrVoid, What: s.arg},
		Body: cgen.Stmts{
			cgen.Var{Type: s.ptrNode, What: s.node1, Init: s.arg},
			cgen.Var{
				Type: s.PtrTeam,
				What: s.team,
				Init: cgen.Arrow{Expr: s.node1, Name: s.nodeTeam},
			},
			cgen.Var{
				Type: cgen.PtrdiffT,
				What: s.nt,
				Init: cgen.Arrow{Expr: s.team, Name: s.teamNt},
			},
			cgen.Var{
				Type: s.ptrHub,
				What: s.hub,
				Init: cgen.Arrow{Expr: s.team, Name: s.teamHub},
			},
			cgen.Var{
				Type: s.ptrNode,
				What: s.nodes,
				Init: cgen.Arrow{Expr: s.team, Name: s.teamNodes},
			},
			cgen.Var{
				Type: cgen.SizeT,
				What: s.role,
				Init: cgen.Sub{Expr1: s.node1, Expr2: s.nodes},
			},
			s.lockNode1,
			cgen.For{Body: cgen.Stmts{
				cgen.Var{Type: s.ptrTask, What: s.task, Init: s.taskField},
				cgen.If{
					Cond: cgen.IsZero{Expr: s.task},
					Then: cgen.Stmts{
						wait{s.node1, s.nodeCond, s.nodeMut},
						cgen.Continue,
					},
				},
				cgen.Var{Type: cgen.Int64T, What: s.np, Init: s.npField1},
				cgen.If{
					Cond: cgen.CmpL{Expr1: s.np, Expr2: cgen.Zero},
					Then: cgen.Stmts{
						s.unlockNode1,
						cgen.Return{Expr: cgen.Zero},
					},
				},
				cgen.Assign{Expr1: s.taskField, Expr2: cgen.Zero},
				s.local(),
				s.unlockNode1,
				s.lockHub,
				s.nonlocal(),
				cgen.Var{
					Type: cgen.PtrdiffT,
					What: s.pending,
					Init: cgen.DecPre{
						Expr: cgen.Arrow{Expr: s.hub, Name: s.hubPending},
					},
				},
				s.unlockHub,
				cgen.If1{
					Cond: cgen.IsZero{Expr: s.pending},
					Then: signal{s.hub, s.hubCond},
				},
				s.lockNode1,
			}},
		},
	}.Append(s.to)
}

func (p *Prep) stage5() {
	s := &stage5{
		Prep:    p,
		arg:     vb(p.name("arg")),
		callee:  vb(p.name("callee")),
		hand:    vb(p.name("hand")),
		hub:     vb(p.name("hub")),
		mask:    vb(p.name("mask")),
		nd:      vb(p.name("nd")),
		node1:   vb(p.name("node")),
		node2:   vb(p.name("node")),
		nodes:   vb(p.name("nodes")),
		np:      vb(p.name("np")),
		nt:      vb(p.name("nt")),
		offset:  vb(p.name("offset")),
		pending: vb(p.name("pending")),
		pt:      vb(p.name("pt")),
		role:    vb(p.name("role")),
		target:  vb(p.name("target")),
		task:    vb(p.name("task")),
		team:    vb(p.name("team")),
		wrapped: vb(p.name("wrapped")),
	}
	s.hullField = cgen.Arrow{Expr: s.task, Name: s.taskHull}
	s.lockHub = lock{s.hub, s.hubMut}
	s.lockNode1 = lock{s.node1, s.nodeMut}
	s.lockNode2 = lock{s.node2, s.nodeMut}
	s.maskField = cgen.Arrow{Expr: s.hub, Name: s.hubMask}
	s.npField1 = cgen.Arrow{Expr: s.node1, Name: s.nodeNp}
	s.npField2 = cgen.Arrow{Expr: s.node2, Name: s.nodeNp}
	s.offsetField = cgen.Arrow{Expr: s.hub, Name: s.hubOffset}
	s.ptField1 = cgen.Arrow{Expr: s.node1, Name: s.nodePt}
	s.ptField2 = cgen.Arrow{Expr: s.node2, Name: s.nodePt}
	s.statusField = cgen.Arrow{Expr: s.hub, Name: s.hubStatus}
	s.taskField = cgen.Arrow{Expr: s.node1, Name: s.nodeTask}
	s.unlockHub = unlock{s.hub, s.hubMut}
	s.unlockNode1 = unlock{s.node1, s.nodeMut}
	s.unlockNode2 = unlock{s.node2, s.nodeMut}
	s.fn()
	s.newline()
}

func (p *Prep) stage6() {
	var (
		team   = vb(p.name("team"))
		nodes  = vb(p.name("nodes"))
		node   = vb(p.name("node"))
		stop   = vb(p.name("stop"))
		unwind = cgen.Arrow{Expr: team, Name: p.teamUnwind}
		hub    = vb(p.name("hub"))
	)
	field := func(a string) cgen.Gen {
		return cgen.Dot{Expr: unwind, Name: a}
	}
	loop := func(a ...cgen.Gen) cgen.Gen {
		return cgen.For{
			Init: cgen.Var{Type: p.ptrNode, What: node, Init: nodes},
			Cond: cgen.CmpNE{Expr1: node, Expr2: stop},
			Post: cgen.IncPre{Expr: node},
			Body: cgen.Stmts(a),
		}
	}
	reps := func(a cgen.Gen) cgen.Gen {
		return cgen.Assign{
			Expr1: stop,
			Expr2: cgen.Add{Expr1: nodes, Expr2: a},
		}
	}
	free := func(a cgen.Gen) cgen.Gen {
		return cgen.Call{Func: cgen.Free, Args: a}
	}
	p.to = cgen.StaticFuncDef{
		ReturnType: cgen.Void,
		Name:       p.destroy,
		Params:     cgen.Param{Type: p.PtrTeam, What: team},
		Body: cgen.Stmts{
			cgen.If1{
				Cond: cgen.IsZero{Expr: team},
				Then: cgen.Return{},
			},
			cgen.Var{
				Type: p.ptrNode,
				What: nodes,
				Init: cgen.Arrow{Expr: team, Name: p.teamNodes},
			},
			cgen.Var{
				Type: p.ptrNode,
				What: stop,
				Init: cgen.Add{Expr1: nodes, Expr2: field(p.unwindJoin)},
			},
			loop(
				lock{node, p.nodeMut},
				cgen.Assign{
					Expr1: cgen.Arrow{Expr: node, Name: p.nodeNp},
					Expr2: cgen.NegOne,
				},
				cgen.Assign{
					Expr1: cgen.Arrow{Expr: node, Name: p.nodeTask},
					Expr2: cgen.Cast{Type: p.ptrTask, Expr: cgen.One},
				},
				unlock{node, p.nodeMut},
				signal{node, p.nodeCond},
			),
			loop(join{node, p.nodeThr}),
			reps(field(p.unwindNodeConds)),
			loop(destroyCond{node, p.nodeCond}),
			reps(field(p.unwindNodeMuts)),
			loop(destroyMut{node, p.nodeMut}),
			cgen.Var{
				Type: p.ptrHub,
				What: hub,
				Init: cgen.Arrow{Expr: team, Name: p.teamHub},
			},
			cgen.If{
				Cond: field(p.unwindHubCond),
				Then: cgen.Stmts{destroyCond{hub, p.hubCond}},
			},
			cgen.If{
				Cond: field(p.unwindHubMut),
				Then: cgen.Stmts{destroyMut{hub, p.hubMut}},
			},
			free(field(p.unwindNodes)),
			free(field(p.unwindHub)),
			free(team),
		},
	}.Append(p.to)
	p.newline()
}

func (p *Prep) stage7Up4(root, self string) {
	var (
		team   = vb(p.name("team"))
		nt     = vb(p.name("nt"))
		nodes  = vb(p.name("nodes"))
		node   = vb(p.name("node"))
		cnt    = cgen.Sub{Expr1: node, Expr2: nodes}
		cntOne = cgen.Add{Expr1: cnt, Expr2: cgen.One}
		unwind = cgen.Arrow{Expr: team, Name: p.teamUnwind}
		muts   = cgen.Dot{Expr: unwind, Name: p.unwindNodeMuts}
		conds  = cgen.Dot{Expr: unwind, Name: p.unwindNodeConds}
		join   = cgen.Dot{Expr: unwind, Name: p.unwindJoin}
	)
	set := func(nm, nc, nj cgen.Gen) cgen.Gen {
		return cgen.Stmts{
			cgen.Assign{Expr1: muts, Expr2: nm},
			cgen.Assign{Expr1: conds, Expr2: nc},
			cgen.Assign{Expr1: join, Expr2: nj},
		}
	}
	p.to = cgen.StaticFuncDef{
		ReturnType: cgen.PtrChar,
		Name:       self,
		Params: cgen.CommaSpaced{
			cgen.Param{Type: p.PtrTeam, What: team},
			cgen.Param{Type: cgen.PtrdiffT, What: nt},
		},
		Body: cgen.Stmts{
			cgen.Var{
				Type: p.ptrNode,
				What: nodes,
				Init: cgen.Arrow{Expr: team, Name: p.teamNodes},
			},
			cgen.For{
				Init: cgen.Var{Type: p.ptrNode, What: node, Init: nodes},
				Cond: cgen.CmpNE{
					Expr1: node,
					Expr2: cgen.Add{Expr1: nodes, Expr2: nt},
				},
				Post: cgen.IncPre{Expr: node},
				Body: cgen.Stmts{
					&errmsg.ReturnedErrnoIf{
						Ctx: p.emc,
						Call: cgen.Call{
							Func: cgen.PthreadMutexInit,
							Args: cgen.CommaSpaced{
								cgen.AddrArrow{Expr: node, Name: p.nodeMut},
								cgen.Zero,
							},
						},
						Unwind: set(cnt, cnt, cnt),
					},
					cgen.Assign{
						Expr1: cgen.Arrow{Expr: node, Name: p.nodeTask},
						Expr2: cgen.Zero,
					},
					&errmsg.ReturnedErrnoIf{
						Ctx: p.emc,
						Call: cgen.Call{
							Func: cgen.PthreadCondInit,
							Args: cgen.CommaSpaced{
								cgen.AddrArrow{Expr: node, Name: p.nodeCond},
								cgen.Zero,
							},
						},
						Unwind: set(cntOne, cnt, cnt),
					},
					cgen.Assign{
						Expr1: cgen.Arrow{Expr: node, Name: p.nodeTeam},
						Expr2: team,
					},
					&errmsg.ReturnedErrnoIf{
						Ctx: p.emc,
						Call: cgen.Call{
							Func: cgen.PthreadCreate,
							Args: cgen.CommaSpaced{
								cgen.AddrArrow{Expr: node, Name: p.nodeThr},
								cgen.Zero, vb(p.main), node,
							},
						},
						Unwind: set(cntOne, cntOne, cnt),
					},
				},
			},
			set(nt, nt, nt),
			cgen.Return{Expr: cgen.Zero},
		},
	}.Append(p.to)
	p.newline()
}

func (p *Prep) stage7Up3(root, self string) {
	var (
		up     = p.name(root)
		team   = vb(p.name("team"))
		nt     = vb(p.name("nt"))
		hub    = vb(p.name("hub"))
		unwind = cgen.Arrow{Expr: team, Name: p.teamUnwind}
	)
	p.stage7Up4(root, up)
	p.to = cgen.StaticFuncDef{
		ReturnType: cgen.PtrChar,
		Name:       self,
		Params: cgen.CommaSpaced{
			cgen.Param{Type: p.PtrTeam, What: team},
			cgen.Param{Type: cgen.PtrdiffT, What: nt},
		},
		Body: cgen.Stmts{
			cgen.Var{
				Type: p.ptrHub,
				What: hub,
				Init: cgen.Arrow{Expr: team, Name: p.teamHub},
			},
			&errmsg.ReturnedErrnoIf{
				Ctx: p.emc,
				Call: cgen.Call{
					Func: cgen.PthreadMutexInit,
					Args: cgen.CommaSpaced{
						cgen.AddrArrow{Expr: hub, Name: p.hubMut},
						cgen.Zero,
					},
				},
			},
			cgen.Assign{
				Expr1: cgen.Dot{Expr: unwind, Name: p.unwindHubMut},
				Expr2: cgen.One,
			},
			&errmsg.ReturnedErrnoIf{
				Ctx: p.emc,
				Call: cgen.Call{
					Func: cgen.PthreadCondInit,
					Args: cgen.CommaSpaced{
						cgen.AddrArrow{Expr: hub, Name: p.hubCond},
						cgen.Zero,
					},
				},
			},
			cgen.Assign{
				Expr1: cgen.Dot{Expr: unwind, Name: p.unwindHubCond},
				Expr2: cgen.One,
			},
			cgen.Return{Expr: cgen.Call{
				Func: vb(up),
				Args: cgen.CommaSpaced{team, nt},
			}},
		},
	}.Append(p.to)
	p.newline()
}

func (p *Prep) stage7Up2(root, self string) {
	var (
		up   = p.name(root)
		team = vb(p.name("team"))
		nt   = vb(p.name("nt"))
		size = vb(p.name("size"))
		each = cgen.Sizeof{What: vb(p.nodeType)}
		addr = vb(p.name("addr"))
		line = cgen.IntLit(p.cacheLine)
	)
	p.stage7Up3(root, up)
	p.to = cgen.StaticFuncDef{
		ReturnType: cgen.PtrChar,
		Name:       self,
		Params: cgen.CommaSpaced{
			cgen.Param{Type: p.PtrTeam, What: team},
			cgen.Param{Type: cgen.PtrdiffT, What: nt},
		},
		Body: cgen.Stmts{
			cgen.Var{
				Type: cgen.SizeT,
				What: size,
				Init: cgen.Mul{Expr1: nt, Expr2: each},
			},
			&errmsg.FormatIf{
				Ctx: p.emc,
				Cond: cgen.CmpNE{
					Expr1: cgen.Quo{Expr1: size, Expr2: each},
					Expr2: cgen.Cast{Type: cgen.SizeT, Expr: nt},
				},
				Format: "too many threads",
			},
			cgen.Var{
				Type: cgen.PtrVoid,
				What: addr,
				Init: cgen.Call{
					Func: cgen.Malloc,
					Args: cgen.Add{Expr1: size, Expr2: line - 1},
				},
			},
			&errmsg.ErrnoIf{
				Ctx:  p.emc,
				Cond: cgen.IsZero{Expr: addr},
			},
			cgen.Assign{
				Expr1: cgen.Dot{
					Expr: cgen.Arrow{Expr: team, Name: p.teamUnwind},
					Name: p.unwindNodes,
				},
				Expr2: addr,
			},
			cgen.Assign{
				Expr1: cgen.Arrow{Expr: team, Name: p.teamNodes},
				Expr2: cgen.Cast{
					Type: cgen.PtrVoid,
					Expr: cgen.Paren{Inner: round{
						cgen.Cast{Type: cgen.SizeT, Expr: addr},
						line,
					}},
				},
			},
			cgen.Return{Expr: cgen.Call{
				Func: vb(up),
				Args: cgen.CommaSpaced{team, nt},
			}},
		},
	}.Append(p.to)
	p.newline()
}

func (p *Prep) stage7Up1(root, self string) {
	var (
		up   = p.name(root)
		team = vb(p.name("team"))
		nt   = vb(p.name("nt"))
		size = vb(p.name("size"))
		line = cgen.IntLit(p.cacheLine)
		addr = vb(p.name("addr"))
	)
	p.stage7Up2(root, up)
	p.to = cgen.StaticFuncDef{
		ReturnType: cgen.PtrChar,
		Name:       self,
		Params: cgen.CommaSpaced{
			cgen.Param{Type: p.PtrTeam, What: team},
			cgen.Param{Type: cgen.PtrdiffT, What: nt},
		},
		Body: cgen.Stmts{
			cgen.Assign{
				Expr1: cgen.Arrow{Expr: team, Name: p.teamNt},
				Expr2: nt,
			},
			cgen.Var{
				Type: cgen.SizeT,
				What: size,
				Init: cgen.Sizeof{What: vb(p.hubType)},
			},
			cgen.AddAssign{
				Expr1: size,
				Expr2: cgen.Mul{
					Expr1: cgen.Sizeof{What: cgen.Long},
					Expr2: cgen.Paren{Inner: cgen.Add{
						Expr1: cgen.Quo{
							Expr1: cgen.Cast{Type: cgen.SizeT, Expr: nt},
							Expr2: cgen.BitsPerLong,
						},
						Expr2: cgen.One,
					}},
				},
			},
			cgen.Assign{
				Expr1: size,
				Expr2: round{size, line},
			},
			cgen.Var{
				Type: cgen.PtrVoid,
				What: addr,
				Init: cgen.Call{
					Func: cgen.Malloc,
					Args: cgen.Add{Expr1: size, Expr2: line - 1},
				},
			},
			&errmsg.ErrnoIf{
				Ctx:  p.emc,
				Cond: cgen.IsZero{Expr: addr},
			},
			cgen.Assign{
				Expr1: cgen.Dot{
					Expr: cgen.Arrow{Expr: team, Name: p.teamUnwind},
					Name: p.unwindHub,
				},
				Expr2: addr,
			},
			cgen.Assign{
				Expr1: cgen.Arrow{Expr: team, Name: p.teamHub},
				Expr2: cgen.Cast{
					Type: cgen.PtrVoid,
					Expr: cgen.Paren{Inner: round{
						cgen.Cast{Type: cgen.SizeT, Expr: addr},
						line,
					}},
				},
			},
			cgen.Return{Expr: cgen.Call{
				Func: vb(up),
				Args: cgen.CommaSpaced{team, nt},
			}},
		},
	}.Append(p.to)
	p.newline()
}

func (p *Prep) stage7() {
	var (
		root = p.create + "Up"
		up   = p.name(root)
		team = vb(p.name("team"))
		nt   = vb(p.name("nt"))
		addr = vb(p.name("addr"))
		err  = vb(p.name("err"))
	)
	p.stage7Up1(root, up)
	p.to = cgen.StaticFuncDef{
		ReturnType: cgen.PtrChar,
		Name:       p.create,
		Params: cgen.CommaSpaced{
			cgen.Param{
				Type: cgen.Ptr{Type: p.PtrTeam},
				What: team,
			},
			cgen.Param{Type: cgen.PtrdiffT, What: nt},
		},
		Body: cgen.Stmts{
			&errmsg.FormatIf{
				Ctx:    p.emc,
				Cond:   cgen.CmpL{Expr1: nt, Expr2: cgen.One},
				Format: "too few threads",
			},
			cgen.Var{
				Type: cgen.PtrVoid,
				What: addr,
				Init: cgen.Call{
					Func: cgen.Calloc,
					Args: cgen.CommaSpaced{
						cgen.One,
						cgen.Sizeof{What: vb(p.teamType)},
					},
				},
			},
			&errmsg.ErrnoIf{
				Ctx:  p.emc,
				Cond: cgen.IsZero{Expr: addr},
			},
			cgen.Var{
				Type: cgen.PtrChar,
				What: err,
				Init: cgen.Call{
					Func: vb(up),
					Args: cgen.CommaSpaced{addr, nt},
				},
			},
			cgen.If{
				Cond: cgen.Unlikely{
					Cond: cgen.IsNonzero{Expr: err},
				},
				Then: cgen.Stmts{cgen.Call{
					Func: vb(p.destroy), Args: addr,
				}},
				Else: cgen.Stmts{cgen.Assign{
					Expr1: cgen.At{Expr: team},
					Expr2: addr,
				}},
			},
			cgen.Return{Expr: err},
		},
	}.Append(p.to)
	p.newline()
}

func (p *Prep) stage8() {
	var (
		thr  = vb(p.name("thr"))
		team = vb(p.name("team"))
		idx  = vb(p.name("idx"))
	)
	p.to = cgen.StaticFuncDef{
		ReturnType: cgen.PtrChar,
		Name:       p.pthreadT,
		Params: cgen.CommaLines{
			cgen.Param{Type: cgen.PtrPthreadT, What: thr},
			cgen.Param{Type: p.PtrTeam, What: team},
			cgen.Param{Type: cgen.PtrdiffT, What: idx},
		},
		Body: cgen.Stmts{
			&errmsg.FormatIf{
				Ctx: p.emc,
				Cond: cgen.Lor{
					Expr1: cgen.CmpL{Expr1: idx, Expr2: cgen.Zero},
					Expr2: cgen.CmpGE{
						Expr1: idx,
						Expr2: cgen.Arrow{Expr: team, Name: p.teamNt},
					},
				},
				Format: "bad thread idx",
			},
			cgen.Assign{
				Expr1: cgen.At{Expr: thr},
				Expr2: cgen.Dot{
					Expr: cgen.Elem{
						Arr: cgen.Arrow{Expr: team, Name: p.teamNodes},
						Idx: idx,
					},
					Name: p.nodeThr,
				},
			},
			cgen.Return{Expr: cgen.Zero},
		},
	}.Append(p.to)
	p.newline()
}

type stage9 struct {
	*Prep
	team    cgen.Gen
	task    cgen.Gen
	nd      cgen.Gen
	tot     cgen.Gen
	hull    cgen.Gen
	i       cgen.Gen
	nt      cgen.Gen
	each    cgen.Gen
	more    cgen.Gen
	plus    cgen.Gen
	pt      cgen.Gen
	hub     cgen.Gen
	node    cgen.Gen
	carry   cgen.Gen
	pending cgen.Gen
}

func (s *stage9) divide() cgen.Gen {
	return cgen.Stmts{
		cgen.Var{
			Type: cgen.PtrdiffT,
			What: s.nd,
			Init: cgen.Arrow{Expr: s.task, Name: s.taskNd},
		},
		cgen.If1{
			Cond: cgen.CmpL{Expr1: s.nd, Expr2: cgen.One},
			Then: cgen.Return{},
		},
		cgen.Var{
			Type: cgen.Int64T,
			What: s.tot,
			Init: cgen.Elem{Arr: s.hull, Idx: cgen.Zero},
		},
		cgen.For{
			Init: cgen.Var{Type: cgen.PtrdiffT, What: s.i, Init: cgen.One},
			Cond: cgen.CmpL{Expr1: s.i, Expr2: s.nd},
			Post: cgen.MulAssign{
				Expr1: s.tot,
				Expr2: cgen.Elem{
					Arr: s.hull,
					Idx: cgen.IncPost{Expr: s.i},
				},
			},
		},
		cgen.Var{
			Type: cgen.PtrdiffT,
			What: s.nt,
			Init: cgen.Arrow{Expr: s.team, Name: s.teamNt},
		},
		cgen.Var{
			Type: cgen.Int64T,
			What: s.each,
			Init: cgen.Quo{Expr1: s.tot, Expr2: s.nt},
		},
		cgen.Var{
			Type: cgen.PtrdiffT,
			What: s.more,
			Init: cgen.Rem{Expr1: s.tot, Expr2: s.nt},
		},
		cgen.Var{
			Type: cgen.Int64T,
			What: cgen.Elem{Arr: s.plus, Idx: s.maxNd},
		},
		cgen.Call{
			Func: vb(s.put),
			Args: cgen.CommaSpaced{s.nd, s.hull, s.plus, s.each},
		},
		cgen.Var{
			Type: cgen.Int64T,
			What: cgen.Elem{Arr: s.pt, Idx: s.maxNd},
			Init: cgen.Zeros,
		},
	}
}

func (s *stage9) launch() cgen.Gen {
	return cgen.Stmts{
		cgen.Var{
			Type: s.ptrNode,
			What: s.node,
			Init: cgen.Arrow{Expr: s.team, Name: s.teamNodes},
		},
		cgen.For{
			Init: cgen.Var{Type: cgen.PtrdiffT, What: s.i, Init: cgen.Zero},
			Post: cgen.IncPre{Expr: s.node},
			Body: cgen.Stmts{
				lock{s.node, s.nodeMut},
				cgen.Var{
					Type: cgen.Int64T,
					What: s.carry,
					Init: cgen.CmpL{Expr1: s.i, Expr2: s.more},
				},
				cgen.Assign{
					Expr1: cgen.Arrow{Expr: s.node, Name: s.nodeNp},
					Expr2: cgen.Add{Expr1: s.each, Expr2: s.carry},
				},
				cgen.Call{
					Func: cgen.Memcpy,
					Args: cgen.CommaSpaced{
						cgen.Arrow{Expr: s.node, Name: s.nodePt},
						s.pt, cgen.Sizeof{What: s.pt},
					},
				},
				cgen.Assign{
					Expr1: cgen.Arrow{Expr: s.node, Name: s.nodeTask},
					Expr2: s.task,
				},
				unlock{s.node, s.nodeMut},
				signal{s.node, s.nodeCond},
				cgen.If1{
					Cond: cgen.CmpE{
						Expr1: cgen.IncPre{Expr: s.i},
						Expr2: s.nt,
					},
					Then: cgen.Break,
				},
				cgen.Call{
					Func: vb(s.add),
					Args: cgen.CommaSpaced{s.nd, s.hull, s.pt, s.plus, s.carry},
				},
			},
		},
	}
}

func (s *stage9) fn() {
	s.to = cgen.StaticFuncDef{
		ReturnType: cgen.Void,
		Name:       s.do,
		Params: cgen.CommaSpaced{
			cgen.Param{Type: s.PtrTeam, What: s.team},
			cgen.Param{Type: s.ptrTask, What: s.task},
		},
		Body: cgen.Stmts{
			s.divide(),
			cgen.Var{
				Type: s.ptrHub,
				What: s.hub,
				Init: cgen.Arrow{Expr: s.team, Name: s.teamHub},
			},
			lock{s.hub, s.hubMut},
			s.launch(),
			cgen.Assign{
				Expr1: cgen.Arrow{Expr: s.hub, Name: s.hubOffset},
				Expr2: cgen.Zero,
			},
			cgen.Assign{
				Expr1: cgen.Arrow{Expr: s.hub, Name: s.hubMask},
				Expr2: cgen.NegOne,
			},
			cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT,
					What: s.i,
					Init: cgen.Quo{
						Expr1: cgen.Cast{Type: cgen.SizeT, Expr: s.nt},
						Expr2: cgen.BitsPerLong,
					},
				},
				Cond: cgen.CmpGE{Expr1: s.i, Expr2: cgen.Zero},
				Body: cgen.Stmts{cgen.Assign{
					Expr1: cgen.Elem{
						Arr: cgen.Arrow{Expr: s.hub, Name: s.hubStatus},
						Idx: cgen.DecPost{Expr: s.i},
					},
					Expr2: cgen.NegOne,
				}},
			},
			cgen.For{
				Init: cgen.Assign{Expr1: s.pending, Expr2: s.nt},
				Cond: s.pending,
				Body: cgen.Stmts{wait{s.hub, s.hubCond, s.hubMut}},
			},
			unlock{s.hub, s.hubMut},
		},
	}.Append(s.to)
}

func (p *Prep) stage9() {
	s := &stage9{
		Prep:  p,
		team:  vb(p.name("team")),
		task:  vb(p.name("task")),
		nd:    vb(p.name("nd")),
		tot:   vb(p.name("tot")),
		i:     vb(p.name("i")),
		nt:    vb(p.name("nt")),
		each:  vb(p.name("each")),
		more:  vb(p.name("more")),
		plus:  vb(p.name("plus")),
		pt:    vb(p.name("pt")),
		hub:   vb(p.name("hub")),
		node:  vb(p.name("node")),
		carry: vb(p.name("carry")),
	}
	s.hull = cgen.Arrow{Expr: s.task, Name: s.taskHull}
	s.pending = cgen.Arrow{Expr: s.hub, Name: s.hubPending}
	s.fn()
}

type Destroy struct {
	*Ctx
	Team cgen.Gen
}

func (d *Destroy) Append(to []byte) []byte {
	return cgen.Stmts{cgen.Call{
		Func: vb(d.destroy),
		Args: d.Team,
	}}.Append(to)
}

type Create struct {
	*Ctx
	Team   cgen.Gen
	Nt     cgen.Gen
	Unwind cgen.Gen
}

func (c *Create) Append(to []byte) []byte {
	var (
		err = vb(c.name("err"))
		ret = cgen.Return{Expr: err}
	)
	cond := cgen.Unlikely{
		Cond: cgen.IsNonzero{Expr: err},
	}
	var follow cgen.Gen
	if c.Unwind == nil {
		follow = cgen.If1{Cond: cond, Then: ret}
	} else {
		follow = cgen.If{
			Cond: cond,
			Then: cgen.Stmts{c.Unwind, ret},
		}
	}
	return cgen.Stmts{
		cgen.Var{
			Type: cgen.PtrChar,
			What: err,
			Init: cgen.Call{
				Func: vb(c.create),
				Args: cgen.CommaSpaced{c.Team, c.Nt},
			},
		},
		follow,
	}.Append(to)
}

type PthreadT struct {
	*Ctx
	Thr  cgen.Gen
	Team cgen.Gen
	Idx  cgen.Gen
}

func (p *PthreadT) Append(to []byte) []byte {
	return cgen.Stmts{cgen.Return{
		Expr: cgen.Call{
			Func: vb(p.pthreadT),
			Args: cgen.CommaSpaced{p.Thr, p.Team, p.Idx},
		},
	}}.Append(to)
}

type Callee struct {
	*Ctx
	Name string
	Task cgen.Gen
	Pt   cgen.Gen
}

func (c *Callee) Any() cgen.Gen {
	return cgen.Arrow{Expr: c.Task, Name: c.taskAny}
}

func (c *Callee) Nd() cgen.Gen {
	return cgen.Arrow{Expr: c.Task, Name: c.taskNd}
}

func (c *Callee) Hull() cgen.Gen {
	return cgen.Arrow{Expr: c.Task, Name: c.taskHull}
}

func (c *Callee) Func(body cgen.Gen) cgen.Gen {
	return cgen.StaticFuncDef{
		ReturnType: cgen.Void,
		Name:       c.Name,
		Params: cgen.CommaSpaced{
			cgen.Param{Type: c.ptrTask, What: c.Task},
			cgen.Param{Type: cgen.PtrInt64T, What: c.Pt},
		},
		Body: body,
	}
}

type Do struct {
	*Ctx
	Callee cgen.Gen
	Any    cgen.Gen
	Hull   []cgen.Gen
	Team   cgen.Gen
}

func (d *Do) Append(to []byte) []byte {
	task := vb(d.name("task"))
	tf := func(a string) cgen.Gen {
		return cgen.Dot{Expr: task, Name: a}
	}
	nd := cgen.IntLit(len(d.Hull))
	if nd > maxNd {
		panic("bug")
	}
	stmts := cgen.Stmts{
		cgen.Var{Type: vb(d.taskType), What: task},
		cgen.Assign{Expr1: tf(d.taskCallee), Expr2: d.Callee},
		cgen.Assign{Expr1: tf(d.taskAny), Expr2: d.Any},
		cgen.Assign{Expr1: tf(d.taskNd), Expr2: nd},
	}
	hull := tf(d.taskHull)
	for i, expr := range d.Hull {
		stmts = append(stmts, cgen.Assign{
			Expr1: cgen.Elem{
				Arr: hull,
				Idx: cgen.IntLit(i),
			},
			Expr2: expr,
		})
	}
	stmts = append(stmts, cgen.Call{
		Func: vb(d.do),
		Args: cgen.CommaSpaced{
			d.Team, cgen.Addr{Expr: task},
		},
	})
	return stmts.Append(to)
}
