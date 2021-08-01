package errmsg

import (
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
)

type Ctx struct {
	nms       nmsrc.Src
	msgPrefix string
	funcName  string
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src) *Ctx {
	prefix := pl.Config.Prefix
	return &Ctx{
		nms:       nms,
		msgPrefix: prefix,
		funcName:  nms.Name(prefix + "Errmsg"),
	}
}

func (c *Ctx) name(prefix string) cgen.Gen {
	return cgen.Vb(c.nms.Name(prefix))
}

type Prep struct {
	*Ctx
	lineNum cgen.Gen
	format  cgen.Gen
}

func (p *Prep) body() cgen.Gen {
	msg := p.name("msg")
	pre := cgen.DoubleQuoted(p.msgPrefix + ": line %td: ")
	const plenty = 1 << 8
	size := cgen.IntLit(len(pre) + plenty)
	step := p.name("step")
	ap := p.name("ap")
	return cgen.Stmts{
		cgen.Var{
			Type: cgen.PtrChar,
			What: msg,
			Init: cgen.Call{Func: cgen.Malloc, Args: size},
		},
		cgen.Var{
			Type: cgen.Int,
			What: step,
			Init: cgen.Call{
				Func: cgen.Sprintf,
				Args: cgen.CommaSpaced{msg, pre, p.lineNum},
			},
		},
		cgen.Var{Type: cgen.VaList, What: ap},
		cgen.Call{
			Func: cgen.VaStart,
			Args: cgen.CommaSpaced{ap, p.format},
		},
		cgen.Call{
			Func: cgen.Vsnprintf,
			Args: cgen.CommaSpaced{
				cgen.Add{Expr1: msg, Expr2: step},
				cgen.Sub{Expr1: size, Expr2: step},
				p.format, ap,
			},
		},
		cgen.Call{Func: cgen.VaEnd, Args: ap},
		cgen.Return{Expr: msg},
	}
}

func (p *Prep) Append(to []byte) []byte {
	p.lineNum = p.name("lineNum")
	p.format = p.name("format")
	return cgen.StaticFuncDef{
		ReturnType: cgen.PtrChar,
		Name:       p.funcName,
		Params: cgen.CommaSpaced{
			cgen.Param{Type: cgen.PtrdiffT, What: p.lineNum},
			cgen.Param{Type: cgen.PtrChar, What: p.format},
			cgen.Ellipsis,
		},
		Body: p.body(),
	}.Append(to)
}

type FormatIf struct {
	*Ctx
	Cond   cgen.Gen
	Format string
	Args   []cgen.Gen
	Unwind cgen.Gen
}

func (f *FormatIf) Append(to []byte) []byte {
	args := make(cgen.CommaSpaced, 2+len(f.Args))
	args[0] = cgen.LineNum
	args[1] = cgen.DoubleQuoted(f.Format)
	copy(args[2:], f.Args)
	call := cgen.Call{
		Func: cgen.Vb(f.funcName),
		Args: args,
	}
	var then cgen.Stmts
	if f.Unwind == nil {
		then = cgen.Stmts{cgen.Return{Expr: call}}
	} else {
		msg := f.name("msg")
		then = cgen.Stmts{
			cgen.Var{Type: cgen.PtrChar, What: msg, Init: call},
			f.Unwind,
			cgen.Return{Expr: msg},
		}
	}
	return cgen.Stmts{cgen.If{
		Cond: cgen.Unlikely{Cond: f.Cond},
		Then: then,
	}}.Append(to)
}

const errnoFormat = "errno %d"

type ErrnoIf struct {
	*Ctx
	Cond   cgen.Gen
	Unwind cgen.Gen
}

func (e *ErrnoIf) Append(to []byte) []byte {
	return (&FormatIf{
		Ctx:    e.Ctx,
		Cond:   e.Cond,
		Format: errnoFormat,
		Args:   []cgen.Gen{cgen.Errno},
		Unwind: e.Unwind,
	}).Append(to)
}

type ReturnedErrnoIf struct {
	*Ctx
	Call   cgen.Gen
	Unwind cgen.Gen
}

func (r *ReturnedErrnoIf) Append(to []byte) []byte {
	err := r.name("err")
	return cgen.Stmts{
		cgen.Var{Type: cgen.Int, What: err, Init: r.Call},
		&FormatIf{
			Ctx:    r.Ctx,
			Cond:   err,
			Format: errnoFormat,
			Args:   []cgen.Gen{err},
			Unwind: r.Unwind,
		},
	}.Append(to)
}
