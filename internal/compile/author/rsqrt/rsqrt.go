package rsqrt

import (
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
)

func vb(s string) cgen.Gen {
	return cgen.Vb(s)
}

type Ctx struct {
	platform raw.Platform
	nms      nmsrc.Src
	funcName string
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src) *Ctx {
	return &Ctx{
		platform: pl.Config.Platform,
		nms:      nms,
		funcName: nms.Name(pl.Config.Prefix + "Rsqrt"),
	}
}

func (c *Ctx) Prep() cgen.Gen {
	switch c.platform {
	case raw.AVX512Float32:
		return c.m512()
	default:
		panic("bug")
	}
}

func (c *Ctx) name(s string) cgen.Gen {
	return vb(c.nms.Name(s))
}

func (c *Ctx) m512() cgen.Gen {
	var (
		x = c.name("x")
		y = c.name("y")
		z = c.name("z")
		a = c.name("a")
		b = c.name("b")
	)
	return cgen.StaticFuncDef{
		ReturnType: avx.M512,
		Name:       c.funcName,
		Params:     cgen.Param{Type: avx.M512, What: x},
		Body: cgen.Stmts{
			cgen.Var{
				Type: avx.M512, What: y,
				Init: avx.Mm512Rsqrt14Ps{x},
			},
			cgen.Var{
				Type: avx.M512, What: z,
				Init: avx.Mm512MulPs{x, y},
			},
			cgen.Var{
				Type: avx.M512, What: a,
				Init: avx.Mm512MulPs{
					y, avx.Mm512Set1PsLit(0.5),
				},
			},
			cgen.Var{
				Type: avx.M512, What: b,
				Init: avx.Mm512FnmaddPs{
					y, z, avx.Mm512Set1PsLit(3),
				},
			},
			cgen.Return{
				Expr: avx.Mm512MulPs{a, b},
			},
		},
	}
}

type Call struct {
	*Ctx
	Arg cgen.Gen
}

func (c *Call) Append(to []byte) []byte {
	return cgen.Call{
		Func: vb(c.funcName),
		Args: c.Arg,
	}.Append(to)
}
