package exp

import (
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
)

type Ctx struct {
	platform raw.Platform
	nms      nmsrc.Src
	funcName string
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src) *Ctx {
	return &Ctx{
		platform: pl.Config.Platform,
		nms:      nms,
		funcName: nms.Name(pl.Config.Prefix + "Exp"),
	}
}

func (c *Ctx) name(a string) cgen.Gen {
	return cgen.Vb(c.nms.Name(a))
}

type Prep struct {
	*Ctx
	to []byte
}

func (p *Prep) Append(to []byte) []byte {
	p.to = to
	switch p.platform {
	case raw.AVX512Float32:
		p.m512()
	default:
		panic("bug")
	}
	return p.to
}

func (p *Prep) m512() {
	var (
		x = p.name("x")
		t = p.name("t")
		r = p.name("r")
		f = p.name("f")
		g = p.name("g")
		y = p.name("y")
	)
	p.to = cgen.StaticFuncDef{
		ReturnType: avx.M512,
		Name:       p.funcName,
		Params:     cgen.Param{Type: avx.M512, What: x},
		Body: cgen.Stmts{
			cgen.Assign{
				Expr1: x,
				Expr2: avx.Mm512MaxPs{x, avx.Mm512Set1PsLit(-87.33654)},
			},
			cgen.Assign{
				Expr1: x,
				Expr2: avx.Mm512MinPs{x, avx.Mm512Set1PsLit(88.72284)},
			},
			cgen.Var{
				Type: avx.M512, What: t,
				Init: avx.Mm512MulPs{x, avx.Mm512Set1PsLit(1.442695)},
			},
			cgen.Var{
				Type: avx.M512, What: r,
				Init: avx.Mm512RoundscalePs{t, avx.FroundToNearestIntNoExc},
			},
			cgen.Var{
				Type: avx.M512, What: f,
				Init: avx.Mm512FmaddPs{r, avx.Mm512Set1PsLit(-0.69314575), x},
			},
			cgen.Assign{
				Expr1: f,
				Expr2: avx.Mm512FmaddPs{r, avx.Mm512Set1PsLit(-1.4286068e-6), f},
			},
			cgen.Var{
				Type: avx.M512, What: g, Init: avx.Mm512Set1PsLit(0.04194439),
			},
			cgen.Assign{
				Expr1: g,
				Expr2: avx.Mm512FmaddPs{g, f, avx.Mm512Set1PsLit(0.16800667)},
			},
			cgen.Assign{
				Expr1: g,
				Expr2: avx.Mm512FmaddPs{g, f, avx.Mm512Set1PsLit(0.49999994)},
			},
			cgen.Assign{
				Expr1: g,
				Expr2: avx.Mm512FmaddPs{g, f, avx.Mm512Set1PsLit(0.9999569)},
			},
			cgen.Assign{
				Expr1: g,
				Expr2: avx.Mm512FmaddPs{g, f, avx.Mm512Set1PsLit(0.99999964)},
			},
			cgen.Var{
				Type: avx.M512i, What: y,
				Init: avx.Mm512SlliEpi32{
					avx.Mm512CvtpsEpi32{t}, cgen.IntLit(23),
				},
			},
			cgen.Return{Expr: avx.Mm512Castsi512Ps{
				avx.Mm512AddEpi32{y, avx.Mm512CastpsSi512{g}},
			}},
		},
	}.Append(p.to)
}

type Call struct {
	*Ctx
	Arg cgen.Gen
}

func (c *Call) Append(to []byte) []byte {
	return cgen.Call{
		Func: cgen.Vb(c.funcName),
		Args: c.Arg,
	}.Append(to)
}
