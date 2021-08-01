package act

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
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src) *Ctx {
	return &Ctx{
		platform: pl.Config.Platform,
		nms:      nms,
	}
}

func (c *Ctx) name(s string) string {
	return c.nms.Name(s)
}

func vb(s string) cgen.Gen {
	return cgen.Vb(s)
}

type ReLU struct {
	*Ctx
	NegSlope float32
	Var      cgen.Gen
}

func (r *ReLU) Append(to []byte) []byte {
	if r.NegSlope == 1 {
		return to
	}
	switch r.platform {
	case raw.AVX512Float32:
		return r.m512(to)
	default:
		panic("bug")
	}
}

func (r *ReLU) m512(to []byte) []byte {
	stmts := make(cgen.Stmts, 2)
	switch r.NegSlope {
	case 0:
		stmts[0] = cgen.Assign{
			Expr1: r.Var,
			Expr2: avx.Mm512MaxPs{avx.Mm512SetzeroPs, r.Var},
		}
	default:
		mask := vb(r.name("mask"))
		stmts[0] = cgen.Var{
			Type: avx.Mmask16, What: mask,
			Init: avx.Mm512CmpPsMask{
				r.Var, avx.Mm512SetzeroPs, avx.CmpLtOq,
			},
		}
		stmts[1] = cgen.Assign{
			Expr1: r.Var,
			Expr2: avx.Mm512MaskMulPs{
				r.Var, mask,
				r.Var, avx.Mm512Set1PsLit(r.NegSlope),
			},
		}
	}
	return stmts.Append(to)
}
