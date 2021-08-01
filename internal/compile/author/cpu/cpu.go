package cpu

import (
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/errmsg"
	"NN-512/internal/raw"
)

type Chk struct {
	Platform raw.Platform
	Emc      *errmsg.Ctx
}

func (c *Chk) Append(to []byte) []byte {
	switch c.Platform {
	case raw.AVX512Float32:
		return c.m512().Append(to)
	default:
		panic("bug")
	}
}

func (c *Chk) m512() cgen.Gen {
	call := cgen.Call{
		Func: cgen.CpuSupports,
		Args: cgen.DoubleQuoted("avx512f"),
	}
	return &errmsg.FormatIf{
		Ctx:    c.Emc,
		Cond:   cgen.IsZero{Expr: call},
		Format: "CPU does not support AVX512F",
	}
}
