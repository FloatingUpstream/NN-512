package tobuild

import (
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/plan"
	"NN-512/internal/raw"
	"strings"
)

func Gen(pl *plan.Plan) cgen.Gen {
	var insert string
	switch pl.Config.Platform {
	case raw.AVX512Float32:
		insert = "-mavx512f"
	default:
		panic("bug")
	}
	return cgen.Comment{
		"To build an object file:",
		strings.Join([]string{
			"gcc",
			"-c",
			"-w",
			"-std=c99",
			"-pthread",
			"-Ofast",
			insert,
			pl.Config.Prefix + ".c",
		}, " "),
	}
}
