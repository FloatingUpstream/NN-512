package include

import (
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/plan"
	"NN-512/internal/raw"
)

var alwaysH = [...]string{
	"pthread.h",
	"stddef.h",
}

var alwaysC = [...]string{
	"errno.h",
	"stdarg.h",
	"stdint.h",
	"stdio.h",
	"stdlib.h",
	"string.h",
}

func H() cgen.Gen {
	var (
		n  = len(alwaysH)
		gs = make(cgen.Gens, n)
	)
	for i, name := range &alwaysH {
		gs[i] = cgen.Preprocessor{
			Head: cgen.Include,
			Tail: cgen.AngleBracketed(name),
		}
	}
	return gs
}

func C(pl *plan.Plan) cgen.Gen {
	var gs cgen.Gens
	inc := func(a cgen.Gen) {
		g := cgen.Preprocessor{Head: cgen.Include, Tail: a}
		gs = append(gs, g)
	}
	sys := func(a string) { inc(cgen.AngleBracketed(a)) }
	usr := func(a string) { inc(cgen.DoubleQuoted(a)) }
	newline := func() { gs = append(gs, cgen.Newline) }
	for _, name := range &alwaysC {
		sys(name)
	}
	newline()
	switch pl.Config.Platform {
	case raw.AVX512Float32:
		sys("immintrin.h")
	default:
		panic("bug")
	}
	newline()
	usr(pl.Config.Prefix + ".h")
	return gs
}
