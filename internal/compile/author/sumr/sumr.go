package sumr

import (
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
)

func il(i int) cgen.Gen {
	return cgen.IntLit(i)
}

func mix(a, b cgen.Stmts) cgen.Stmts {
	var (
		tot = len(a) + len(b)
		ret = make(cgen.Stmts, tot)
		n   = 0
	)
	for i := 0; n < tot; i++ {
		if i < len(a) {
			ret[n] = a[i]
			n++
		}
		if i < len(b) {
			ret[n] = b[i]
			n++
		}
	}
	return ret
}

type Pack struct {
	Platform raw.Platform
	Nms      nmsrc.Src
	Vars     []cgen.Gen
}

func (p *Pack) Append(to []byte) []byte {
	var gen cgen.Gen
	switch p.Platform {
	case raw.AVX512Float32:
		gen = &m512Pack{Pack: p}
	default:
		panic("bug")
	}
	return gen.Append(to)
}

func (p *Pack) name(s string) cgen.Gen {
	return cgen.Vb(p.Nms.Name(s))
}

type m512Pack struct {
	*Pack
	pmEven cgen.Gen
	pmOdd  cgen.Gen
	pm1Lo  cgen.Gen
	pm1Hi  cgen.Gen
	pm4Lo  cgen.Gen
	pm4Hi  cgen.Gen
}

func (m *m512Pack) Append(to []byte) []byte {
	n := len(m.Vars)
	switch {
	case n == 0:
		return to
	case n > 16:
		panic("bug")
	}
	gs := make(cgen.Gens, 3)
	switch {
	case n == 1 || n > 8:
		gs[1] = m.fold(m.Vars, 1)
	default:
		gs[1] = m.fold(m.Vars, 2)
		var (
			lower = m.Vars[0]
			upper = m.name("upper")
		)
		decl := cgen.Var{
			Type: avx.M512, What: upper,
		}
		assn := cgen.Assign{
			Expr1: lower,
		}
		if n == 2 {
			decl.Init = avx.Mm512ShufflePs{
				lower, lower, il(3<<2 | 1),
			}
			assn.Expr2 = avx.Mm512ShufflePs{
				lower, lower, il(2<<2 | 0),
			}
		} else {
			m.pmOdd = m.name("pmOdd")
			decl.Init = avx.Mm512PermutexvarPs{
				m.pmOdd, lower,
			}
			m.pmEven = m.name("pmEven")
			assn.Expr2 = avx.Mm512PermutexvarPs{
				m.pmEven, lower,
			}
		}
		gs[2] = cgen.Stmts{
			decl,
			assn,
			cgen.Assign{
				Expr1: lower,
				Expr2: avx.Mm512AddPs{
					lower, upper,
				},
			},
		}
	}
	gs[0] = m.pms()
	return gs.Append(to)
}

func (m *m512Pack) fold(vs []cgen.Gen, w int) cgen.Stmts {
	var (
		stmts = make(cgen.Stmts, 4)
		lower = vs[0]
		upper = m.name("upper")
	)
	decl := cgen.Var{
		Type: avx.M512, What: upper,
	}
	if n := len(vs); n == 1 {
		if w < 8 {
			stmts[0] = m.fold(vs, w*2)
		}
		switch w {
		case 1:
			decl.Init = avx.Mm512ShufflePs{
				lower, lower, il(1),
			}
		case 2:
			decl.Init = avx.Mm512ShufflePs{
				lower, lower, il(3<<2 | 2),
			}
		case 4:
			decl.Init = avx.Mm512ShuffleF32x4{
				lower, lower, il(1),
			}
		case 8:
			decl.Init = avx.Mm512ShuffleF32x4{
				lower, lower, il(3<<2 | 2),
			}
		}
	} else {
		if w < 8 {
			var (
				n2  = n >> 1
				n1  = n - n2
				vs1 = make([]cgen.Gen, n1)
				vs2 = make([]cgen.Gen, n2)
			)
			for i, v := range vs {
				if i&1 == 0 {
					vs1[i>>1] = v
				} else {
					vs2[i>>1] = v
				}
			}
			stmts[0] = mix(
				m.fold(vs1, w*2),
				m.fold(vs2, w*2),
			)
		}
		v := vs[1]
		assn := cgen.Assign{
			Expr1: lower,
		}
		switch w {
		case 1:
			if m.pm1Lo == nil {
				m.pm1Lo = m.name("pm1Lo")
				m.pm1Hi = m.name("pm1Hi")
			}
			decl.Init = avx.Mm512Permutex2varPs{
				lower, m.pm1Hi, v,
			}
			assn.Expr2 = avx.Mm512Permutex2varPs{
				lower, m.pm1Lo, v,
			}
		case 2:
			decl.Init = avx.Mm512ShufflePs{
				lower, v, il(3<<6 | 2<<4 | 3<<2 | 2),
			}
			assn.Expr2 = avx.Mm512ShufflePs{
				lower, v, il(1<<6 | 0<<4 | 1<<2 | 0),
			}
		case 4:
			if m.pm4Lo == nil {
				m.pm4Lo = m.name("pm4Lo")
				m.pm4Hi = m.name("pm4Hi")
			}
			decl.Init = avx.Mm512Permutex2varPs{
				lower, m.pm4Hi, v,
			}
			assn.Expr2 = avx.Mm512Permutex2varPs{
				lower, m.pm4Lo, v,
			}
		case 8:
			decl.Init = avx.Mm512ShuffleF32x4{
				lower, v, il(3<<6 | 2<<4 | 3<<2 | 2),
			}
			assn.Expr2 = avx.Mm512ShuffleF32x4{
				lower, v, il(1<<6 | 0<<4 | 1<<2 | 0),
			}
		}
		stmts[2] = assn
	}
	stmts[1] = decl
	stmts[3] = cgen.Assign{
		Expr1: lower,
		Expr2: avx.Mm512AddPs{
			lower, upper,
		},
	}
	return stmts
}

func (m *m512Pack) pms() cgen.Gen {
	decl := func(pm cgen.Gen, fn func(int) int) cgen.Gen {
		if pm == nil {
			return nil
		}
		set := make(avx.Mm512SetEpi32, 16)
		for i := 0; i < 16; i++ {
			set[15-i] = il(fn(i))
		}
		return cgen.Var{
			Type: avx.M512i, What: pm,
			Init: set,
		}
	}
	return cgen.Stmts{
		decl(m.pmEven, func(i int) int {
			return 0 + i*2
		}),
		decl(m.pmOdd, func(i int) int {
			return 1 + i*2
		}),
		decl(m.pm1Lo, func(i int) int {
			return i&^1 + i&1*16
		}),
		decl(m.pm1Hi, func(i int) int {
			return i | 1 + i&1*16
		}),
		decl(m.pm4Lo, func(i int) int {
			return i&^4 + i&4*4
		}),
		decl(m.pm4Hi, func(i int) int {
			return i | 4 + i&4*4
		}),
	}
}
