package bn

import (
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/rsqrt"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
	"fmt"
)

type Ctx struct {
	prefix   string
	platform raw.Platform
	nms      nmsrc.Src
	rc       *rsqrt.Ctx
	dedup    map[string]string
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src, rc *rsqrt.Ctx) *Ctx {
	return &Ctx{
		prefix:   pl.Config.Prefix + "Bn",
		platform: pl.Config.Platform,
		nms:      nms,
		rc:       rc,
		dedup:    make(map[string]string),
	}
}

func (c *Ctx) maBytes() int {
	switch c.platform {
	case raw.AVX512Float32:
		return 8
	default:
		panic("bug")
	}
}

func (c *Ctx) name(s string) string {
	return c.nms.Name(s)
}

func vb(s string) cgen.Gen {
	return cgen.Vb(s)
}

func il(i int) cgen.Gen {
	return cgen.IntLit(i)
}

func cast(stride int) cgen.Gen {
	return cgen.Cast{
		Type: cgen.PtrdiffT,
		Expr: il(stride),
	}
}

func addr(ptr, stride, idx cgen.Gen) cgen.Gen {
	return cgen.Add{
		Expr1: ptr,
		Expr2: cgen.Mul{
			Expr1: stride,
			Expr2: idx,
		},
	}
}

type Simplify struct {
	*Ctx
	Channels  int
	Epsilon   float32
	Means     cgen.Gen
	Variances cgen.Gen
	Scales    cgen.Gen
	Shifts    cgen.Gen
	Mas       cgen.Gen
	funcName  string
	means     cgen.Gen
	variances cgen.Gen
	scales    cgen.Gen
	shifts    cgen.Gen
	mas       cgen.Gen
}

func (s *Simplify) Append(to []byte) []byte {
	return cgen.Stmts{cgen.Call{
		Func: vb(s.funcName),
		Args: cgen.CommaLines{
			s.Means,
			s.Variances,
			s.Scales,
			s.Shifts,
			s.Mas,
		},
	}}.Append(to)
}

func (s *Simplify) MasBytes() int {
	return s.Channels * s.maBytes()
}

func (s *Simplify) Prep() cgen.Gen {
	const label = "Simplify"
	sig := fmt.Sprintf(label+" %d %g", s.Channels, s.Epsilon)
	if prior, ok := s.dedup[sig]; ok {
		s.funcName = prior
		return nil
	}
	s.funcName = s.name(s.prefix + label)
	s.dedup[sig] = s.funcName
	return cgen.Gens{s.funcDef(), cgen.Newline}
}

func (s *Simplify) funcDef() cgen.Gen {
	s.means = vb(s.name("means"))
	s.variances = vb(s.name("variances"))
	s.scales = vb(s.name("scales"))
	s.shifts = vb(s.name("shifts"))
	s.mas = vb(s.name("mas"))
	return cgen.StaticFuncDef{
		ReturnType: cgen.Void,
		Name:       s.funcName,
		Params: cgen.CommaLines{
			cgen.Param{Type: cgen.RestrictPtrFloat, What: s.means},
			cgen.Param{Type: cgen.RestrictPtrFloat, What: s.variances},
			cgen.Param{Type: cgen.RestrictPtrFloat, What: s.scales},
			cgen.Param{Type: cgen.RestrictPtrFloat, What: s.shifts},
			cgen.Param{Type: cgen.RestrictPtrChar, What: s.mas},
		},
		Body: s.body(),
	}
}

func (s *Simplify) body() cgen.Gen {
	switch s.platform {
	case raw.AVX512Float32:
		return s.m512()
	default:
		panic("bug")
	}
}

func (s *Simplify) m512() cgen.Gen {
	const (
		unroll    = 5
		lanes     = 16
		laneBytes = 4
	)
	var (
		iters = s.Channels / (unroll * lanes)
		after = s.Channels % (unroll * lanes)
		eps   = vb(s.name("eps"))
		xlo   = vb(s.name("xlo"))
		xhi   cgen.Gen
	)
	ld := func(to, ptr, i cgen.Gen, j, n int) cgen.Gen {
		var (
			stmt = cgen.Var{Type: avx.M512, What: to}
			from = addr(ptr, cast(lanes), il(j))
		)
		if iters > 0 {
			from = addr(from, cast(unroll*lanes), i)
		}
		if n == lanes {
			stmt.Init = avx.Mm512LoaduPs{from}
		} else {
			stmt.Init = avx.Mm512MaskzLoaduPs{
				il(1<<uint(n) - 1), from,
			}
		}
		return stmt
	}
	st := func(lo, hi, i cgen.Gen, j, n int) cgen.Gen {
		const half = lanes * laneBytes
		var (
			stmts = make(cgen.Stmts, 2)
			alo   = addr(s.mas, cast(half), il(j*2+0))
			ahi   = addr(s.mas, cast(half), il(j*2+1))
		)
		if iters > 0 {
			alo = addr(alo, cast(unroll*half*2), i)
			ahi = addr(ahi, cast(unroll*half*2), i)
		}
		if nn := n * 2; nn < lanes {
			stmts[0] = avx.Mm512MaskStoreuPs{
				alo, il(1<<uint(nn) - 1), lo,
			}
		} else {
			stmts[0] = avx.Mm512StoreuPs{alo, lo}
			if nn -= lanes; nn == lanes {
				stmts[1] = avx.Mm512StoreuPs{ahi, hi}
			} else if nn > 0 {
				stmts[1] = avx.Mm512MaskStoreuPs{
					ahi, il(1<<uint(nn) - 1), hi,
				}
			}
		}
		return stmts
	}
	deck := func(i cgen.Gen, j, n int) []cgen.Gen {
		var (
			gs  = make([]cgen.Gen, 10)
			va  = vb(s.name("va"))
			rcp = vb(s.name("rcp"))
			sc  = vb(s.name("sc"))
			mul = vb(s.name("mul"))
			me  = vb(s.name("me"))
			sh  = vb(s.name("sh"))
			add = vb(s.name("add"))
			lo  = vb(s.name("lo"))
			hi  cgen.Gen
		)
		gs[0] = ld(va, s.variances, i, j, n)
		gs[1] = cgen.Var{
			Type: avx.M512, What: rcp,
			Init: &rsqrt.Call{
				Ctx: s.rc,
				Arg: avx.Mm512AddPs{eps, va},
			},
		}
		gs[2] = ld(sc, s.scales, i, j, n)
		gs[3] = cgen.Var{
			Type: avx.M512, What: mul,
			Init: avx.Mm512MulPs{rcp, sc},
		}
		gs[4] = ld(me, s.means, i, j, n)
		gs[5] = ld(sh, s.shifts, i, j, n)
		gs[6] = cgen.Var{
			Type: avx.M512, What: add,
			Init: avx.Mm512FnmaddPs{
				me, mul, sh,
			},
		}
		gs[7] = cgen.Var{
			Type: avx.M512, What: lo,
			Init: avx.Mm512Permutex2varPs{
				mul, xlo, add,
			},
		}
		if n > lanes/2 {
			hi = vb(s.name("hi"))
			gs[8] = cgen.Var{
				Type: avx.M512, What: hi,
				Init: avx.Mm512Permutex2varPs{
					mul, xhi, add,
				},
			}
		}
		gs[9] = st(lo, hi, i, j, n)
		return gs
	}
	shuf := func(a [][]cgen.Gen) cgen.Stmts {
		var (
			n     = len(a[0])
			stmts = make(cgen.Stmts, len(a)*n)
			i     = 0
		)
		for j := 0; j < n; j++ {
			for k := range a {
				stmts[i] = a[k][j]
				i++
			}
		}
		return stmts
	}
	var (
		stmts = make(cgen.Stmts, 5)
		lower = make(avx.Mm512SetEpi32, lanes)
		upper = make(avx.Mm512SetEpi32, lanes)
	)
	stmts[0] = cgen.Var{
		Type: avx.M512, What: eps,
		Init: avx.Mm512Set1PsLit(s.Epsilon),
	}
	for i := 0; i < lanes; i++ {
		x := i>>1 + lanes*(i&1)
		lower[lanes-1-i] = il(x)
		upper[lanes-1-i] = il(x + lanes/2)
	}
	stmts[1] = cgen.Var{
		Type: avx.M512i, What: xlo,
		Init: lower,
	}
	if s.Channels > lanes/2 {
		xhi = vb(s.name("xhi"))
		stmts[2] = cgen.Var{
			Type: avx.M512i, What: xhi,
			Init: upper,
		}
	}
	if iters > 0 {
		var (
			inner = make([][]cgen.Gen, unroll)
			i     = vb(s.name("i"))
		)
		for j := 0; j < unroll; j++ {
			inner[j] = deck(i, j, lanes)
		}
		stmts[3] = cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT, What: i,
				Init: cgen.Zero,
			},
			Cond: cgen.CmpL{
				Expr1: i, Expr2: il(iters),
			},
			Post: cgen.IncPre{Expr: i},
			Body: shuf(inner),
		}
	}
	if after > 0 {
		var (
			full  = after / lanes
			part  = after % lanes
			outer = make([][]cgen.Gen, full, full+1)
			i     = il(iters)
		)
		for j := 0; j < full; j++ {
			outer[j] = deck(i, j, lanes)
		}
		if part > 0 {
			last := deck(i, full, part)
			outer = append(outer, last)
		}
		stmts[4] = shuf(outer)
	}
	return stmts
}

type Offset struct {
	*Ctx
	Mas     cgen.Gen
	Channel cgen.Gen
}

func (o *Offset) Append(to []byte) []byte {
	var (
		stride = cast(o.maBytes())
		expr   = addr(o.Mas, stride, o.Channel)
	)
	return expr.Append(to)
}

type Load struct {
	*Ctx
	Mas     cgen.Gen
	Channel cgen.Gen
	Mul     cgen.Gen
	Add     cgen.Gen
	Cnt     int
	Spread  int
}

func (l *Load) Append(to []byte) []byte {
	switch l.platform {
	case raw.AVX512Float32:
		return l.m512(to)
	default:
		panic("bug")
	}
}

func (l *Load) m512(to []byte) []byte {
	if l.Cnt == 0 {
		return l.m512Broadcast(to)
	}
	return l.m512Singles(to)
}

func (l *Load) m512Broadcast(to []byte) []byte {
	var (
		stmts = make(cgen.Stmts, 2)
		a1    = cgen.Cast{Type: cgen.PtrFloat, Expr: l.Mas}
		a2    = addr(a1, cast(2), l.Channel)
		a3    = cgen.Gen(cgen.Paren{Inner: a2})
	)
	if l.Mul != nil {
		stmts[0] = cgen.Var{
			Type: avx.M512, What: l.Mul,
			Init: avx.Mm512Set1Ps{cgen.Elem{
				Arr: a3, Idx: cgen.Zero,
			}},
		}
	}
	if l.Add != nil {
		stmts[1] = cgen.Var{
			Type: avx.M512, What: l.Add,
			Init: avx.Mm512Set1Ps{cgen.Elem{
				Arr: a3, Idx: cgen.One,
			}},
		}
	}
	return stmts.Append(to)
}

func (l *Load) m512Singles(to []byte) []byte {
	const (
		lanes     = 16
		laneBytes = 4
	)
	var (
		stmts  = make(cgen.Stmts, 6)
		spread = l.Spread
		even   = make(avx.Mm512SetEpi32, lanes)
		odd    = make(avx.Mm512SetEpi32, lanes)
		pmMul  = vb(l.name("pmMul"))
		pmAdd  = vb(l.name("pmAdd"))
		mul    cgen.Gen
		add    cgen.Gen
	)
	if spread == 0 {
		spread = 1
	}
	for i := 0; i < lanes; i++ {
		j, k := lanes-1-i, i/spread*2
		even[j], odd[j] = il(k), il(k+1)
	}
	stmts[0] = cgen.Var{
		Type: avx.M512i, What: pmMul,
		Init: even,
	}
	stmts[1] = cgen.Var{
		Type: avx.M512i, What: pmAdd,
		Init: odd,
	}
	lo := &Offset{
		Ctx:     l.Ctx,
		Mas:     l.Mas,
		Channel: l.Channel,
	}
	if n := l.Cnt * 2; n <= lanes {
		mas := vb(l.name("mas"))
		stmts[2] = cgen.Var{
			Type: avx.M512, What: mas,
			Init: avx.Mm512MaskzLoaduPs{
				il(1<<uint(n) - 1), lo,
			},
		}
		mul = avx.Mm512PermutexvarPs{
			pmMul, mas,
		}
		add = avx.Mm512PermutexvarPs{
			pmAdd, mas,
		}
	} else {
		var (
			masLo = vb(l.name("masLo"))
			masHi = vb(l.name("masHi"))
		)
		stmts[2] = cgen.Var{
			Type: avx.M512, What: masLo,
			Init: avx.Mm512LoaduPs{lo},
		}
		hi := cgen.Add{
			Expr1: lo,
			Expr2: cast(lanes * laneBytes),
		}
		stmts[3] = cgen.Var{
			Type: avx.M512, What: masHi,
			Init: avx.Mm512MaskzLoaduPs{
				il(1<<uint(n-lanes) - 1), hi,
			},
		}
		mul = avx.Mm512Permutex2varPs{
			masLo, pmMul, masHi,
		}
		add = avx.Mm512Permutex2varPs{
			masLo, pmAdd, masHi,
		}
	}
	stmts[4] = cgen.Var{
		Type: avx.M512, What: l.Mul,
		Init: mul,
	}
	stmts[5] = cgen.Var{
		Type: avx.M512, What: l.Add,
		Init: add,
	}
	return stmts.Append(to)
}

type Apply struct {
	*Ctx
	Mul  cgen.Gen
	Add  cgen.Gen
	To   cgen.Gen
	Mask cgen.Gen
}

func (a *Apply) Append(to []byte) []byte {
	switch a.platform {
	case raw.AVX512Float32:
		return a.m512(to)
	default:
		panic("bug")
	}
}

func (a *Apply) m512(to []byte) []byte {
	assn := cgen.Assign{
		Expr1: a.To,
	}
	if a.Mask == nil {
		assn.Expr2 = avx.Mm512FmaddPs{
			a.To, a.Mul, a.Add,
		}
	} else {
		assn.Expr2 = avx.Mm512MaskFmaddPs{
			a.To, a.Mask, a.Mul, a.Add,
		}
	}
	return cgen.Stmts{
		assn,
	}.Append(to)
}
