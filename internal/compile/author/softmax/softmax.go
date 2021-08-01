package softmax

import (
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/exp"
	"NN-512/internal/compile/author/threader"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
	"fmt"
)

type Ctx struct {
	prefix   string
	platform raw.Platform
	nms      nmsrc.Src
	tc       *threader.Ctx
	ec       *exp.Ctx
	dedup    map[string]string
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src, tc *threader.Ctx, ec *exp.Ctx) *Ctx {
	return &Ctx{
		prefix:   pl.Config.Prefix + "Softmax",
		platform: pl.Config.Platform,
		nms:      nms,
		tc:       tc,
		ec:       ec,
		dedup:    make(map[string]string),
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

func cast(pitch int) cgen.Gen {
	return cgen.Cast{
		Type: cgen.PtrdiffT,
		Expr: il(pitch),
	}
}

func addr(ptr, pitch, idx cgen.Gen) cgen.Gen {
	return cgen.Add{
		Expr1: ptr,
		Expr2: cgen.Mul{Expr1: pitch, Expr2: idx},
	}
}

type Call struct {
	*Ctx
	Team     cgen.Gen
	Tensors  []cgen.Gen
	Shapes   []Shape
	funcName string
}

func (c *Call) Prep() cgen.Gen {
	sig := fmt.Sprintf("%v", c.Shapes)
	if prior, ok := c.dedup[sig]; ok {
		c.funcName = prior
		return nil
	}
	c.funcName = c.name(c.prefix)
	c.dedup[sig] = c.funcName
	return cgen.Gens{
		&funcDef{
			Ctx:      c.Ctx,
			funcName: c.funcName,
			shapes:   c.Shapes,
		},
		cgen.Newline,
	}
}

func (c *Call) Append(to []byte) []byte {
	var (
		tensors = vb(c.name("tensors"))
		ptrs    = cgen.CommaLines(c.Tensors)
	)
	return cgen.Stmts{
		cgen.Var{
			Type: cgen.PtrChar,
			What: cgen.Elem{Arr: tensors},
			Init: cgen.Brace{Inner: ptrs},
		},
		cgen.Call{
			Func: vb(c.funcName),
			Args: cgen.CommaSpaced{
				c.Team, tensors,
			},
		},
	}.Append(to)
}

type Shape struct {
	Channels    int
	Height      int
	Width       int
	ElemBytes   int
	Pitch1Bytes int
	Pitch2Bytes int
}

type funcDef struct {
	*Ctx
	funcName string
	shapes   []Shape
}

func (f *funcDef) Append(to []byte) []byte {
	var lanes int
	switch f.platform {
	case raw.AVX512Float32:
		lanes = 16
	default:
		panic("bug")
	}
	const (
		packed = iota
		linear
		planar
	)
	var (
		form     = packed
		shape    = &f.shapes[0]
		channels = shape.Channels
		height   = shape.Height
		width    = shape.Width
		spatial  = height * width
		elem     = shape.ElemBytes
		nt       = len(f.shapes)
		pitches1 = make([]cgen.Gen, nt)
		pitches2 = make([]cgen.Gen, nt)
	)
	if spatial*2 > lanes || channels < 2 {
		form = linear
	}
	for i := range f.shapes {
		var (
			at     = &f.shapes[i]
			pitch1 = at.Pitch1Bytes
			pitch2 = at.Pitch2Bytes
		)
		if pitch1 != width*elem {
			form = planar
		} else if form == packed &&
			pitch2 != spatial*elem {
			form = linear
		}
		pitches1[i] = cast(pitch1)
		pitches2[i] = cast(pitch2)
	}
	var (
		gens    = make(cgen.Gens, 3)
		team    = vb(f.name("team"))
		tensors = vb(f.name("tensors"))
		body    cgen.Gen
	)
	if form == packed {
		stmts := make(cgen.Stmts, 2)
		stmts[0] = cgen.Cast{
			Type: cgen.Void, Expr: team,
		}
		switch f.platform {
		case raw.AVX512Float32:
			stmts[1] = &m512Packed{
				Ctx:      f.Ctx,
				channels: channels,
				spatial:  spatial,
				tensors:  tensors,
				nt:       nt,
			}
		}
		body = stmts
	} else {
		var (
			callee = f.name(f.funcName + "Callee")
			hull   []cgen.Gen
		)
		if form == linear {
			switch f.platform {
			case raw.AVX512Float32:
				gens[0] = &m512Linear{
					Ctx:      f.Ctx,
					funcName: callee,
					channels: channels,
					spatial:  spatial,
					pitches:  pitches2,
				}
			}
			hull = []cgen.Gen{
				il((spatial + lanes - 1) / lanes),
			}
		} else {
			switch f.platform {
			case raw.AVX512Float32:
				gens[0] = &m512Planar{
					Ctx:      f.Ctx,
					funcName: callee,
					channels: channels,
					width:    width,
					pitches1: pitches1,
					pitches2: pitches2,
				}
			}
			hull = []cgen.Gen{
				il((width + lanes - 1) / lanes),
				il(height),
			}
		}
		gens[1] = cgen.Newline
		body = &threader.Do{
			Ctx:    f.tc,
			Callee: vb(callee),
			Any:    tensors,
			Hull:   hull,
			Team:   team,
		}
	}
	gens[2] = cgen.StaticFuncDef{
		ReturnType: cgen.Void,
		Name:       f.funcName,
		Params: cgen.CommaSpaced{
			cgen.Param{
				Type: f.tc.PtrTeam,
				What: team,
			},
			cgen.Param{
				Type: cgen.PtrPtrChar,
				What: tensors,
			},
		},
		Body: body,
	}
	return gens.Append(to)
}

type m512Planar struct {
	*Ctx
	funcName string
	channels int
	width    int
	pitches1 []cgen.Gen
	pitches2 []cgen.Gen
}

func (m *m512Planar) Append(to []byte) []byte {
	const (
		lanes     = 16
		laneBytes = 4
	)
	var (
		n       = len(m.pitches1)
		stmts   = make(cgen.Stmts, 4+n)
		tensors = vb(m.name("tensors"))
		w       = vb(m.name("w"))
		h       = vb(m.name("h"))
		mask    = vb(m.name("mask"))
		ptrs    = make([]cgen.Gen, n)
		max     = vb(m.name("max"))
		sum     = vb(m.name("sum"))
	)
	callee := &threader.Callee{
		Ctx:  m.tc,
		Name: m.funcName,
		Task: vb(m.name("task")),
		Pt:   vb(m.name("pt")),
	}
	stmts[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: tensors,
		Init: callee.Any(),
	}
	stmts[1] = cgen.Var{
		Type: cgen.PtrdiffT, What: w,
		Init: cgen.Elem{Arr: callee.Pt, Idx: cgen.Zero},
	}
	stmts[2] = cgen.Var{
		Type: cgen.PtrdiffT, What: h,
		Init: cgen.Elem{Arr: callee.Pt, Idx: cgen.One},
	}
	stmts[3] = cgen.Var{
		Type: avx.Mmask16, What: mask,
		Init: cgen.Ternary{
			Cond: cgen.CmpL{
				Expr1: w,
				Expr2: il(m.width / lanes),
			},
			Then: il(1<<lanes - 1),
			Else: il(1<<uint(m.width%lanes) - 1),
		},
	}
	for i := range ptrs {
		ptrs[i] = vb(m.name("ptr"))
		var (
			a1 = cgen.Elem{Arr: tensors, Idx: il(i)}
			a2 = addr(a1, m.pitches1[i], h)
			a3 = addr(a2, cast(lanes*laneBytes), w)
		)
		stmts[4+i] = cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptrs[i], Init: a3,
		}
	}
	return callee.Func(cgen.Gens{
		stmts,
		&m512Max{
			Ctx:      m.Ctx,
			ptr:      ptrs[0],
			pitch:    m.pitches2[0],
			loopCnt:  m.channels,
			loopMask: mask,
			max:      max,
		},
		&m512Exp{
			Ctx:      m.Ctx,
			ldPtr:    ptrs[0],
			ldPitch:  m.pitches2[0],
			stPtr:    ptrs[1],
			stPitch:  m.pitches2[1],
			loopCnt:  m.channels,
			loopMask: mask,
			sub:      max,
			sum:      sum,
		},
		&m512Denom{
			Ctx:      m.Ctx,
			ptrs:     ptrs[1:],
			pitches:  m.pitches2[1:],
			loopCnt:  il(m.channels),
			loopMask: mask,
			divisor:  sum,
		},
	}).Append(to)
}

type m512Linear struct {
	*Ctx
	funcName string
	channels int
	spatial  int
	pitches  []cgen.Gen
}

func (m *m512Linear) Append(to []byte) []byte {
	const (
		lanes     = 16
		laneBytes = 4
	)
	var (
		n       = len(m.pitches)
		stmts   = make(cgen.Stmts, 3+n)
		tensors = vb(m.name("tensors"))
		i       = vb(m.name("i"))
		mask    = vb(m.name("mask"))
		ptrs    = make([]cgen.Gen, n)
		max     = vb(m.name("max"))
		sum     = vb(m.name("sum"))
	)
	callee := &threader.Callee{
		Ctx:  m.tc,
		Name: m.funcName,
		Task: vb(m.name("task")),
		Pt:   vb(m.name("pt")),
	}
	stmts[0] = cgen.Var{
		Type: cgen.PtrPtrChar, What: tensors,
		Init: callee.Any(),
	}
	stmts[1] = cgen.Var{
		Type: cgen.PtrdiffT, What: i,
		Init: cgen.Elem{Arr: callee.Pt, Idx: cgen.Zero},
	}
	stmts[2] = cgen.Var{
		Type: avx.Mmask16, What: mask,
		Init: cgen.Ternary{
			Cond: cgen.CmpL{
				Expr1: i,
				Expr2: il(m.spatial / lanes),
			},
			Then: il(1<<lanes - 1),
			Else: il(1<<uint(m.spatial%lanes) - 1),
		},
	}
	for j := range ptrs {
		ptrs[j] = vb(m.name("ptr"))
		var (
			a1 = cgen.Elem{Arr: tensors, Idx: il(j)}
			a2 = addr(a1, cast(lanes*laneBytes), i)
		)
		stmts[3+j] = cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptrs[j], Init: a2,
		}
	}
	return callee.Func(cgen.Gens{
		stmts,
		&m512Max{
			Ctx:      m.Ctx,
			ptr:      ptrs[0],
			pitch:    m.pitches[0],
			loopCnt:  m.channels,
			loopMask: mask,
			max:      max,
		},
		&m512Exp{
			Ctx:      m.Ctx,
			ldPtr:    ptrs[0],
			ldPitch:  m.pitches[0],
			stPtr:    ptrs[1],
			stPitch:  m.pitches[1],
			loopCnt:  m.channels,
			loopMask: mask,
			sub:      max,
			sum:      sum,
		},
		&m512Denom{
			Ctx:      m.Ctx,
			ptrs:     ptrs[1:],
			pitches:  m.pitches[1:],
			loopCnt:  il(m.channels),
			loopMask: mask,
			divisor:  sum,
		},
	}).Append(to)
}

type m512Packed struct {
	*Ctx
	channels int
	spatial  int
	tensors  cgen.Gen
	nt       int
}

func (m *m512Packed) Append(to []byte) []byte {
	var (
		ptrs     = make([]cgen.Gen, m.nt)
		loadPtrs = make(cgen.Stmts, m.nt)
	)
	for i := range ptrs {
		ptrs[i] = vb(m.name("ptr"))
		loadPtrs[i] = cgen.Var{
			Type: cgen.RestrictPtrChar,
			What: ptrs[i],
			Init: cgen.Elem{
				Arr: m.tensors,
				Idx: il(i),
			},
		}
	}
	const (
		lanes     = 16
		laneBytes = 4
	)
	loopChans := lanes / m.spatial
	if loopChans > m.channels {
		loopChans = m.channels
	}
	var (
		loopLanes = loopChans * m.spatial
		pitch     = cast(loopLanes * laneBytes)
		loopCnt   = m.channels / loopChans
		edgeLanes = m.channels % loopChans * m.spatial
		loopMask  = il(1<<uint(loopLanes) - 1)
		edgeMask  cgen.Gen
	)
	if edgeLanes > 0 {
		edgeMask = il(1<<uint(edgeLanes) - 1)
	}
	var (
		max     = vb(m.name("max"))
		sum     = vb(m.name("sum"))
		pitches = make([]cgen.Gen, m.nt-1)
	)
	for i := range pitches {
		pitches[i] = pitch
	}
	return cgen.Gens{
		loadPtrs,
		&m512Max{
			Ctx:      m.Ctx,
			ptr:      ptrs[0],
			pitch:    pitch,
			loopCnt:  loopCnt,
			loopMask: loopMask,
			edgeMask: edgeMask,
			max:      max,
		},
		&m512Fold{
			Ctx:  m.Ctx,
			dat:  max,
			cnt:  loopChans,
			each: m.spatial,
			op:   foldMax,
		},
		&m512Exp{
			Ctx:      m.Ctx,
			ldPtr:    ptrs[0],
			ldPitch:  pitch,
			stPtr:    ptrs[1],
			stPitch:  pitch,
			loopCnt:  loopCnt,
			loopMask: loopMask,
			edgeMask: edgeMask,
			sub:      max,
			sum:      sum,
		},
		&m512Fold{
			Ctx:  m.Ctx,
			dat:  sum,
			cnt:  loopChans,
			each: m.spatial,
			op:   foldAdd,
		},
		&m512Denom{
			Ctx:      m.Ctx,
			ptrs:     ptrs[1:],
			pitches:  pitches,
			loopCnt:  il(loopCnt),
			loopMask: loopMask,
			edgeMask: edgeMask,
			divisor:  sum,
		},
	}.Append(to)
}

type m512Max struct {
	*Ctx
	ptr      cgen.Gen
	pitch    cgen.Gen
	loopCnt  int
	loopMask cgen.Gen
	edgeMask cgen.Gen
	max      cgen.Gen
}

func (m *m512Max) Append(to []byte) []byte {
	const unroll = 16
	nparts := m.loopCnt
	if nparts > unroll {
		nparts = unroll
	}
	parts := make([]cgen.Gen, nparts)
	parts[0] = m.max
	for i := 1; i < nparts; i++ {
		parts[i] = vb(m.name("max"))
	}
	first := make(cgen.Stmts, nparts)
	for i := range first {
		first[i] = cgen.Var{
			Type: avx.M512, What: parts[i],
			Init: avx.Mm512MaskzLoaduPs{
				m.loopMask,
				addr(m.ptr, m.pitch, il(i)),
			},
		}
	}
	stmts := make(cgen.Stmts, 6)
	stmts[0] = first
	if remain := m.loopCnt - nparts; remain > 0 {
		var (
			iters = remain / unroll
			after = remain % unroll
		)
		fill := func(s cgen.Stmts, i cgen.Gen, n int) {
			for j := 0; j < n; j++ {
				var (
					dat  = vb(m.name("dat"))
					part = parts[j]
				)
				from1 := addr(m.ptr, m.pitch, il(j))
				from2 := addr(from1, m.pitch, cgen.Mul{
					Expr1: il(unroll),
					Expr2: i,
				})
				s[unroll*0+j] = cgen.Var{
					Type: avx.M512, What: dat,
					Init: avx.Mm512MaskzLoaduPs{
						m.loopMask, from2,
					},
				}
				s[unroll*1+j] = cgen.Assign{
					Expr1: part,
					Expr2: avx.Mm512MaxPs{
						part, dat,
					},
				}
			}
		}
		if iters > 0 {
			var (
				body = make(cgen.Stmts, unroll*2)
				i    = vb(m.name("i"))
			)
			fill(body, i, unroll)
			stmts[1] = cgen.For{
				Init: cgen.Var{
					Type: cgen.PtrdiffT, What: i,
					Init: cgen.One,
				},
				Cond: cgen.CmpLE{
					Expr1: i, Expr2: il(iters),
				},
				Post: cgen.IncPre{Expr: i},
				Body: body,
			}
		}
		var (
			tail = make(cgen.Stmts, unroll*2)
			i    = il(iters + 1)
		)
		fill(tail, i, after)
		stmts[2] = tail
	}
	if m.edgeMask != nil {
		var (
			dat  = vb(m.name("dat"))
			idx  = il(m.loopCnt)
			part = parts[nparts-1]
		)
		stmts[3] = cgen.Var{
			Type: avx.M512, What: dat,
			Init: avx.Mm512MaskzLoaduPs{
				m.edgeMask,
				addr(m.ptr, m.pitch, idx),
			},
		}
		stmts[4] = cgen.Assign{
			Expr1: part,
			Expr2: avx.Mm512MaskMaxPs{
				part, m.edgeMask,
				part, dat,
			},
		}
	}
	fold := make(cgen.Stmts, nparts-1)
	for i := 0; nparts > 1; {
		cnt := nparts >> 1
		nparts -= cnt
		for j := 0; j < cnt; j++ {
			fold[i] = cgen.Assign{
				Expr1: parts[j],
				Expr2: avx.Mm512MaxPs{
					parts[j],
					parts[nparts+j],
				},
			}
			i++
		}
	}
	stmts[5] = fold
	return stmts.Append(to)
}

type m512Exp struct {
	*Ctx
	ldPtr    cgen.Gen
	ldPitch  cgen.Gen
	stPtr    cgen.Gen
	stPitch  cgen.Gen
	loopCnt  int
	loopMask cgen.Gen
	edgeMask cgen.Gen
	sub      cgen.Gen
	sum      cgen.Gen
}

func (m *m512Exp) Append(to []byte) []byte {
	const unroll = 16
	var (
		iters = m.loopCnt / unroll
		more  = m.loopCnt % unroll
		neg   = vb(m.name("neg"))
	)
	ae := func(ptr, pitch, i, j cgen.Gen) cgen.Gen {
		expr := addr(ptr, pitch, j)
		if iters > 0 {
			expr = addr(expr, pitch, cgen.Mul{
				Expr1: il(unroll),
				Expr2: i,
			})
		}
		return expr
	}
	ld := func(dat, mask, i, j cgen.Gen) cgen.Gen {
		return cgen.Var{
			Type: avx.M512, What: dat,
			Init: avx.Mm512MaskzLoaduPs{
				mask,
				ae(m.ldPtr, m.ldPitch, i, j),
			},
		}
	}
	add := func(dat, mask cgen.Gen) cgen.Gen {
		if mask == m.loopMask {
			return avx.Mm512AddPs{m.sum, dat}
		}
		return avx.Mm512MaskAddPs{m.sum, mask, m.sum, dat}
	}
	op := func(dat, mask cgen.Gen) cgen.Gen {
		return cgen.Stmts{
			cgen.Assign{
				Expr1: dat,
				Expr2: &exp.Call{
					Ctx: m.ec,
					Arg: avx.Mm512AddPs{neg, dat},
				},
			},
			cgen.Assign{
				Expr1: m.sum,
				Expr2: add(dat, mask),
			},
		}
	}
	st := func(dat, mask, i, j cgen.Gen) cgen.Gen {
		return avx.Mm512MaskStoreuPs{
			ae(m.stPtr, m.stPitch, i, j),
			mask, dat,
		}
	}
	fill := func(s cgen.Stmts, mask, i cgen.Gen, j, n int) {
		for stop := j + n; j < stop; j++ {
			var (
				dat = vb(m.name("dat"))
				jj  = il(j)
			)
			s[unroll*1-1-j] = ld(dat, mask, i, jj)
			s[unroll*2-1-j] = op(dat, mask)
			s[unroll*3-1-j] = st(dat, mask, i, jj)
		}
	}
	stmts := make(cgen.Stmts, 4)
	stmts[0] = cgen.Var{
		Type: avx.M512, What: m.sum,
		Init: avx.Mm512SetzeroPs,
	}
	stmts[1] = cgen.Var{
		Type: avx.M512, What: neg,
		Init: avx.Mm512SubPs{m.sum, m.sub},
	}
	if iters > 0 {
		var (
			inner = make(cgen.Stmts, unroll*3)
			i     = vb(m.name("i"))
		)
		fill(inner, m.loopMask, i, 0, unroll)
		stmts[3] = cgen.For{
			Init: cgen.Var{
				Type: cgen.PtrdiffT, What: i,
				Init: il(iters - 1),
			},
			Cond: cgen.CmpGE{
				Expr1: i, Expr2: cgen.Zero,
			},
			Post: cgen.DecPre{Expr: i},
			Body: inner,
		}
	}
	var (
		outer = make(cgen.Stmts, unroll*3)
		i     = il(iters)
	)
	fill(outer, m.loopMask, i, 0, more)
	if m.edgeMask != nil {
		fill(outer, m.edgeMask, i, more, 1)
	}
	stmts[2] = outer
	return stmts.Append(to)
}

type foldOp int

const (
	foldAdd foldOp = iota
	foldMax
)

type m512Fold struct {
	*Ctx
	dat  cgen.Gen
	cnt  int
	each int
	op   foldOp
}

func (m *m512Fold) Append(to []byte) []byte {
	var stmts cgen.Stmts
	assign := func(a cgen.Gen) {
		stmts = append(stmts, cgen.Assign{
			Expr1: m.dat, Expr2: a,
		})
	}
	perm := func(a []cgen.Gen) cgen.Gen {
		p := vb(m.name("p"))
		stmts = append(stmts, cgen.Var{
			Type: avx.M512i, What: p,
			Init: avx.Mm512SetEpi32(a),
		})
		return avx.Mm512PermutexvarPs{p, m.dat}
	}
	call := func(a ...cgen.Gen) cgen.Gen {
		switch m.op {
		case foldAdd:
			return avx.Mm512MaskAddPs(a)
		case foldMax:
			return avx.Mm512MaskMaxPs(a)
		default:
			panic("bug")
		}
	}
	const lanes = 16
	for have := m.cnt; have > 1; {
		stop := have * m.each
		fold := have >> 1
		have -= fold
		elem := have * m.each
		from := make([]cgen.Gen, lanes)
		i := lanes - 1
		for ; elem < stop; elem++ {
			from[i] = il(elem)
			i--
		}
		for ; i >= 0; i-- {
			from[i] = cgen.Zero
		}
		mask := 1<<uint(fold*m.each) - 1
		assign(call(
			m.dat, il(mask),
			m.dat, perm(from),
		))
	}
	elem := 0
	from := make([]cgen.Gen, lanes)
	for i := lanes - 1; i >= 0; i-- {
		from[i] = il(elem)
		if elem++; elem == m.each {
			elem = 0
		}
	}
	assign(perm(from))
	return stmts.Append(to)
}

type m512Denom struct {
	*Ctx
	ptrs     []cgen.Gen
	pitches  []cgen.Gen
	loopCnt  cgen.Gen
	loopMask cgen.Gen
	edgeMask cgen.Gen
	divisor  cgen.Gen
}

func (m *m512Denom) Append(to []byte) []byte {
	var (
		outer = make(cgen.Stmts, 2, 3)
		rcp   = vb(m.name("rcp"))
		i     = vb(m.name("i"))
	)
	outer[0] = cgen.Var{
		Type: avx.M512, What: rcp,
		Init: avx.Mm512DivPs{
			avx.Mm512Set1PsLit(1),
			m.divisor,
		},
	}
	iter := func(mask, idx cgen.Gen) cgen.Gen {
		var (
			inner = make(cgen.Stmts, 2+len(m.ptrs))
			dat   = vb(m.name("dat"))
		)
		inner[0] = cgen.Var{
			Type: avx.M512, What: dat,
			Init: avx.Mm512MaskzLoaduPs{
				mask,
				addr(m.ptrs[0], m.pitches[0], idx),
			},
		}
		inner[1] = cgen.Assign{
			Expr1: dat,
			Expr2: avx.Mm512MulPs{rcp, dat},
		}
		for j, ptr := range m.ptrs {
			inner[2+j] = avx.Mm512MaskStoreuPs{
				addr(ptr, m.pitches[j], idx),
				mask, dat,
			}
		}
		return inner
	}
	outer[1] = cgen.For{
		Init: cgen.Var{
			Type: cgen.PtrdiffT, What: i,
			Init: cgen.Zero,
		},
		Cond: cgen.CmpL{Expr1: i, Expr2: m.loopCnt},
		Post: cgen.IncPre{Expr: i},
		Body: iter(m.loopMask, i),
	}
	if m.edgeMask != nil {
		outer = append(outer, iter(
			m.edgeMask, m.loopCnt,
		))
	}
	return outer.Append(to)
}
