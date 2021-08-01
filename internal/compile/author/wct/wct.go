package wct

import (
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/trans"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
)

func min(x, y int) int {
	if x <= y {
		return x
	}
	return y
}

func vb(s string) cgen.Gen {
	return cgen.Vb(s)
}

func fl(f float64) cgen.Gen {
	return cgen.FloatLit(f)
}

func il(i int) cgen.Gen {
	return cgen.IntLit(i)
}

func void(a cgen.Gen) cgen.Gen {
	return cgen.Cast{
		Type: cgen.Void,
		Expr: a,
	}
}

func mix(a, b cgen.Stmts) cgen.Stmts {
	var (
		n   = len(a) + len(b)
		ret = make(cgen.Stmts, n)
	)
	for i, aa := range a {
		ret[i*2] = aa
		ret[i*2+1] = b[i]
	}
	return ret
}

type Wts struct {
	Platform raw.Platform
	Nms      nmsrc.Src
	Blks     int
	In       [3]cgen.Gen
	Out      [16]cgen.Gen
}

func (w *Wts) Append(to []byte) []byte {
	switch w.Platform {
	case raw.AVX512Float32:
		return w.m512().Append(to)
	default:
		panic("bug")
	}
}

func (w *Wts) tmp() cgen.Gen {
	s := w.Nms.Name("tmp")
	return vb(s)
}

func (w *Wts) m512() cgen.Gen {
	var (
		vecs1 [12]cgen.Gen
		vecs2 [16]cgen.Gen
	)
	matmul := func(vecs []cgen.Gen) cgen.Stmts {
		var (
			stmts = make(cgen.Stmts, 0, 9)
			more  [6]cgen.Gen
		)
		assn := func(i int, expr cgen.Gen) {
			var (
				stmt cgen.Gen
				vec  = more[i]
			)
			if vec == nil {
				vec = w.tmp()
				more[i] = vec
				stmt = cgen.Var{
					Type: avx.M512, What: vec,
					Init: expr,
				}
			} else {
				stmt = cgen.Assign{
					Expr1: vec,
					Expr2: expr,
				}
			}
			stmts = append(stmts, stmt)
		}
		coeff := func(a float64) cgen.Gen {
			return avx.Mm512Set1PsLit(a)
		}
		fmadd := func(a, b cgen.Gen, c float64) cgen.Gen {
			return avx.Mm512FmaddPs{b, coeff(c), a}
		}
		fnmadd := func(a, b cgen.Gen, c float64) cgen.Gen {
			return avx.Mm512FnmaddPs{b, coeff(c), a}
		}
		add := func(a, b cgen.Gen) cgen.Gen {
			return avx.Mm512AddPs{a, b}
		}
		sub := func(a, b cgen.Gen) cgen.Gen {
			return avx.Mm512SubPs{a, b}
		}
		assn(0, fmadd(vecs[2], vecs[0], 4))
		assn(1, add(vecs[0], vecs[2]))
		assn(2, fmadd(vecs[0], vecs[2], 4))
		assn(3, add(vecs[1], more[1]))
		assn(4, fmadd(more[2], vecs[1], 2))
		assn(2, fnmadd(more[2], vecs[1], 2))
		assn(5, fnmadd(more[0], vecs[1], 2))
		assn(0, fmadd(more[0], vecs[1], 2))
		assn(1, sub(more[1], vecs[1]))
		vecs[1] = more[3]
		vecs[7] = vecs[2]
		vecs[2] = more[1]
		vecs[3] = more[4]
		vecs[4] = more[2]
		vecs[5] = more[0]
		vecs[6] = more[5]
		return stmts
	}
	shuf := func(a, b cgen.Gen, c int) cgen.Gen {
		c |= (c + 1) << 2
		c |= c << 4
		return avx.Mm512ShuffleF32x4{
			a, b, il(c),
		}
	}
	layer6 := func(rows int) cgen.Gen {
		var (
			n     = w.Blks
			stmts = make(cgen.Stmts, 0, n*4)
		)
		for i := 0; i < rows; i += 2 {
			var (
				vec1  = vecs2[i]
				vec2  = vecs2[i+1]
				loBlk = i / 8 * 2
				hiBlk = loBlk + 1
				frag  = i / 2 % 4
			)
			stmts = append(
				stmts, cgen.Var{
					Type: avx.M512,
					What: w.Out[loBlk*4+frag],
					Init: shuf(vec1, vec2, 0),
				},
			)
			if hiBlk == n {
				continue
			}
			stmts = append(
				stmts, cgen.Var{
					Type: avx.M512,
					What: w.Out[hiBlk*4+frag],
					Init: shuf(vec1, vec2, 2),
				},
			)
		}
		return stmts
	}
	layer5 := func() cgen.Gen {
		const y = 1.0 / 9
		ms := [8]float64{
			1,
			y * -2, y * -2,
			y / 10, y / 10,
			y / 20, y / 20,
			1,
		}
		rows := 8
		if w.Blks > 2 {
			rows = 16
		}
		stmts := make(cgen.Stmts, rows)
		mul := func(row int, expr cgen.Gen) {
			if row >= rows {
				return
			}
			vec := vecs2[row]
			stmts[row] = cgen.Assign{
				Expr1: vec,
				Expr2: avx.Mm512MulPs{
					vec, expr,
				},
			}
		}
		for i := 0; i < 8; i++ {
			set := make(avx.Mm512SetPs, 16)
			for j := 0; j < 8; j++ {
				m := fl(ms[i] * ms[j])
				set[15-j] = m
				set[7-j] = m
			}
			mul(i, set)
			mul(i+8, set)
		}
		return cgen.Gens{
			stmts,
			layer6(rows),
		}
	}
	layer4 := func() cgen.Gen {
		stmts := matmul(vecs2[:8])
		if w.Blks > 2 {
			stmts = mix(
				stmts,
				matmul(vecs2[8:]),
			)
		}
		return cgen.Gens{
			stmts,
			layer5(),
		}
	}
	layer3 := func() cgen.Gen {
		var (
			n     = w.Blks
			stmts = make(cgen.Stmts, 0, n/2*3)
		)
		for i := 0; i < n; i += 2 {
			for j := 0; j < 3; j++ {
				vec1 := vecs1[i*3+j]
				vecs2[i/2*8+j] = vec1
				if i+1 == n {
					continue
				}
				var (
					vec2 = vecs1[(i+1)*3+j]
					both = shuf(vec1, vec2, 0)
				)
				stmts = append(
					stmts, cgen.Assign{
						Expr1: vec1,
						Expr2: both,
					},
				)
			}
		}
		return cgen.Gens{
			stmts,
			layer4(),
		}
	}
	layer2 := func() cgen.Gen {
		cols := w.Blks * 3
		for i := 8; i < cols; i++ {
			vecs1[i] = w.tmp()
		}
		return cgen.Gens{
			&trans.Pose{
				Platform: w.Platform,
				Nms:      w.Nms,
				Rows:     8,
				Cols:     cols,
				Vars:     vecs1[:],
			},
			layer3(),
		}
	}
	layer1 := func() cgen.Gen {
		copy(vecs1[:3], w.In[:])
		return cgen.Gens{
			matmul(vecs1[:8]),
			layer2(),
		}
	}
	return layer1()
}

type Dats struct {
	Platform raw.Platform
	Nms      nmsrc.Src
	Blks     int
	LZCols   [4]int
	TZCols   [4]int
	In       [16]cgen.Gen
	Out      [16]cgen.Gen
}

func (d *Dats) Append(to []byte) []byte {
	switch d.Platform {
	case raw.AVX512Float32:
		return d.m512().Append(to)
	default:
		panic("bug")
	}
}

func (d *Dats) tmp() cgen.Gen {
	s := d.Nms.Name("tmp")
	return vb(s)
}

func (d *Dats) m512() cgen.Gen {
	var (
		tpd [16]cgen.Gen
	)
	matmul := func(in []cgen.Gen) cgen.Stmts {
		var (
			stmts = make(cgen.Stmts, 0, 32)
			more  [4]cgen.Gen
		)
		add := func(a, b cgen.Gen) cgen.Gen {
			switch {
			case a == nil:
				return b
			case b == nil:
				return a
			}
			return avx.Mm512AddPs{a, b}
		}
		sub := func(a, b cgen.Gen) cgen.Gen {
			switch {
			case b == nil:
				return a
			case a == nil:
				a = avx.Mm512SetzeroPs
			}
			return avx.Mm512SubPs{a, b}
		}
		coeff := func(a float64) cgen.Gen {
			return avx.Mm512Set1PsLit(a)
		}
		fmadd := func(a, b cgen.Gen, c float64) cgen.Gen {
			switch {
			case b == nil:
				return a
			case a == nil:
				return avx.Mm512MulPs{
					b, coeff(c),
				}
			}
			return avx.Mm512FmaddPs{
				b, coeff(c), a,
			}
		}
		fnmadd := func(a, b cgen.Gen, c float64) cgen.Gen {
			switch {
			case b == nil:
				return a
			case a == nil:
				a = avx.Mm512SetzeroPs
			}
			return avx.Mm512FnmaddPs{
				b, coeff(c), a,
			}
		}
		assn := func(vec *cgen.Gen, expr cgen.Gen) {
			var stmt cgen.Gen
			switch {
			case expr == nil:
				*vec = nil
			case *vec == nil:
				*vec = d.tmp()
				stmt = cgen.Var{
					Type: avx.M512, What: *vec,
					Init: expr,
				}
			default:
				stmt = cgen.Assign{
					Expr1: *vec,
					Expr2: expr,
				}
			}
			stmts = append(stmts, stmt)
		}
		emit := func(row int, vec cgen.Gen) {
			var stmt cgen.Gen
			if vec == nil {
				vec = d.tmp()
				stmt = cgen.Var{
					Type: avx.M512, What: vec,
					Init: avx.Mm512SetzeroPs,
				}
			}
			stmts = append(stmts, stmt)
			in[row] = vec
		}
		assn(&more[0], add(in[1], in[5]))
		assn(&more[1], sub(in[4], in[2]))
		assn(&more[2], add(in[2], in[6]))
		assn(&in[0], sub(in[0], in[6]))
		assn(&more[0], fmadd(more[0], in[3], -4.25))
		assn(&more[2], fmadd(more[2], in[4], -4.25))
		assn(&in[0], fmadd(in[0], more[1], 5.25))
		assn(&more[1], fmadd(in[6], in[2], 0.25))
		assn(&in[2], fmadd(in[6], in[2], 4))
		assn(&more[3], sub(more[2], more[0]))
		assn(&more[2], add(more[0], more[2]))
		assn(&more[0], fmadd(in[5], in[1], 0.25))
		assn(&more[1], fmadd(more[1], in[4], -1.25))
		assn(&in[4], fmadd(in[2], in[4], -5))
		assn(&more[0], fmadd(more[0], in[3], -1.25))
		assn(&in[6], fmadd(more[1], more[0], 2))
		assn(&more[1], fnmadd(more[1], more[0], 2))
		assn(&more[0], fmadd(in[1], in[5], 0.25))
		assn(&in[1], sub(in[7], in[1]))
		assn(&more[0], fmadd(more[0], in[3], -1.25))
		assn(&in[3], sub(in[3], in[5]))
		assn(&in[3], fmadd(in[1], in[3], 5.25))
		assn(&in[2], fmadd(in[4], more[0], 2))
		assn(&in[4], fnmadd(in[4], more[0], 2))
		emit(0, in[0])
		emit(1, more[2])
		emit(5, in[2])
		emit(2, more[3])
		emit(7, in[3])
		emit(3, in[6])
		emit(6, in[4])
		emit(4, more[1])
		return stmts
	}
	layer6 := func() cgen.Gen {
		var (
			stmts cgen.Stmts
			rows  = 8
		)
		if d.Blks > 1 {
			rows = 16
		}
		for i := 0; i < rows; i += 2 {
			var (
				row1   = tpd[i]
				row2   = tpd[i+1]
				used   = false
				loBlk  = i / 8
				hiBlk  = loBlk + 2
				frag   = i / 2 % 4
				loExpr cgen.Gen
				hiExpr cgen.Gen
			)
			if d.TZCols[loBlk] == 8 {
				loExpr = avx.Mm512SetzeroPs
			} else {
				loExpr = avx.Mm512ShuffleF32x4{
					row1, row2,
					il(1<<6 | 0<<4 | 1<<2 | 0<<0),
				}
				used = true
			}
			stmts = append(
				stmts, cgen.Var{
					Type: avx.M512,
					What: d.Out[loBlk*4+frag],
					Init: loExpr,
				},
			)
			if hiBlk < d.Blks {
				if d.TZCols[hiBlk] == 8 {
					hiExpr = avx.Mm512SetzeroPs
				} else {
					hiExpr = avx.Mm512ShuffleF32x4{
						row1, row2,
						il(3<<6 | 2<<4 | 3<<2 | 2<<0),
					}
					used = true
				}
				stmts = append(
					stmts, cgen.Var{
						Type: avx.M512,
						What: d.Out[hiBlk*4+frag],
						Init: hiExpr,
					},
				)
			}
			if !used {
				stmts = append(
					stmts,
					void(row1),
					void(row2),
				)
			}
		}
		return stmts
	}
	layer5 := func() cgen.Gen {
		stmts := matmul(tpd[:8])
		if d.Blks > 1 {
			stmts = mix(
				stmts,
				matmul(tpd[8:]),
			)
		}
		return cgen.Gens{
			stmts,
			layer6(),
		}
	}
	layer4 := func(rows int) cgen.Gen {
		mask := func(i int) int {
			if i >= d.Blks {
				return 0
			}
			var (
				first = d.LZCols[i]
				past  = 8 - d.TZCols[i]
				n     = past - first
				ones  = 1<<uint(n) - 1
			)
			return ones << uint(first)
		}
		var (
			lo    = mask(0) | mask(2)
			hi    = mask(1) | mask(3)
			keep  = hi<<8 | lo
			stmts cgen.Stmts
		)
		for i, row := range d.In[:rows] {
			if keep&1 == 0 {
				stmts = append(
					stmts, void(row),
				)
			} else {
				tpd[i] = row
			}
			keep >>= 1
		}
		return cgen.Gens{
			stmts,
			layer5(),
		}
	}
	layer3 := func() cgen.Gen {
		var (
			rows int
			cols int
			tp   cgen.Gen
		)
		switch d.Blks {
		case 1:
			rows = 8
			cols = 8 - d.TZCols[0]
		case 2:
			rows = 8
			cols = 16 - d.TZCols[1]
			if cols == 8 {
				cols -= d.TZCols[0]
			}
		case 3:
			rows = 16
			cols = 16 - d.TZCols[1]
			if cols == 8 {
				cols -= min(d.TZCols[0], d.TZCols[2])
			}
		case 4:
			rows = 16
			cols = 16 - min(d.TZCols[1], d.TZCols[3])
			if cols == 8 {
				cols -= min(d.TZCols[0], d.TZCols[2])
			}
		default:
			panic("bug")
		}
		if cols == 0 {
			stmts := make(cgen.Stmts, rows)
			for i := range stmts {
				stmts[i] = void(d.In[i])
			}
			tp = stmts
		} else {
			for i := rows; i < cols; i++ {
				d.In[i] = d.tmp()
			}
			tp = &trans.Pose{
				Platform: d.Platform,
				Nms:      d.Nms,
				Rows:     rows,
				Cols:     cols,
				Vars:     d.In[:],
			}
		}
		return cgen.Gens{
			tp,
			layer4(cols),
		}
	}
	layer2 := func() cgen.Gen {
		stmts := matmul(d.In[:8])
		if d.Blks > 2 {
			stmts = mix(
				stmts,
				matmul(d.In[8:]),
			)
		}
		return cgen.Gens{
			stmts,
			layer3(),
		}
	}
	layer1 := func() cgen.Gen {
		for i, lz := range d.LZCols[:d.Blks] {
			if lz+d.TZCols[i] == 8 {
				d.LZCols[i] = 0
				d.TZCols[i] = 8
			}
		}
		return layer2()
	}
	return layer1()
}

type Sums struct {
	Platform raw.Platform
	Nms      nmsrc.Src
	Blks     int
	Cols     [4]int
	In       [16]cgen.Gen
	Out      [12]cgen.Gen
}

func (s *Sums) Append(to []byte) []byte {
	switch s.Platform {
	case raw.AVX512Float32:
		return s.m512().Append(to)
	default:
		panic("bug")
	}
}

func (s *Sums) tmp() cgen.Gen {
	t := s.Nms.Name("tmp")
	return vb(t)
}

func (s *Sums) m512() cgen.Gen {
	var (
		vecs [16]cgen.Gen
	)
	decl := func(vec, expr cgen.Gen) cgen.Gen {
		return cgen.Var{
			Type: avx.M512, What: vec,
			Init: expr,
		}
	}
	matmul := func(in, out []cgen.Gen) cgen.Stmts {
		const (
			n1 = 8
			n2 = 20
			n3 = 6
		)
		add := func(a, b cgen.Gen, c float64) cgen.Gen {
			return avx.Mm512AddPs{a, b}
		}
		sub := func(a, b cgen.Gen, c float64) cgen.Gen {
			return avx.Mm512SubPs{a, b}
		}
		fmadd := func(a, b cgen.Gen, c float64) cgen.Gen {
			coeff := avx.Mm512Set1PsLit(c)
			return avx.Mm512FmaddPs{
				b, coeff, a,
			}
		}
		prog := [n2]struct {
			f func(a, b cgen.Gen, c float64) cgen.Gen
			a int
			b int
			c float64
		}{
			{add, ^1, ^2, 0},
			{add, ^3, ^4, 0},
			{sub, ^3, ^4, 0},
			{sub, ^1, ^2, 0},
			{add, ^5, ^6, 0},
			{sub, ^5, ^6, 0},
			{fmadd, 3, 2, 2},
			{fmadd, 3, 2, 8},
			{add, 1, 0, 0},
			{fmadd, 6, 5, 16},
			{fmadd, 7, 5, 4},
			{add, 5, 3, 0},
			{fmadd, 0, 1, 4},
			{fmadd, 0, 1, 16},
			{add, 8, ^0, 0},
			{add, 11, ^7, 0},
			{fmadd, 14, 4, 32},
			{fmadd, 12, 4, 8},
			{fmadd, 15, 2, 32},
			{fmadd, 13, 4, 2},
		}
		taps := [n3]int{
			16, 9, 17, 10, 19, 18,
		}
		var (
			used  [n1]bool
			nodes [n2]cgen.Gen
			trace func(int)
			stmts = make(cgen.Stmts, n1+n2+n3)
		)
		trace = func(i int) {
			switch {
			case i < 0:
				used[^i] = true
				return
			case nodes[i] != nil:
				return
			}
			nodes[i] = s.tmp()
			trace(prog[i].a)
			trace(prog[i].b)
		}
		for i, vec := range out[:n3] {
			if vec == nil {
				continue
			}
			tap := taps[i]
			trace(tap)
			stmts[n1+n2+i] = decl(
				vec, nodes[tap],
			)
		}
		for i, vec := range in[:n1] {
			if !used[i] {
				stmts[i] = void(vec)
			}
		}
		sym := func(i int) cgen.Gen {
			if i < 0 {
				return in[^i]
			}
			return nodes[i]
		}
		for i, vec := range &nodes {
			if vec == nil {
				continue
			}
			do := &prog[i]
			stmts[n1+i] = decl(
				vec, do.f(
					sym(do.a),
					sym(do.b),
					do.c,
				),
			)
		}
		return stmts
	}
	layer4 := func() cgen.Gen {
		stmts := matmul(
			vecs[:], s.Out[:],
		)
		if s.Blks > 1 {
			stmts = mix(
				stmts, matmul(
					vecs[8:], s.Out[6:],
				),
			)
		}
		return stmts
	}
	layer3 := func() cgen.Gen {
		var (
			decls cgen.Stmts
			zero  = avx.Mm512SetzeroPs
		)
		fill := func(n int) {
			for i, vec := range vecs[:n] {
				if vec != nil {
					continue
				}
				vec = s.tmp()
				vecs[i] = vec
				decls = append(
					decls,
					decl(vec, zero),
				)
			}
		}
		var (
			rows = 0
			cols = 8
			tp   cgen.Gen
		)
		for i, vec := range vecs[:12] {
			if vec != nil {
				rows = i + 1
			}
		}
		if s.Blks > 1 {
			cols = 16
		}
		if rows == 0 {
			fill(cols)
		} else {
			fill(rows)
			for i := rows; i < cols; i++ {
				vecs[i] = s.tmp()
			}
			tp = &trans.Pose{
				Platform: s.Platform,
				Nms:      s.Nms,
				Rows:     rows,
				Cols:     cols,
				Vars:     vecs[:],
			}
		}
		return cgen.Gens{
			decls,
			tp,
			layer4(),
		}
	}
	layer2 := func() cgen.Gen {
		stmts := matmul(
			s.In[:], vecs[:],
		)
		if s.Blks > 2 {
			stmts = mix(
				stmts, matmul(
					s.In[8:], vecs[6:],
				),
			)
		}
		return cgen.Gens{
			stmts,
			layer3(),
		}
	}
	layer1 := func() cgen.Gen {
		n := s.Blks
		if n < 1 || n > 4 {
			panic("bug")
		}
		for i := 0; i < n; i += 2 {
			rows := s.Cols[i]
			if i+1 < n {
				rows |= s.Cols[i+1]
			}
			for j := 0; j < 6; j++ {
				if rows&1 != 0 {
					k := i/2*6 + j
					vecs[k] = s.tmp()
				}
				rows >>= 1
			}
		}
		return layer2()
	}
	return layer1()
}
