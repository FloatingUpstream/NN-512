package trans

import (
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
)

func il(i int) cgen.Gen {
	return cgen.IntLit(i)
}

type Pose struct {
	Platform raw.Platform
	Nms      nmsrc.Src
	Rows     int
	Cols     int
	Vars     []cgen.Gen
	stmts    cgen.Stmts
}

func (p *Pose) Append(to []byte) []byte {
	if p.Rows < 1 || p.Cols < 1 {
		panic("bug")
	}
	if p.Rows == 1 && p.Cols == 1 {
		return to
	}
	switch p.Platform {
	case raw.AVX512Float32:
		(&m512Pose{Pose: p}).Stmts()
	default:
		panic("bug")
	}
	return p.stmts.Append(to)
}

func (p *Pose) tmp() cgen.Gen {
	return cgen.Vb(p.Nms.Name("tmp"))
}

func (p *Pose) stmt(s cgen.Gen) {
	p.stmts = append(p.stmts, s)
}

type m512Var struct {
	name cgen.Gen
	mask int
}

type m512Pose struct {
	*Pose
	src *[16]m512Var
	dst *[16]m512Var
	in1 m512Var
	in2 m512Var
}

func (m *m512Pose) Stmts() {
	if m.Rows > 16 || m.Cols > 16 {
		panic("bug")
	}
	m.stage1()
	m.stage2()
	m.stage3()
	m.stage4()
}

func (m *m512Pose) stage1() {
	m.dst = new([16]m512Var)
	for i := 0; i < 8; i++ {
		row := i * 2
		m.op1(row+0, row+0, row+1)
		m.op2(row+1, row+0, row+1)
	}
}

func (m *m512Pose) stage2() {
	m.src = m.dst
	m.dst = new([16]m512Var)
	for i := 0; i < 4; i++ {
		row := i * 4
		m.op3(row+0, row+0, row+2)
		m.op4(row+1, row+0, row+2)
		m.op3(row+2, row+1, row+3)
		m.op4(row+3, row+1, row+3)
	}
}

func (m *m512Pose) stage3() {
	m.src = m.dst
	m.dst = new([16]m512Var)
	for i := 0; i < 2; i++ {
		row := i * 8
		m.op5(row+0, row+0, row+4)
		m.op6(row+4, row+0, row+4)
		m.op5(row+1, row+1, row+5)
		m.op6(row+5, row+1, row+5)
		m.op5(row+2, row+2, row+6)
		m.op6(row+6, row+2, row+6)
		m.op5(row+3, row+3, row+7)
		m.op6(row+7, row+3, row+7)
	}
}

func (m *m512Pose) stage4() {
	m.src = m.dst
	m.dst = nil
	for row := 0; row < 8; row++ {
		m.op5(row+0, row+0, row+8)
		m.op6(row+8, row+0, row+8)
	}
}

func (m *m512Pose) attach(row1, row2 int) bool {
	if m.src == nil {
		m.in1 = m512Var{}
		m.in2 = m512Var{}
		mask := 1<<uint(m.Cols) - 1
		if row1 < m.Rows {
			m.in1.name = m.Vars[row1]
			m.in1.mask = mask
		}
		if row2 < m.Rows {
			m.in2.name = m.Vars[row2]
			m.in2.mask = mask
		}
	} else {
		m.in1 = m.src[row1]
		m.in2 = m.src[row2]
	}
	var (
		name1 = m.in1.name
		name2 = m.in2.name
	)
	if name1 == nil {
		if name2 == nil {
			return false
		}
		m.in1.name = name2
	} else if name2 == nil {
		m.in2.name = name1
	}
	return true
}

func (m *m512Pose) do(row, mask int, expr cgen.Gen) {
	var name cgen.Gen
	switch {
	case m.dst == nil:
		name = m.Vars[row]
		if row < m.Rows {
			m.stmt(cgen.Assign{
				Expr1: name,
				Expr2: expr,
			})
			return
		}
	default:
		name = m.tmp()
		m.dst[row].name = name
		m.dst[row].mask = mask
	}
	m.stmt(cgen.Var{
		Type: avx.M512, What: name,
		Init: expr,
	})
}

func (m *m512Pose) eq(row, mask int, name cgen.Gen) {
	switch {
	case m.src == nil || m.dst == nil:
		m.do(row, mask, name)
	default:
		m.dst[row].name = name
		m.dst[row].mask = mask
	}
}

func (m *m512Pose) op1(row1, row2, row3 int) {
	if !m.attach(row2, row3) {
		return
	}
	var (
		mask = 0
		k1   = m.in1.mask
		k2   = m.in2.mask
	)
	for i := 0; i < 4; i++ {
		var (
			bit1 = 1 << uint(i*4)
			bit2 = bit1 << 1
		)
		mask |= k1 & bit1 << 0
		mask |= k2 & bit1 << 1
		mask |= k1 & bit2 << 1
		mask |= k2 & bit2 << 2
	}
	if mask == 0 {
		return
	}
	if mask&(1<<12|1<<8|1<<4|1<<0) == mask {
		m.eq(row1, mask, m.in1.name)
		return
	}
	m.do(row1, mask, avx.Mm512UnpackloPs{
		m.in1.name, m.in2.name,
	})
}

func (m *m512Pose) op2(row1, row2, row3 int) {
	if !m.attach(row2, row3) {
		return
	}
	var (
		mask = 0
		k1   = m.in1.mask
		k2   = m.in2.mask
	)
	for i := 0; i < 4; i++ {
		var (
			bit1 = 4 << uint(i*4)
			bit2 = bit1 << 1
		)
		mask |= k1 & bit1 >> 2
		mask |= k2 & bit1 >> 1
		mask |= k1 & bit2 >> 1
		mask |= k2 & bit2 >> 0
	}
	if mask == 0 {
		return
	}
	if mask&(8<<12|8<<8|8<<4|8<<0) == mask {
		m.eq(row1, mask, m.in2.name)
		return
	}
	m.do(row1, mask, avx.Mm512UnpackhiPs{
		m.in1.name, m.in2.name,
	})
}

func (m *m512Pose) op3(row1, row2, row3 int) {
	if !m.attach(row2, row3) {
		return
	}
	mask := 0
	for i := 0; i < 4; i++ {
		bits := 3 << uint(i*4)
		mask |= m.in1.mask & bits << 0
		mask |= m.in2.mask & bits << 2
	}
	if mask == 0 {
		return
	}
	if mask&(3<<12|3<<8|3<<4|3<<0) == mask {
		m.eq(row1, mask, m.in1.name)
		return
	}
	m.do(row1, mask, avx.Mm512ShufflePs{
		m.in1.name, m.in2.name,
		il(1<<6 | 0<<4 | 1<<2 | 0<<0),
	})
}

func (m *m512Pose) op4(row1, row2, row3 int) {
	if !m.attach(row2, row3) {
		return
	}
	mask := 0
	for i := 0; i < 4; i++ {
		bits := 12 << uint(i*4)
		mask |= m.in1.mask & bits >> 2
		mask |= m.in2.mask & bits >> 0
	}
	if mask == 0 {
		return
	}
	if mask&(12<<12|12<<8|12<<4|12<<0) == mask {
		m.eq(row1, mask, m.in2.name)
		return
	}
	m.do(row1, mask, avx.Mm512ShufflePs{
		m.in1.name, m.in2.name,
		il(3<<6 | 2<<4 | 3<<2 | 2<<0),
	})
}

func (m *m512Pose) op5(row1, row2, row3 int) {
	if !m.attach(row2, row3) {
		return
	}
	var (
		k1 = m.in1.mask
		k2 = m.in2.mask
	)
	const (
		bits1 = 15 << 0
		bits2 = 15 << 8
	)
	mask := k1 & bits1 >> 0
	mask |= k1 & bits2 >> 4
	mask |= k2 & bits1 << 8
	mask |= k2 & bits2 << 4
	if mask == 0 {
		return
	}
	if mask&bits1 == mask {
		m.eq(row1, mask, m.in1.name)
		return
	}
	m.do(row1, mask, avx.Mm512ShuffleF32x4{
		m.in1.name, m.in2.name,
		il(2<<6 | 0<<4 | 2<<2 | 0<<0),
	})
}

func (m *m512Pose) op6(row1, row2, row3 int) {
	if !m.attach(row2, row3) {
		return
	}
	var (
		k1 = m.in1.mask
		k2 = m.in2.mask
	)
	const (
		bits1 = 15 << 4
		bits2 = 15 << 12
	)
	mask := k1 & bits1 >> 4
	mask |= k1 & bits2 >> 8
	mask |= k2 & bits1 << 4
	mask |= k2 & bits2 << 0
	if mask == 0 {
		return
	}
	if mask&bits2 == mask {
		m.eq(row1, mask, m.in2.name)
		return
	}
	m.do(row1, mask, avx.Mm512ShuffleF32x4{
		m.in1.name, m.in2.name,
		il(3<<6 | 1<<4 | 3<<2 | 1<<0),
	})
}
