package quadfft

import (
	"NN-512/internal/compile/author/avx"
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
	"math"
)

func fl(f float64) cgen.Gen {
	return cgen.FloatLit(f)
}

func il(i int) cgen.Gen {
	return cgen.IntLit(i)
}

type Fwd struct {
	Platform raw.Platform
	Nms      nmsrc.Src
	In       [16]cgen.Gen
	Out      [16]cgen.Gen
}

func (F *Fwd) Append(to []byte) []byte {
	switch F.Platform {
	case raw.AVX512Float32:
		return F.m512().Append(to)
	default:
		panic("bug")
	}
}

func (F *Fwd) m512() cgen.Gen {
	var (
		stmts  cgen.Stmts
		in     []cgen.Gen
		out    []cgen.Gen
		coeffs [6]cgen.Gen
		perms  [2]cgen.Gen
		nodes  [80]cgen.Gen
	)
	stmt := func(st cgen.Gen) {
		stmts = append(stmts, st)
	}
	decl := func(t, id, expr cgen.Gen) cgen.Gen {
		if id == nil {
			fft := F.Nms.Name("fft")
			id = cgen.Vb(fft)
		}
		stmt(cgen.Var{
			Type: t, What: id,
			Init: expr,
		})
		return id
	}
	add := func(a, b cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512AddPs{a, b},
		)
	}
	sub := func(a, b cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512SubPs{a, b},
		)
	}
	bcast := func(i int) cgen.Gen {
		return avx.Mm512Set1PsLit(
			[2]float64{
				math.Sqrt2 * 0.5,
				0.5,
			}[i],
		)
	}
	fmadd := func(a, b, c cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512FmaddPs{a, b, c},
		)
	}
	fnmsub := func(a, b, c cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512FnmsubPs{a, b, c},
		)
	}
	fnmadd := func(a, b, c cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512FnmaddPs{a, b, c},
		)
	}
	coeff := func(i int) cgen.Gen {
		cf := coeffs[i]
		switch cf {
		case nil:
			var (
				neg1 = il(-1)
				neg2 = fl(-math.Sqrt2 * 0.5)
				pos1 = il(1)
				pos2 = fl(math.Sqrt2 * 0.5)
				zero = il(0)
				expr cgen.Gen
			)
			switch i {
			case 0:
				expr = avx.Mm512SetPs{
					neg1, neg1, neg1, neg1,
					neg1, neg1, neg1, neg1,
					pos1, pos1, pos1, pos1,
					pos1, pos1, pos1, pos1,
				}
			case 1:
				expr = avx.Mm512SetPs{
					neg2, neg2, zero, zero,
					pos2, pos2, pos1, pos1,
					pos1, pos1, pos1, pos1,
					pos1, pos1, pos1, pos1,
				}
			case 2:
				expr = avx.Mm512SetPs{
					pos2, pos2, pos1, pos1,
					pos2, pos2, zero, zero,
					zero, zero, zero, zero,
					zero, zero, zero, zero,
				}
			case 3:
				expr = avx.Mm512SetPs{
					neg1, neg1, neg1, neg1,
					pos1, pos1, pos1, pos1,
					neg1, neg1, neg1, neg1,
					pos1, pos1, pos1, pos1,
				}
			case 4:
				expr = avx.Mm512SetPs{
					neg1, neg1, pos1, pos1,
					neg1, neg1, pos1, pos1,
					neg1, neg1, pos1, pos1,
					neg1, neg1, pos1, pos1,
				}
			case 5:
				expr = avx.Mm512SetPs{
					neg1, pos1, neg1, pos1,
					neg1, pos1, zero, zero,
					neg1, pos1, neg1, pos1,
					neg1, pos1, zero, zero,
				}
			}
			cf = decl(avx.M512, nil, expr)
			coeffs[i] = cf
		default:
			stmt(nil)
		}
		return cf
	}
	shuf := func(i int, node cgen.Gen) cgen.Gen {
		var (
			ctrl int
			expr cgen.Gen
		)
		switch i {
		case 0, 2:
			ctrl = 1<<6 | 0<<4 | 3<<2 | 2<<0
		case 1:
			ctrl = 2<<6 | 3<<4 | 0<<2 | 1<<0
		}
		switch i {
		case 0, 1:
			expr = avx.Mm512ShuffleF32x4{
				node, node, il(ctrl),
			}
		case 2:
			expr = avx.Mm512ShufflePs{
				node, node, il(ctrl),
			}
		}
		return expr
	}
	mul := func(a, b cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512MulPs{a, b},
		)
	}
	blend := func(i int, node0, node1 cgen.Gen) cgen.Gen {
		var (
			mask cgen.Gen
			expr cgen.Gen
		)
		switch i {
		case 0, 1:
			mask = il(0xc0c0)
		case 2:
			mask = il(0x5555)
		case 3:
			mask = il(0xa8a8)
		case 4:
			mask = il(0x5656)
		case 5:
			mask = il(0xfcfc)
		}
		switch i {
		case 0, 2, 3, 4:
			expr = avx.Mm512MaskMovPs{
				node0, mask, node1,
			}
		case 1:
			expr = avx.Mm512MaskSubPs{
				node0, mask,
				avx.Mm512SetzeroPs, node1,
			}
		case 5:
			expr = avx.Mm512MaskMulPs{
				node0, mask,
				node1, bcast(1),
			}
		}
		return decl(avx.M512, nil, expr)
	}
	perm := func(i int, node cgen.Gen) cgen.Gen {
		pm := perms[i]
		switch pm {
		case nil:
			var (
				set = make(avx.Mm512SetEpi32, 16)
				tbl []int
			)
			switch i {
			case 0:
				tbl = []int{13, 9, 5, 3}
			case 1:
				tbl = []int{11, 15, 7, 1}
			}
			for j := range set {
				set[j] = il(tbl[j%8/2] - j/8)
			}
			pm = decl(avx.M512i, nil, set)
			perms[i] = pm
		default:
			stmt(nil)
		}
		return decl(
			avx.M512, nil,
			avx.Mm512PermutexvarPs{
				pm, node,
			},
		)
	}
	layer18 := func() {
		decl(avx.M512, out[0], nodes[78])
		decl(avx.M512, out[1], nodes[79])
		decl(avx.M512, out[2], nodes[62])
		decl(avx.M512, out[3], nodes[63])
		decl(avx.M512, out[4], nodes[64])
		decl(avx.M512, out[5], nodes[65])
		decl(avx.M512, out[6], nodes[66])
		decl(avx.M512, out[7], nodes[67])
	}
	layer17 := func() {
		nodes[78] = blend(5, nodes[76], nodes[76])
		nodes[79] = blend(5, nodes[77], nodes[77])
		layer18()
	}
	layer16 := func() {
		nodes[76] = blend(3, nodes[74], nodes[73])
		nodes[77] = blend(4, nodes[75], nodes[73])
		layer17()
	}
	layer15 := func() {
		nodes[74] = blend(2, nodes[71], nodes[72])
		nodes[75] = blend(3, nodes[68], nodes[72])
		layer16()
	}
	layer14 := func() {
		cf := coeff(5)
		nodes[72] = fmadd(nodes[68], cf, nodes[69])
		nodes[73] = fnmadd(nodes[71], cf, nodes[70])
		layer15()
	}
	layer13 := func() {
		nodes[68] = perm(0, nodes[60])
		nodes[69] = perm(1, nodes[60])
		nodes[70] = perm(0, nodes[61])
		nodes[71] = perm(1, nodes[61])
		layer14()
	}
	layer12 := func() {
		cf := coeff(4)
		nodes[60] = fmadd(nodes[52], cf, shuf(2, nodes[52]))
		nodes[61] = fmadd(nodes[53], cf, shuf(2, nodes[53]))
		nodes[62] = fmadd(nodes[54], cf, shuf(2, nodes[54]))
		nodes[63] = fmadd(nodes[55], cf, shuf(2, nodes[55]))
		nodes[64] = fmadd(nodes[56], cf, shuf(2, nodes[56]))
		nodes[65] = fmadd(nodes[57], cf, shuf(2, nodes[57]))
		nodes[66] = fmadd(nodes[58], cf, shuf(2, nodes[58]))
		nodes[67] = fmadd(nodes[59], cf, shuf(2, nodes[59]))
		layer13()
	}
	layer11 := func() {
		nodes[52] = blend(0, nodes[44], nodes[45])
		nodes[53] = blend(1, nodes[45], nodes[44])
		nodes[54] = blend(0, nodes[46], nodes[47])
		nodes[55] = blend(1, nodes[47], nodes[46])
		nodes[56] = blend(0, nodes[48], nodes[49])
		nodes[57] = blend(1, nodes[49], nodes[48])
		nodes[58] = blend(0, nodes[50], nodes[51])
		nodes[59] = blend(1, nodes[51], nodes[50])
		layer12()
	}
	layer10 := func() {
		cf := coeff(3)
		nodes[44] = fmadd(nodes[36], cf, shuf(1, nodes[36]))
		nodes[45] = fmadd(nodes[37], cf, shuf(1, nodes[37]))
		nodes[46] = fmadd(nodes[38], cf, shuf(1, nodes[38]))
		nodes[47] = fmadd(nodes[39], cf, shuf(1, nodes[39]))
		nodes[48] = fmadd(nodes[40], cf, shuf(1, nodes[40]))
		nodes[49] = fmadd(nodes[41], cf, shuf(1, nodes[41]))
		nodes[50] = fmadd(nodes[42], cf, shuf(1, nodes[42]))
		nodes[51] = fmadd(nodes[43], cf, shuf(1, nodes[43]))
		layer11()
	}
	layer9 := func() {
		cf := coeff(2)
		nodes[36] = fmadd(nodes[21], cf, nodes[28])
		nodes[37] = fnmadd(nodes[20], cf, nodes[29])
		nodes[38] = fmadd(nodes[23], cf, nodes[30])
		nodes[39] = fnmadd(nodes[22], cf, nodes[31])
		nodes[40] = fmadd(nodes[25], cf, nodes[32])
		nodes[41] = fnmadd(nodes[24], cf, nodes[33])
		nodes[42] = fmadd(nodes[27], cf, nodes[34])
		nodes[43] = fnmadd(nodes[26], cf, nodes[35])
		layer10()
	}
	layer8 := func() {
		cf := coeff(1)
		nodes[28] = mul(nodes[20], cf)
		nodes[29] = mul(nodes[21], cf)
		nodes[30] = mul(nodes[22], cf)
		nodes[31] = mul(nodes[23], cf)
		nodes[32] = mul(nodes[24], cf)
		nodes[33] = mul(nodes[25], cf)
		nodes[34] = mul(nodes[26], cf)
		nodes[35] = mul(nodes[27], cf)
		layer9()
	}
	layer7 := func() {
		cf := coeff(0)
		nodes[20] = fmadd(nodes[14], cf, shuf(0, nodes[14]))
		nodes[21] = fmadd(nodes[15], cf, shuf(0, nodes[15]))
		nodes[22] = fmadd(nodes[16], cf, shuf(0, nodes[16]))
		nodes[23] = fmadd(nodes[17], cf, shuf(0, nodes[17]))
		nodes[24] = fmadd(nodes[9], cf, shuf(0, nodes[9]))
		nodes[25] = fmadd(nodes[11], cf, shuf(0, nodes[11]))
		nodes[26] = fmadd(nodes[18], cf, shuf(0, nodes[18]))
		nodes[27] = fmadd(nodes[19], cf, shuf(0, nodes[19]))
		layer8()
	}
	layer6 := func() {
		bc := bcast(0)
		nodes[14] = add(nodes[8], nodes[10])
		nodes[15] = sub(nodes[8], nodes[10])
		nodes[16] = fmadd(nodes[12], bc, nodes[1])
		nodes[17] = fnmsub(nodes[13], bc, nodes[5])
		nodes[18] = fnmadd(nodes[12], bc, nodes[1])
		nodes[19] = fnmadd(nodes[13], bc, nodes[5])
		layer7()
	}
	layer5 := func() {
		nodes[8] = add(nodes[0], nodes[4])
		nodes[9] = sub(nodes[0], nodes[4])
		nodes[10] = add(nodes[2], nodes[6])
		nodes[11] = sub(nodes[6], nodes[2])
		nodes[12] = sub(nodes[3], nodes[7])
		nodes[13] = add(nodes[3], nodes[7])
		layer6()
	}
	layer4 := func() {
		nodes[0] = add(in[0], in[8])
		nodes[1] = sub(in[0], in[8])
		nodes[2] = add(in[2], in[10])
		nodes[3] = sub(in[2], in[10])
		nodes[4] = add(in[4], in[12])
		nodes[5] = sub(in[4], in[12])
		nodes[6] = add(in[6], in[14])
		nodes[7] = sub(in[6], in[14])
		layer5()
	}
	layer3 := func(from, to int) cgen.Stmts {
		stmts = nil
		in = F.In[from:]
		out = F.Out[to:]
		layer4()
		return stmts
	}
	layer2 := func() cgen.Gen {
		toMix := [2]cgen.Stmts{
			layer3(0, 0),
			layer3(1, 8),
		}
		var (
			n     = len(toMix[0])
			mixed = make(cgen.Stmts, 2*n)
		)
		for i := range mixed {
			mixed[i] = toMix[i&1][i>>1]
		}
		return mixed
	}
	layer1 := func() cgen.Gen {
		for i := 0; i < 16; i++ {
			if F.In[i] == nil {
				F.In[i] = avx.Mm512SetzeroPs
			}
		}
		return layer2()
	}
	return layer1()
}

type Bwd struct {
	Platform raw.Platform
	Nms      nmsrc.Src
	In       [16]cgen.Gen
	Out      [16]cgen.Gen
}

func (B *Bwd) Append(to []byte) []byte {
	switch B.Platform {
	case raw.AVX512Float32:
		return B.m512().Append(to)
	default:
		panic("bug")
	}
}

func (B *Bwd) m512() cgen.Gen {
	var (
		stmts  cgen.Stmts
		in     []cgen.Gen
		out    []cgen.Gen
		perms  [2]cgen.Gen
		coeffs [6]cgen.Gen
		nodes  [84]cgen.Gen
	)
	stmt := func(st cgen.Gen) {
		stmts = append(stmts, st)
	}
	decl := func(t, id, expr cgen.Gen) cgen.Gen {
		if id == nil {
			ifft := B.Nms.Name("ifft")
			id = cgen.Vb(ifft)
		}
		stmt(cgen.Var{
			Type: t, What: id,
			Init: expr,
		})
		return id
	}
	perm := func(i int, node cgen.Gen) cgen.Gen {
		pm := perms[i]
		switch pm {
		case nil:
			var (
				set = make(avx.Mm512SetEpi32, 16)
				tbl []int
			)
			switch i {
			case 0:
				tbl = []int{12, 14, 14, 12, 10, 10, 9, 8}
			case 1:
				tbl = []int{13, 15, 15, 13, 11, 11, 8, 9}
			}
			for j := range set {
				set[j] = il(tbl[j%8] - j&8)
			}
			pm = decl(avx.M512i, nil, set)
			perms[i] = pm
		default:
			stmt(nil)
		}
		return decl(
			avx.M512, nil,
			avx.Mm512PermutexvarPs{
				pm, node,
			},
		)
	}
	coeff := func(i int) cgen.Gen {
		cf := coeffs[i]
		switch cf {
		case nil:
			var (
				neg1 = il(-1)
				neg2 = fl(-math.Sqrt2 * 0.5)
				pos1 = il(1)
				pos2 = fl(math.Sqrt2 * 0.5)
				zero = il(0)
				expr cgen.Gen
			)
			switch i {
			case 0:
				expr = avx.Mm512SetPs{
					pos1, neg1, pos1, neg1,
					pos1, neg1, zero, zero,
					pos1, neg1, pos1, neg1,
					pos1, neg1, zero, zero,
				}
			case 1:
				expr = avx.Mm512SetPs{
					neg1, pos1, neg1, pos1,
					neg1, pos1, neg1, pos1,
					neg1, pos1, neg1, pos1,
					neg1, pos1, neg1, pos1,
				}
			case 2:
				expr = avx.Mm512SetPs{
					neg2, pos1, pos2, pos1,
					zero, pos1, pos1, pos1,
					neg2, pos1, pos2, pos1,
					zero, pos1, pos1, pos1,
				}
			case 3:
				expr = avx.Mm512SetPs{
					pos2, zero, pos2, zero,
					pos1, zero, zero, zero,
					pos2, zero, pos2, zero,
					pos1, zero, zero, zero,
				}
			case 4:
				expr = avx.Mm512SetPs{
					neg1, neg1, pos1, pos1,
					neg1, neg1, pos1, pos1,
					neg1, neg1, pos1, pos1,
					neg1, neg1, pos1, pos1,
				}
			case 5:
				expr = avx.Mm512SetPs{
					neg1, neg1, neg1, neg1,
					pos1, pos1, pos1, pos1,
					neg1, neg1, neg1, neg1,
					pos1, pos1, pos1, pos1,
				}
			}
			cf = decl(avx.M512, nil, expr)
			coeffs[i] = cf
		default:
			stmt(nil)
		}
		return cf
	}
	blend := func(i int, node0, node1 cgen.Gen) cgen.Gen {
		var (
			mask cgen.Gen
			expr cgen.Gen
		)
		switch i {
		case 0, 1:
			mask = il(0xfdfd)
		case 2, 3:
			mask = il(0xc0c0)
		}
		switch i {
		case 0:
			expr = avx.Mm512MaskFmaddPs{
				node0, mask, coeff(0),
				node1,
			}
		case 1:
			expr = avx.Mm512MaskFnmaddPs{
				node0, mask, coeff(0),
				node1,
			}
		case 2:
			expr = avx.Mm512MaskSubPs{
				node0, mask,
				avx.Mm512SetzeroPs, node1,
			}
		case 3:
			expr = avx.Mm512MaskMovPs{
				node0, mask, node1,
			}
		}
		return decl(avx.M512, nil, expr)
	}
	shuf := func(i int, node cgen.Gen) cgen.Gen {
		var (
			ctrl int
			expr cgen.Gen
		)
		switch i {
		case 0, 2:
			ctrl = 2<<6 | 3<<4 | 0<<2 | 1<<0
		case 1:
			ctrl = 1<<6 | 0<<4 | 3<<2 | 2<<0
		}
		switch i {
		case 0, 1:
			expr = avx.Mm512ShufflePs{
				node, node, il(ctrl),
			}
		case 2:
			expr = avx.Mm512ShuffleF32x4{
				node, node, il(ctrl),
			}
		}
		return expr
	}
	fmadd := func(a, b, c cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512FmaddPs{a, b, c},
		)
	}
	mul := func(a, b cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512MulPs{a, b},
		)
	}
	fnmadd := func(a, b, c cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512FnmaddPs{a, b, c},
		)
	}
	fnmsub := func(a, b, c cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512FnmsubPs{a, b, c},
		)
	}
	add := func(a, b cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512AddPs{a, b},
		)
	}
	sub := func(a, b cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512SubPs{a, b},
		)
	}
	bcast := func(i int) cgen.Gen {
		return avx.Mm512Set1PsLit(
			[3]float64{
				1.0 / 32,
				1.0 / 64,
				math.Sqrt2 * 0.5,
			}[i],
		)
	}
	fmsub := func(a, b, c cgen.Gen) cgen.Gen {
		return decl(
			avx.M512, nil,
			avx.Mm512FmsubPs{a, b, c},
		)
	}
	emit := func(i int, node cgen.Gen) {
		id := out[i]
		switch id {
		case nil:
			stmt(cgen.Cast{
				Type: cgen.Void,
				Expr: node,
			})
		default:
			decl(avx.M512, id, node)
		}
	}
	layer16 := func() {
		emit(0, nodes[76])
		emit(1, nodes[78])
		emit(2, nodes[80])
		emit(3, nodes[82])
		emit(4, nodes[77])
		emit(5, nodes[79])
		emit(6, nodes[81])
		emit(7, nodes[83])
	}
	layer15 := func() {
		bc := bcast(1)
		nodes[76] = fmadd(nodes[72], bc, nodes[62])
		nodes[77] = fnmadd(nodes[72], bc, nodes[62])
		nodes[78] = fmadd(nodes[74], bc, nodes[64])
		nodes[79] = fnmadd(nodes[74], bc, nodes[64])
		nodes[80] = fnmadd(nodes[75], bc, nodes[63])
		nodes[81] = fmadd(nodes[75], bc, nodes[63])
		nodes[82] = fmadd(nodes[73], bc, nodes[65])
		nodes[83] = fnmadd(nodes[73], bc, nodes[65])
		layer16()
	}
	layer14 := func() {
		nodes[72] = add(nodes[68], nodes[69])
		nodes[73] = sub(nodes[68], nodes[69])
		nodes[74] = add(nodes[70], nodes[71])
		nodes[75] = sub(nodes[70], nodes[71])
		layer15()
	}
	layer13 := func() {
		bc := bcast(2)
		nodes[68] = fnmadd(nodes[66], bc, nodes[58])
		nodes[69] = fmadd(nodes[66], bc, nodes[58])
		nodes[70] = fmadd(nodes[67], bc, nodes[59])
		nodes[71] = fmsub(nodes[67], bc, nodes[59])
		layer14()
	}
	layer12 := func() {
		bc := bcast(1)
		nodes[62] = fmadd(nodes[54], bc, nodes[60])
		nodes[63] = fmsub(nodes[54], bc, nodes[60])
		nodes[64] = fmadd(nodes[55], bc, nodes[61])
		nodes[65] = fmsub(nodes[55], bc, nodes[61])
		nodes[66] = add(nodes[56], nodes[57])
		nodes[67] = sub(nodes[56], nodes[57])
		layer13()
	}
	layer11 := func() {
		bc := bcast(0)
		nodes[54] = add(nodes[46], nodes[47])
		nodes[55] = sub(nodes[46], nodes[47])
		nodes[56] = sub(nodes[48], nodes[52])
		nodes[57] = add(nodes[49], nodes[53])
		nodes[58] = add(nodes[48], nodes[52])
		nodes[59] = sub(nodes[49], nodes[53])
		nodes[60] = mul(nodes[50], bc)
		nodes[61] = mul(nodes[51], bc)
		layer12()
	}
	layer10 := func() {
		cf := coeff(5)
		nodes[46] = fmadd(nodes[38], cf, shuf(2, nodes[38]))
		nodes[47] = fmadd(nodes[39], cf, shuf(2, nodes[39]))
		nodes[48] = fmadd(nodes[40], cf, shuf(2, nodes[40]))
		nodes[49] = fmadd(nodes[41], cf, shuf(2, nodes[41]))
		nodes[50] = fmadd(nodes[42], cf, shuf(2, nodes[42]))
		nodes[51] = fnmsub(nodes[43], cf, shuf(2, nodes[43]))
		nodes[52] = fmadd(nodes[44], cf, shuf(2, nodes[44]))
		nodes[53] = fmadd(nodes[45], cf, shuf(2, nodes[45]))
		layer11()
	}
	layer9 := func() {
		nodes[38] = blend(2, nodes[30], nodes[31])
		nodes[39] = blend(3, nodes[31], nodes[30])
		nodes[40] = blend(2, nodes[32], nodes[33])
		nodes[41] = blend(3, nodes[33], nodes[32])
		nodes[42] = blend(2, nodes[34], nodes[35])
		nodes[43] = blend(3, nodes[35], nodes[34])
		nodes[44] = blend(2, nodes[36], nodes[37])
		nodes[45] = blend(3, nodes[37], nodes[36])
		layer10()
	}
	layer8 := func() {
		cf := coeff(4)
		nodes[30] = fmadd(nodes[22], cf, shuf(1, nodes[22]))
		nodes[31] = fmadd(nodes[23], cf, shuf(1, nodes[23]))
		nodes[32] = fmadd(nodes[24], cf, shuf(1, nodes[24]))
		nodes[33] = fmadd(nodes[25], cf, shuf(1, nodes[25]))
		nodes[34] = fmadd(nodes[26], cf, shuf(1, nodes[26]))
		nodes[35] = fmadd(nodes[27], cf, shuf(1, nodes[27]))
		nodes[36] = fmadd(nodes[28], cf, shuf(1, nodes[28]))
		nodes[37] = fmadd(nodes[29], cf, shuf(1, nodes[29]))
		layer9()
	}
	layer7 := func() {
		cf := coeff(3)
		nodes[22] = fnmadd(nodes[7], cf, nodes[14])
		nodes[23] = fmadd(nodes[6], cf, nodes[15])
		nodes[24] = fnmadd(nodes[9], cf, nodes[16])
		nodes[25] = fmadd(nodes[8], cf, nodes[17])
		nodes[26] = fnmadd(nodes[11], cf, nodes[18])
		nodes[27] = fmadd(nodes[10], cf, nodes[19])
		nodes[28] = fnmadd(nodes[13], cf, nodes[20])
		nodes[29] = fmadd(nodes[12], cf, nodes[21])
		layer8()
	}
	layer6 := func() {
		cf := coeff(2)
		nodes[14] = mul(nodes[6], cf)
		nodes[15] = mul(nodes[7], cf)
		nodes[16] = mul(nodes[8], cf)
		nodes[17] = mul(nodes[9], cf)
		nodes[18] = mul(nodes[10], cf)
		nodes[19] = mul(nodes[11], cf)
		nodes[20] = mul(nodes[12], cf)
		nodes[21] = mul(nodes[13], cf)
		layer7()
	}
	layer5 := func() {
		cf := coeff(1)
		nodes[6] = fmadd(nodes[4], cf, shuf(0, nodes[4]))
		nodes[7] = fmadd(nodes[5], cf, shuf(0, nodes[5]))
		nodes[8] = fmadd(in[2], cf, shuf(0, in[2]))
		nodes[9] = fmadd(in[3], cf, shuf(0, in[3]))
		nodes[10] = fmadd(in[4], cf, shuf(0, in[4]))
		nodes[11] = fmadd(in[5], cf, shuf(0, in[5]))
		nodes[12] = fmadd(in[6], cf, shuf(0, in[6]))
		nodes[13] = fmadd(in[7], cf, shuf(0, in[7]))
		layer6()
	}
	layer4 := func() {
		nodes[4] = blend(0, nodes[3], nodes[0])
		nodes[5] = blend(1, nodes[2], nodes[1])
		layer5()
	}
	layer3 := func() {
		nodes[0] = perm(0, in[0])
		nodes[1] = perm(1, in[0])
		nodes[2] = perm(0, in[1])
		nodes[3] = perm(1, in[1])
		layer4()
	}
	layer2 := func(i int) cgen.Stmts {
		if B.In[i] == nil {
			return nil
		}
		stmts = nil
		in = B.In[i:]
		out = B.Out[i:]
		layer3()
		return stmts
	}
	layer1 := func() cgen.Gen {
		toMix := [2]cgen.Stmts{
			layer2(0),
			layer2(8),
		}
		if toMix[1] == nil {
			return toMix[0]
		}
		var (
			n     = len(toMix[0])
			mixed = make(cgen.Stmts, 2*n)
		)
		for i := range mixed {
			mixed[i] = toMix[i&1][i>>1]
		}
		return mixed
	}
	return layer1()
}
