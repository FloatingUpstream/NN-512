package avx

import "NN-512/internal/compile/author/cgen"

var (
	CmpLtOq cgen.Gen = cgen.Vb("_CMP_LT_OQ")
	M256i   cgen.Gen = cgen.Vb("__m256i")
	M512    cgen.Gen = cgen.Vb("__m512")
	M512i   cgen.Gen = cgen.Vb("__m512i")
	Mmask16 cgen.Gen = cgen.Vb("__mmask16")
)

const (
	mmFroundToNearestInt cgen.Vb = "_MM_FROUND_TO_NEAREST_INT"
	mmFroundNoExc        cgen.Vb = "_MM_FROUND_NO_EXC"
)

var FroundToNearestIntNoExc cgen.Gen = cgen.Or{
	Expr1: mmFroundToNearestInt,
	Expr2: mmFroundNoExc,
}

func call(to []byte, fn string, args []cgen.Gen) []byte {
	return cgen.Call{
		Func: cgen.Vb(fn),
		Args: cgen.CommaSpaced(args),
	}.Append(to)
}

type Mm512AddEpi32 []cgen.Gen

func (m Mm512AddEpi32) Append(to []byte) []byte {
	return call(to, "_mm512_add_epi32", m)
}

type Mm512AddPs []cgen.Gen

func (m Mm512AddPs) Append(to []byte) []byte {
	return call(to, "_mm512_add_ps", m)
}

type Mm512AlignrEpi32 []cgen.Gen

func (m Mm512AlignrEpi32) Append(to []byte) []byte {
	return call(to, "_mm512_alignr_epi32", m)
}

type Mm512CastpsSi512 []cgen.Gen

func (m Mm512CastpsSi512) Append(to []byte) []byte {
	return call(to, "_mm512_castps_si512", m)
}

type Mm512Castsi256Si512 []cgen.Gen

func (m Mm512Castsi256Si512) Append(to []byte) []byte {
	return call(to, "_mm512_castsi256_si512", m)
}

type Mm512Castsi512Ps []cgen.Gen

func (m Mm512Castsi512Ps) Append(to []byte) []byte {
	return call(to, "_mm512_castsi512_ps", m)
}

type Mm512Castsi512Si256 []cgen.Gen

func (m Mm512Castsi512Si256) Append(to []byte) []byte {
	return call(to, "_mm512_castsi512_si256", m)
}

type Mm512CmpPsMask []cgen.Gen

func (m Mm512CmpPsMask) Append(to []byte) []byte {
	return call(to, "_mm512_cmp_ps_mask", m)
}

type Mm512CvtphPs []cgen.Gen

func (m Mm512CvtphPs) Append(to []byte) []byte {
	return call(to, "_mm512_cvtph_ps", m)
}

type Mm512CvtpsEpi32 []cgen.Gen

func (m Mm512CvtpsEpi32) Append(to []byte) []byte {
	return call(to, "_mm512_cvtps_epi32", m)
}

type Mm512CvtpsPh []cgen.Gen

func (m Mm512CvtpsPh) Append(to []byte) []byte {
	return call(to, "_mm512_cvtps_ph", m)
}

type Mm512DivPs []cgen.Gen

func (m Mm512DivPs) Append(to []byte) []byte {
	return call(to, "_mm512_div_ps", m)
}

type Mm512Extracti64x4Epi64 []cgen.Gen

func (m Mm512Extracti64x4Epi64) Append(to []byte) []byte {
	return call(to, "_mm512_extracti64x4_epi64", m)
}

type Mm512FmaddPs []cgen.Gen

func (m Mm512FmaddPs) Append(to []byte) []byte {
	return call(to, "_mm512_fmadd_ps", m)
}

type Mm512FmsubPs []cgen.Gen

func (m Mm512FmsubPs) Append(to []byte) []byte {
	return call(to, "_mm512_fmsub_ps", m)
}

type Mm512FnmaddPs []cgen.Gen

func (m Mm512FnmaddPs) Append(to []byte) []byte {
	return call(to, "_mm512_fnmadd_ps", m)
}

type Mm512FnmsubPs []cgen.Gen

func (m Mm512FnmsubPs) Append(to []byte) []byte {
	return call(to, "_mm512_fnmsub_ps", m)
}

type Mm512Inserti64x4 []cgen.Gen

func (m Mm512Inserti64x4) Append(to []byte) []byte {
	return call(to, "_mm512_inserti64x4", m)
}

type Mm512LoaduPs []cgen.Gen

func (m Mm512LoaduPs) Append(to []byte) []byte {
	return call(to, "_mm512_loadu_ps", m)
}

type Mm512LoaduSi512 []cgen.Gen

func (m Mm512LoaduSi512) Append(to []byte) []byte {
	return call(to, "_mm512_loadu_si512", m)
}

type Mm512Mask3FmaddPs []cgen.Gen

func (m Mm512Mask3FmaddPs) Append(to []byte) []byte {
	return call(to, "_mm512_mask3_fmadd_ps", m)
}

type Mm512Mask3FnmaddPs []cgen.Gen

func (m Mm512Mask3FnmaddPs) Append(to []byte) []byte {
	return call(to, "_mm512_mask3_fnmadd_ps", m)
}

type Mm512MaskAddPs []cgen.Gen

func (m Mm512MaskAddPs) Append(to []byte) []byte {
	return call(to, "_mm512_mask_add_ps", m)
}

type Mm512MaskFmaddPs []cgen.Gen

func (m Mm512MaskFmaddPs) Append(to []byte) []byte {
	return call(to, "_mm512_mask_fmadd_ps", m)
}

type Mm512MaskFnmaddPs []cgen.Gen

func (m Mm512MaskFnmaddPs) Append(to []byte) []byte {
	return call(to, "_mm512_mask_fnmadd_ps", m)
}

type Mm512MaskMaxPs []cgen.Gen

func (m Mm512MaskMaxPs) Append(to []byte) []byte {
	return call(to, "_mm512_mask_max_ps", m)
}

type Mm512MaskMovPs []cgen.Gen

func (m Mm512MaskMovPs) Append(to []byte) []byte {
	return call(to, "_mm512_mask_mov_ps", m)
}

type Mm512MaskMulPs []cgen.Gen

func (m Mm512MaskMulPs) Append(to []byte) []byte {
	return call(to, "_mm512_mask_mul_ps", m)
}

type Mm512MaskStoreuEpi32 []cgen.Gen

func (m Mm512MaskStoreuEpi32) Append(to []byte) []byte {
	return call(to, "_mm512_mask_storeu_epi32", m)
}

type Mm512MaskStoreuPs []cgen.Gen

func (m Mm512MaskStoreuPs) Append(to []byte) []byte {
	return call(to, "_mm512_mask_storeu_ps", m)
}

type Mm512MaskSubPs []cgen.Gen

func (m Mm512MaskSubPs) Append(to []byte) []byte {
	return call(to, "_mm512_mask_sub_ps", m)
}

type Mm512MaskzLoaduEpi32 []cgen.Gen

func (m Mm512MaskzLoaduEpi32) Append(to []byte) []byte {
	return call(to, "_mm512_maskz_loadu_epi32", m)
}

type Mm512MaskzLoaduPs []cgen.Gen

func (m Mm512MaskzLoaduPs) Append(to []byte) []byte {
	return call(to, "_mm512_maskz_loadu_ps", m)
}

type Mm512MaxPs []cgen.Gen

func (m Mm512MaxPs) Append(to []byte) []byte {
	return call(to, "_mm512_max_ps", m)
}

type Mm512MinPs []cgen.Gen

func (m Mm512MinPs) Append(to []byte) []byte {
	return call(to, "_mm512_min_ps", m)
}

type Mm512MulPs []cgen.Gen

func (m Mm512MulPs) Append(to []byte) []byte {
	return call(to, "_mm512_mul_ps", m)
}

type Mm512Permutex2varPs []cgen.Gen

func (m Mm512Permutex2varPs) Append(to []byte) []byte {
	return call(to, "_mm512_permutex2var_ps", m)
}

type Mm512PermutexvarPs []cgen.Gen

func (m Mm512PermutexvarPs) Append(to []byte) []byte {
	return call(to, "_mm512_permutexvar_ps", m)
}

type Mm512RoundscalePs []cgen.Gen

func (m Mm512RoundscalePs) Append(to []byte) []byte {
	return call(to, "_mm512_roundscale_ps", m)
}

type Mm512Rsqrt14Ps []cgen.Gen

func (m Mm512Rsqrt14Ps) Append(to []byte) []byte {
	return call(to, "_mm512_rsqrt14_ps", m)
}

type Mm512Set1Ps []cgen.Gen

func (m Mm512Set1Ps) Append(to []byte) []byte {
	return call(to, "_mm512_set1_ps", m)
}

type Mm512Set1PsLit cgen.FloatLit

func (m Mm512Set1PsLit) Append(to []byte) []byte {
	return Mm512Set1Ps{cgen.FloatLit(m)}.Append(to)
}

type Mm512SetEpi32 []cgen.Gen

func (m Mm512SetEpi32) Append(to []byte) []byte {
	return call(to, "_mm512_set_epi32", m)
}

type Mm512SetPs []cgen.Gen

func (m Mm512SetPs) Append(to []byte) []byte {
	return call(to, "_mm512_set_ps", m)
}

var Mm512SetzeroPs cgen.Gen = cgen.Call{
	Func: cgen.Vb("_mm512_setzero_ps"),
}

type Mm512ShuffleF32x4 []cgen.Gen

func (m Mm512ShuffleF32x4) Append(to []byte) []byte {
	return call(to, "_mm512_shuffle_f32x4", m)
}

type Mm512ShuffleI32x4 []cgen.Gen

func (m Mm512ShuffleI32x4) Append(to []byte) []byte {
	return call(to, "_mm512_shuffle_i32x4", m)
}

type Mm512ShufflePs []cgen.Gen

func (m Mm512ShufflePs) Append(to []byte) []byte {
	return call(to, "_mm512_shuffle_ps", m)
}

type Mm512SlliEpi32 []cgen.Gen

func (m Mm512SlliEpi32) Append(to []byte) []byte {
	return call(to, "_mm512_slli_epi32", m)
}

type Mm512StoreuEpi32 []cgen.Gen

func (m Mm512StoreuEpi32) Append(to []byte) []byte {
	return call(to, "_mm512_storeu_epi32", m)
}

type Mm512StoreuPs []cgen.Gen

func (m Mm512StoreuPs) Append(to []byte) []byte {
	return call(to, "_mm512_storeu_ps", m)
}

type Mm512SubPs []cgen.Gen

func (m Mm512SubPs) Append(to []byte) []byte {
	return call(to, "_mm512_sub_ps", m)
}

type Mm512UnpackhiPs []cgen.Gen

func (m Mm512UnpackhiPs) Append(to []byte) []byte {
	return call(to, "_mm512_unpackhi_ps", m)
}

type Mm512UnpackloPs []cgen.Gen

func (m Mm512UnpackloPs) Append(to []byte) []byte {
	return call(to, "_mm512_unpacklo_ps", m)
}
