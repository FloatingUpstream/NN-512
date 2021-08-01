package rand

type Src [2]uint64

func New(bits1, bits2 uint64) *Src {
	return &Src{bits1, bits2 | 1}
}

func (s *Src) Uint32() uint32 {
	var (
		x = s[0]
		y = uint32(x >> 59)
		z = uint32((x>>18 ^ x) >> 27)
	)
	s[0] = s[1] + x*6364136223846793005
	return z>>y | z<<(-y&31)
}

func (s *Src) Below(n uint32) uint32 {
	for min := -n % n; ; {
		r := s.Uint32()
		if r >= min {
			return r % n
		}
	}
}
