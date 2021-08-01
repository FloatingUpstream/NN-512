package nmsrc

import "strconv"

type Src struct {
	m map[string]int
}

func New() Src {
	return Src{
		m: make(map[string]int),
	}
}

func (s Src) Name(prefix string) string {
	i := s.m[prefix] + 1
	s.m[prefix] = i
	return prefix + strconv.Itoa(i)
}
