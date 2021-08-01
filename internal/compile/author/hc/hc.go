package hc

import "NN-512/internal/compile/author/cgen"

type Section int

const (
	HFirst Section = iota
	HPragmaOnce
	HLicense
	HInclude
	HLinkage1
	HParams1
	HNet
	HEngine
	HParams2
	HLinkage2
	HLast
	CFirst
	CToBuild
	CLicense
	CInclude
	CErrmsg
	CThreader
	CExp
	CSoftmax
	CRsqrt
	CBn
	CElwi
	CGlopl
	CTwopl
	CThrpl
	CFc
	COne
	CThree
	CStrider
	CLoom
	CNet
	CEngine
	CLast
	sectionCount
)

type Sections struct {
	a [sectionCount][]byte
}

func (s *Sections) Append(to Section, from ...cgen.Gen) {
	for _, gen := range from {
		if gen != nil {
			s.a[to] = gen.Append(s.a[to])
		}
	}
}

func (s *Sections) Join() (h, c []byte) {
	h = s.join(HFirst, HLast)
	c = s.join(CFirst, CLast)
	return
}

func (s *Sections) join(first, last Section) (to []byte) {
	const (
		brace1  = '{'
		brace2  = '}'
		newline = '\n'
		paren1  = '('
		paren2  = ')'
		tab     = '\t'
	)
	var prev byte
	var indent []byte
	for _, from := range s.a[first : last+1] {
		for _, curr := range from {
			switch curr {
			case newline:
				if prev == brace1 || prev == paren1 {
					indent = append(indent, tab)
				}
			default:
				if prev == newline {
					if curr == brace2 || curr == paren2 {
						indent = indent[:len(indent)-1]
					}
					to = append(to, indent...)
				}
			}
			to = append(to, curr)
			prev = curr
		}
	}
	return
}
