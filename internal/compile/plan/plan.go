package plan

import "NN-512/internal/raw"

type Param struct {
	Tensor string
	NCHW   [4]int
}

type Pile struct {
	Channels    int
	Height      int
	Width       int
	ElemBytes   int
	Pitch1Bytes int
	Pitch2Bytes int
	SizeBytes   int
	OffsetBytes int
	Writers     []*Span
	Readers     []*Span
}

type Span struct {
	Piles   []*Pile
	Offsets []int
	Tensors []string
	Counts  []int
	Op      *Op
}

type Mod struct {
	Nodes  []raw.Node
	Params []Param
	From   []*Span
}

type Op struct {
	Nodes     []raw.Node
	Params    [][]Param
	ParamMods [][2][]Mod
	From      []*Span
	FromMods  [][]Mod
	To        []*Span
	ToMods    [][]Mod
}

type Plan struct {
	Config *raw.Config
	Seq    []*Op
}
