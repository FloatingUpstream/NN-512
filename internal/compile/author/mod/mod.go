package mod

type Kind int

const (
	Add Kind = iota
	Bn
	ReLU
)

type Op struct {
	Kind
	Int   int
	Float float32
}
