package loom

import (
	"NN-512/internal/raw"
	"NN-512/internal/serve/example"
	"NN-512/internal/serve/rand"
	"fmt"
)

func Prep() (parts []string) {
	put := func(a ...string) {
		parts = append(parts, a...)
	}
	Example := func(off int) {
		src := rand.New(
			0xc973e6e610658b14,
			0x65058b487d5a4fd9,
		)
		for n := off; n > 0; n-- {
			_ = src.Uint32()
		}
		R := func(first, last int) int {
			var (
				n = uint32(last + 1 - first)
				r = int(src.Below(n))
			)
			return first + r
		}
		const (
			in  = "in"
			out = "out"
		)
		var (
			groups = R(1, 8)
			inCh   = R(1, 1024) * groups
			outCh  = R(1, 1024) * groups
		)
		conv := &raw.Conv{
			FromTensor: in,
			ToTensor:   out,
			ToChannels: outCh,
			FilterH:    R(1, 8),
			FilterW:    R(1, 8),
			StrideH:    R(1, 4),
			StrideW:    R(1, 4),
			PaddingH:   R(0, 3),
			PaddingW:   R(0, 3),
			DilationH:  R(1, 4),
			DilationW:  R(1, 4),
			Groups:     groups,
		}
		desc := fmt.Sprintf(
			"Filter%dx%d Stride%dx%d Dilation%dx%d",
			conv.FilterH,
			conv.FilterW,
			conv.StrideH,
			conv.StrideW,
			conv.DilationH,
			conv.DilationW,
		)
		put(example.Prep(
			desc,
			&raw.Input{
				ToTensor: in,
				Channels: inCh,
				Height:   R(1, 128),
				Width:    R(1, 128),
			},
			conv,
			&raw.Output{
				FromTensor: out,
			},
		))
	}
	layer2 := func() {
		Example(955)
		Example(424)
		Example(300)
		Example(365)
		Example(904)
	}
	layer1 := func() {
		put(
			"Generates efficient code for "+
				"convolutions with arbitrary "+
				"filter shape, stride, and dilation",
			"The input data tensor is split into "+
				"disjoint subtensors modulo the stride, "+
				"while being packed for the cache",
			"The weight tensor is split similarly, "+
				"taking dilation into account",
			"Corresponding subtensors are multiplied, "+
				"with accumulation at heightwise offsets",
			"The output data tensor is formed by "+
				"combining accumulators at "+
				"widthwise offsets",
		)
		layer2()
	}
	layer1()
	return
}
