package wct

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
	Example := func(off, mul int, tag string) {
		src := rand.New(
			0x90472ae923f130ac,
			0x7eb7b48a3952e346,
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
			outC   = R(1, 256*mul*mul) * groups
			inC    = R(1, 256*mul*mul) * groups
			inH    = R(1, 16*mul)
			inW    = R(1, 16*mul)
			padH   = R(0, 2)
			padW   = R(0, 2)
		)
		conv := &raw.Conv{
			FromTensor: in,
			ToTensor:   out,
			ToChannels: outC,
			FilterH:    3,
			FilterW:    3,
			StrideH:    1,
			StrideW:    1,
			PaddingH:   padH,
			PaddingW:   padW,
			DilationH:  1,
			DilationW:  1,
			Groups:     groups,
		}
		desc := fmt.Sprintf(
			"Filter%dx%d %s",
			conv.FilterH,
			conv.FilterW,
			tag,
		)
		put(example.Prep(
			desc,
			&raw.Input{
				ToTensor: in,
				Channels: inC,
				Height:   inH,
				Width:    inW,
			},
			conv,
			&raw.Output{
				FromTensor: out,
			},
		))
	}
	layer2 := func() {
		Example(111, 1, "small tensor")
		Example(275, 2, "medium tensor")
		Example(461, 4, "large tensor")
	}
	layer1 := func() {
		put(
			"Generates very efficient code "+
				"for Filter3x3 convolutions",
			"Winograd-Cook-Toom-Lavin convolution "+
				"with an 8x8 tile",
			"4-way tile transforms fully utilize "+
				"512-bit vector registers (and "+
				"amortize transposition costs)",
			"Winograd-domain data is packed for "+
				"persistence in L2 cache",
			"Winograd-domain weights are streamed "+
				"in half-precision to conserve "+
				"memory bandwidth",
		)
		layer2()
	}
	layer1()
	return
}
