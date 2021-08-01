package one

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
	Example := func(off, stride int) {
		src := rand.New(
			0xa1dc9f499481013c,
			0x8a6a01a02ea5242b,
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
			outC   = R(1, 4096) * groups
			inC    = R(1, 4096) * groups
			inH    = R(1, 256)
			inW    = R(1, 256)
		)
		conv := &raw.Conv{
			FromTensor: in,
			ToTensor:   out,
			ToChannels: outC,
			FilterH:    1,
			FilterW:    1,
			StrideH:    stride,
			StrideW:    stride,
			PaddingH:   0,
			PaddingW:   0,
			DilationH:  1,
			DilationW:  1,
			Groups:     groups,
		}
		desc := fmt.Sprintf(
			"Filter%dx%d Stride%dx%d",
			conv.FilterH,
			conv.FilterW,
			conv.StrideH,
			conv.StrideW,
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
		Example(177, 1)
		Example(224, 2)
		Example(399, 3)
		Example(404, 4)
	}
	layer1 := func() {
		put(
			"Generates very efficient code for "+
				"Filter1x1 convolutions (including "+
				"those that are not Stride1x1)",
			"Single-precision matrix multiplication "+
				"(making full use of the large "+
				"vector register file)",
			"The input data tensor is subsampled and "+
				"packed for persistence in L2 cache",
			"The weight tensor is packed and streamed "+
				"for broadcast multiplication",
		)
		layer2()
	}
	layer1()
	return
}
