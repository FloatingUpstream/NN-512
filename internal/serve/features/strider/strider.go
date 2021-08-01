package strider

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
	Example := func(off, filtH, filtW int) {
		src := rand.New(
			0x8992e83e921c1ede,
			0x81cf1e175ed2341f,
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
			inCh   = R(1, 128) * groups
			outCh  = R(1, 128) * groups
			padH   = R(0, filtH-1)
			padW   = R(0, filtW-1)
		)
		conv := &raw.Conv{
			FromTensor: in,
			ToTensor:   out,
			ToChannels: outCh,
			FilterH:    filtH,
			FilterW:    filtW,
			StrideH:    2,
			StrideW:    2,
			PaddingH:   padH,
			PaddingW:   padW,
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
				Channels: inCh,
				Height:   R(1, 32),
				Width:    R(1, 32),
			},
			conv,
			&raw.Output{
				FromTensor: out,
			},
		))
	}
	layer2 := func() {
		Example(300, 3, 3)
		Example(434, 4, 3)
		Example(513, 5, 5)
		Example(720, 7, 7)
		Example(801, 8, 9)
	}
	layer1 := func() {
		put(
			"Generates very efficient code "+
				"for Stride2x2 convolutions "+
				"(in particular, Filter7x7 Stride2x2)",
			"Fourier convolution with a 16x16 tile; "+
				"four 8x8 FFTs per tile, "+
				"interleaved modulo the stride",
			"Fourier-domain data is packed for "+
				"persistence in L2 cache",
			"Fourier-domain weights are streamed "+
				"in half-precision to conserve "+
				"memory bandwidth",
			"The 16x16 tiles are deinterleaved, "+
				"multiplied, and accumulated "+
				"(yielding 8x8 tiles)",
			"The inverse transform (IFFT) operates "+
				"on 8x8 tiles, four at a time",
		)
		layer2()
	}
	layer1()
	return
}
