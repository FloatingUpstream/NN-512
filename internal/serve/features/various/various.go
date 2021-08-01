package various

import (
	"NN-512/internal/raw"
	"NN-512/internal/serve/example"
	"NN-512/internal/serve/rand"
)

func Prep() (parts []string) {
	put := func(part string) {
		parts = append(parts, part)
	}
	Example := func(off, idx int) {
		src := rand.New(
			0x11f41482faead1d0,
			0xa5f78cf4be8c95e9,
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
			inC  = R(1, 2048)
			inH  = R(1, 512)
			inW  = R(1, 512)
			desc string
			node raw.Node
		)
		switch idx {
		case 1:
			desc = "Fully connected " +
				"(half-precision weights)"
			node = &raw.FullyConnected{
				FromTensor: in,
				ToTensor:   out,
				ToChannels: R(1, 2048),
			}
		case 2, 3:
			var (
				kind raw.PoolingKind
				op   string
			)
			switch idx {
			case 2:
				kind = raw.Max3x3Stride2
				op = "Max"
			case 3:
				kind = raw.Avg3x3Stride2
				op = "Avg"
			}
			desc = op + " pooling " +
				"(3x3 window, 2x2 stride)"
			node = &raw.Pooling{
				FromTensor: in,
				ToTensor:   out,
				Kind:       kind,
				PaddingH:   R(0, 2),
				PaddingW:   R(0, 2),
			}
		case 4, 5:
			var (
				kind raw.PoolingKind
				op   string
			)
			switch idx {
			case 4:
				kind = raw.Max2x2Stride2
				op = "Max"
			case 5:
				kind = raw.Avg2x2Stride2
				op = "Avg"
			}
			desc = op + " pooling " +
				"(2x2 window, 2x2 stride)"
			node = &raw.Pooling{
				FromTensor: in,
				ToTensor:   out,
				Kind:       kind,
				PaddingH:   R(0, 1),
				PaddingW:   R(0, 1),
			}
		case 6, 7:
			var (
				kind raw.PoolingKind
				op   string
			)
			switch idx {
			case 6:
				inH = R(1, 8)
				inW = R(1, 8)
				kind = raw.MaxGlobal
				op = "max"
			case 7:
				inH = R(1, 64)
				inW = R(1, 64)
				kind = raw.AvgGlobal
				op = "avg"
			}
			desc = "Global " + op + " pooling"
			node = &raw.Pooling{
				FromTensor: in,
				ToTensor:   out,
				Kind:       kind,
				PaddingH:   0,
				PaddingW:   0,
			}
		case 8, 9:
			var size string
			switch idx {
			case 8:
				inH = R(1, 4)
				inW = R(1, 4)
				size = "small"
			case 9:
				size = "large"
			}
			desc = "Softmax (" + size + " channel)"
			node = &raw.Softmax{
				FromTensor: in,
				ToTensor:   out,
			}
		default:
			panic("bug")
		}
		put(example.Prep(
			desc,
			&raw.Input{
				ToTensor: in,
				Channels: inC,
				Height:   inH,
				Width:    inW,
			},
			node,
			&raw.Output{
				FromTensor: out,
			},
		))
	}
	layer2 := func() {
		Example(110, 1)
		Example(245, 2)
		Example(370, 3)
		Example(490, 4)
		Example(590, 5)
		Example(612, 6)
		Example(700, 7)
		Example(808, 8)
		Example(900, 9)
	}
	layer1 := func() {
		put("Generates efficient code for " +
			"various other tensor operations")
		layer2()
	}
	layer1()
	return
}
