package elwi

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
	Example := func(off, idx int) {
		src := rand.New(
			0xe9292d7994694105,
			0x3f813d1fd0ca2b93,
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
		negSlope := func() float32 {
			if src.Uint32()&1 == 0 {
				return 0
			}
			const denom = 16
			var (
				Int   = R(1, denom-1)
				numer = float32(Int)
			)
			return numer / denom
		}
		const (
			in1  = "in1"
			in2  = "in2"
			in3  = "in3"
			bn1  = "bn1"
			bn2  = "bn2"
			bn3  = "bn3"
			bn4  = "bn4"
			eps  = 0.00001
			act1 = "act1"
			act2 = "act2"
			add1 = "add1"
			add2 = "add2"
		)
		var (
			pre   = bn2
			preC  = R(1, 128)
			preH  = R(1, 32)
			preW  = R(1, 32)
			post  string
			postC int
			postH int
			postW int
			desc  string
			node  raw.Node
		)
		switch idx {
		case 1, 2, 3, 4:
			var (
				filter int
				stride = 1
			)
			switch idx {
			case 1:
				preC = R(1, 2048)
				filter = 1
			case 2:
				filter = 3
			case 3:
				filter = 7
				stride = 2
			case 4:
				filter = 4
			}
			post = "conv"
			postC = R(1, 128)
			postH = (preH-filter)/stride + 1
			postW = (preW-filter)/stride + 1
			conv := &raw.Conv{
				FromTensor: pre,
				ToTensor:   post,
				ToChannels: postC,
				FilterH:    filter,
				FilterW:    filter,
				StrideH:    stride,
				StrideW:    stride,
				PaddingH:   0,
				PaddingW:   0,
				DilationH:  1,
				DilationW:  1,
				Groups:     1,
			}
			desc = fmt.Sprintf(
				"Convolution "+
					"(Filter%dx%d "+
					"Stride%dx%d)",
				conv.FilterH,
				conv.FilterW,
				conv.StrideH,
				conv.StrideW,
			)
			node = conv
		case 5, 6, 7:
			var (
				kind raw.PoolingKind
				side int
				pad  = 0
			)
			switch idx {
			case 5:
				kind = raw.Max3x3Stride2
				side = 3
				pad = 1
			case 6:
				kind = raw.Avg2x2Stride2
				side = 2
			case 7:
				kind = raw.AvgGlobal
			}
			post = "pool"
			postC = preC
			switch idx {
			case 5, 6:
				postH = (preH+2*pad-side)/2 + 1
				postW = (preW+2*pad-side)/2 + 1
				desc = fmt.Sprintf(
					"%dx%d window",
					side, side,
				)
			case 7:
				postH = 1
				postW = 1
				desc = "global"
			}
			desc = "Pooling (" + desc + ")"
			node = &raw.Pooling{
				FromTensor: pre,
				ToTensor:   post,
				Kind:       kind,
				PaddingH:   pad,
				PaddingW:   pad,
			}
		case 8:
			post = "fc"
			postC = R(1, 128)
			postH = 1
			postW = 1
			desc = "Fully connected"
			node = &raw.FullyConnected{
				FromTensor: pre,
				ToTensor:   post,
				ToChannels: postC,
			}
		default:
			panic("bug")
		}
		put(example.Prep(
			desc,
			&raw.Input{
				ToTensor: in1,
				Channels: preC,
				Height:   preH,
				Width:    preW,
			},
			&raw.Input{
				ToTensor: in2,
				Channels: preC,
				Height:   preH,
				Width:    preW,
			},
			&raw.Input{
				ToTensor: in3,
				Channels: postC,
				Height:   postH,
				Width:    postW,
			},
			&raw.BatchNorm{
				FromTensor: in1,
				ToTensor:   bn1,
				Epsilon:    eps,
			},
			&raw.Activation{
				FromTensor: bn1,
				ToTensor:   act1,
				Kind:       raw.ReLU,
				Param:      negSlope(),
			},
			&raw.Add{
				FromTensor1: act1,
				FromTensor2: in2,
				ToTensor:    add1,
			},
			&raw.BatchNorm{
				FromTensor: add1,
				ToTensor:   bn2,
				Epsilon:    eps,
			},
			node,
			&raw.BatchNorm{
				FromTensor: post,
				ToTensor:   bn3,
				Epsilon:    eps,
			},
			&raw.Activation{
				FromTensor: bn3,
				ToTensor:   act2,
				Kind:       raw.ReLU,
				Param:      negSlope(),
			},
			&raw.Add{
				FromTensor1: act2,
				FromTensor2: in3,
				ToTensor:    add2,
			},
			&raw.BatchNorm{
				FromTensor: add2,
				ToTensor:   bn4,
				Epsilon:    eps,
			},
			&raw.Output{
				FromTensor: bn4,
			},
		))
	}
	layer2 := func() {
		Example(143, 1)
		Example(260, 2)
		Example(333, 3)
		Example(409, 4)
		Example(512, 5)
		Example(690, 6)
		Example(777, 7)
		Example(822, 8)
	}
	layer1 := func() {
		put(
			"Integrates elementwise operations "+
				"directly into the code generated "+
				"for more complex tensor operations",
			"Batch normalization is implemented "+
				"by modifying weights and biases "+
				"during packing (if possible)",
			"Remaining elementwise operations are "+
				"applied to data already present "+
				"in registers (during inference)",
		)
		layer2()
	}
	layer1()
	return
}
