package densenet

import (
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
	"strconv"
)

const (
	denseNet121 = "DenseNet121"
	denseNet169 = "DenseNet169"
	denseNet201 = "DenseNet201"
	denseNet265 = "DenseNet265"
)

const (
	growthRate  = 32
	initial     = 2 * growthRate
	bottleneck  = 4 * growthRate
	compression = 0.5
)

type tensor struct {
	name     string
	channels int
}

type state struct {
	variant string
	text    []byte
	nms     nmsrc.Src
}

func (st *state) line(a string) {
	st.text = append(st.text, a...)
	st.text = append(st.text, '\n')
}

func (st *state) bnr(t1 tensor) tensor {
	t2 := st.nms.Name("bn")
	st.line("BatchNorm FromTensor=" + t1.name + " ToTensor=" + t2 + " Epsilon=0.00001")
	t3 := st.nms.Name("relu")
	st.line("Activation FromTensor=" + t2 + " ToTensor=" + t3 + " Kind=ReLU Param=0")
	return tensor{t3, t1.channels}
}

func (st *state) dense(t1 tensor) tensor {
	t2 := st.bnr(t1)
	t3 := tensor{st.nms.Name("one"), bottleneck}
	st.line("Conv FromTensor=" + t2.name + " ToTensor=" + t3.name + " ToChannels=" + strconv.Itoa(t3.channels) +
		" FilterH=1 FilterW=1 StrideH=1 StrideW=1 PaddingH=0 PaddingW=0 DilationH=1 DilationW=1 Groups=1")
	t4 := st.bnr(t3)
	t5 := tensor{st.nms.Name("three"), growthRate}
	st.line("Conv FromTensor=" + t4.name + " ToTensor=" + t5.name + " ToChannels=" + strconv.Itoa(t5.channels) +
		" FilterH=3 FilterW=3 StrideH=1 StrideW=1 PaddingH=1 PaddingW=1 DilationH=1 DilationW=1 Groups=1")
	t6 := tensor{st.nms.Name("concat"), t1.channels + t5.channels}
	st.line("Concat FromTensor1=" + t1.name + " FromTensor2=" + t5.name + " ToTensor=" + t6.name)
	return t6
}

func (st *state) trans(t1 tensor) tensor {
	t2 := st.bnr(t1)
	t3 := tensor{st.nms.Name("one"), int(float64(t2.channels) * compression)}
	st.line("Conv FromTensor=" + t2.name + " ToTensor=" + t3.name + " ToChannels=" + strconv.Itoa(t3.channels) +
		" FilterH=1 FilterW=1 StrideH=1 StrideW=1 PaddingH=0 PaddingW=0 DilationH=1 DilationW=1 Groups=1")
	t4 := tensor{st.nms.Name("pool"), t3.channels}
	st.line("Pooling FromTensor=" + t3.name + " ToTensor=" + t4.name + " Kind=Avg2x2Stride2 PaddingH=0 PaddingW=0")
	return t4
}

func (st *state) config() {
	const head = "Config"
	st.text = append(st.text, head...)
	for _, seg := range raw.Guide[head].Segs {
		st.text = append(st.text, " "+seg.Label+raw.Binder...)
		if seg.Label == "Prefix" {
			st.text = append(st.text, st.variant...)
		} else {
			st.text = append(st.text, seg.Default...)
		}
	}
	st.text = append(st.text, '\n')
}

func (st *state) prologue() tensor {
	const t1 = "image"
	st.line("Input ToTensor=" + t1 + " Channels=3 Height=224 Width=224")
	t2 := tensor{"sevenDS", initial}
	st.line("Conv FromTensor=" + t1 + " ToTensor=" + t2.name + " ToChannels=" + strconv.Itoa(t2.channels) +
		" FilterH=7 FilterW=7 StrideH=2 StrideW=2 PaddingH=3 PaddingW=3 DilationH=1 DilationW=1 Groups=1")
	t3 := st.bnr(t2)
	t4 := tensor{st.nms.Name("pool"), t3.channels}
	st.line("Pooling FromTensor=" + t3.name + " ToTensor=" + t4.name + " Kind=Max3x3Stride2 PaddingH=1 PaddingW=1")
	return t4
}

func (st *state) stage1(t1 tensor) tensor {
	for i := 0; i < 6; i++ {
		t1 = st.dense(t1)
	}
	return st.trans(t1)
}

func (st *state) stage2(t1 tensor) tensor {
	for i := 0; i < 12; i++ {
		t1 = st.dense(t1)
	}
	return st.trans(t1)
}

func (st *state) stage3(t1 tensor) tensor {
	var reps int
	switch st.variant {
	case denseNet121:
		reps = 24
	case denseNet169:
		reps = 32
	case denseNet201:
		reps = 48
	case denseNet265:
		reps = 64
	default:
		panic("bug")
	}
	for i := 0; i < reps; i++ {
		t1 = st.dense(t1)
	}
	return st.trans(t1)
}

func (st *state) stage4(t1 tensor) tensor {
	var reps int
	switch st.variant {
	case denseNet121:
		reps = 16
	case denseNet169, denseNet201:
		reps = 32
	case denseNet265:
		reps = 48
	default:
		panic("bug")
	}
	for i := 0; i < reps; i++ {
		t1 = st.dense(t1)
	}
	return t1
}

func (st *state) epilogue(t1 tensor) {
	t2 := st.bnr(t1)
	t3 := st.nms.Name("pool")
	st.line("Pooling FromTensor=" + t2.name + " ToTensor=" + t3 + " Kind=AvgGlobal PaddingH=0 PaddingW=0")
	const t4 = "fc"
	st.line("FullyConnected FromTensor=" + t3 + " ToTensor=" + t4 + " ToChannels=1000")
	const t5 = "prob"
	st.line("Softmax FromTensor=" + t4 + " ToTensor=" + t5)
	st.line("Output FromTensor=" + t5)
}

func gen(variant string) []byte {
	st := &state{
		variant: variant,
		nms:     nmsrc.New(),
	}
	st.config()
	t1 := st.prologue()
	t2 := st.stage1(t1)
	t3 := st.stage2(t2)
	t4 := st.stage3(t3)
	t5 := st.stage4(t4)
	st.epilogue(t5)
	return st.text
}

func DenseNet121() []byte { return gen(denseNet121) }
func DenseNet169() []byte { return gen(denseNet169) }
func DenseNet201() []byte { return gen(denseNet201) }
func DenseNet265() []byte { return gen(denseNet265) }
