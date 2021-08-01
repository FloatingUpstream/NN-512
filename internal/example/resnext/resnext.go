package resnext

import (
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
)

const (
	resNeXt50  = "ResNeXt50"
	resNeXt101 = "ResNeXt101"
	resNeXt152 = "ResNeXt152"
)

type state struct {
	variant string
	text    []byte
	nms     nmsrc.Src
}

func (st *state) line(a string) {
	st.text = append(st.text, a...)
	st.text = append(st.text, '\n')
}

func (st *state) bn(t1 string) string {
	t2 := st.nms.Name("bn")
	st.line("BatchNorm FromTensor=" + t1 + " ToTensor=" + t2 + " Epsilon=0.00002")
	return t2
}

func (st *state) relu(t1 string) string {
	t2 := st.nms.Name("relu")
	st.line("Activation FromTensor=" + t1 + " ToTensor=" + t2 + " Kind=ReLU Param=0")
	return t2
}

func (st *state) add(t1, t2 string) string {
	t3 := st.nms.Name("add")
	st.line("Add FromTensor1=" + t1 + " FromTensor2=" + t2 + " ToTensor=" + t3)
	return t3
}

func (st *state) one(t1, toChannels, stride string) string {
	var t2 string
	switch stride {
	case "1":
		t2 = st.nms.Name("one")
	case "2":
		t2 = st.nms.Name("oneDS")
	default:
		panic("bug")
	}
	st.line("Conv FromTensor=" + t1 + " ToTensor=" + t2 + " ToChannels=" + toChannels +
		" FilterH=1 FilterW=1 StrideH=" + stride + " StrideW=" + stride +
		" PaddingH=0 PaddingW=0 DilationH=1 DilationW=1 Groups=1")
	return t2
}

func (st *state) three(t1, toChannels, stride string) string {
	var t2 string
	switch stride {
	case "1":
		t2 = st.nms.Name("three")
	case "2":
		t2 = st.nms.Name("threeDS")
	default:
		panic("bug")
	}
	st.line("Conv FromTensor=" + t1 + " ToTensor=" + t2 + " ToChannels=" + toChannels +
		" FilterH=3 FilterW=3 StrideH=" + stride + " StrideW=" + stride +
		" PaddingH=1 PaddingW=1 DilationH=1 DilationW=1 Groups=32")
	return t2
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

func (st *state) prologue() string {
	const t1 = "image"
	st.line("Input ToTensor=" + t1 + " Channels=3 Height=224 Width=224")
	t2 := st.bn(t1)
	const t3 = "sevenDS"
	st.line("Conv FromTensor=" + t2 + " ToTensor=" + t3 + " ToChannels=64" +
		" FilterH=7 FilterW=7 StrideH=2 StrideW=2 PaddingH=3 PaddingW=3 DilationH=1 DilationW=1 Groups=1")
	t4 := st.relu(st.bn(t3))
	t5 := st.nms.Name("pool")
	st.line("Pooling FromTensor=" + t4 + " ToTensor=" + t5 + " Kind=Max3x3Stride2 PaddingH=1 PaddingW=1")
	return t5
}

func (st *state) stage1(t1 string) string {
	t2 := st.bn(st.one(t1, "256", "1"))
	for i := 0; i < 3; i++ {
		t3 := st.relu(st.bn(st.one(t1, "128", "1")))
		t4 := st.relu(st.bn(st.three(t3, "128", "1")))
		t5 := st.bn(st.one(t4, "256", "1"))
		t1 = st.relu(st.add(t2, t5))
		t2 = t1
	}
	return t1
}

func (st *state) stage2(t1 string) string {
	var reps int
	switch st.variant {
	case resNeXt50, resNeXt101:
		reps = 4
	case resNeXt152:
		reps = 8
	default:
		panic("bug")
	}
	stride := "2"
	t2 := st.bn(st.one(t1, "512", stride))
	for i := 0; i < reps; i++ {
		t3 := st.relu(st.bn(st.one(t1, "256", "1")))
		t4 := st.relu(st.bn(st.three(t3, "256", stride)))
		t5 := st.bn(st.one(t4, "512", "1"))
		t1 = st.relu(st.add(t2, t5))
		stride = "1"
		t2 = t1
	}
	return t1
}

func (st *state) stage3(t1 string) string {
	var reps int
	switch st.variant {
	case resNeXt50:
		reps = 6
	case resNeXt101:
		reps = 23
	case resNeXt152:
		reps = 36
	default:
		panic("bug")
	}
	stride := "2"
	t2 := st.bn(st.one(t1, "1024", stride))
	for i := 0; i < reps; i++ {
		t3 := st.relu(st.bn(st.one(t1, "512", "1")))
		t4 := st.relu(st.bn(st.three(t3, "512", stride)))
		t5 := st.bn(st.one(t4, "1024", "1"))
		t1 = st.relu(st.add(t2, t5))
		stride = "1"
		t2 = t1
	}
	return t1
}

func (st *state) stage4(t1 string) string {
	stride := "2"
	t2 := st.bn(st.one(t1, "2048", stride))
	for i := 0; i < 3; i++ {
		t3 := st.relu(st.bn(st.one(t1, "1024", "1")))
		t4 := st.relu(st.bn(st.three(t3, "1024", stride)))
		t5 := st.bn(st.one(t4, "2048", "1"))
		t1 = st.relu(st.add(t2, t5))
		stride = "1"
		t2 = t1
	}
	return t1
}

func (st *state) epilogue(t1 string) {
	t2 := st.nms.Name("pool")
	st.line("Pooling FromTensor=" + t1 + " ToTensor=" + t2 + " Kind=AvgGlobal PaddingH=0 PaddingW=0")
	const t3 = "fc"
	st.line("FullyConnected FromTensor=" + t2 + " ToTensor=" + t3 + " ToChannels=1000")
	const t4 = "prob"
	st.line("Softmax FromTensor=" + t3 + " ToTensor=" + t4)
	st.line("Output FromTensor=" + t4)
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

func ResNeXt50() []byte  { return gen(resNeXt50) }
func ResNeXt101() []byte { return gen(resNeXt101) }
func ResNeXt152() []byte { return gen(resNeXt152) }
