package raw

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

type Node interface {
	LineNumber() int
	FromTensors() []string
	ToTensors() []string
	ParamTensors() []string
}

type Platform int

const (
	AVX512Float32 Platform = iota
)

var PlatformStrings = []string{
	AVX512Float32: "AVX512Float32",
}

type Config struct {
	LineNum                int
	Prefix                 string
	Platform               Platform
	L1DataCachePerThread   int
	L2CachePerThreadExL1   int
	L3CachePerThreadExL1L2 int
}

func (c *Config) LineNumber() int        { return c.LineNum }
func (c *Config) FromTensors() []string  { return nil }
func (c *Config) ToTensors() []string    { return nil }
func (c *Config) ParamTensors() []string { return nil }

type Input struct {
	LineNum  int
	ToTensor string
	Channels int
	Height   int
	Width    int
}

func (i *Input) LineNumber() int        { return i.LineNum }
func (i *Input) FromTensors() []string  { return nil }
func (i *Input) ToTensors() []string    { return []string{i.ToTensor} }
func (i *Input) ParamTensors() []string { return nil }

type Output struct {
	LineNum    int
	FromTensor string
}

func (o *Output) LineNumber() int        { return o.LineNum }
func (o *Output) FromTensors() []string  { return []string{o.FromTensor} }
func (o *Output) ToTensors() []string    { return nil }
func (o *Output) ParamTensors() []string { return nil }

type ActivationKind int

const (
	ReLU ActivationKind = iota
)

var ActivationStrings = []string{
	ReLU: "ReLU",
}

type Activation struct {
	LineNum    int
	FromTensor string
	ToTensor   string
	Kind       ActivationKind
	Param      float32
}

func (a *Activation) LineNumber() int        { return a.LineNum }
func (a *Activation) FromTensors() []string  { return []string{a.FromTensor} }
func (a *Activation) ToTensors() []string    { return []string{a.ToTensor} }
func (a *Activation) ParamTensors() []string { return nil }

type BatchNorm struct {
	LineNum         int
	FromTensor      string
	ToTensor        string
	MeansTensor     string
	VariancesTensor string
	ScalesTensor    string
	ShiftsTensor    string
	Epsilon         float32
}

func (b *BatchNorm) LineNumber() int       { return b.LineNum }
func (b *BatchNorm) FromTensors() []string { return []string{b.FromTensor} }
func (b *BatchNorm) ToTensors() []string   { return []string{b.ToTensor} }

func (b *BatchNorm) ParamTensors() []string {
	return []string{
		b.MeansTensor,
		b.VariancesTensor,
		b.ScalesTensor,
		b.ShiftsTensor,
	}
}

type PoolingKind int

const (
	Max2x2Stride2 PoolingKind = iota
	Max3x3Stride2
	MaxGlobal
	Avg2x2Stride2
	Avg3x3Stride2
	AvgGlobal
)

var PoolingStrings = []string{
	Max2x2Stride2: "Max2x2Stride2",
	Max3x3Stride2: "Max3x3Stride2",
	MaxGlobal:     "MaxGlobal",
	Avg2x2Stride2: "Avg2x2Stride2",
	Avg3x3Stride2: "Avg3x3Stride2",
	AvgGlobal:     "AvgGlobal",
}

type Pooling struct {
	LineNum    int
	FromTensor string
	ToTensor   string
	Kind       PoolingKind
	PaddingH   int
	PaddingW   int
}

func (p *Pooling) LineNumber() int        { return p.LineNum }
func (p *Pooling) FromTensors() []string  { return []string{p.FromTensor} }
func (p *Pooling) ToTensors() []string    { return []string{p.ToTensor} }
func (p *Pooling) ParamTensors() []string { return nil }

type Softmax struct {
	LineNum    int
	FromTensor string
	ToTensor   string
}

func (s *Softmax) LineNumber() int        { return s.LineNum }
func (s *Softmax) FromTensors() []string  { return []string{s.FromTensor} }
func (s *Softmax) ToTensors() []string    { return []string{s.ToTensor} }
func (s *Softmax) ParamTensors() []string { return nil }

type FullyConnected struct {
	LineNum       int
	FromTensor    string
	ToTensor      string
	WeightsTensor string
	BiasesTensor  string
	ToChannels    int
}

func (f *FullyConnected) LineNumber() int        { return f.LineNum }
func (f *FullyConnected) FromTensors() []string  { return []string{f.FromTensor} }
func (f *FullyConnected) ToTensors() []string    { return []string{f.ToTensor} }
func (f *FullyConnected) ParamTensors() []string { return []string{f.WeightsTensor, f.BiasesTensor} }

type Conv struct {
	LineNum       int
	FromTensor    string
	ToTensor      string
	WeightsTensor string
	BiasesTensor  string
	ToChannels    int
	FilterH       int
	FilterW       int
	StrideH       int
	StrideW       int
	PaddingH      int
	PaddingW      int
	DilationH     int
	DilationW     int
	Groups        int
}

func (c *Conv) LineNumber() int        { return c.LineNum }
func (c *Conv) FromTensors() []string  { return []string{c.FromTensor} }
func (c *Conv) ToTensors() []string    { return []string{c.ToTensor} }
func (c *Conv) ParamTensors() []string { return []string{c.WeightsTensor, c.BiasesTensor} }

type Add struct {
	LineNum     int
	FromTensor1 string
	FromTensor2 string
	ToTensor    string
}

func (a *Add) LineNumber() int        { return a.LineNum }
func (a *Add) FromTensors() []string  { return []string{a.FromTensor1, a.FromTensor2} }
func (a *Add) ToTensors() []string    { return []string{a.ToTensor} }
func (a *Add) ParamTensors() []string { return nil }

type Concat struct {
	LineNum     int
	FromTensor1 string
	FromTensor2 string
	ToTensor    string
}

func (c *Concat) LineNumber() int        { return c.LineNum }
func (c *Concat) FromTensors() []string  { return []string{c.FromTensor1, c.FromTensor2} }
func (c *Concat) ToTensors() []string    { return []string{c.ToTensor} }
func (c *Concat) ParamTensors() []string { return nil }

type Seg struct {
	Doc     string
	Label   string
	Default string
	Choices []string
	Parse   func(string) (interface{}, error)
}

type Tail struct {
	Doc   string
	Segs  []*Seg
	Parse func(int, []interface{}) Node
}

var Guide = make(map[string]*Tail)

const Binder = "="

func Parse(text string) ([]Node, error) {
	const (
		pre = "parse failed: "
		wln = pre + "line %d: "
		eg  = wln + "expected %s" + Binder + "%s (for example)"
	)
	if n := len(text); n == 0 {
		return nil, nil
	} else if text[n-1] != '\n' {
		return nil, errors.New(pre + "expected final newline")
	}
	var nodes []Node
	const (
		headSpace int = iota
		headToken
		tailSpace
		tailToken
	)
	phase := headSpace
	i, lineHead, line := 0, 0, 1
	var tail *Tail
	var vals []interface{}
	for j, jj := range text {
		if !unicode.IsSpace(jj) {
			if phase == headSpace {
				phase, i, lineHead = headToken, j, line
			} else if phase == tailSpace {
				phase, i = tailToken, j
			}
			continue
		}
		if phase == headToken {
			phase = tailSpace
			if tail = Guide[text[i:j]]; tail == nil {
				heads := make([]string, 0, len(Guide))
				for head := range Guide {
					heads = append(heads, head)
				}
				sort.Strings(heads)
				msg := fmt.Sprintf(wln+"%s", line, errExpected(heads).Error())
				return nil, errors.New(msg)
			}
		} else if phase == tailToken {
			seg := tail.Segs[len(vals)]
			parts := strings.Split(text[i:j], Binder)
			if len(parts) != 2 || parts[0] != seg.Label {
				msg := fmt.Sprintf(eg, line, seg.Label, seg.Default)
				return nil, errors.New(msg)
			}
			val, err := seg.Parse(parts[1])
			if err != nil {
				msg := fmt.Sprintf(wln+"%s: %s", line, seg.Label, err.Error())
				return nil, errors.New(msg)
			}
			vals = append(vals, val)
			if len(vals) == len(tail.Segs) {
				nodes = append(nodes, tail.Parse(lineHead, vals))
				phase, vals = headSpace, vals[:0]
			} else {
				phase = tailSpace
			}
		}
		if jj == '\n' {
			line += 1
		}
	}
	if phase == tailSpace {
		seg := tail.Segs[len(vals)]
		msg := fmt.Sprintf(eg, line, seg.Label, seg.Default)
		return nil, errors.New(msg)
	}
	return nodes, nil
}

const (
	identStr   = `^[a-zA-Z][a-zA-Z0-9]*$`
	nonNegStr  = `^0|[1-9][0-9]*$`
	posIntStr  = `^[1-9][0-9]*$`
	memSizeStr = `^([1-9][0-9]*)([km](i?b)?)?$`
	floatStr   = `^-?(0|[1-9][0-9]*)(\.[0-9]+)?$`
)

var (
	identRE   = regexp.MustCompile(identStr)
	nonNegRE  = regexp.MustCompile(nonNegStr)
	posIntRE  = regexp.MustCompile(posIntStr)
	memSizeRE = regexp.MustCompile(memSizeStr)
	floatRE   = regexp.MustCompile(floatStr)
)

const (
	identDoc   = "Must be a letter followed by zero or more letters/digits: " + identStr
	nonNegDoc  = "Must be a non-negative integer: " + nonNegStr
	posIntDoc  = "Must be a positive integer: " + posIntStr
	memSizeDoc = "A positive integer with an optional suffix like k, K, KB, KiB, m, M, MB, MiB. " +
		"The K suffixes multiply by 1024. The M suffixes multiply by the square of 1024. " +
		"After conversion to lowercase: " + memSizeStr
	floatDoc = "Must be a simple float: " + floatStr
)

const (
	affixData      = "Data"
	affixMeans     = "Means"
	affixVariances = "Variances"
	affixScales    = "Scales"
	affixShifts    = "Shifts"
	affixWeights   = "Weights"
	affixBiases    = "Biases"
)

var (
	errGap      = errors.New("unexpected gap after " + Binder)
	errRejected = errors.New("rejected")
)

func errMatch(a, b string) error {
	return errors.New(a + "does not match " + b)
}

func errExpected(a []string) error {
	return errors.New("expected " + strings.Join(a, " or "))
}

func ident(a, b string) (interface{}, error) {
	if !identRE.MatchString(a) {
		if a == "" {
			return nil, errGap
		}
		return nil, errMatch("", identStr)
	}
	return a + b, nil
}

func nonNeg(a string, r int) (interface{}, error) {
	if !nonNegRE.MatchString(a) {
		if a == "" {
			return nil, errGap
		}
		return nil, errMatch("", nonNegStr)
	}
	n, err := strconv.Atoi(a)
	if err != nil {
		return nil, err
	}
	if n >= r {
		return nil, errRejected
	}
	return n, nil
}

func posInt(a string, r int) (interface{}, error) {
	if !posIntRE.MatchString(a) {
		if a == "" {
			return nil, errGap
		}
		return nil, errMatch("", posIntStr)
	}
	n, err := strconv.Atoi(a)
	if err != nil {
		return nil, err
	}
	if n >= r {
		return nil, errRejected
	}
	return n, nil
}

func memSize(a string, r int) (interface{}, error) {
	aa := memSizeRE.FindStringSubmatch(strings.ToLower(a))
	if aa == nil {
		if a == "" {
			return nil, errGap
		}
		return nil, errMatch("as lowercase, ", memSizeStr)
	}
	n, err := strconv.Atoi(aa[1])
	if err != nil {
		return nil, err
	}
	if n >= r {
		return nil, errRejected
	}
	if aa[2] != "" {
		n <<= 10
		if aa[2][0] == 'm' {
			n <<= 10
		}
		if n >= r {
			return nil, errRejected
		}
	}
	return n, nil
}

func float(a string) (interface{}, error) {
	if !floatRE.MatchString(a) {
		if a == "" {
			return nil, errGap
		}
		return nil, errMatch("", floatStr)
	}
	n, err := strconv.ParseFloat(a, 32)
	if err != nil {
		return nil, err
	}
	return float32(n), nil
}

func fromTensor(x, y, z string) *Seg {
	return &Seg{
		Doc: "Read from a pre-existing data tensor with this name. " + x +
			identDoc,
		Label:   "FromTensor" + y,
		Default: "from" + y,
		Parse: func(a string) (interface{}, error) {
			return ident(a, z)
		},
	}
}

func toTensor(x, y, z string) *Seg {
	return &Seg{
		Doc: "Write to a new data tensor with this name. " + x +
			identDoc,
		Label:   "ToTensor" + y,
		Default: "to" + y,
		Parse: func(a string) (interface{}, error) {
			return ident(a, z)
		},
	}
}

func paddingH(d string) *Seg {
	return &Seg{
		Doc: "Implicit heightwise padding of FromTensor. " +
			"This is the number of all-zero rows to implicitly concatenate " +
			"at the top of each feature map, before the first explicit row. " +
			"The same number of all-zero rows is implicitly concatenated " +
			"at the bottom of each feature map, after the last explicit row. " +
			nonNegDoc,
		Label:   "PaddingH",
		Default: d,
		Parse: func(a string) (interface{}, error) {
			return nonNeg(a, 1<<48)
		},
	}
}

func paddingW(d string) *Seg {
	return &Seg{
		Doc: "Implicit widthwise padding of FromTensor. " +
			"This is the number of all-zero columns to implicitly concatenate " +
			"on the left side of each feature map, before the first explicit column. " +
			"The same number of all-zero columns is implicitly concatenated " +
			"on the right side of each feature map, after the last explicit column. " +
			nonNegDoc,
		Label:   "PaddingW",
		Default: d,
		Parse: func(a string) (interface{}, error) {
			return nonNeg(a, 1<<48)
		},
	}
}

func initConfigPrefix() *Seg {
	return &Seg{
		Doc: "A string used for filenames, function names, etc. " +
			identDoc,
		Label:   "Prefix",
		Default: "NN512",
		Parse: func(a string) (interface{}, error) {
			return ident(a, "")
		},
	}
}

func initConfigPlatform() *Seg {
	return &Seg{
		Doc: "The kind of C99 code to generate. " +
			PlatformStrings[AVX512Float32] + " denotes " +
			"x86-64 AVX-512 Foundation (AVX512F) and 32-bit floating point.",
		Label:   "Platform",
		Default: PlatformStrings[AVX512Float32],
		Choices: PlatformStrings,
		Parse: func(a string) (interface{}, error) {
			for i, s := range PlatformStrings {
				if a == s {
					return Platform(i), nil
				}
			}
			if a == "" {
				return nil, errGap
			}
			return nil, errExpected(PlatformStrings)
		},
	}
}

func initConfigL1DataCachePerThread() *Seg {
	return &Seg{
		Doc: "Size in bytes of each L1D cache divided by the number of threads that share each L1D cache. " +
			memSizeDoc,
		Label:   "L1DataCachePerThread",
		Default: "32KiB",
		Parse: func(a string) (interface{}, error) {
			return memSize(a, 1<<30)
		},
	}
}

func initConfigL2CachePerThreadExL1() *Seg {
	return &Seg{
		Doc: "Size in bytes of each L2 cache divided by the number of threads that share each L2 cache. " +
			"This size must exclude the L1 overlap if L2 is inclusive. " +
			memSizeDoc,
		Label:   "L2CachePerThreadExL1",
		Default: "960KiB",
		Parse: func(a string) (interface{}, error) {
			return memSize(a, 1<<30)
		},
	}
}

func initConfigL3CachePerThreadExL1L2() *Seg {
	return &Seg{
		Doc: "Size in bytes of the L3 cache divided by the number of threads that share the L3 cache. " +
			"This size must exclude the L1/L2 overlap if L3 is inclusive. " +
			memSizeDoc,
		Label:   "L3CachePerThreadExL1L2",
		Default: "1408KiB",
		Parse: func(a string) (interface{}, error) {
			return memSize(a, 1<<30)
		},
	}
}

func initConfig() {
	Guide["Config"] = &Tail{
		Doc: "Settings for the code generator.",
		Segs: []*Seg{
			initConfigPrefix(),
			initConfigPlatform(),
			initConfigL1DataCachePerThread(),
			initConfigL2CachePerThreadExL1(),
			initConfigL3CachePerThreadExL1L2(),
		},
		Parse: func(l int, a []interface{}) Node {
			return &Config{
				LineNum:                l,
				Prefix:                 a[0].(string),
				Platform:               a[1].(Platform),
				L1DataCachePerThread:   a[2].(int),
				L2CachePerThreadExL1:   a[3].(int),
				L3CachePerThreadExL1L2: a[4].(int),
			}
		},
	}
}

func initInputToTensor() *Seg {
	return &Seg{
		Doc: "A name for this input data tensor. " +
			"The corresponding inference function parameter in the generated code " +
			"has the same name but with \"" + affixData + "\" appended. " +
			identDoc,
		Label:   "ToTensor",
		Default: "image",
		Parse: func(a string) (interface{}, error) {
			return ident(a, affixData)
		},
	}
}

func initInputChannels() *Seg {
	return &Seg{
		Doc: "The number of feature maps for this input data tensor. " +
			"This is the C in CHW (the outermost/slowest dimension) " +
			"and has a stride of H*W*sizeof(float) bytes. " +
			posIntDoc,
		Label:   "Channels",
		Default: "3",
		Parse: func(a string) (interface{}, error) {
			return posInt(a, 1<<48)
		},
	}
}

func initInputHeight() *Seg {
	return &Seg{
		Doc: "The spatial height dimension of this input data tensor. " +
			"For an image tensor the height is usually the number of pixel rows. " +
			"This is the H in CHW (the outermost/slowest spatial dimension) " +
			"and has a stride of W*sizeof(float) bytes. " +
			posIntDoc,
		Label:   "Height",
		Default: "224",
		Parse: func(a string) (interface{}, error) {
			return posInt(a, 1<<48)
		},
	}
}

func initInputWidth() *Seg {
	return &Seg{
		Doc: "The spatial width dimension of this input data tensor. " +
			"For an image tensor the width is usually the number of pixels per row. " +
			"This is the W in CHW (the innermost/fastest dimension) " +
			"and has a stride of sizeof(float) bytes. " +
			posIntDoc,
		Label:   "Width",
		Default: "224",
		Parse: func(a string) (interface{}, error) {
			return posInt(a, 1<<48)
		},
	}
}

func initInput() {
	Guide["Input"] = &Tail{
		Doc: "Declare an input data tensor parameter for the generated inference function. " +
			"Input data must be in CHW format, 32-bit floating point, fully packed. " +
			"The inference code reads the input tensor memory but never writes to it.",
		Segs: []*Seg{
			initInputToTensor(),
			initInputChannels(),
			initInputHeight(),
			initInputWidth(),
		},
		Parse: func(l int, a []interface{}) Node {
			return &Input{
				LineNum:  l,
				ToTensor: a[0].(string),
				Channels: a[1].(int),
				Height:   a[2].(int),
				Width:    a[3].(int),
			}
		},
	}
}

func initOutputFromTensor() *Seg {
	return &Seg{
		Doc: "The name of a data tensor that will be written back to the user as output. " +
			"Must not be the name of an input tensor and must not be the same as another output. " +
			"The corresponding inference function parameter in the generated code " +
			"has a matching name but with \"" + affixData + "\" appended. " +
			identDoc,
		Label:   "FromTensor",
		Default: "prob",
		Parse: func(a string) (interface{}, error) {
			return ident(a, affixData)
		},
	}
}

func initOutput() {
	Guide["Output"] = &Tail{
		Doc: "Declare an output data tensor parameter for the generated inference function. " +
			"The user allocates output tensor memory and passes a pointer into the inference function. " +
			"There, output data is written in CHW format, 32-bit floating point, fully packed.",
		Segs: []*Seg{
			initOutputFromTensor(),
		},
		Parse: func(l int, a []interface{}) Node {
			return &Output{
				LineNum:    l,
				FromTensor: a[0].(string),
			}
		},
	}
}

func initActivationKind() *Seg {
	return &Seg{
		Doc: "The kind of activation function to apply. " +
			ActivationStrings[ReLU] + " means that if X is positive then F(X)=X else F(X)=X*C " +
			"where C is the constant negative slope parameter.",
		Label:   "Kind",
		Default: ActivationStrings[ReLU],
		Choices: ActivationStrings,
		Parse: func(a string) (interface{}, error) {
			for i, s := range ActivationStrings {
				if a == s {
					return ActivationKind(i), nil
				}
			}
			if a == "" {
				return nil, errGap
			}
			return nil, errExpected(ActivationStrings)
		},
	}
}

func initActivationParam() *Seg {
	return &Seg{
		Doc: "A parameter for the activation function. " +
			"For " + ActivationStrings[ReLU] + " this is the negative slope parameter " +
			"(0 gives standard ReLU, 0.1 gives a typical leaky ReLU, " +
			"-1 gives absolute value, 1 gives the identity function, etc.). " +
			floatDoc,
		Label:   "Param",
		Default: "0",
		Parse: func(a string) (interface{}, error) {
			return float(a)
		},
	}
}

func initActivation() {
	Guide["Activation"] = &Tail{
		Doc: "Generate code to apply an elementwise activation function.",
		Segs: []*Seg{
			fromTensor("", "", affixData),
			toTensor("", "", affixData),
			initActivationKind(),
			initActivationParam(),
		},
		Parse: func(l int, a []interface{}) Node {
			return &Activation{
				LineNum:    l,
				FromTensor: a[0].(string),
				ToTensor:   a[1].(string),
				Kind:       a[2].(ActivationKind),
				Param:      a[3].(float32),
			}
		},
	}
}

func initBatchNormEpsilon() *Seg {
	return &Seg{
		Doc: "A small positive number added to the variance to avoid division by zero. " +
			"Should match the value that was used for this purpose during training. " +
			floatDoc,
		Label:   "Epsilon",
		Default: "0.001",
		Parse: func(a string) (interface{}, error) {
			f, err := float(a)
			if err == nil && f.(float32) < 1e-5 {
				return nil, errRejected
			}
			return f, err
		},
	}
}

func initBatchNorm() {
	Guide["BatchNorm"] = &Tail{
		Doc: "Generate code to apply batch normalization with per-channel mean, variance, scale, and shift parameters. " +
			"Let X be an element of FromTensor and let Y be the corresponding element of ToTensor that will be computed. " +
			"X and Y are at the same CHW coordinate in their respective tensors and the channel part of that coordinate " +
			"selects a mean M, a variance V, a scale S, and a shift H. Then Y=S*(X-M)/SQRT(V+E)+H " +
			"where E is the constant epsilon parameter (to avoid division by zero).",
		Segs: []*Seg{
			fromTensor("", "", affixData),
			toTensor("The user passes the mean, variance, scale, and shift parameter tensors into the "+
				"generated initialization code through struct fields that have this same name but "+
				"with \""+affixMeans+"\", \""+affixVariances+"\", \""+affixScales+"\", and \""+affixShifts+"\" appended "+
				"(each of these parameter tensors is an array of 32-bit floats, one float per data tensor channel). ",
				"", ""),
			initBatchNormEpsilon(),
		},
		Parse: func(l int, a []interface{}) Node {
			t := a[1].(string)
			return &BatchNorm{
				LineNum:         l,
				FromTensor:      a[0].(string),
				ToTensor:        t + affixData,
				MeansTensor:     t + affixMeans,
				VariancesTensor: t + affixVariances,
				ScalesTensor:    t + affixScales,
				ShiftsTensor:    t + affixShifts,
				Epsilon:         a[2].(float32),
			}
		},
	}
}

func initPoolingKind() *Seg {
	return &Seg{
		Doc: "The kind of pooling operation to apply. " +
			PoolingStrings[Max2x2Stride2] + " and " + PoolingStrings[Avg2x2Stride2] + " produce a single value " +
			"for each 2x2 window and there is no overlap between adjacent windows. " +
			PoolingStrings[Max3x3Stride2] + " and " + PoolingStrings[Avg3x3Stride2] + " produce a single value " +
			"for each 3x3 window and adjacent windows overlap. " +
			PoolingStrings[MaxGlobal] + " and " + PoolingStrings[AvgGlobal] + " produce a single value " +
			"for each feature map.",
		Label:   "Kind",
		Default: PoolingStrings[Max2x2Stride2],
		Choices: PoolingStrings,
		Parse: func(a string) (interface{}, error) {
			for i, s := range PoolingStrings {
				if a == s {
					return PoolingKind(i), nil
				}
			}
			if a == "" {
				return nil, errGap
			}
			return nil, errExpected(PoolingStrings)
		},
	}
}

func initPooling() {
	Guide["Pooling"] = &Tail{
		Doc: "Generate code to apply a standard window pooling or global pooling operation. " +
			"Padding affects window placement but padding values never participate in max/avg calculations. " +
			"Therefore the padding must be small enough that every window will contain at least one non-padding value. " +
			"Each (H+2*PaddingH)x(W+2*PaddingW) feature map in FromTensor yields a corresponding feature map in ToTensor. " +
			"For RxR window pooling with a stride of S the height of every feature map in ToTensor is ((H+2*PaddingH)-R)/S+1 " +
			"where the division by S truncates toward zero; the dividend must not be negative. The width formula is analogous. " +
			"For global pooling there is no padding and every feature map in ToTensor is 1x1.",
		Segs: []*Seg{
			fromTensor("", "", affixData),
			toTensor("", "", affixData),
			initPoolingKind(),
			paddingH("0"),
			paddingW("0"),
		},
		Parse: func(l int, a []interface{}) Node {
			return &Pooling{
				LineNum:    l,
				FromTensor: a[0].(string),
				ToTensor:   a[1].(string),
				Kind:       a[2].(PoolingKind),
				PaddingH:   a[3].(int),
				PaddingW:   a[4].(int),
			}
		},
	}
}

func initSoftmax() {
	Guide["Softmax"] = &Tail{
		Doc: "Generate code to compute softmax along the channel dimension " +
			"independently for each spatial (height, width) location. " +
			"FromTensor and ToTensor have the same number of channels, " +
			"the same height, and the same width.",
		Segs: []*Seg{
			fromTensor("", "", affixData),
			toTensor("", "", affixData),
		},
		Parse: func(l int, a []interface{}) Node {
			return &Softmax{
				LineNum:    l,
				FromTensor: a[0].(string),
				ToTensor:   a[1].(string),
			}
		},
	}
}

func initFullyConnectedToChannels() *Seg {
	return &Seg{
		Doc: "The number of feature maps in ToTensor (each feature map has height 1 and width 1). " +
			"This is also the number of filters in the weight parameter tensor (the K in KCHW) " +
			"and the number of biases in the bias parameter tensor. " +
			posIntDoc,
		Label:   "ToChannels",
		Default: "1000",
		Parse: func(a string) (interface{}, error) {
			return posInt(a, 1<<48)
		},
	}
}

func initFullyConnected() {
	Guide["FullyConnected"] = &Tail{
		Doc: "Generate code to implement a fully connected layer. " +
			"Suppose FromTensor has C channels, height H, and width W. " +
			"The weight parameter tensor consists of K filters (where K is the ToChannels parameter) " +
			"and each filter is structurally identical to FromTensor (C channels, height H, width W). " +
			"The weight parameter tensor is in KCHW format, 32-bit floating point, fully packed " +
			"(filter number is the outermost/slowest dimension; the rest is like an input data tensor). " +
			"Each filter element is multiplied by the FromTensor element that has the same CHW coordinate. " +
			"The bias parameter tensor is an array of K 32-bit floats (one float for each filter), " +
			"fully packed. ToTensor has K channels, height 1, and width 1.",
		Segs: []*Seg{
			fromTensor("", "", affixData),
			toTensor("The user passes the weight parameter tensor into the generated initialization code "+
				"through a struct field that has this same name but with \""+affixWeights+"\" appended. "+
				"Similarly the bias parameter tensor (\""+affixBiases+"\" is appended). ", "", ""),
			initFullyConnectedToChannels(),
		},
		Parse: func(l int, a []interface{}) Node {
			t := a[1].(string)
			return &FullyConnected{
				LineNum:       l,
				FromTensor:    a[0].(string),
				ToTensor:      t + affixData,
				WeightsTensor: t + affixWeights,
				BiasesTensor:  t + affixBiases,
				ToChannels:    a[2].(int),
			}
		},
	}
}

func initConvToChannels() *Seg {
	return &Seg{
		Doc: "The number of feature maps in ToTensor. " +
			"This is also the number of filters in the weight parameter tensor (the K in KCHW) " +
			"and the number of biases in the bias parameter tensor. " +
			posIntDoc,
		Label:   "ToChannels",
		Default: "64",
		Parse: func(a string) (interface{}, error) {
			return posInt(a, 1<<48)
		},
	}
}

func initConvFilter(x, y string) *Seg {
	return &Seg{
		Doc: "The undilated spatial " + x + " of each filter in " +
			"the weight parameter tensor (the " + y + " in KCHW). " +
			posIntDoc,
		Label:   "Filter" + y,
		Default: "3",
		Parse: func(a string) (interface{}, error) {
			return posInt(a, 1<<24)
		},
	}
}

func initConvStride(x, y, z string) *Seg {
	return &Seg{
		Doc: "The " + x + " step between adjacent " + y + " of the filtering position grid " +
			"(the " + x + " subsampling ratio). " +
			posIntDoc,
		Label:   "Stride" + z,
		Default: "1",
		Parse: func(a string) (interface{}, error) {
			return posInt(a, 1<<48)
		},
	}
}

func initConvDilation(x, y, z string) *Seg {
	return &Seg{
		Doc: "The " + x + " filter dilation factor. 1 means no dilation (ordinary cross-correlation). " +
			"2 means the filter is multiplied against FromTensor in a spatially sparse (spread out) way " +
			"just as if one all-zero " + y + " had been inserted between each pair of adjacent " + y + "s " +
			"in the filter. 3 is like if two all-zero " + y + "s had been inserted. And so on. " +
			posIntDoc,
		Label:   "Dilation" + z,
		Default: "1",
		Parse: func(a string) (interface{}, error) {
			return posInt(a, 1<<24)
		},
	}
}

func initConvGroups() *Seg {
	return &Seg{
		Doc: "The number of disjoint cross-correlation operations to perform (no shared data, " +
			"no shared filters). Suppose FromTensor has C channels and ToTensor has K channels " +
			"(ToChannels is K). Let G be the number of groups (both C and K must be divisible " +
			"by G). Then there are K filters in the weight parameter tensor and each of them " +
			"has C/G channels. The first operation applies the first K/G filters to the first " +
			"C/G FromTensor channels to produce the first K/G ToTensor channels. The second " +
			"operation applies the second K/G filters to the second C/G FromTensor channels " +
			"to produce the second K/G ToTensor channels. And so on. " +
			posIntDoc,
		Label:   "Groups",
		Default: "1",
		Parse: func(a string) (interface{}, error) {
			return posInt(a, 1<<48)
		},
	}
}

func initConv() {
	Guide["Conv"] = &Tail{
		Doc: "Generate code to perform cross-correlation. Suppose FromTensor has C channels, height H, " +
			"and width W. ToTensor has K (= ToChannels) channels. A formula for the height of ToTensor is " +
			"((H+2*PaddingH)-(1+(FilterH-1)*DilationH))/StrideH+1 in which the division truncates toward " +
			"zero and the dividend must not be negative. The width of ToTensor is calculated analogously. " +
			"There are K filters in the weight parameter tensor and each of them has C/Groups channels, " +
			"a height of FilterH, and a width of FilterW. The weight parameter tensor is in KCHW format, " +
			"32-bit floating point, fully packed (filter number is the outermost/slowest dimension and " +
			"otherwise the layout is just like an input data tensor). The bias parameter tensor is an " +
			"array of K 32-bit floats (one float for each filter), fully packed.",
		Segs: []*Seg{
			fromTensor("", "", affixData),
			toTensor("The user passes the weight parameter tensor into the generated initialization code "+
				"through a struct field that has this same name but with \""+affixWeights+"\" appended. "+
				"Similarly the bias parameter tensor (\""+affixBiases+"\" is appended). ", "", ""),
			initConvToChannels(),
			initConvFilter("height", "H"),
			initConvFilter("width", "W"),
			initConvStride("heightwise", "rows", "H"),
			initConvStride("widthwise", "columns", "W"),
			paddingH("1"),
			paddingW("1"),
			initConvDilation("heightwise", "row", "H"),
			initConvDilation("widthwise", "column", "W"),
			initConvGroups(),
		},
		Parse: func(l int, a []interface{}) Node {
			t := a[1].(string)
			return &Conv{
				LineNum:       l,
				FromTensor:    a[0].(string),
				ToTensor:      t + affixData,
				WeightsTensor: t + affixWeights,
				BiasesTensor:  t + affixBiases,
				ToChannels:    a[2].(int),
				FilterH:       a[3].(int),
				FilterW:       a[4].(int),
				StrideH:       a[5].(int),
				StrideW:       a[6].(int),
				PaddingH:      a[7].(int),
				PaddingW:      a[8].(int),
				DilationH:     a[9].(int),
				DilationW:     a[10].(int),
				Groups:        a[11].(int),
			}
		},
	}
}

func initAdd() {
	Guide["Add"] = &Tail{
		Doc: "Generate code for the elementwise addition of two data tensors. " +
			"FromTensor1, FromTensor2, and ToTensor are all structurally identical " +
			"(same number of channels, same height, same width).",
		Segs: []*Seg{
			fromTensor("", "1", affixData),
			fromTensor("", "2", affixData),
			toTensor("", "", affixData),
		},
		Parse: func(l int, a []interface{}) Node {
			return &Add{
				LineNum:     l,
				FromTensor1: a[0].(string),
				FromTensor2: a[1].(string),
				ToTensor:    a[2].(string),
			}
		},
	}
}

func initConcat() {
	Guide["Concat"] = &Tail{
		Doc: "Generate code to concatenate two tensors along the channel dimension. " +
			"FromTensor1 and FromTensor2 must have matching spatial extents " +
			"(the same height H and the same width W). " +
			"If FromTensor1 has C1 channels and FromTensor2 has C2 channels " +
			"then ToTensor has C1+C2 channels, height H, and width W. " +
			"The feature maps of FromTensor1 go first (they are assigned channel numbers starting with zero) " +
			"and the feature maps of FromTensor2 go next (they are assigned channel numbers starting with C1).",
		Segs: []*Seg{
			fromTensor("The feature maps of this tensor get the low channel numbers in ToTensor. ", "1", affixData),
			fromTensor("The feature maps of this tensor get the high channel numbers in ToTensor. ", "2", affixData),
			toTensor("", "", affixData),
		},
		Parse: func(l int, a []interface{}) Node {
			return &Concat{
				LineNum:     l,
				FromTensor1: a[0].(string),
				FromTensor2: a[1].(string),
				ToTensor:    a[2].(string),
			}
		},
	}
}

func init() {
	initConfig()
	initInput()
	initOutput()
	initActivation()
	initBatchNorm()
	initPooling()
	initSoftmax()
	initFullyConnected()
	initConv()
	initAdd()
	initConcat()
}
