package compile

import (
	"NN-512/internal/compile/author"
	"NN-512/internal/compile/plan"
	"NN-512/internal/raw"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"
)

type Result struct {
	Name string
	H, C []byte
}

func Compile(text string) (*Result, error) {
	nodes, err := raw.Parse(text)
	if err != nil {
		return nil, err
	}
	st := &state{
		nodes: nodes,
	}
	if err := st.stages(); err != nil {
		return nil, errors.New("compile failed: " + err.Error())
	}
	return &Result{
		Name: st.config.Prefix,
		H:    st.h,
		C:    st.c,
	}, nil
}

const origNameShield = "A"

func origName(tensor string) string {
	i := strings.LastIndexFunc(tensor, unicode.IsUpper)
	return tensor[:i]
}

func anError(msg, tensor string, lines ...int) error {
	var pre string
	if n := len(lines); n != 0 {
		if n > 2 {
			panic("bug")
		}
		l0 := lines[0]
		if n == 1 || l0 == lines[1] {
			pre = fmt.Sprintf("line %d: ", l0)
		} else {
			l1 := lines[1]
			if l0 > l1 {
				l0, l1 = l1, l0
			}
			pre = fmt.Sprintf("lines %d and %d: ", l0, l1)
		}
	}
	if tensor != "" {
		pre += origName(tensor) + ": "
	}
	return errors.New(pre + msg)
}

type arc struct {
	tensor string
	attach int
}

type shape struct {
	tensor string
	nchw   [4]int
}

type state struct {
	nodes   []raw.Node
	config  *raw.Config
	inputs  []int
	outputs []int
	fanins  [][]arc
	fanouts [][]arc
	shapes  [][]shape
	plan    plan.Plan
	h, c    []byte
}

var stages = [...]func(*state) error{
	(*state).stage1,
	(*state).stage2,
	(*state).stage3,
	(*state).stage4,
	(*state).stage5,
	(*state).stage6,
	(*state).stage7,
	(*state).stage8,
	(*state).stage9,
	(*state).stage10,
	(*state).stage11,
	(*state).stage12,
	(*state).stage13,
	(*state).stage14,
	(*state).stage15,
	(*state).stage16,
	(*state).stage17,
}

func (st *state) stages() error {
	for _, stage := range &stages {
		if err := stage(st); err != nil {
			return err
		}
	}
	return nil
}

func (st *state) stage1() error {
	for i, node := range st.nodes {
		if config, ok := node.(*raw.Config); ok {
			if st.config != nil {
				return anError("second Config", "", config.LineNum)
			}
			st.config = config
			copy(st.nodes[1:], st.nodes[:i])
			st.nodes = st.nodes[1:]
		}
	}
	if st.config == nil {
		return anError("no Config", "")
	}
	return nil
}

func (st *state) stage2() error {
	const directly = "Input is directly connected to Output"
	seen := make(map[string]int)
	for i, node := range st.nodes {
		switch at := node.(type) {
		case *raw.Input:
			to := at.ToTensor
			if prev := seen[to]; prev == 0 {
				seen[to] = -at.LineNum
				st.inputs = append(st.inputs, i)
			} else if prev < 0 {
				return anError("Inputs have the same ToTensor", to, -prev, at.LineNum)
			} else {
				return anError(directly, to, prev, at.LineNum)
			}
		case *raw.Output:
			from := at.FromTensor
			if prev := seen[from]; prev == 0 {
				seen[from] = at.LineNum
				st.outputs = append(st.outputs, i)
			} else if prev < 0 {
				return anError(directly, from, -prev, at.LineNum)
			} else {
				return anError("Outputs have the same FromTensor", from, prev, at.LineNum)
			}
		}
	}
	if len(st.inputs) == 0 {
		return anError("no Input", "")
	}
	if len(st.outputs) == 0 {
		return anError("no Output", "")
	}
	return nil
}

func (st *state) stage3() error {
	n := len(st.nodes)
	use := make(map[string][]int, n)
	gen := make(map[string]int, n)
	for i, node := range st.nodes {
		for _, tensor := range node.FromTensors() {
			use[tensor] = append(use[tensor], i)
		}
		for _, tensor := range node.ToTensors() {
			if j, ok := gen[tensor]; ok {
				ii, jj := node.LineNumber(), st.nodes[j].LineNumber()
				return anError("tensor is produced more than once", tensor, ii, jj)
			}
			gen[tensor] = i
		}
	}
	st.fanins = make([][]arc, n)
	st.fanouts = make([][]arc, n)
	insert := func(fan *[]arc, z arc) {
		for _, have := range *fan {
			if have == z {
				return
			}
		}
		*fan = append(*fan, z)
	}
	for i, node := range st.nodes {
		for _, tensor := range node.FromTensors() {
			if j, ok := gen[tensor]; ok {
				if i == j {
					return anError("self-loop", tensor, node.LineNumber())
				}
				insert(&st.fanins[i], arc{tensor, j})
				continue
			}
			line := node.LineNumber()
			return anError("tensor is consumed but never produced", tensor, line)
		}
		for _, tensor := range node.ToTensors() {
			for _, j := range use[tensor] {
				insert(&st.fanouts[i], arc{tensor, j})
			}
		}
	}
	return nil
}

func (st *state) stage4() error {
	const (
		white byte = iota
		gray
		black
	)
	type frame struct {
		to   []arc
		from int
	}
	v := 0
	n := len(st.nodes)
	color := make([]byte, n)
	stack := make([]frame, n)
	for _, i := range st.outputs {
		v += 1
		color[i] = gray
		stack[0] = frame{st.fanins[i], i}
		for j := 0; j >= 0; {
			top := &stack[j]
			if len(top.to) == 0 {
				color[top.from] = black
				j -= 1
				continue
			}
			arc0 := top.to[0]
			top.to = top.to[1:]
			k := arc0.attach
			if color[k] == white {
				v += 1
				color[k] = gray
				j += 1
				stack[j] = frame{st.fanins[k], k}
				continue
			}
			if color[k] == black {
				continue
			}
			prev := st.nodes[k].LineNumber()
			curr := st.nodes[top.from].LineNumber()
			return anError("circular dependency", arc0.tensor, prev, curr)
		}
	}
	if v == n {
		return nil
	}
	for i := n - 1; ; i-- {
		if color[i] == white {
			node := st.nodes[i]
			line := node.LineNumber()
			tensor := node.ToTensors()[0]
			return anError("no path to an Output", tensor, line)
		}
	}
}

func (st *state) stage5FeedsOutput(i int) bool {
	fanout := st.fanouts[i]
	for j := range fanout {
		a := fanout[j].attach
		if _, ok := st.nodes[a].(*raw.Output); ok {
			return true
		}
	}
	return false
}

func (st *state) stage5DetachFanin(i int) {
	fanin := st.fanins[i]
	st.fanins[i] = nil
	for j := range fanin {
		a := fanin[j].attach
		fanout := st.fanouts[a]
		n := len(fanout)
		for k := 0; k < n; k++ {
			if fanout[k].attach == i {
				n -= 1
				fanout[k] = fanout[n]
				k -= 1
			}
		}
		st.fanouts[a] = fanout[:n]
	}
}

func (st *state) stage5Unique(a []arc) []arc {
	n := len(a)
	for i := 1; i < n; i++ {
		for j := 0; j < i; j++ {
			if a[j] == a[i] {
				n -= 1
				a[i] = a[n]
				i -= 1
				break
			}
		}
	}
	return a[:n]
}

func (st *state) stage5Replace(this, with int) {
	st.stage5DetachFanin(this)
	tensors1 := st.nodes[this].ToTensors()
	tensors2 := st.nodes[with].ToTensors()
	match := func(tensor string) string {
		for i, ii := range tensors1 {
			if ii == tensor {
				return tensors2[i]
			}
		}
		return tensor
	}
	fanout1 := st.fanouts[this]
	fanout2 := st.fanouts[with]
	for i := range fanout1 {
		a := fanout1[i].attach
		fanin := st.fanins[a]
		for j := range fanin {
			if fanin[j].attach == this {
				tensor := match(fanin[j].tensor)
				fanin[j] = arc{tensor, with}
				fanout2 = append(fanout2, arc{tensor, a})
			}
		}
		st.fanins[a] = st.stage5Unique(fanin)
	}
	st.nodes[this] = nil
	st.fanouts[this] = nil
	st.fanouts[with] = st.stage5Unique(fanout2)
	for i := range fanout1 {
		a := fanout1[i].attach
		switch aa := st.nodes[a].(type) {
		case *raw.Activation:
			aa.FromTensor = match(aa.FromTensor)
		case *raw.Add:
			aa.FromTensor1 = match(aa.FromTensor1)
			aa.FromTensor2 = match(aa.FromTensor2)
		case *raw.BatchNorm:
			aa.FromTensor = match(aa.FromTensor)
		case *raw.Concat:
			aa.FromTensor1 = match(aa.FromTensor1)
			aa.FromTensor2 = match(aa.FromTensor2)
		case *raw.Conv:
			aa.FromTensor = match(aa.FromTensor)
		case *raw.FullyConnected:
			aa.FromTensor = match(aa.FromTensor)
		case *raw.Pooling:
			aa.FromTensor = match(aa.FromTensor)
		case *raw.Softmax:
			aa.FromTensor = match(aa.FromTensor)
		default:
			panic("bug")
		}
	}
}

func (st *state) stage5() error {
	const (
		activation int = iota
		add
		concat
		pooling
		softmax
	)
	type sig struct {
		s1, s2 string
		i1, i2 int
		i3, i4 int
	}
	n := len(st.nodes)
	seen := make([]bool, n)
	repl := make(map[sig]int, n)
	var dedup func(int)
	dedup = func(i int) {
		fanin := st.fanins[i]
		saved := make([]int, len(fanin))
		for j := range fanin {
			saved[j] = fanin[j].attach
		}
		for _, j := range saved {
			if !seen[j] {
				seen[j] = true
				dedup(j)
			}
		}
		var subst int
		switch at := st.nodes[i].(type) {
		case *raw.Activation:
			i3 := int(math.Float32bits(at.Param))
			sg := sig{
				s1: at.FromTensor,
				i1: activation,
				i2: int(at.Kind),
				i3: i3,
			}
			if subst = repl[sg]; subst == 0 {
				repl[sg] = i + 1
				if at.Kind == raw.ReLU {
					sg.i3 = -1
					if subst = repl[sg]; subst == 0 {
						if at.Param <= 0 {
							sg.s1 = at.ToTensor
							repl[sg] = i + 1
						}
					} else {
						sg.i3 = i3
						delete(repl, sg)
					}
				}
			}
		case *raw.Add:
			sg := sig{s1: at.FromTensor1, s2: at.FromTensor2, i1: add}
			if subst = repl[sg]; subst == 0 {
				repl[sg] = i + 1
				sg.s1, sg.s2 = sg.s2, sg.s1
				repl[sg] = i + 1
			}
		case *raw.Concat:
			sg := sig{s1: at.FromTensor1, s2: at.FromTensor2, i1: concat}
			if subst = repl[sg]; subst == 0 {
				repl[sg] = i + 1
			}
		case *raw.Pooling:
			sg := sig{
				s1: at.FromTensor,
				i1: pooling,
				i2: int(at.Kind),
				i3: at.PaddingH,
				i4: at.PaddingW,
			}
			if subst = repl[sg]; subst == 0 {
				repl[sg] = i + 1
			}
		case *raw.Softmax:
			sg := sig{s1: at.FromTensor, i1: softmax}
			if subst = repl[sg]; subst == 0 {
				repl[sg] = i + 1
			}
		}
		if subst != 0 && !st.stage5FeedsOutput(i) {
			st.stage5Replace(i, subst-1)
		}
	}
	for _, i := range st.outputs {
		dedup(i)
	}
	return nil
}

func (st *state) stage6Check(nchw *[4]int) error {
	x := 1
	for _, y := range nchw {
		if y <= 0 {
			return errors.New("tensor is empty")
		}
		xy := x * y
		if xy/x != y || xy >= 1<<48 {
			return errors.New("tensor is too large")
		}
		x = xy
	}
	return nil
}

func (st *state) stage6Add(at *raw.Add, from1, from2 *shape) ([]shape, error) {
	if from1.nchw != from2.nchw {
		const chw = "%s is %dx%dx%d"
		msg := fmt.Sprintf(chw+" but "+chw+" (CxHxW mismatch)",
			origName(from1.tensor), from1.nchw[1], from1.nchw[2], from1.nchw[3],
			origName(from2.tensor), from2.nchw[1], from2.nchw[2], from2.nchw[3])
		return nil, anError(msg, at.ToTensor, at.LineNum)
	}
	return []shape{{at.ToTensor, from1.nchw}}, nil
}

func (st *state) stage6BatchNorm(at *raw.BatchNorm, from *shape) []shape {
	perChannel := [4]int{1, from.nchw[1], 1, 1}
	return []shape{
		{at.ToTensor, from.nchw},
		{at.MeansTensor, perChannel},
		{at.VariancesTensor, perChannel},
		{at.ScalesTensor, perChannel},
		{at.ShiftsTensor, perChannel},
	}
}

func (st *state) stage6Concat(at *raw.Concat, from1, from2 *shape) ([]shape, error) {
	h1, w1 := from1.nchw[2], from1.nchw[3]
	h2, w2 := from2.nchw[2], from2.nchw[3]
	if h1 != h2 || w1 != w2 {
		const hw = "%s is spatially %dx%d"
		msg := fmt.Sprintf(hw+" but "+hw+" (HxW mismatch)",
			origName(from1.tensor), h1, w1,
			origName(from2.tensor), h2, w2)
		return nil, anError(msg, at.ToTensor, at.LineNum)
	}
	c := from1.nchw[1] + from2.nchw[1]
	shapes := []shape{{at.ToTensor, [4]int{1, c, h1, w1}}}
	if err := st.stage6Check(&shapes[0].nchw); err != nil {
		return nil, anError(err.Error(), at.ToTensor, at.LineNum)
	}
	return shapes, nil
}

func (st *state) stage6Conv(at *raw.Conv, from *shape) ([]shape, error) {
	const notDivisible = " has %d channels (not divisible by %d groups)"
	c, k, g := from.nchw[1], at.ToChannels, at.Groups
	if c%g != 0 {
		msg := fmt.Sprintf("FromTensor"+notDivisible, c, g)
		return nil, anError(msg, at.FromTensor, at.LineNum)
	}
	if k%g != 0 {
		msg := fmt.Sprintf("ToTensor"+notDivisible, k, g)
		return nil, anError(msg, at.ToTensor, at.LineNum)
	}
	h := from.nchw[2] + 2*at.PaddingH - (1 + (at.FilterH-1)*at.DilationH)
	w := from.nchw[3] + 2*at.PaddingW - (1 + (at.FilterW-1)*at.DilationW)
	if h >= 0 {
		h = h/at.StrideH + 1
	}
	if w >= 0 {
		w = w/at.StrideW + 1
	}
	shapes := []shape{
		{at.ToTensor, [4]int{1, k, h, w}},
		{at.WeightsTensor, [4]int{k, c / g, at.FilterH, at.FilterW}},
		{at.BiasesTensor, [4]int{1, k, 1, 1}},
	}
	if err := st.stage6Check(&shapes[0].nchw); err != nil {
		return nil, anError(err.Error(), at.ToTensor, at.LineNum)
	}
	if err := st.stage6Check(&shapes[1].nchw); err != nil {
		return nil, anError(err.Error(), at.WeightsTensor+origNameShield, at.LineNum)
	}
	return shapes, nil
}

func (st *state) stage6FullyConnected(at *raw.FullyConnected, from *shape) ([]shape, error) {
	k := at.ToChannels
	shapes := []shape{
		{at.ToTensor, [4]int{1, k, 1, 1}},
		{at.WeightsTensor, [4]int{k, from.nchw[1], from.nchw[2], from.nchw[3]}},
		{at.BiasesTensor, [4]int{1, k, 1, 1}},
	}
	if err := st.stage6Check(&shapes[1].nchw); err != nil {
		return nil, anError(err.Error(), at.WeightsTensor+origNameShield, at.LineNum)
	}
	return shapes, nil
}

func (st *state) stage6Input(at *raw.Input) ([]shape, error) {
	shapes := []shape{
		{at.ToTensor, [4]int{1, at.Channels, at.Height, at.Width}},
	}
	if err := st.stage6Check(&shapes[0].nchw); err != nil {
		return nil, anError(err.Error(), at.ToTensor, at.LineNum)
	}
	return shapes, nil
}

func (st *state) stage6Pooling(at *raw.Pooling, from *shape) ([]shape, error) {
	var side int
	const stride = 2
	switch at.Kind {
	case raw.Max2x2Stride2, raw.Avg2x2Stride2:
		side = 2
	case raw.Max3x3Stride2, raw.Avg3x3Stride2:
		side = 3
	case raw.MaxGlobal, raw.AvgGlobal:
		side = 1
	default:
		panic("bug")
	}
	if at.PaddingH >= side || at.PaddingW >= side {
		return nil, anError("too much padding", at.ToTensor, at.LineNum)
	}
	c, h, w := from.nchw[1], 1, 1
	if side != 1 {
		h = from.nchw[2] + 2*at.PaddingH - side
		w = from.nchw[3] + 2*at.PaddingW - side
		if h >= 0 {
			h = h/stride + 1
		}
		if w >= 0 {
			w = w/stride + 1
		}
	}
	shapes := []shape{{at.ToTensor, [4]int{1, c, h, w}}}
	if err := st.stage6Check(&shapes[0].nchw); err != nil {
		return nil, anError(err.Error(), at.ToTensor, at.LineNum)
	}
	return shapes, nil
}

func (st *state) stage6Fill(i int) (err error) {
	slot := &st.shapes[i]
	if *slot != nil {
		return
	}
	fanin := st.fanins[i]
	for j := range fanin {
		k := fanin[j].attach
		if err = st.stage6Fill(k); err != nil {
			return
		}
	}
	lookup := func(tensor string) *shape {
		for j := range fanin {
			if fanin[j].tensor == tensor {
				shapes := st.shapes[fanin[j].attach]
				for k := range shapes {
					if shapes[k].tensor == tensor {
						return &shapes[k]
					}
				}
			}
		}
		panic("bug")
	}
	switch at := st.nodes[i].(type) {
	case *raw.Activation:
		from := lookup(at.FromTensor)
		*slot = []shape{{at.ToTensor, from.nchw}}
	case *raw.Add:
		from1 := lookup(at.FromTensor1)
		from2 := lookup(at.FromTensor2)
		*slot, err = st.stage6Add(at, from1, from2)
	case *raw.BatchNorm:
		from := lookup(at.FromTensor)
		*slot = st.stage6BatchNorm(at, from)
	case *raw.Concat:
		from1 := lookup(at.FromTensor1)
		from2 := lookup(at.FromTensor2)
		*slot, err = st.stage6Concat(at, from1, from2)
	case *raw.Conv:
		from := lookup(at.FromTensor)
		*slot, err = st.stage6Conv(at, from)
	case *raw.FullyConnected:
		from := lookup(at.FromTensor)
		*slot, err = st.stage6FullyConnected(at, from)
	case *raw.Input:
		*slot, err = st.stage6Input(at)
	case *raw.Output:
		from := lookup(at.FromTensor)
		*slot = []shape{*from}
	case *raw.Pooling:
		from := lookup(at.FromTensor)
		*slot, err = st.stage6Pooling(at, from)
	case *raw.Softmax:
		from := lookup(at.FromTensor)
		*slot = []shape{{at.ToTensor, from.nchw}}
	default:
		panic("bug")
	}
	return
}

func (st *state) stage6() error {
	n := len(st.nodes)
	st.shapes = make([][]shape, n)
	for _, i := range st.outputs {
		if err := st.stage6Fill(i); err != nil {
			return err
		}
	}
	return nil
}

func (st *state) stage7() error {
	in := len(st.inputs)
	st.plan = plan.Plan{
		Config: st.config,
		Seq:    make([]*plan.Op, in, len(st.nodes)),
	}
	ops := make([]*plan.Op, len(st.nodes))
	var mirror func(int)
	mirror = func(i int) {
		fanin := st.fanins[i]
		for j := range fanin {
			if k := fanin[j].attach; ops[k] == nil {
				mirror(k)
			}
		}
		node := st.nodes[i]
		op := &plan.Op{Nodes: []raw.Node{node}}
		switch node.(type) {
		case *raw.Input:
			in--
			st.plan.Seq[in] = op
		default:
			st.plan.Seq = append(st.plan.Seq, op)
		}
		ops[i] = op
		shapes := st.shapes[i]
		params := node.ParamTensors()
		op.Params = [][]plan.Param{make([]plan.Param, len(params))}
		op.ParamMods = make([][2][]plan.Mod, 1)
		for j, tensor := range params {
			var nchw *[4]int
			for k := range shapes {
				if shapes[k].tensor == tensor {
					nchw = &shapes[k].nchw
					break
				}
			}
			op.Params[0][j] = plan.Param{
				Tensor: tensor,
				NCHW:   *nchw,
			}
		}
		from := node.FromTensors()
		op.From = make([]*plan.Span, len(from))
		op.FromMods = make([][]plan.Mod, len(from))
		for j, tensor := range from {
			var pile *plan.Pile
			for k := range fanin {
				if fanin[k].tensor == tensor {
					for _, span := range ops[fanin[k].attach].To {
						if span.Tensors[0] == tensor {
							pile = span.Piles[0]
							break
						}
					}
					break
				}
			}
			span := &plan.Span{
				Piles:   []*plan.Pile{pile},
				Offsets: []int{0},
				Tensors: []string{tensor},
				Counts:  []int{pile.Channels},
				Op:      op,
			}
			pile.Readers = append(pile.Readers, span)
			op.From[j] = span
		}
		to := node.ToTensors()
		op.To = make([]*plan.Span, len(to))
		op.ToMods = make([][]plan.Mod, len(to))
		for j, tensor := range to {
			var nchw *[4]int
			for k := range shapes {
				if shapes[k].tensor == tensor {
					nchw = &shapes[k].nchw
					break
				}
			}
			pile := &plan.Pile{
				Channels: nchw[1],
				Height:   nchw[2],
				Width:    nchw[3],
			}
			span := &plan.Span{
				Piles:   []*plan.Pile{pile},
				Offsets: []int{0},
				Tensors: []string{tensor},
				Counts:  []int{nchw[1]},
				Op:      op,
			}
			pile.Writers = []*plan.Span{span}
			op.To[j] = span
		}
	}
	for _, i := range st.outputs {
		mirror(i)
	}
	return nil
}

func (st *state) stage8Pre(op1 *plan.Op) {
	var mods []plan.Mod
	for span1 := op1.From[0]; ; {
		pile1 := span1.Piles[0]
		span2 := pile1.Writers[0]
		op2 := span2.Op
		if _, ok := op2.Nodes[0].(*raw.BatchNorm); !ok {
			break
		}
		mods = append(mods, plan.Mod{
			Nodes:  op2.Nodes,
			Params: op2.Params[0],
		})
		span3 := op2.From[0]
		pile2 := span3.Piles[0]
		span1.Piles[0] = pile2
		if len(pile1.Readers) == 1 {
			for i, ii := range pile2.Readers {
				if ii == span3 {
					pile2.Readers[i] = span1
					break
				}
			}
			*op2 = plan.Op{}
			continue
		}
		pile2.Readers = append(pile2.Readers, span1)
		for i, ii := range pile1.Readers {
			if ii == span1 {
				j := len(pile1.Readers) - 1
				pile1.Readers[i] = pile1.Readers[j]
				pile1.Readers[j] = nil
				pile1.Readers = pile1.Readers[:j]
				break
			}
		}
		op2.Nodes = []raw.Node{op2.Nodes[0]}
		params := make([]plan.Param, len(op2.Params[0]))
		copy(params, op2.Params[0])
		op2.Params[0] = params
	}
	for i, j := 0, len(mods)-1; i < j; i, j = i+1, j-1 {
		mods[i], mods[j] = mods[j], mods[i]
	}
	op1.ParamMods[0][0] = mods
}

func (st *state) stage8Post(op1 *plan.Op) {
	var mods []plan.Mod
	for span1 := op1.To[0]; ; {
		pile1 := span1.Piles[0]
		if len(pile1.Readers) != 1 {
			break
		}
		span2 := pile1.Readers[0]
		op2 := span2.Op
		if _, ok := op2.Nodes[0].(*raw.BatchNorm); !ok {
			break
		}
		mods = append(mods, plan.Mod{
			Nodes:  op2.Nodes,
			Params: op2.Params[0],
		})
		span3 := op2.To[0]
		pile2 := span3.Piles[0]
		span1.Piles[0] = pile2
		pile2.Writers[0] = span1
		*op2 = plan.Op{}
	}
	op1.ParamMods[0][1] = mods
}

func (st *state) stage8() error {
	for _, op := range st.plan.Seq {
		if op.Nodes == nil {
			continue
		}
		switch node := op.Nodes[0].(type) {
		case *raw.Conv:
			if node.PaddingH == 0 &&
				node.PaddingW == 0 {
				st.stage8Pre(op)
			}
			st.stage8Post(op)
		case *raw.FullyConnected:
			st.stage8Pre(op)
			st.stage8Post(op)
		}
	}
	return nil
}

func (st *state) stage9() error {
	for _, op1 := range st.plan.Seq {
		if op1.Nodes == nil {
			continue
		}
		if _, ok := op1.Nodes[0].(*raw.Add); !ok {
			continue
		}
		for i, span1 := range op1.From {
			pile1 := span1.Piles[0]
			if len(pile1.Readers) != 1 {
				continue
			}
			span2 := pile1.Writers[0]
			op2 := span2.Op
			if _, ok := op2.Nodes[0].(*raw.Add); !ok {
				continue
			}
			op1.Nodes = append(op1.Nodes, op2.Nodes...)
			op1.Params = append(op1.Params, op2.Params...)
			op1.ParamMods = append(op1.ParamMods, op2.ParamMods...)
			for _, span3 := range op2.From {
				span3.Op = op1
			}
			op1.From[i] = op2.From[0]
			op1.From = append(op1.From, op2.From[1:]...)
			op1.FromMods = append(op1.FromMods, op2.FromMods[1:]...)
			*op2 = plan.Op{}
		}
	}
	return nil
}

func (st *state) stage10Via(op1 *plan.Op) int {
	var i int
	for j, span1 := range op1.From {
		pile1 := span1.Piles[0]
		span2 := pile1.Writers[0]
		op2 := span2.Op
		switch op2.Nodes[0].(type) {
		case *raw.Activation, *raw.BatchNorm:
			if i == 0 {
				i = j + 1
			} else {
				return -1
			}
		}
	}
	if i == 0 {
		return 0
	}
	return i - 1
}

func (st *state) stage10Pre(op1 *plan.Op, pass int) {
	for i, span1 := range op1.From {
		var mods []plan.Mod
		for {
			pile1 := span1.Piles[0]
			span2 := pile1.Writers[0]
			op2 := span2.Op
			mod := plan.Mod{Nodes: op2.Nodes}
			var span3 *plan.Span
			switch mod.Nodes[0].(type) {
			case *raw.Activation:
				span3 = op2.From[0]
			case *raw.Add:
				if pass >= 2 {
					break
				}
				if len(pile1.Readers) != 1 {
					break
				}
				j := st.stage10Via(op2)
				if j < 0 {
					break
				}
				mod.From = op2.From
				span3 = mod.From[j]
				k := len(mod.From) - 1
				mod.From[j] = mod.From[k]
				mod.From[k] = nil
				mod.From = mod.From[:k]
				for _, span4 := range mod.From {
					span4.Op = op1
				}
			case *raw.BatchNorm:
				if pass >= 3 {
					break
				}
				mod.Params = op2.Params[0]
				span3 = op2.From[0]
			}
			if span3 == nil {
				break
			}
			mods = append(mods, mod)
			pile2 := span3.Piles[0]
			span1.Piles[0] = pile2
			if len(pile1.Readers) == 1 {
				for j, span4 := range pile2.Readers {
					if span4 == span3 {
						pile2.Readers[j] = span1
						break
					}
				}
				*op2 = plan.Op{}
				continue
			}
			pile2.Readers = append(pile2.Readers, span1)
			for j, span4 := range pile1.Readers {
				if span4 == span1 {
					k := len(pile1.Readers) - 1
					pile1.Readers[j] = pile1.Readers[k]
					pile1.Readers[k] = nil
					pile1.Readers = pile1.Readers[:k]
					break
				}
			}
			op2.Nodes = []raw.Node{op2.Nodes[0]}
			if mod.Params != nil {
				op2.Params[0] = make([]plan.Param, len(mod.Params))
				copy(op2.Params[0], mod.Params)
			}
		}
		for j, k := 0, len(mods)-1; j < k; j, k = j+1, k-1 {
			mods[j], mods[k] = mods[k], mods[j]
		}
		op1.FromMods[i] = mods
	}
}

func (st *state) stage10Post(op1 *plan.Op, pass int) {
	for i, span1 := range op1.To {
		var mods []plan.Mod
		for {
			pile1 := span1.Piles[0]
			if len(pile1.Readers) != 1 {
				break
			}
			span2 := pile1.Readers[0]
			op2 := span2.Op
			mod := plan.Mod{Nodes: op2.Nodes}
			switch mod.Nodes[0].(type) {
			case *raw.Activation:
			case *raw.Add:
				if pass > 2 {
					mod.Nodes = nil
					break
				}
				var j int
				for k, span3 := range op2.From {
					if span3 == span2 {
						j = k
					} else if span3.Piles[0].ElemBytes != pass {
						mod.Nodes = nil
						break
					}
				}
				if mod.Nodes == nil {
					break
				}
				mod.From = op2.From
				k := len(mod.From) - 1
				mod.From[j] = mod.From[k]
				mod.From[k] = nil
				mod.From = mod.From[:k]
				for _, span3 := range mod.From {
					span3.Op = op1
				}
			case *raw.BatchNorm:
				if pass > 3 {
					mod.Nodes = nil
					break
				}
				mod.Params = op2.Params[0]
			default:
				mod.Nodes = nil
			}
			if mod.Nodes == nil {
				break
			}
			mods = append(mods, mod)
			span3 := op2.To[0]
			pile2 := span3.Piles[0]
			span1.Piles[0] = pile2
			pile2.Writers[0] = span1
			*op2 = plan.Op{}
		}
		op1.ToMods[i] = mods
	}
}

func (st *state) stage10() error {
	pass := 1
	for ; pass <= 3; pass++ {
		for _, op := range st.plan.Seq {
			if op.Nodes == nil {
				continue
			}
			switch z := op.Nodes[0]; pass {
			case 1:
				switch z.(type) {
				case *raw.Conv, *raw.Pooling:
					st.stage10Pre(op, pass)
					st.stage10Post(op, pass)
				case *raw.FullyConnected:
					st.stage10Post(op, pass)
				}
				for _, span := range op.To {
					span.Piles[0].ElemBytes = pass
				}
			case 2:
				switch z.(type) {
				case *raw.Add:
					st.stage10Pre(op, pass)
					st.stage10Post(op, pass)
				}
				for _, span := range op.To {
					span.Piles[0].ElemBytes = pass
				}
			case 3:
				switch z.(type) {
				case *raw.BatchNorm:
					st.stage10Pre(op, pass)
					st.stage10Post(op, pass)
				}
			}
		}
	}
	for i := len(st.plan.Seq) - 1; i >= 0; i-- {
		op := st.plan.Seq[i]
		if op.Nodes == nil {
			continue
		}
		switch op.Nodes[0].(type) {
		case *raw.Activation:
			st.stage10Pre(op, pass)
		}
	}
	n := 0
	for i, op := range st.plan.Seq {
		st.plan.Seq[i] = nil
		if op.Nodes == nil {
			continue
		}
		for _, span := range op.To {
			span.Piles[0].ElemBytes = 0
		}
		st.plan.Seq[n] = op
		n += 1
	}
	st.plan.Seq = st.plan.Seq[:n]
	return nil
}

func (st *state) stage11Reduce(mods []plan.Mod) []plan.Mod {
	phase := 0
	for i := range mods {
		node := mods[i].Nodes[0]
		if node, ok := node.(*raw.Activation); ok {
			if node.Kind == raw.ReLU {
				if phase++; phase == 2 {
					break
				}
				continue
			}
		}
		phase = 0
	}
	if phase != 2 {
		return mods
	}
	var first int
	var slopes []float32
	keep, have := 0, len(mods)
	for i := 0; i <= have; i++ {
		if i < have {
			node := mods[i].Nodes[0]
			if node, ok := node.(*raw.Activation); ok {
				if node.Kind == raw.ReLU {
					if len(slopes) == 0 {
						first = i
					}
					slopes = append(slopes, node.Param)
					continue
				}
			}
		}
		if run := len(slopes); run != 0 {
			if run == 1 {
				mods[keep] = mods[first]
			} else {
				var param float32 = 1
				for _, slope := range slopes {
					if param *= slope; param <= 0 {
						break
					}
				}
				cross := *mods[first+run-1].Nodes[0].(*raw.Activation)
				cross.Param = param
				nodes := make([]raw.Node, 1, 1+run)
				nodes[0] = &cross
				for j := first; j < first+run; j++ {
					nodes = append(nodes, mods[j].Nodes[0])
				}
				mods[keep] = plan.Mod{Nodes: nodes}
			}
			keep += 1
			slopes = slopes[:0]
		}
		if i < have {
			mods[keep] = mods[i]
			keep += 1
		}
	}
	for i := keep; i < have; i++ {
		mods[i] = plan.Mod{}
	}
	return mods[:keep]
}

func (st *state) stage11Absorb(op *plan.Op) {
	at, ok := op.Nodes[0].(*raw.Activation)
	if !ok || at.Kind != raw.ReLU {
		return
	}
	mods := op.FromMods[0]
	i := len(mods) - 1
	if i < 0 {
		return
	}
	nodes := mods[i].Nodes
	node, ok := nodes[0].(*raw.Activation)
	if !ok || node.Kind != raw.ReLU {
		return
	}
	mods[i] = plan.Mod{}
	op.FromMods[0] = mods[:i]
	param := node.Param
	if param > 0 {
		param *= at.Param
	}
	cross := *at
	cross.Param = param
	if len(nodes) > 1 {
		nodes = nodes[1:]
	}
	cnt := 1 + len(nodes) + 1
	all := make([]raw.Node, 1, cnt)
	all[0] = &cross
	all = append(all, nodes...)
	all = append(all, at)
	op.Nodes = all
	op.Params = make([][]plan.Param, cnt)
	op.ParamMods = make([][2][]plan.Mod, cnt)
}

func (st *state) stage11() error {
	for _, op := range st.plan.Seq {
		for i, mods := range op.FromMods {
			op.FromMods[i] = st.stage11Reduce(mods)
		}
		for i, mods := range op.ToMods {
			op.ToMods[i] = st.stage11Reduce(mods)
		}
		st.stage11Absorb(op)
	}
	return nil
}

func (st *state) stage12Compatible(node1, node2 raw.Node) bool {
	switch node1 := node1.(type) {
	case *raw.Conv:
		node2, ok := node2.(*raw.Conv)
		return ok &&
			node1.FilterH == node2.FilterH &&
			node1.FilterW == node2.FilterW &&
			node1.StrideH == node2.StrideH &&
			node1.StrideW == node2.StrideW &&
			node1.PaddingH == node2.PaddingH &&
			node1.PaddingW == node2.PaddingW &&
			node1.DilationH == node2.DilationH &&
			node1.DilationW == node2.DilationW &&
			node1.Groups == node2.Groups &&
			node1.Groups == 1
	}
	return false
}

func (st *state) stage12AddsEq(from1, from2 []*plan.Span) bool {
	cnt := len(from1)
	if len(from2) != cnt {
		return false
	}
	used := make([]bool, cnt)
	for _, span1 := range from1 {
		found := false
		for i, span2 := range from2 {
			if !used[i] &&
				span1.Piles[0] == span2.Piles[0] &&
				span1.Offsets[0] == span2.Offsets[0] {
				used[i] = true
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func (st *state) stage12Common(mods1, mods2 []plan.Mod, allow bool) int {
	n1, n2 := len(mods1), len(mods2)
	if n1 > n2 {
		n1 = n2
	}
	for i := 0; i < n1; i++ {
		switch node1 := mods1[i].Nodes[0].(type) {
		case *raw.Activation:
			node2, ok := mods2[i].Nodes[0].(*raw.Activation)
			if ok && node1.Kind == node2.Kind && node1.Param == node2.Param {
				continue
			}
		case *raw.Add:
			if !allow {
				break
			}
			_, ok := mods2[i].Nodes[0].(*raw.Add)
			if ok && st.stage12AddsEq(mods1[i].From, mods2[i].From) {
				continue
			}
		case *raw.BatchNorm:
			if !allow {
				break
			}
			node2, ok := mods2[i].Nodes[0].(*raw.BatchNorm)
			if ok && node1.MeansTensor == node2.MeansTensor {
				continue
			}
		default:
			panic("bug")
		}
		return i
	}
	return n1
}

func (st *state) stage12Stackable(span1, span2 *plan.Span) bool {
	if span1.Offsets[0] != span2.Offsets[0] {
		return false
	}
	op1, op2 := span1.Op, span2.Op
	if op1 == op2 {
		return false
	}
	nodes1, nodes2 := op1.Nodes, op2.Nodes
	if len(nodes1) != 1 || len(nodes2) != 1 {
		return false
	}
	if !st.stage12Compatible(nodes1[0], nodes2[0]) {
		return false
	}
	from1, from2 := op1.From, op2.From
	if len(from1) != 1 || len(from2) != 1 {
		return false
	}
	if from1[0] != span1 || from2[0] != span2 {
		return false
	}
	mods1, mods2 := op1.FromMods[0], op2.FromMods[0]
	if len(mods1) != len(mods2) {
		return false
	}
	if len(mods1) != st.stage12Common(mods1, mods2, true) {
		return false
	}
	if len(op1.To) != 1 || len(op2.To) != 1 {
		return false
	}
	return true
}

func (st *state) stage12Clone(a *plan.Span) *plan.Span {
	return &plan.Span{
		Piles:   []*plan.Pile{a.Piles[0]},
		Offsets: []int{a.Offsets[0]},
		Tensors: []string{a.Tensors[0]},
		Counts:  []int{a.Counts[0]},
		Op:      a.Op,
	}
}

func (st *state) stage12Indirect(op1 *plan.Op, broadcast bool) *plan.Op {
	span1 := op1.To[0]
	pile1 := span1.Piles[0]
	readers, split := pile1.Readers, 0
	for i, span2 := range readers {
		op2 := span2.Op
		switch op2.Nodes[0].(type) {
		case *raw.Concat, *raw.Output:
		case *raw.FullyConnected, *raw.Softmax:
			if !broadcast {
				continue
			}
		default:
			if !broadcast {
				continue
			}
			mod := true
			for _, span3 := range op2.From {
				if span3 == span2 {
					mod = false
					break
				}
			}
			if !mod {
				continue
			}
		}
		readers[i] = readers[split]
		readers[split] = span2
		split += 1
	}
	if split == 0 {
		return nil
	}
	span2 := st.stage12Clone(readers[0])
	span3 := st.stage12Clone(readers[0])
	op2 := &plan.Op{
		Nodes: []raw.Node{
			&raw.Activation{
				LineNum:    op1.Nodes[0].LineNumber(),
				FromTensor: span2.Tensors[0],
				ToTensor:   span3.Tensors[0],
				Kind:       raw.ReLU,
				Param:      1,
			},
		},
		Params:    make([][]plan.Param, 1),
		ParamMods: make([][2][]plan.Mod, 1),
		From:      []*plan.Span{span2},
		FromMods:  make([][]plan.Mod, 1),
		To:        []*plan.Span{span3},
		ToMods:    make([][]plan.Mod, 1),
	}
	span2.Op = op2
	span3.Op = op2
	pile2 := &plan.Pile{
		Channels: pile1.Channels,
		Height:   pile1.Height,
		Width:    pile1.Width,
		Writers:  []*plan.Span{span3},
		Readers:  make([]*plan.Span, split),
	}
	span3.Piles[0] = pile2
	copy(pile2.Readers, readers)
	for _, span4 := range pile2.Readers {
		span4.Piles[0] = pile2
	}
	remain := 1 + len(readers) - split
	pile1.Readers = readers[:remain]
	readers[0] = span2
	copy(readers[1:], readers[split:])
	for i, n := remain, len(readers); i < n; i++ {
		readers[i] = nil
	}
	return op2
}

func (st *state) stage12Detach(mods []plan.Mod) {
	for i := range mods {
		switch mods[i].Nodes[0].(type) {
		case *raw.Add:
			for _, span1 := range mods[i].From {
				pile := span1.Piles[0]
				j := -1
				for k, span2 := range pile.Readers {
					if span2 == span1 {
						j = k
						break
					}
				}
				copy(pile.Readers[j:], pile.Readers[j+1:])
				k := len(pile.Readers) - 1
				pile.Readers[k] = nil
				pile.Readers = pile.Readers[:k]
			}
		}
	}
}

func (st *state) stage12Broadcast(mods1 []plan.Mod, pile1 *plan.Pile) {
	for _, span1 := range pile1.Readers {
		op1 := span1.Op
		i := -1
		for j, span2 := range op1.From {
			if span2 == span1 {
				i = j
				break
			}
		}
		mods2 := op1.FromMods[i]
		mods3 := make([]plan.Mod, len(mods1)+len(mods2))
		op1.FromMods[i] = mods3
		copy(mods3[len(mods1):], mods2)
		for j := range mods1 {
			nodes1 := mods1[j].Nodes
			nodes2 := make([]raw.Node, len(nodes1))
			mods3[j].Nodes = nodes2
			copy(nodes2, nodes1)
			switch nodes1[0].(type) {
			case *raw.Add:
				from1 := mods1[j].From
				from2 := make([]*plan.Span, len(from1))
				mods3[j].From = from2
				for k, span2 := range from1 {
					span3 := st.stage12Clone(span2)
					span3.Op = op1
					from2[k] = span3
					pile2 := span3.Piles[0]
					pile2.Readers = append(pile2.Readers, span3)
				}
			case *raw.BatchNorm:
				params1 := mods1[j].Params
				params2 := make([]plan.Param, len(params1))
				mods3[j].Params = params2
				copy(params2, params1)
			}
		}
	}
}

func (st *state) stage12Fanout(edits map[*plan.Op][2]*plan.Op, loose []*plan.Op, fused *plan.Op) {
	op1 := loose[0]
	mods1 := op1.ToMods[0]
	for _, op2 := range loose[1:] {
		mods2 := op2.ToMods[0]
		n := st.stage12Common(mods1, mods2, false)
		mods1 = mods1[:n]
	}
	for _, op2 := range loose {
		mods2 := op2.ToMods[0]
		broadcast := len(mods1) < len(mods2)
		op3 := st.stage12Indirect(op2, broadcast)
		edits[op2] = [2]*plan.Op{fused, op3}
	}
	for _, op2 := range loose {
		mods2 := op2.ToMods[0][len(mods1):]
		if len(mods2) != 0 {
			st.stage12Detach(mods2)
			span2 := op2.To[0]
			pile2 := span2.Piles[0]
			st.stage12Broadcast(mods2, pile2)
		}
	}
	span1 := op1.To[0]
	pile1 := span1.Piles[0]
	span3 := &plan.Span{
		Piles:   []*plan.Pile{nil},
		Offsets: []int{0},
		Tensors: make([]string, len(loose)),
		Counts:  make([]int, len(loose)),
		Op:      fused,
	}
	pile3 := &plan.Pile{
		Height:  pile1.Height,
		Width:   pile1.Width,
		Writers: []*plan.Span{span3},
	}
	span3.Piles[0] = pile3
	fused.To = []*plan.Span{span3}
	fused.ToMods = [][]plan.Mod{make([]plan.Mod, len(mods1))}
	copy(fused.ToMods[0], mods1)
	for i, op2 := range loose {
		span2 := op2.To[0]
		pile2 := span2.Piles[0]
		span3.Tensors[i] = span2.Tensors[0]
		span3.Counts[i] = span2.Counts[0]
		offset := pile3.Channels
		pile3.Channels += pile2.Channels
		for _, span4 := range pile2.Readers {
			span4.Piles[0] = pile3
			span4.Offsets[0] = offset
			pile3.Readers = append(pile3.Readers, span4)
		}
	}
}

func (st *state) stage12Fusion(edits map[*plan.Op][2]*plan.Op, loose []*plan.Op) {
	fused := &plan.Op{
		Nodes:     make([]raw.Node, len(loose)),
		Params:    make([][]plan.Param, len(loose)),
		ParamMods: make([][2][]plan.Mod, len(loose)),
		From:      loose[0].From,
		FromMods:  loose[0].FromMods,
	}
	for i, op := range loose {
		fused.Nodes[i] = op.Nodes[0]
		fused.Params[i] = op.Params[0]
		fused.ParamMods[i] = op.ParamMods[0]
	}
	fused.From[0].Op = fused
	mods := fused.FromMods[0]
	for i := range mods {
		switch mods[i].Nodes[0].(type) {
		case *raw.Add:
			for _, span := range mods[i].From {
				span.Op = fused
			}
		}
	}
	st.stage12Fanout(edits, loose, fused)
	for _, op := range loose {
		*op = plan.Op{}
	}
}

func (st *state) stage12Readers(edits map[*plan.Op][2]*plan.Op, pile *plan.Pile) {
	var loose []*plan.Op
	n := len(pile.Readers)
	for i := 0; i < n; i++ {
		span1 := pile.Readers[i]
		for j := i + 1; j < n; j++ {
			span2 := pile.Readers[j]
			if !st.stage12Stackable(span1, span2) {
				continue
			}
			if len(loose) == 0 {
				loose = append(loose, span1.Op)
			}
			loose = append(loose, span2.Op)
			n -= 1
			pile.Readers[j] = pile.Readers[n]
			pile.Readers[n] = nil
			pile.Readers = pile.Readers[:n]
			j -= 1
		}
		if len(loose) == 0 {
			continue
		}
		for _, op := range loose[1:] {
			st.stage12Detach(op.FromMods[0])
		}
		st.stage12Fusion(edits, loose)
		loose = loose[:0]
		n = len(pile.Readers)
		if i >= n {
			i = n - 1
		}
		for pile.Readers[i] != span1 {
			i -= 1
		}
	}
}

func (st *state) stage12() error {
	edits := make(map[*plan.Op][2]*plan.Op)
	for _, op := range st.plan.Seq {
		if len(op.From) == 0 {
			continue
		}
		span := op.From[0]
		pile := span.Piles[0]
		if pile.ElemBytes != 0 {
			continue
		}
		pile.ElemBytes = -1
		st.stage12Readers(edits, pile)
	}
	if n := len(edits); n != 0 {
		seq := make([]*plan.Op, 0, len(st.plan.Seq)+n)
		seen := make(map[*plan.Op]bool, n)
		for _, op := range st.plan.Seq {
			if op.Nodes != nil {
				seq = append(seq, op)
				continue
			}
			ed := edits[op]
			if fused := ed[0]; !seen[fused] {
				seen[fused] = true
				seq = append(seq, fused)
			}
			if indirect := ed[1]; indirect != nil {
				seq = append(seq, indirect)
			}
		}
		st.plan.Seq = seq
	}
	for _, op := range st.plan.Seq {
		for _, span := range op.To {
			span.Piles[0].ElemBytes = 0
		}
	}
	return nil
}

func (st *state) stage13Fork(op1 *plan.Op) {
	for _, span1 := range op1.To {
		if len(span1.Tensors) != 1 {
			continue
		}
		pile1 := span1.Piles[0]
		first := true
		var moving []*plan.Span
		n := len(pile1.Readers)
		for i := 0; i < n; i++ {
			span2 := pile1.Readers[i]
			op2 := span2.Op
			switch op2.Nodes[0].(type) {
			case *raw.Concat, *raw.Output:
				if first {
					first = false
					break
				}
				moving = append(moving, span2)
				n -= 1
				pile1.Readers[i] = pile1.Readers[n]
				pile1.Readers[n] = nil
				pile1.Readers = pile1.Readers[:n]
				i -= 1
			}
		}
		for _, span2 := range moving {
			pile2 := &plan.Pile{
				Channels: pile1.Channels,
				Height:   pile1.Height,
				Width:    pile1.Width,
				Writers:  []*plan.Span{span1},
				Readers:  []*plan.Span{span2},
			}
			span1.Piles = append(span1.Piles, pile2)
			span1.Offsets = append(span1.Offsets, 0)
			span2.Piles[0] = pile2
		}
	}
}

func (st *state) stage13IndirectFork(edits map[*plan.Op][]*plan.Op, op1 *plan.Op) {
	for _, span1 := range op1.To {
		if len(span1.Tensors) != 1 {
			continue
		}
		pile1 := span1.Piles[0]
		var moving []*plan.Span
		n := len(pile1.Readers)
		for i := 0; i < n; i++ {
			span2 := pile1.Readers[i]
			op2 := span2.Op
			switch op2.Nodes[0].(type) {
			case *raw.Concat, *raw.Output:
				moving = append(moving, span2)
				n -= 1
				pile1.Readers[i] = pile1.Readers[n]
				pile1.Readers[n] = nil
				i -= 1
			}
		}
		if len(moving) == 0 {
			continue
		}
		span2 := &plan.Span{
			Piles:   []*plan.Pile{pile1},
			Offsets: []int{0},
			Tensors: []string{moving[0].Tensors[0]},
			Counts:  []int{moving[0].Counts[0]},
		}
		pile1.Readers[n] = span2
		pile1.Readers = pile1.Readers[:n+1]
		op2 := &plan.Op{
			Nodes: []raw.Node{
				&raw.Activation{
					LineNum:    op1.Nodes[0].LineNumber(),
					FromTensor: span2.Tensors[0],
					ToTensor:   span2.Tensors[0],
					Kind:       raw.ReLU,
					Param:      1,
				},
			},
			Params:    make([][]plan.Param, 1),
			ParamMods: make([][2][]plan.Mod, 1),
			From:      []*plan.Span{span2},
			FromMods:  make([][]plan.Mod, 1),
			To:        make([]*plan.Span, 1),
			ToMods:    make([][]plan.Mod, 1),
		}
		span2.Op = op2
		edits[op1] = append(edits[op1], op2)
		span3 := &plan.Span{
			Piles:   make([]*plan.Pile, len(moving)),
			Offsets: make([]int, len(moving)),
			Tensors: []string{span2.Tensors[0]},
			Counts:  []int{span2.Counts[0]},
			Op:      op2,
		}
		op2.To[0] = span3
		for i, span4 := range moving {
			pile2 := &plan.Pile{
				Channels: pile1.Channels,
				Height:   pile1.Height,
				Width:    pile1.Width,
				Writers:  []*plan.Span{span3},
				Readers:  []*plan.Span{span4},
			}
			span3.Piles[i] = pile2
			span4.Piles[0] = pile2
		}
	}
}

func (st *state) stage13Fission(edits map[*plan.Op][]*plan.Op, op1 *plan.Op) {
	span1 := op1.To[0]
	pile1 := span1.Piles[0]
	first := true
	var moving []*plan.Span
	n := len(pile1.Readers)
	for i := 0; i < n; i++ {
		span2 := pile1.Readers[i]
		op2 := span2.Op
		switch op2.Nodes[0].(type) {
		case *raw.Concat, *raw.Output:
			if first {
				first = false
				break
			}
			moving = append(moving, span2)
			n -= 1
			pile1.Readers[i] = pile1.Readers[n]
			pile1.Readers[n] = nil
			i -= 1
		}
	}
	if len(moving) == 0 {
		return
	}
	pile1.Readers = pile1.Readers[:n]
	span2 := op1.From[0]
	span3 := op1.From[1]
	pile2 := span2.Piles[0]
	pile3 := span3.Piles[0]
	for _, span4 := range moving {
		pile4 := &plan.Pile{
			Channels: pile1.Channels,
			Height:   pile1.Height,
			Width:    pile1.Width,
			Writers:  []*plan.Span{nil},
			Readers:  []*plan.Span{span4},
		}
		span4.Piles[0] = pile4
		span5 := &plan.Span{
			Piles:   []*plan.Pile{pile4},
			Offsets: []int{span1.Offsets[0]},
			Tensors: []string{span1.Tensors[0]},
			Counts:  []int{span1.Counts[0]},
		}
		pile4.Writers[0] = span5
		op2 := &plan.Op{
			Nodes:     []raw.Node{op1.Nodes[0]},
			Params:    make([][]plan.Param, 1),
			ParamMods: make([][2][]plan.Mod, 1),
			From:      make([]*plan.Span, 2),
			FromMods:  make([][]plan.Mod, 2),
			To:        []*plan.Span{span5},
			ToMods:    make([][]plan.Mod, 1),
		}
		edits[op1] = append(edits[op1], op2)
		span5.Op = op2
		span6 := &plan.Span{
			Piles:   []*plan.Pile{pile2},
			Offsets: []int{span2.Offsets[0]},
			Tensors: []string{span2.Tensors[0]},
			Counts:  []int{span2.Counts[0]},
			Op:      op2,
		}
		pile2.Readers = append(pile2.Readers, span6)
		op2.From[0] = span6
		span7 := &plan.Span{
			Piles:   []*plan.Pile{pile3},
			Offsets: []int{span3.Offsets[0]},
			Tensors: []string{span3.Tensors[0]},
			Counts:  []int{span3.Counts[0]},
			Op:      op2,
		}
		pile3.Readers = append(pile3.Readers, span7)
		op2.From[1] = span7
	}
}

func (st *state) stage13() error {
	edits := make(map[*plan.Op][]*plan.Op)
	for i := len(st.plan.Seq) - 1; i >= 0; i-- {
		op := st.plan.Seq[i]
		switch op.Nodes[0].(type) {
		case *raw.Concat:
			st.stage13Fission(edits, op)
		case *raw.Input:
			st.stage13IndirectFork(edits, op)
		default:
			st.stage13Fork(op)
		}
	}
	if len(edits) != 0 {
		cnt := len(st.plan.Seq)
		for _, insert := range edits {
			cnt += len(insert)
		}
		seq := make([]*plan.Op, 0, cnt)
		for _, op := range st.plan.Seq {
			seq = append(seq, op)
			if insert, ok := edits[op]; ok {
				seq = append(seq, insert...)
			}
		}
		st.plan.Seq = seq
	}
	return nil
}

func (st *state) stage14Rewire(op *plan.Op, pile1, pile2 *plan.Pile, offset int) {
	pile1.Writers = append(pile1.Writers, pile2.Writers...)
	for _, span := range pile2.Writers {
		for i, pile3 := range span.Piles {
			if pile3 == pile2 {
				span.Piles[i] = pile1
				span.Offsets[i] += offset
			}
		}
	}
	for _, span := range pile2.Readers {
		if span.Op != op {
			pile1.Readers = append(pile1.Readers, span)
			span.Piles[0] = pile1
			span.Offsets[0] += offset
		}
	}
}

func (st *state) stage14Bypass(op *plan.Op) {
	pile1 := op.To[0].Piles[0]
	pile2 := op.From[0].Piles[0]
	pile3 := op.From[1].Piles[0]
	fanin := len(pile2.Writers) + len(pile3.Writers)
	pile1.Writers = make([]*plan.Span, 0, fanin)
	st.stage14Rewire(op, pile1, pile2, 0)
	st.stage14Rewire(op, pile1, pile3, pile2.Channels)
}

func (st *state) stage14() error {
	n := 0
	for i, op := range st.plan.Seq {
		switch op.Nodes[0].(type) {
		case *raw.Concat:
			st.stage14Bypass(op)
			*op = plan.Op{}
			continue
		}
		if n < i {
			st.plan.Seq[n] = op
			st.plan.Seq[i] = nil
		}
		n += 1
	}
	st.plan.Seq = st.plan.Seq[:n]
	return nil
}

func (st *state) stage15InputOutput(pile *plan.Pile) bool {
	for _, span := range pile.Writers {
		switch span.Op.Nodes[0].(type) {
		case *raw.Input:
			return true
		}
	}
	for _, span := range pile.Readers {
		switch span.Op.Nodes[0].(type) {
		case *raw.Output:
			return true
		}
	}
	return false
}

func (st *state) stage15Include(pile *plan.Pile) bool {
	for _, span := range pile.Readers {
		op := span.Op
		switch op.Nodes[0].(type) {
		case *raw.FullyConnected:
			if op.From[0] == span {
				return false
			}
		}
	}
	return true
}

func (st *state) stage15Edit(pile *plan.Pile) {
	if pile.ElemBytes != 0 {
		return
	}
	elem := 4
	pitch1 := pile.Width * elem
	pitch2 := pile.Height * pitch1
	size := pile.Channels * pitch2
	offset := 0
	if st.stage15InputOutput(pile) {
		offset = -1
	} else if st.stage15Include(pile) {
		switch st.config.Platform {
		case raw.AVX512Float32:
			pad := func(y int) int {
				const line = 1 << 6
				if y <= line {
					return y
				}
				return (y+line-1)&-line | line
			}
			pitch2 = pad(pitch2)
			size = pad(pile.Channels * pitch2)
		default:
			panic("bug")
		}
	}
	pile.ElemBytes = elem
	pile.Pitch1Bytes = pitch1
	pile.Pitch2Bytes = pitch2
	pile.SizeBytes = size
	pile.OffsetBytes = offset
}

func (st *state) stage15() error {
	for _, op := range st.plan.Seq {
		for _, span := range op.To {
			for _, pile := range span.Piles {
				st.stage15Edit(pile)
			}
		}
	}
	return nil
}

type stage16Cell struct {
	pile  *plan.Pile
	first int
	last  int
	guide []int
}

func stage16Rank(cells []stage16Cell, i1, i2 int) bool {
	cell1 := &cells[i1]
	cell2 := &cells[i2]
	size1 := cell1.pile.SizeBytes
	size2 := cell2.pile.SizeBytes
	if size1 != size2 {
		return size1 > size2
	}
	diff1 := cell1.last - cell1.first
	diff2 := cell2.last - cell2.first
	if diff1 != diff2 {
		return diff1 > diff2
	}
	return i1 < i2
}

func (st *state) stage16Guides(cells []stage16Cell) {
	n := len(cells)
	for i1 := range cells {
		c1 := &cells[i1]
		if c1.pile.OffsetBytes < 0 {
			continue
		}
		last := c1.last
		for i2 := i1 + 1; i2 < n; i2++ {
			c2 := &cells[i2]
			if last < c2.first {
				break
			}
			if c2.pile.OffsetBytes < 0 {
				continue
			}
			if stage16Rank(cells, i1, i2) {
				c2.guide = append(c2.guide, i1)
				continue
			}
			c1.guide = append(c1.guide, i2)
		}
	}
}

func (st *state) stage16Cells() []stage16Cell {
	guess := len(st.plan.Seq)
	index := make(map[*plan.Pile]int, guess)
	cells := make([]stage16Cell, 0, guess)
	for i, op := range st.plan.Seq {
		for j, span1 := range op.From {
			cells[index[span1.Piles[0]]].last = i
			mods := op.FromMods[j]
			for k := range mods {
				for _, span2 := range mods[k].From {
					cells[index[span2.Piles[0]]].last = i
				}
			}
		}
		for j, span1 := range op.To {
			for _, pile := range span1.Piles {
				if at, ok := index[pile]; ok {
					cells[at].last = i
					continue
				}
				index[pile] = len(cells)
				cells = append(cells, stage16Cell{
					pile:  pile,
					first: i,
					last:  i,
				})
			}
			mods := op.ToMods[j]
			for k := range mods {
				for _, span2 := range mods[k].From {
					cells[index[span2.Piles[0]]].last = i
				}
			}
		}
	}
	st.stage16Guides(cells)
	return cells
}

type stage16ByRank struct {
	cells []stage16Cell
	seq   []int
}

func (by *stage16ByRank) Len() int {
	return len(by.seq)
}

func (by *stage16ByRank) Less(i, j int) bool {
	return stage16Rank(by.cells, by.seq[i], by.seq[j])
}

func (by *stage16ByRank) Swap(i, j int) {
	by.seq[i], by.seq[j] = by.seq[j], by.seq[i]
}

type stage16ByOffset struct {
	cells []stage16Cell
	guide []int
}

func (by *stage16ByOffset) Len() int {
	return len(by.guide)
}

func (by *stage16ByOffset) Less(i, j int) bool {
	ii := by.cells[by.guide[i]].pile.OffsetBytes
	jj := by.cells[by.guide[j]].pile.OffsetBytes
	return ii < jj
}

func (by *stage16ByOffset) Swap(i, j int) {
	by.guide[i], by.guide[j] = by.guide[j], by.guide[i]
}

func (st *state) stage16() error {
	cells := st.stage16Cells()
	byRank := &stage16ByRank{
		cells: cells,
		seq:   make([]int, len(cells)),
	}
	for i := range byRank.seq {
		byRank.seq[i] = i
	}
	sort.Sort(byRank)
	byOffset := &stage16ByOffset{
		cells: cells,
	}
	for _, i1 := range byRank.seq {
		cell1 := &cells[i1]
		pile1 := cell1.pile
		if pile1.OffsetBytes < 0 {
			continue
		}
		offset1, size1 := 0, pile1.SizeBytes
		byOffset.guide = cell1.guide
		sort.Sort(byOffset)
		for _, i2 := range byOffset.guide {
			pile2 := cells[i2].pile
			offset2 := pile2.OffsetBytes
			if offset1+size1 <= offset2 {
				break
			}
			min := offset2 + pile2.SizeBytes
			if offset1 < min {
				offset1 = min
			}
		}
		pile1.OffsetBytes = offset1
	}
	return nil
}

func (st *state) stage17() error {
	st.h, st.c = author.Implement(&st.plan)
	return nil
}
