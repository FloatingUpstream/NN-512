package params

import (
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/plan"
	"fmt"
	"sort"
)

const (
	indent = space + space + space + space
	space  = " "
	suffix = "Params"
)

func Name(pl *plan.Plan) string {
	return pl.Config.Prefix + suffix
}

func Fwd(name string) cgen.Gen {
	comment := cgen.Comment{
		`All weights, biases, and other trained parameters are passed into`,
		`the initialization code through the ` + suffix + ` struct that is declared`,
		`just below this comment. The corresponding struct definition can be`,
		`found near the end of this header file.`,
		``,
		`Each field of the ` + suffix + ` struct is an array of float that holds a`,
		`parameter tensor in NCHW format with no padding. The struct fields`,
		`are ordered by name, lexically bytewise. If you concatenate all the`,
		`trained parameter tensors to a file in this same format and order`,
		`you can load the struct as follows (error checking omitted here):`,
		``,
		indent + `size_t size = sizeof(` + name + `);`,
		indent + name + `* to = malloc(size);`,
		indent + `FILE* from = fopen("` + suffix + `File", "r");`,
		indent + `fread(to, size, 1, from);`,
		indent + `fclose(from);`,
		``,
		`Be careful to match endianness (and floating point format).`,
	}
	return cgen.Gens{
		comment,
		cgen.Newline,
		cgen.StructFwd(name),
	}
}

type byTensor []*plan.Param

func (by byTensor) Len() int {
	return len(by)
}

func (by byTensor) Less(i, j int) bool {
	return by[i].Tensor < by[j].Tensor
}

func (by byTensor) Swap(i, j int) {
	by[i], by[j] = by[j], by[i]
}

func gather(pl *plan.Plan) (by byTensor) {
	take := func(ps []plan.Param) {
		for i := range ps {
			by = append(by, &ps[i])
		}
	}
	mods := func(a [][]plan.Mod) {
		for _, ms := range a {
			for i := range ms {
				take(ms[i].Params)
			}
		}
	}
	for _, op := range pl.Seq {
		for i, ps := range op.Params {
			take(ps)
			mods(op.ParamMods[i][:])
		}
		mods(op.FromMods)
		mods(op.ToMods)
	}
	sort.Sort(by)
	return
}

func fields(pl *plan.Plan) cgen.Gen {
	by := gather(pl)
	const cols = 2
	table := cgen.Table{
		Flat: make([]cgen.Gen, 0, len(by)*cols),
		Cols: cols,
	}
	var tensor string
	for _, param := range by {
		if param.Tensor == tensor {
			continue
		}
		tensor = param.Tensor
		product := 1
		for _, each := range &param.NCHW {
			product *= each
		}
		table.Flat = append(table.Flat, cgen.Field{
			Type: cgen.Float,
			What: cgen.Elem{
				Arr: cgen.Vb(tensor),
				Idx: cgen.IntLit(product),
			},
		})
		table.Flat = append(table.Flat, cgen.Comment{
			fmt.Sprintf("%dx%dx%dx%d",
				param.NCHW[0],
				param.NCHW[1],
				param.NCHW[2],
				param.NCHW[3]),
		})
	}
	return table
}

func Def(pl *plan.Plan, name string) cgen.Gen {
	return cgen.Gens{
		cgen.Comment{
			`The fields of the following struct have been sorted by name using`,
			`Go's "<" string comparison operator (bytewise lexical string sort).`,
			`Tensor dimensions are NxCxHxW where N is the outermost/slowest and`,
			`W is the innermost/fastest. There is no padding anywhere.`,
		},
		cgen.Newline,
		cgen.StructDef{
			Name:   name,
			Fields: fields(pl),
			Attrs:  cgen.Packed,
		},
	}
}
