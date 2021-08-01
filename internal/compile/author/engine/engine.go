package engine

import (
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/author/errmsg"
	"NN-512/internal/compile/author/net"
	"NN-512/internal/compile/author/threader"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
	"fmt"
	"sort"
)

func il(i int) cgen.Gen {
	return cgen.IntLit(i)
}

func vb(s string) cgen.Gen {
	return cgen.Vb(s)
}

func ptr(t cgen.Gen) cgen.Gen {
	return cgen.Ptr{Type: t}
}

type tensor struct {
	name   string
	chans  int
	height int
	width  int
}

type tensors []*tensor

func (ts tensors) Len() int {
	return len(ts)
}

func (ts tensors) Less(i, j int) bool {
	return ts[i].name < ts[j].name
}

func (ts tensors) Swap(i, j int) {
	ts[i], ts[j] = ts[j], ts[i]
}

type Ctx struct {
	nms              nmsrc.Src
	emc              *errmsg.Ctx
	tc               *threader.Ctx
	nc               *net.Ctx
	structName       string
	StructNet        string
	StructTeam       string
	structAlloc      string
	StructAlign      string
	Alignment        int
	Split            int
	createName       string
	pthreadTName     string
	inferenceName    string
	InferenceEng     cgen.Gen
	inferenceTensors tensors
	destroyName      string
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src, emc *errmsg.Ctx, tc *threader.Ctx, nc *net.Ctx) *Ctx {
	var (
		structName = pl.Config.Prefix + "Engine"
		alignment  = nc.Alignment
	)
	split := func() (bytes int) {
		for _, op := range pl.Seq {
			for _, span := range op.To {
				for _, pile := range span.Piles {
					first := pile.OffsetBytes
					if first < 0 {
						continue
					}
					past := first + pile.SizeBytes
					if bytes < past {
						bytes = past
					}
				}
			}
		}
		bytes += alignment - 1
		bytes &= -alignment
		return
	}
	Tensors := func() (ts tensors) {
		put := func(t *tensor) {
			ts = append(ts, t)
		}
		for _, op := range pl.Seq {
			switch node := op.Nodes[0].(type) {
			case *raw.Input:
				put(&tensor{
					name:   node.ToTensor,
					chans:  node.Channels,
					height: node.Height,
					width:  node.Width,
				})
			case *raw.Output:
				var (
					span = op.From[0]
					pile = span.Piles[0]
				)
				put(&tensor{
					name:   node.FromTensor,
					chans:  pile.Channels,
					height: pile.Height,
					width:  pile.Width,
				})
			}
		}
		sort.Sort(ts)
		return
	}
	return &Ctx{
		nms:              nms,
		emc:              emc,
		tc:               tc,
		nc:               nc,
		structName:       structName,
		StructNet:        nms.Name("net"),
		StructTeam:       nms.Name("team"),
		structAlloc:      nms.Name("alloc"),
		StructAlign:      nms.Name("align"),
		Alignment:        alignment,
		Split:            split(),
		createName:       structName + "Create",
		pthreadTName:     structName + "PthreadT",
		inferenceName:    structName + "Inference",
		InferenceEng:     vb(nms.Name("eng")),
		inferenceTensors: Tensors(),
		destroyName:      structName + "Destroy",
	}
}

func (c *Ctx) Comment() cgen.Gen {
	const (
		space  = ` `
		indent = space + space + space + space
	)
	var comment cgen.Comment
	text := func(lines ...string) {
		comment = append(comment, lines...)
	}
	text(
		`An Engine performs inference. It contains inference threads, scratch`,
		`memory, and a pointer to the Net. Any number of Engines can share the`,
		`same Net (and perform inference in parallel) because the Net is never`,
		`modified. For best performance the number of inference threads should`,
		`not exceed the number of CPU cores.`,
		``,
		indent+c.nc.StructName+`* net;`,
		``,
		indent+`... Create net ...`,
		``,
		indent+c.structName+`* engine; // For example, 4 inference threads:`,
		indent+`char* err = `+c.createName+`(&engine, net, 4);`,
		``,
		indent+`if (err) { // Nonzero err means failure; engine is unmodified.`,
		indent+indent+`printf("%s\n", err); // Explain the failure, add a newline.`,
		indent+indent+`free(err); // Free the error string to avoid a memory leak.`,
		``,
		indent+indent+`... Destroy net ...`,
		``,
		indent+indent+`exit(1); // Exit, or propagate the failure some other way.`,
		indent+`}`,
		``,
		indent+`... Use the POSIX threads API to adjust engine's threads ...`,
		indent+`... Use engine to perform inference (dependent on net) ...`,
		``,
		indent+c.destroyName+`(engine); // Terminate threads, free memory.`,
		``,
		indent+`... Destroy net ...`,
		``,
		`The POSIX threads API can be used to adjust an Engine's threads. If`,
		`an Engine has N threads, those threads are indexed 0, 1, 2, ..., N-1`,
		`and a pthread_t identifier is associated with each index. To set the`,
		`CPU affinity mask for the first inference thread, for example:`,
		``,
		indent+`pthread_t thread; // The first thread has index 0:`,
		indent+`char* err = `+c.pthreadTName+`(engine, 0, &thread);`,
		``,
		indent+`assert(!err); // Can only fail if the thread index is invalid.`,
		``,
		indent+`pthread_setaffinity_np(thread, ...); // Details omitted.`,
		``,
		`The inference function reads floats from (one or more) input tensors`,
		`and writes floats to (one or more) output tensors. All the input and`,
		`output tensors are owned (allocated and freed) by the caller and are`,
		`in CHW format, 32-bit floating point, fully packed (in other words,`,
		`C has the largest pitch, W has the smallest pitch, and there is no`,
		`padding anywhere).`,
		``,
	)
	for _, t := range c.inferenceTensors {
		text(fmt.Sprintf(
			indent+`float* %s = malloc(sizeof(float)*%d*%d*%d);`,
			t.name, t.chans, t.height, t.width,
		))
	}
	text(
		``,
		indent+`for (...) { // Reuse the input and output tensors.`,
		``,
		indent+indent+`... Write the input floats ...`,
		``,
		indent+indent+c.inferenceName+`( // This function cannot fail.`,
		indent+indent+indent+`engine, // Pass an Engine as the first argument.`,
	)
	for x, t := range c.inferenceTensors {
		var follow string
		switch x {
		case 0:
			follow = `, // The tensor arguments are sorted by name.`
		case len(c.inferenceTensors) - 1:
		default:
			follow = `,`
		}
		text(fmt.Sprintf(
			indent+indent+indent+`%s%s`,
			t.name, follow,
		))
	}
	text(
		indent+indent+`);`,
		``,
		indent+indent+`... Read the output floats ...`,
		``,
		indent+`}`,
		``,
	)
	for _, t := range c.inferenceTensors {
		text(fmt.Sprintf(
			indent+`free(%s);`,
			t.name,
		))
	}
	text(
		``,
		`The tensor parameters of the inference function are ordered by name,`,
		`lexically bytewise. In other words, the function parameters have been`,
		`sorted by name using Go's "<" string comparison operator (a bytewise`,
		`lexical string sort).`,
	)
	return comment
}

func (c *Ctx) StructFwd() cgen.Gen {
	return cgen.StructFwd(c.structName)
}

func (c *Ctx) StructDef() cgen.Gen {
	return cgen.StructDef{
		Name: c.structName,
		Fields: cgen.Stmts{
			cgen.Field{
				Type: ptr(vb(c.nc.StructName)),
				What: vb(c.StructNet),
			},
			cgen.Field{
				Type: c.tc.PtrTeam,
				What: vb(c.StructTeam),
			},
			cgen.Field{
				Type: cgen.PtrChar,
				What: vb(c.structAlloc),
			},
			cgen.Field{
				Type: cgen.PtrChar,
				What: vb(c.StructAlign),
			},
		},
	}
}

func (c *Ctx) CreateDecl() cgen.Gen {
	return cgen.FuncDecl{
		ReturnType: cgen.PtrChar,
		Name:       c.createName,
		Params: cgen.CommaLines{
			ptr(ptr(vb(c.structName))),
			ptr(vb(c.nc.StructName)),
			cgen.Param{
				Type: cgen.PtrdiffT,
				What: vb("threads"),
			},
		},
	}
}

func (c *Ctx) CreateDef(bytes int) cgen.Gen {
	var (
		paramEng     = vb(c.nms.Name("eng"))
		paramNet     = vb(c.nms.Name("net"))
		paramThreads = vb(c.nms.Name("threads"))
		eng          = vb(c.nms.Name("eng"))
		alloc        = vb(c.nms.Name("alloc"))
	)
	freeEng := cgen.Call{
		Func: cgen.Free, Args: eng,
	}
	freeAlloc := cgen.Call{
		Func: cgen.Free, Args: alloc,
	}
	field := func(nm string) cgen.Gen {
		return cgen.Arrow{
			Expr: eng, Name: nm,
		}
	}
	align := cgen.Cast{
		Type: cgen.PtrVoid,
		Expr: cgen.Paren{
			Inner: cgen.And{
				Expr1: cgen.Paren{
					Inner: cgen.Add{
						Expr1: cgen.Cast{
							Type: cgen.SizeT,
							Expr: alloc,
						},
						Expr2: il(c.Alignment - 1),
					},
				},
				Expr2: il(-c.Alignment),
			},
		},
	}
	return cgen.FuncDef{
		ReturnType: cgen.PtrChar,
		Name:       c.createName,
		Params: cgen.CommaLines{
			cgen.Param{
				Type: ptr(ptr(vb(c.structName))),
				What: paramEng,
			},
			cgen.Param{
				Type: ptr(vb(c.nc.StructName)),
				What: paramNet,
			},
			cgen.Param{
				Type: cgen.PtrdiffT,
				What: paramThreads,
			},
		},
		Body: cgen.Stmts{
			cgen.Var{
				Type: ptr(vb(c.structName)),
				What: eng,
				Init: cgen.Call{
					Func: cgen.Malloc,
					Args: cgen.Sizeof{
						What: vb(c.structName),
					},
				},
			},
			&errmsg.ErrnoIf{
				Ctx:  c.emc,
				Cond: cgen.IsZero{Expr: eng},
			},
			cgen.Var{
				Type: cgen.PtrChar,
				What: alloc,
				Init: cgen.Call{
					Func: cgen.Malloc,
					Args: il(
						c.Alignment - 1 +
							c.Split +
							bytes,
					),
				},
			},
			&errmsg.ErrnoIf{
				Ctx:    c.emc,
				Cond:   cgen.IsZero{Expr: alloc},
				Unwind: freeEng,
			},
			cgen.Assign{
				Expr1: field(c.structAlloc),
				Expr2: alloc,
			},
			cgen.Assign{
				Expr1: field(c.StructAlign),
				Expr2: align,
			},
			&threader.Create{
				Ctx: c.tc,
				Team: cgen.Addr{
					Expr: field(c.StructTeam),
				},
				Nt: paramThreads,
				Unwind: cgen.Stmts{
					freeEng,
					freeAlloc,
				},
			},
			cgen.Assign{
				Expr1: field(c.StructNet),
				Expr2: paramNet,
			},
			cgen.Assign{
				Expr1: cgen.At{
					Expr: paramEng,
				},
				Expr2: eng,
			},
			cgen.Return{
				Expr: il(0),
			},
		},
	}
}

func (c *Ctx) PthreadTDecl() cgen.Gen {
	return cgen.FuncDecl{
		ReturnType: cgen.PtrChar,
		Name:       c.pthreadTName,
		Params: cgen.CommaLines{
			ptr(vb(c.structName)),
			cgen.Param{
				Type: cgen.PtrdiffT,
				What: vb("threadIdx"),
			},
			cgen.Param{
				Type: cgen.PtrPthreadT,
				What: vb("to"),
			},
		},
	}
}

func (c *Ctx) PthreadTDef() cgen.Gen {
	var (
		paramEng = vb(c.nms.Name("eng"))
		paramIdx = vb(c.nms.Name("idx"))
		paramTo  = vb(c.nms.Name("to"))
	)
	return cgen.FuncDef{
		ReturnType: cgen.PtrChar,
		Name:       c.pthreadTName,
		Params: cgen.CommaLines{
			cgen.Param{
				Type: ptr(vb(c.structName)),
				What: paramEng,
			},
			cgen.Param{
				Type: cgen.PtrdiffT,
				What: paramIdx,
			},
			cgen.Param{
				Type: cgen.PtrPthreadT,
				What: paramTo,
			},
		},
		Body: &threader.PthreadT{
			Ctx: c.tc,
			Thr: paramTo,
			Team: cgen.Arrow{
				Expr: paramEng,
				Name: c.StructTeam,
			},
			Idx: paramIdx,
		},
	}
}

func (c *Ctx) inferenceParams(isDef bool) cgen.Gen {
	var (
		n     = len(c.inferenceTensors)
		lines = make(cgen.CommaLines, 1+n)
	)
	lines[0] = ptr(vb(c.structName))
	if isDef {
		lines[0] = cgen.Param{
			Type: lines[0],
			What: c.InferenceEng,
		}
	}
	for x, t := range c.inferenceTensors {
		lines[1+x] = cgen.Param{
			Type: cgen.PtrFloat,
			What: vb(t.name),
		}
	}
	return lines
}

func (c *Ctx) InferenceDecl() cgen.Gen {
	return cgen.FuncDecl{
		ReturnType: cgen.Void,
		Name:       c.inferenceName,
		Params:     c.inferenceParams(false),
	}
}

func (c *Ctx) InferenceDef(body cgen.Gen) cgen.Gen {
	return cgen.FuncDef{
		ReturnType: cgen.Void,
		Name:       c.inferenceName,
		Params:     c.inferenceParams(true),
		Body:       body,
	}
}

func (c *Ctx) DestroyDecl() cgen.Gen {
	return cgen.FuncDecl{
		ReturnType: cgen.Void,
		Name:       c.destroyName,
		Params:     ptr(vb(c.structName)),
	}
}

func (c *Ctx) DestroyDef() cgen.Gen {
	paramEng := vb(c.nms.Name("eng"))
	return cgen.FuncDef{
		ReturnType: cgen.Void,
		Name:       c.destroyName,
		Params: cgen.Param{
			Type: ptr(vb(c.structName)),
			What: paramEng,
		},
		Body: cgen.Stmts{
			&threader.Destroy{
				Ctx: c.tc,
				Team: cgen.Arrow{
					Expr: paramEng,
					Name: c.StructTeam,
				},
			},
			cgen.Call{
				Func: cgen.Free,
				Args: cgen.Arrow{
					Expr: paramEng,
					Name: c.structAlloc,
				},
			},
			cgen.Call{
				Func: cgen.Free,
				Args: paramEng,
			},
		},
	}
}
