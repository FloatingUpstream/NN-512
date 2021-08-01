package net

import (
	"NN-512/internal/compile/author/cgen"
	"NN-512/internal/compile/plan"
	"NN-512/internal/nmsrc"
	"NN-512/internal/raw"
)

func vb(s string) cgen.Gen {
	return cgen.Vb(s)
}

type Ctx struct {
	StructName    string
	StructAlloc   string
	StructAlign   string
	Alignment     int
	paramsName    string
	createName    string
	CreateNet     cgen.Gen
	CreateParams  cgen.Gen
	CreateThreads cgen.Gen
	destroyName   string
	destroyNet    cgen.Gen
}

func NewCtx(pl *plan.Plan, nms nmsrc.Src, paramsName string) *Ctx {
	var (
		structName = pl.Config.Prefix + "Net"
		alignment  int
	)
	switch pl.Config.Platform {
	case raw.AVX512Float32:
		alignment = 64
	default:
		panic("bug")
	}
	return &Ctx{
		StructName:    structName,
		StructAlloc:   nms.Name("alloc"),
		StructAlign:   nms.Name("align"),
		Alignment:     alignment,
		paramsName:    paramsName,
		createName:    structName + "Create",
		CreateNet:     vb(nms.Name("net")),
		CreateParams:  vb(nms.Name("params")),
		CreateThreads: vb(nms.Name("threads")),
		destroyName:   structName + "Destroy",
		destroyNet:    vb(nms.Name("net")),
	}
}

func (c *Ctx) Comment() cgen.Gen {
	const (
		space  = " "
		indent = space + space + space + space
	)
	return cgen.Comment{
		`The Net contains weights, biases, and other trained parameters in a`,
		`form that enables efficient inference. It is created from the input`,
		`parameter struct without modifying that struct. The input parameter`,
		`struct is no longer needed once the Net has been created. Threads`,
		`that are used to create the Net are temporary (in particular, those`,
		`threads are not used for inference).`,
		``,
		indent + c.paramsName + `* params = malloc(sizeof(` + c.paramsName + `));`,
		``,
		indent + `... Load params (read from a file, perhaps) ...`,
		``,
		indent + c.StructName + `* net; // For example, 4 threads:`,
		indent + `char* err = ` + c.createName + `(&net, params, 4);`,
		indent + `free(params);`,
		``,
		indent + `if (err) { // Nonzero err indicates failure; net is unmodified.`,
		indent + indent + `printf("%s\n", err); // Explain the failure, add a newline.`,
		indent + indent + `free(err); // Free the error string to avoid a memory leak.`,
		indent + indent + `exit(1); // Exit, or propagate the failure some other way.`,
		indent + `}`,
		``,
		indent + `... Perform all inference that depends on net ...`,
		``,
		indent + c.destroyName + `(net);`,
		``,
		`The Net can be shared and reused without restriction because it is`,
		`never modified (not even temporarily) after being created. The Net`,
		`should be destroyed (to free memory) once all dependent inference`,
		`is complete.`,
	}
}

func (c *Ctx) StructFwd() cgen.Gen {
	return cgen.StructFwd(c.StructName)
}

func (c *Ctx) StructDef() cgen.Gen {
	return cgen.StructDef{
		Name: c.StructName,
		Fields: cgen.Stmts{
			cgen.Field{
				Type: cgen.PtrChar,
				What: vb(c.StructAlloc),
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
			cgen.Ptr{
				Type: cgen.Ptr{
					Type: vb(c.StructName),
				},
			},
			cgen.Ptr{
				Type: vb(c.paramsName),
			},
			cgen.Param{
				Type: cgen.PtrdiffT,
				What: vb("threads"),
			},
		},
	}
}

func (c *Ctx) CreateDef(body cgen.Gen) cgen.Gen {
	return cgen.FuncDef{
		ReturnType: cgen.PtrChar,
		Name:       c.createName,
		Params: cgen.CommaLines{
			cgen.Param{
				Type: cgen.Ptr{
					Type: cgen.Ptr{
						Type: vb(c.StructName),
					},
				},
				What: c.CreateNet,
			},
			cgen.Param{
				Type: cgen.Ptr{
					Type: vb(c.paramsName),
				},
				What: c.CreateParams,
			},
			cgen.Param{
				Type: cgen.PtrdiffT,
				What: c.CreateThreads,
			},
		},
		Body: body,
	}
}

func (c *Ctx) DestroyDecl() cgen.Gen {
	return cgen.FuncDecl{
		ReturnType: cgen.Void,
		Name:       c.destroyName,
		Params: cgen.Ptr{
			Type: vb(c.StructName),
		},
	}
}

func (c *Ctx) DestroyDef() cgen.Gen {
	return cgen.FuncDef{
		ReturnType: cgen.Void,
		Name:       c.destroyName,
		Params: cgen.Param{
			Type: cgen.Ptr{
				Type: vb(c.StructName),
			},
			What: c.destroyNet,
		},
		Body: cgen.Stmts{
			cgen.Call{
				Func: cgen.Free,
				Args: cgen.Arrow{
					Expr: c.destroyNet,
					Name: c.StructAlloc,
				},
			},
			cgen.Call{
				Func: cgen.Free,
				Args: c.destroyNet,
			},
		},
	}
}
