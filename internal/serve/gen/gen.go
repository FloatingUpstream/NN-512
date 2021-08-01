package gen

import (
	"NN-512/internal/compile"
	"NN-512/internal/serve/listing"
)

func Handle(urlPath, graph string) {
	result, err := compile.Compile(graph)
	if err != nil {
		panic(err)
	}
	output := "Output " + result.Name
	listing.Handle(
		urlPath,
		listing.Elem{
			Heading: "Input graph file",
			Content: graph,
		},
		listing.Elem{
			Heading: output + ".h file",
			Content: string(result.H),
		},
		listing.Elem{
			Heading: output + ".c file",
			Content: string(result.C),
		},
	)
}
