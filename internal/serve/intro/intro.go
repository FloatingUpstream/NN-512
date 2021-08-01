package intro

import (
	"fmt"
	"strings"
)

func Prep() string {
	var sb *strings.Builder
	put := func(a ...string) {
		const (
			begin = "<ul><li>%s<ul>"
			each  = "<li>%s</li>"
			end   = "</ul></li></ul>"
		)
		_, _ = fmt.Fprintf(
			sb, begin, a[0],
		)
		for _, elem := range a[1:] {
			_, _ = fmt.Fprintf(
				sb, each, elem,
			)
		}
		_, _ = sb.WriteString(end)
	}
	layer6 := func() {
		put(
			"NN-512 generates specialized code "+
				"for each tensor operation",
			"Guided by a description of the "+
				"target CPU cache hierarchy",
			"Thread-level parallelism is "+
				"maximized while limiting "+
				"synchronization costs",
			"Simplified code is generated "+
				"for tensor edges, exploiting "+
				"tile/vector overhang",
			"Complete knowledge of memory "+
				"layout simplifies addressing",
		)
	}
	layer5 := func() {
		put(
			"NN-512 performs a variety of "+
				"inference graph optimizations",
			"Fusion of elementwise operations "+
				"into adjacent operations",
			"Fusion of similar convolutions "+
				"(as needed for, e.g., ResNet)",
			"Removal of concatenations "+
				"(as needed for, e.g., DenseNet)",
			"End-to-end planning of "+
				"memory layout",
		)
		layer6()
	}
	layer4 := func() {
		put(
			"The generated C99 code is "+
				"human-readable and should be "+
				"compiled with GCC 9.1 or later",
			"Earlier versions of GCC may also "+
				"be used, yielding slightly "+
				"inferior object code",
			"The generated C99 code has no "+
				"dependencies outside the "+
				"C POSIX library",
			"NN-512 is a Go program with "+
				"no dependencies outside "+
				"the Go standard library",
			"The NN-512 compiler executable "+
				"is stand-alone",
		)
		layer5()
	}
	layer3 := func() {
		put(
			"NN-512 is a compiler that "+
				"generates C99 code for "+
				"neural net inference",
			"It takes as input a simple text "+
				"description of a convolutional "+
				"neural net inference graph",
			"It produces as output a stand-alone "+
				"C99 implementation of that graph",
			"The generated C99 code uses "+
				"AVX-512 vector instructions "+
				"to perform inference",
		)
		layer4()
	}
	layer2 := func() {
		_, _ = sb.WriteString(
			"<h2>Introduction</h2>",
		)
		layer3()
	}
	layer1 := func() string {
		sb = new(strings.Builder)
		layer2()
		return sb.String()
	}
	return layer1()
}
