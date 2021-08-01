package features

import (
	"NN-512/internal/serve/features/elwi"
	"NN-512/internal/serve/features/loom"
	"NN-512/internal/serve/features/one"
	"NN-512/internal/serve/features/strider"
	"NN-512/internal/serve/features/various"
	"NN-512/internal/serve/features/wct"
	"fmt"
	"strings"
)

func Prep() string {
	var (
		sb    *strings.Builder
		outer [][]string
		inner []string
	)
	layer4 := func() {
		const (
			begin = "<ul><li>%s<ul>"
			each  = "<li>%s</li>"
			end   = "</ul></li></ul>"
		)
		_, _ = fmt.Fprintf(
			sb, begin, inner[0],
		)
		for _, elem := range inner[1:] {
			_, _ = fmt.Fprintf(
				sb, each, elem,
			)
		}
		_, _ = sb.WriteString(end)
	}
	layer3 := func() {
		_, _ = sb.WriteString(
			"<h2>Main Features</h2>",
		)
		for _, inner = range outer {
			layer4()
		}
	}
	layer2 := func() {
		outer = [][]string{
			loom.Prep(),
			strider.Prep(),
			wct.Prep(),
			one.Prep(),
			various.Prep(),
			elwi.Prep(),
		}
		layer3()
	}
	layer1 := func() string {
		sb = new(strings.Builder)
		layer2()
		return sb.String()
	}
	return layer1()
}
