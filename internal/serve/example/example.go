package example

import (
	"NN-512/internal/raw"
	"NN-512/internal/serve/gen"
	"fmt"
	"strconv"
	"strings"
)

var ctr = 0

func Prep(desc string, nodes ...raw.Node) string {
	var (
		urlSuffix string
		urlPath   string
		sb        *strings.Builder
	)
	text := func(s string) {
		_, _ = sb.WriteString(s)
	}
	line := func(format string, a ...interface{}) {
		_, _ = fmt.Fprintf(sb, format, a...)
		text("\n")
	}
	ftoa := func(f float32) string {
		return strconv.FormatFloat(
			float64(f), 'f', -1, 32,
		)
	}
	layer5 := func() {
		for _, node := range nodes {
			switch node := node.(type) {
			case *raw.Activation:
				line(
					"Activation FromTensor=%s ToTensor=%s Kind=%s Param=%s",
					node.FromTensor,
					node.ToTensor,
					raw.ActivationStrings[node.Kind],
					ftoa(node.Param),
				)
			case *raw.Add:
				line(
					"Add FromTensor1=%s FromTensor2=%s ToTensor=%s",
					node.FromTensor1,
					node.FromTensor2,
					node.ToTensor,
				)
			case *raw.BatchNorm:
				line(
					"BatchNorm FromTensor=%s ToTensor=%s Epsilon=%s",
					node.FromTensor,
					node.ToTensor,
					ftoa(node.Epsilon),
				)
			case *raw.Concat:
				line(
					"Concat FromTensor1=%s FromTensor2=%s ToTensor=%s",
					node.FromTensor1,
					node.FromTensor2,
					node.ToTensor,
				)
			case *raw.Conv:
				line(
					"Conv "+
						"FromTensor=%s "+
						"ToTensor=%s "+
						"ToChannels=%d "+
						"FilterH=%d "+
						"FilterW=%d "+
						"StrideH=%d "+
						"StrideW=%d "+
						"PaddingH=%d "+
						"PaddingW=%d "+
						"DilationH=%d "+
						"DilationW=%d "+
						"Groups=%d",
					node.FromTensor,
					node.ToTensor,
					node.ToChannels,
					node.FilterH,
					node.FilterW,
					node.StrideH,
					node.StrideW,
					node.PaddingH,
					node.PaddingW,
					node.DilationH,
					node.DilationW,
					node.Groups,
				)
			case *raw.FullyConnected:
				line(
					"FullyConnected FromTensor=%s ToTensor=%s ToChannels=%d",
					node.FromTensor,
					node.ToTensor,
					node.ToChannels,
				)
			case *raw.Input:
				line(
					"Input ToTensor=%s Channels=%d Height=%d Width=%d",
					node.ToTensor,
					node.Channels,
					node.Height,
					node.Width,
				)
			case *raw.Output:
				line(
					"Output FromTensor=%s",
					node.FromTensor,
				)
			case *raw.Pooling:
				line(
					"Pooling "+
						"FromTensor=%s "+
						"ToTensor=%s "+
						"Kind=%s "+
						"PaddingH=%d "+
						"PaddingW=%d",
					node.FromTensor,
					node.ToTensor,
					raw.PoolingStrings[node.Kind],
					node.PaddingH,
					node.PaddingW,
				)
			case *raw.Softmax:
				line(
					"Softmax FromTensor=%s ToTensor=%s",
					node.FromTensor,
					node.ToTensor,
				)
			default:
				panic("bug")
			}
		}
	}
	layer4 := func() {
		const head = "Config"
		text(head)
		for _, seg := range raw.Guide[head].Segs {
			text(" ")
			text(seg.Label)
			text(raw.Binder)
			switch seg.Label {
			case "Prefix":
				text("Example")
				text(urlSuffix)
			default:
				text(seg.Default)
			}
		}
		text("\n")
		layer5()
	}
	layer3 := func() {
		sb = new(strings.Builder)
		layer4()
		graph := sb.String()
		gen.Handle(urlPath, graph)
	}
	layer2 := func() string {
		layer3()
		const format = `<a href="%s">Example</a>: %s`
		return fmt.Sprintf(format, urlPath, desc)
	}
	layer1 := func() string {
		ctr++
		const urlPrefix = "/example/"
		urlSuffix = strconv.Itoa(ctr)
		urlPath = urlPrefix + urlSuffix
		return layer2()
	}
	return layer1()
}
