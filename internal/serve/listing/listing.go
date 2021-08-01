package listing

import (
	"NN-512/internal/serve/html"
	"NN-512/internal/serve/util"
	"bytes"
	"fmt"
)

type Elem struct {
	Heading string
	Content string
}

func Handle(urlPath string, elems ...Elem) {
	var (
		buf *bytes.Buffer
	)
	layer4 := func() {
		const (
			files  = "<h2>Files</h2>"
			top1   = `<a href="#0">Top</a>`
			format = `<h3 id="%d">` + top1 + ` || ` +
				`<a href="#%[1]d">%s</a></h3>` +
				`<pre style="font-size:xx-small;">%s</pre>`
			top2 = `<h2>` + top1 + `</h2>`
		)
		_, _ = buf.WriteString(files)
		for i := range elems {
			_, _ = fmt.Fprintf(
				buf, format, 1+i,
				elems[i].Heading,
				elems[i].Content,
			)
		}
		_, _ = buf.WriteString(top2)
	}
	layer3 := func() {
		const (
			back   = `<h2 id="0"><a href="/">Back</a></h2>`
			index  = "<h2>Index</h2><ul>"
			format = `<li><a href="#%d">%s</a></li>`
			end    = "</ul>"
		)
		_, _ = buf.WriteString(back + index)
		for i := range elems {
			_, _ = fmt.Fprintf(
				buf, format, 1+i,
				elems[i].Heading,
			)
		}
		_, _ = buf.WriteString(end)
		layer4()
	}
	layer2 := func() {
		buf = new(bytes.Buffer)
		layer3()
		var (
			body = buf.Bytes()
			Html = html.From(body)
			page = util.NewPage(Html)
		)
		util.Handle(urlPath, page)
	}
	layer1 := func() {
		for i := range elems {
			elems[i].Heading = html.Escape(
				elems[i].Heading,
			)
			elems[i].Content = html.Escape(
				elems[i].Content,
			)
		}
		layer2()
	}
	layer1()
}
