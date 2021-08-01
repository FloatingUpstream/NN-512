package docs

import (
	"NN-512/internal/doc"
	"NN-512/internal/serve/html"
	"NN-512/internal/serve/util"
	"fmt"
)

func Prep() string {
	const (
		urlPrefix = "/docs/"
		urlPath   = urlPrefix + "graph"
	)
	layer2 := func() {
		const (
			back   = `<h2 id="0"><a href="/">Back</a></h2>`
			text   = `<pre style="font-size:xx-small;">%s</pre>`
			top    = `<h2><a href="#0">Top</a></h2>`
			format = back + text + top
		)
		var (
			raw  = doc.Bytes()
			esc  = html.Escape(string(raw))
			body = fmt.Sprintf(format, esc)
			Html = html.FromString(body)
			page = util.NewPage(Html)
		)
		util.Handle(urlPath, page)
	}
	layer1 := func() string {
		layer2()
		const (
			begin  = "<h2>Documentation</h2><ul>"
			item   = `<li>%s <a href="%s">%s</a></li>`
			end    = "</ul>"
			format = begin + item + end
		)
		return fmt.Sprintf(
			format,
			"Syntax and semantics of the",
			urlPath,
			"NN-512 graph language",
		)
	}
	return layer1()
}
