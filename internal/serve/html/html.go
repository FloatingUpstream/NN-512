package html

import (
	"bytes"
	"fmt"
	"strings"
)

const format = `<!DOCTYPE html><html><head>` +
	`<meta http-equiv="Content-Type" content="text/html; charset=utf-8">` +
	`<meta name="viewport" content="width=device-width, initial-scale=1.0">` +
	`<title>NN-512</title></head><body><h1>NN-512</h1>%s</body></html>`

func From(body []byte) []byte {
	var buf bytes.Buffer
	_, _ = fmt.Fprintf(&buf, format, body)
	return buf.Bytes()
}

func FromString(body string) []byte {
	var buf bytes.Buffer
	_, _ = fmt.Fprintf(&buf, format, body)
	return buf.Bytes()
}

var escaper = strings.NewReplacer(
	"\t", "    ",
	"\n", "<br>",
	"&", "&amp;",
	"'", "&#39;",
	"<", "&lt;",
	">", "&gt;",
	`"`, "&#34;",
)

func Escape(pre string) string {
	return escaper.Replace(pre)
}
