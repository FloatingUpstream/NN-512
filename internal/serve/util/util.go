package util

import (
	"NN-512/internal/serve/stats"
	"bytes"
	"compress/gzip"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

func Handle(pattern string, handler http.Handler) {
	http.HandleFunc(
		pattern,
		func(w http.ResponseWriter, r *http.Request) {
			switch {
			case r.URL.Path != pattern:
				http.NotFound(w, r)
			case r.Method != http.MethodGet:
				const (
					msg  = "405 method not allowed"
					code = http.StatusMethodNotAllowed
				)
				w.Header().Set("Allow", http.MethodGet)
				http.Error(w, msg, code)
			default:
				stats.Request(r)
				handler.ServeHTTP(w, r)
			}
		},
	)
}

type Page struct {
	plain []byte
	press []byte
	cLen1 string
	cLen2 string
}

func NewPage(plain []byte) *Page {
	var press bytes.Buffer
	gz, _ := gzip.NewWriterLevel(
		&press, gzip.BestCompression,
	)
	_, _ = gz.Write(plain)
	_ = gz.Close()
	return &Page{
		plain: plain,
		press: press.Bytes(),
		cLen1: strconv.Itoa(len(plain)),
		cLen2: strconv.Itoa(press.Len()),
	}
}

func (p *Page) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var (
		from = p.plain
		cLen = p.cLen1
	)
	for _, val := range r.Header.Values("Accept-Encoding") {
		if strings.Contains(val, "gzip") {
			w.Header().Set("Content-Encoding", "gzip")
			from = p.press
			cLen = p.cLen2
			break
		}
	}
	w.Header().Set("Content-Length", cLen)
	_, _ = w.Write(from)
}

type File struct {
	disp string
	cLen string
	body []byte
}

func NewFile(name string, body []byte) *File {
	const format = `attachment; filename="%s"`
	return &File{
		disp: fmt.Sprintf(format, name),
		cLen: strconv.Itoa(len(body)),
		body: body,
	}
}

func (f *File) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Disposition", f.disp)
	w.Header().Set("Content-Length", f.cLen)
	_, _ = w.Write(f.body)
}
