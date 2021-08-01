package stats

import (
	"NN-512/internal/serve/html"
	"fmt"
	"net/http"
	"strings"
	"sync"
)

var (
	mut   sync.Mutex
	reqs  int
	addrs map[string]struct{}
)

func Prep() {
	addrs = make(map[string]struct{})
	http.HandleFunc(
		"/stats",
		func(w http.ResponseWriter, r *http.Request) {
			const format = `<h2>Stats</h2><ul>` +
				`<li>HTTPS requests: %d</li>` +
				`<li>Unique IP addresses: %d</li>` +
				`</ul>`
			mut.Lock()
			var (
				Reqs  = reqs
				Addrs = len(addrs)
			)
			mut.Unlock()
			var (
				body = fmt.Sprintf(format, Reqs, Addrs)
				Html = html.FromString(body)
			)
			_, _ = w.Write(Html)
		},
	)
}

func Request(r *http.Request) {
	var (
		addr = r.RemoteAddr
		chop = strings.LastIndexByte(addr, ':')
	)
	if chop != -1 {
		addr = addr[:chop]
	}
	mut.Lock()
	reqs++
	addrs[addr] = struct{}{}
	mut.Unlock()
}
