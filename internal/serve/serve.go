package serve

import (
	"NN-512/internal/serve/root"
	"net/http"
)

func Website(addr, cert, key, src, bin string) error {
	err := root.Prep(src, bin)
	if err != nil {
		return err
	}
	return http.ListenAndServeTLS(
		addr, cert, key, nil,
	)
}
