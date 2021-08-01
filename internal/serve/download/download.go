package download

import (
	"NN-512/internal/serve/download/bin"
	"NN-512/internal/serve/download/src"
	"NN-512/internal/version"
	"fmt"
)

func Prep(srcDir, binFile string) (string, error) {
	const (
		prefix = "/download/"
		format = "<h2>Download (Version %d)</h2>" +
			"<ul><li>%s</li><li>%s</li></ul>"
	)
	srcItem, err := src.Prep(
		srcDir, prefix,
	)
	if err != nil {
		return "", err
	}
	binItem, err := bin.Prep(
		binFile, prefix,
	)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf(
		format,
		version.Int,
		srcItem,
		binItem,
	), nil
}
