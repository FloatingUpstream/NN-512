package browse

import (
	"NN-512/internal/example"
	"NN-512/internal/serve/gen"
	"NN-512/internal/serve/listing"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

type keysVals struct {
	keys [][]string
	vals []listing.Elem
}

func (kv keysVals) Len() int {
	return len(kv.keys)
}

func (kv keysVals) Less(i, j int) bool {
	var (
		key1  = kv.keys[i]
		key2  = kv.keys[j]
		last1 = len(key1) - 1
		last2 = len(key2) - 1
	)
	for x := 0; ; x++ {
		switch {
		case x == last1:
			return x < last2 || key1[x] < key2[x]
		case x == last2:
			return false
		case key1[x] < key2[x]:
			return true
		case key1[x] > key2[x]:
			return false
		}
	}
}

func (kv keysVals) Swap(i, j int) {
	kv.keys[i], kv.keys[j] = kv.keys[j], kv.keys[i]
	kv.vals[i], kv.vals[j] = kv.vals[j], kv.vals[i]
}

func Prep(src string) (string, error) {
	var (
		kv       keysVals
		names    []string
		urlPaths []string
	)
	layer5 := func() {
		listing.Handle(
			urlPaths[0],
			kv.vals...,
		)
		for i, name := range names {
			graph := example.Generate(name)
			if graph == nil {
				panic("bug")
			}
			gen.Handle(
				urlPaths[1+i],
				string(graph),
			)
		}
	}
	layer4 := func() string {
		layer5()
		const (
			first  = "<h2>Browse Source Code</h2><ul>"
			format = `<li>%s <a href="%s">%s</a> (%s)</li>`
			last   = "</ul>"
		)
		var sb strings.Builder
		_, _ = sb.WriteString(first)
		_, _ = fmt.Fprintf(
			&sb, format,
			"Source code for the",
			urlPaths[0],
			"NN-512 compiler",
			runtime.Version(),
		)
		for i, name := range names {
			_, _ = fmt.Fprintf(
				&sb, format,
				"Generated code for",
				urlPaths[1+i],
				name,
				"C99",
			)
		}
		_, _ = sb.WriteString(last)
		return sb.String()
	}
	layer3 := func() string {
		n := 1 + len(names)
		urlPaths = make([]string, n)
		const urlPrefix = "/browse/"
		urlPaths[0] = urlPrefix + "NN-512"
		for i, name := range names {
			urlPaths[1+i] = urlPrefix + name
		}
		return layer4()
	}
	layer2 := func() string {
		names = []string{
			"ResNet50",
			"DenseNet121",
			"ResNeXt50",
		}
		return layer3()
	}
	layer1 := func() (string, error) {
		root, err := filepath.EvalSymlinks(src)
		if err != nil {
			return "", err
		}
		info, err := os.Lstat(root)
		switch {
		case err != nil:
			return "", err
		case !info.IsDir():
			return "", errors.New(
				"not a directory: " + src,
			)
		}
		err = filepath.Walk(
			root,
			func(path string, info os.FileInfo, err error) error {
				switch {
				case err != nil:
					return err
				case info.IsDir():
					return nil
				}
				rel, err := filepath.Rel(root, path)
				if err != nil {
					return err
				}
				all, err := ioutil.ReadFile(path)
				if err != nil {
					return err
				}
				const sep = string(filepath.Separator)
				kv.keys = append(
					kv.keys,
					strings.Split(rel, sep),
				)
				kv.vals = append(
					kv.vals,
					listing.Elem{
						Heading: rel,
						Content: string(all),
					},
				)
				return nil
			},
		)
		if err != nil {
			return "", err
		}
		sort.Sort(kv)
		return layer2(), nil
	}
	return layer1()
}
