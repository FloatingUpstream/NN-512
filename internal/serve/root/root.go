package root

import (
	"NN-512/internal/serve/browse"
	"NN-512/internal/serve/docs"
	"NN-512/internal/serve/download"
	"NN-512/internal/serve/email"
	"NN-512/internal/serve/features"
	"NN-512/internal/serve/html"
	"NN-512/internal/serve/intro"
	"NN-512/internal/serve/license"
	"NN-512/internal/serve/stats"
	"NN-512/internal/serve/util"
	"strings"
)

func Prep(src, bin string) error {
	var (
		Intro    string
		Features string
		Download string
		Browse   string
		Docs     string
		Email    string
		License  string
	)
	layer3 := func() error {
		parts := []string{
			Intro,
			Features,
			Download,
			Browse,
			Docs,
			Email,
			License,
		}
		var (
			body = strings.Join(parts, "")
			Html = html.FromString(body)
			page = util.NewPage(Html)
		)
		util.Handle("/", page)
		return nil
	}
	layer2 := func() error {
		var err error
		Download, err = download.Prep(src, bin)
		if err != nil {
			return err
		}
		Browse, err = browse.Prep(src)
		if err != nil {
			return err
		}
		return layer3()
	}
	layer1 := func() error {
		stats.Prep()
		Intro = intro.Prep()
		Features = features.Prep()
		Docs = docs.Prep()
		Email = email.Prep()
		License = license.Prep()
		return layer2()
	}
	return layer1()
}
