package bin

import (
	"NN-512/internal/serve/util"
	"archive/tar"
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"runtime"
	"time"
)

func Prep(binFile, urlPrefix string) (string, error) {
	const (
		urlSuffix = "NN-512.bin.tar.gz"
		tarPrefix = "bin/"
	)
	var (
		urlPath string
		now     time.Time
		buf     *bytes.Buffer
		gz      *gzip.Writer
		tw      *tar.Writer
		size    int64
	)
	layer9 := func() error {
		all, err := ioutil.ReadFile(binFile)
		if err != nil {
			return err
		}
		_, err = tw.Write(all)
		return err
	}
	layer8 := func() error {
		hdr := &tar.Header{
			Name:    tarPrefix + "NN-512",
			Size:    size,
			Mode:    0777,
			ModTime: now,
		}
		err := tw.WriteHeader(hdr)
		if err != nil {
			return err
		}
		return layer9()
	}
	layer7 := func() error {
		info, err := os.Stat(binFile)
		if err != nil {
			return err
		}
		var (
			mode = info.Mode()
			why  string
		)
		switch {
		case !mode.IsRegular():
			why = "not a regular file"
		case mode&0111 == 0:
			why = "not an executable file"
		default:
			size = info.Size()
			return layer8()
		}
		return errors.New(
			why + ": " + binFile,
		)
	}
	layer6 := func() error {
		hdr := &tar.Header{
			Name:    tarPrefix,
			Mode:    0777,
			ModTime: now,
		}
		err := tw.WriteHeader(hdr)
		if err != nil {
			return err
		}
		return layer7()
	}
	layer5 := func() error {
		tw = tar.NewWriter(gz)
		err := layer6()
		if err != nil {
			return err
		}
		return tw.Close()
	}
	layer4 := func() error {
		gz, _ = gzip.NewWriterLevel(
			buf, gzip.BestCompression,
		)
		err := layer5()
		if err != nil {
			return err
		}
		_ = gz.Close()
		return nil
	}
	layer3 := func() error {
		buf = new(bytes.Buffer)
		err := layer4()
		if err != nil {
			return err
		}
		file := util.NewFile(
			urlSuffix,
			buf.Bytes(),
		)
		util.Handle(urlPath, file)
		return nil
	}
	layer2 := func() (string, error) {
		err := layer3()
		if err != nil {
			return "", err
		}
		return fmt.Sprintf(
			"NN-512 compiler executable: "+
				`<a href="%s">%s</a> (%s/%s)`,
			urlPath,
			urlSuffix,
			runtime.GOOS,
			runtime.GOARCH,
		), nil
	}
	layer1 := func() (string, error) {
		urlPath = urlPrefix + urlSuffix
		now = time.Now()
		return layer2()
	}
	return layer1()
}
