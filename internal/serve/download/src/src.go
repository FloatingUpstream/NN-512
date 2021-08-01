package src

import (
	"NN-512/internal/serve/util"
	"archive/tar"
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

func Prep(srcDir, urlPrefix string) (string, error) {
	const (
		urlSuffix = "NN-512.src.tar.gz"
		tarPrefix = "src/"
	)
	var (
		urlPath string
		now     time.Time
		buf     *bytes.Buffer
		gz      *gzip.Writer
		tw      *tar.Writer
		root    string
	)
	layer8 := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		tarPath := filepath.Join(
			tarPrefix+"NN-512", rel,
		)
		mode := info.Mode()
		switch mode & os.ModeType {
		case os.ModeDir:
			hdr := &tar.Header{
				Name:    tarPath + "/",
				Mode:    0777,
				ModTime: now,
			}
			return tw.WriteHeader(hdr)
		case 0:
			if mode&0111 != 0 {
				errPath := filepath.Join(
					srcDir, rel,
				)
				return errors.New(
					"unexpected executable file: " +
						errPath,
				)
			}
			hdr := &tar.Header{
				Name:    tarPath,
				Size:    info.Size(),
				Mode:    0666,
				ModTime: now,
			}
			err := tw.WriteHeader(hdr)
			if err != nil {
				return err
			}
			all, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}
			_, err = tw.Write(all)
			return err
		}
		errPath := filepath.Join(
			srcDir, rel,
		)
		return errors.New(
			"not just a directory " +
				"or regular file: " +
				errPath,
		)
	}
	layer7 := func() error {
		var err error
		root, err = filepath.EvalSymlinks(srcDir)
		if err != nil {
			return err
		}
		info, err := os.Lstat(root)
		if err != nil {
			return err
		}
		if info.IsDir() {
			return filepath.Walk(root, layer8)
		}
		return errors.New(
			"not a directory: " + srcDir,
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
			"NN-512 compiler source code: "+
				`<a href="%s">%s</a> (%s)`,
			urlPath,
			urlSuffix,
			runtime.Version(),
		), nil
	}
	layer1 := func() (string, error) {
		urlPath = urlPrefix + urlSuffix
		now = time.Now()
		return layer2()
	}
	return layer1()
}
