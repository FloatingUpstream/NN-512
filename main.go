// NN-512 (https://NN-512.com)
//
// Copyright (C) 2019 [
//     37ef ced3 3727 60b4
//     3c29 f9c6 dc30 d518
//     f4f3 4106 6964 cab4
//     a06f c1a3 83fd 090e
// ]
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in
//    the documentation and/or other materials provided with the
//    distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package main

import (
	"NN-512/internal/compile"
	"NN-512/internal/doc"
	"NN-512/internal/example"
	"NN-512/internal/serve"
	"NN-512/internal/version"
	"errors"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	newline = "\n"
	space   = " "
	indent  = space + space + space + space
	usage   = newline + "Usage:" + newline + newline + indent + "NN-512" + space
)

func cmdCompile() error {
	if len(os.Args) == 4 {
		from := os.Args[2]
		if from == "-" {
			from = "/dev/stdin"
		}
		text, err := ioutil.ReadFile(from)
		if err != nil {
			return err
		}
		result, err := compile.Compile(string(text))
		if err != nil {
			return err
		}
		prefix := filepath.Join(os.Args[3], result.Name)
		const perm os.FileMode = 0666
		if err := ioutil.WriteFile(prefix+".h", result.H, perm); err != nil {
			return err
		}
		return ioutil.WriteFile(prefix+".c", result.C, perm)
	}
	return errors.New(usage +
		os.Args[1] + space + "GRAPH" + space + "DIR" + newline +
		newline +
		"The GRAPH argument specifies an input file that contains a" + newline +
		"graph language description of a neural net. - means stdin." + newline +
		newline +
		indent + "Example: densenet.graph" + newline +
		indent + "Example: ../graphs/denseNet" + newline +
		indent + "Example: /opt/nets/DenseNet" + newline +
		indent + "Example: -" + newline +
		newline +
		"The DIR argument specifies an output directory where the" + newline +
		"generated C99 files will be written." + newline +
		newline +
		indent + "Example: ." + newline +
		indent + "Example: ../src" + newline +
		indent + "Example: /tmp/" + newline)
}

func cmdDoc() error {
	if len(os.Args) > 2 {
		return errors.New(usage + os.Args[1] + newline)
	}
	_, err := os.Stdout.Write(doc.Bytes())
	return err
}

func cmdExample() error {
	if len(os.Args) == 3 {
		if gen := example.Generate(os.Args[2]); gen != nil {
			_, err := os.Stdout.Write(gen)
			return err
		}
	}
	list := strings.Join(example.Names(), newline+indent)
	return errors.New(usage +
		os.Args[1] + space + "NAME" + newline +
		newline +
		"The NAME argument can be:" + newline +
		newline +
		indent + list + newline)
}

func cmdServe() error {
	if len(os.Args) == 7 {
		var (
			addr = os.Args[2]
			cert = os.Args[3]
			key  = os.Args[4]
			src  = os.Args[5]
			bin  = os.Args[6]
		)
		return serve.Website(addr, cert, key, src, bin)
	}
	return errors.New(usage +
		os.Args[1] + space +
		"ADDR" + space +
		"CERT" + space + "KEY" + space +
		"SRC" + space + "BIN" + newline +
		newline +
		"The ADDR argument specifies the TCP network addresses to" + newline +
		"listen on (in the form \"host:port\"). If the host spec is" + newline +
		"omitted, the server will listen on all available unicast" + newline +
		"and anycast IP addresses of the local system." + newline +
		newline +
		indent + "Example: :https" + newline +
		indent + "Example: :443" + newline +
		indent + "Example: 127.0.0.1:https" + newline +
		indent + "Example: localhost:4321" + newline +
		indent + "Example: 173.230.145.5:443" + newline +
		indent + "Example: [2600:3c01::2]:443" + newline +
		newline +
		"The CERT argument specifies a PEM-encoded HTTPS certificate" + newline +
		"file. If the certificate is signed by a certificate authority," + newline +
		"the file should consist of the server's certificate, then any" + newline +
		"intermediate certificates, then the authority's certificate" + newline +
		"(concatenated in that order)." + newline +
		newline +
		indent + "Example: fullchain.pem" + newline +
		indent + "Example: tls/cert.pem" + newline +
		indent + "Example: /root/certs/my.crt" + newline +
		newline +
		"The KEY argument specifies a PEM-encoded HTTPS private key" + newline +
		"file that matches the previously specified certificate." + newline +
		newline +
		indent + "Example: privkey.pem" + newline +
		indent + "Example: tls/key.pem" + newline +
		indent + "Example: /root/certs/my.key" + newline +
		newline +
		"The SRC argument specifies a directory that contains the Go" + newline +
		"source code of this program." + newline +
		newline +
		indent + "Example: /go/src/NN-512" + newline +
		indent + "Example: /go/src/NN-512/" + newline +
		indent + "Example: ../src/NN-512" + newline +
		newline +
		"The BIN argument specifies this program's executable file." + newline +
		newline +
		indent + "Example: /go/bin/NN-512" + newline +
		indent + "Example: ../bin/NN-512" + newline)
}

func cmdVersion() error {
	if len(os.Args) > 2 {
		return errors.New(usage + os.Args[1] + newline)
	}
	_, err := os.Stdout.WriteString(
		strconv.Itoa(version.Int) + newline,
	)
	return err
}

var cmds = [...]struct {
	name string
	hint string
	call func() error
}{
	{"compile", "Read graph language for a neural net and write C99.", cmdCompile},
	{"doc", "Write documentation for the graph language to stdout.", cmdDoc},
	{"example", "Write graph language for an example neural net to stdout.", cmdExample},
	{"serve", "Serve this program's website as HTML over HTTPS.", cmdServe},
	{"version", "Write the version number of this program to stdout.", cmdVersion},
}

func run() error {
	if len(os.Args) >= 2 {
		arg := os.Args[1]
		for i := range &cmds {
			if cmds[i].name == arg {
				return cmds[i].call()
			}
		}
	}
	max := 0
	for i := range &cmds {
		if alt := len(cmds[i].name); max < alt {
			max = alt
		}
	}
	tot := max + len(indent)
	var list string
	for i := range &cmds {
		name, hint := cmds[i].name, cmds[i].hint
		align := strings.Repeat(space, tot-len(name))
		list += indent + name + align + hint + newline
	}
	return errors.New(usage +
		"COMMAND" + newline +
		newline +
		"The COMMAND argument can be:" + newline +
		newline +
		list)
}

func main() {
	if err := run(); err != nil {
		_, _ = os.Stderr.WriteString(err.Error() + newline)
		os.Exit(1)
	}
	os.Exit(0)
}

var _ uint = 1 << 63
