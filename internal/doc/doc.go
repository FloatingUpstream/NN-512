package doc

import (
	"NN-512/internal/raw"
	"sort"
	"unicode"
)

const (
	empty   = ""
	space   = " "
	dash    = "-"
	newline = "\n"
	indent  = space + space + space + space
	divider = dash + dash + dash + dash + newline
	width   = 80
)

func line(to []byte, dent, text string) []byte {
	to = append(to, dent...)
	to = append(to, text...)
	to = append(to, newline...)
	return to
}

func para(to []byte, dent, text string) []byte {
	to = append(to, newline...)
	fit := width - len(dent)
	var i, j, ij, ik int
	for k, r := range text {
		if unicode.IsSpace(r) {
			if ik > fit && ij != 0 {
				to = line(to, dent, text[i:j])
				i = j + 1
				ik -= ij + 1
			}
			j, ij = k, ik
		}
		ik += 1
	}
	if ik > fit && ij != 0 {
		to = line(to, dent, text[i:j])
		i = j + 1
		ik -= ij + 1
	}
	if ik != 0 {
		to = line(to, dent, text[i:])
	}
	return to
}

func Bytes() (to []byte) {
	heads := make([]string, 0, len(raw.Guide))
	for head := range raw.Guide {
		heads = append(heads, head)
	}
	sort.Strings(heads)
	for i, head := range heads {
		tail := raw.Guide[head]
		if i != 0 {
			to = append(to, newline+divider+newline...)
		}
		to = append(to, head+newline...)
		for _, seg := range tail.Segs {
			to = append(to, indent+seg.Label+raw.Binder+seg.Default+newline...)
		}
		to = para(to, empty, tail.Doc)
		for _, seg := range tail.Segs {
			to = para(to, indent, seg.Label+raw.Binder+space+seg.Doc)
		}
	}
	return
}
