package example

import (
	"NN-512/internal/example/densenet"
	"NN-512/internal/example/resnet"
	"NN-512/internal/example/resnext"
)

var menu = [...]struct {
	name string
	call func() []byte
}{
	{"ResNet50", resnet.ResNet50},
	{"ResNet101", resnet.ResNet101},
	{"ResNet152", resnet.ResNet152},
	{"DenseNet121", densenet.DenseNet121},
	{"DenseNet169", densenet.DenseNet169},
	{"DenseNet201", densenet.DenseNet201},
	{"DenseNet265", densenet.DenseNet265},
	{"ResNeXt50", resnext.ResNeXt50},
	{"ResNeXt101", resnext.ResNeXt101},
	{"ResNeXt152", resnext.ResNeXt152},
}

func Names() []string {
	names := make([]string, len(menu))
	for i := range &menu {
		names[i] = menu[i].name
	}
	return names
}

func Generate(name string) []byte {
	for i := range &menu {
		if menu[i].name == name {
			return menu[i].call()
		}
	}
	return nil
}
