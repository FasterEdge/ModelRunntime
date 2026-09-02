// FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
//go:build linux && cgo

package main

import (
	tflite "github.com/mattn/go-tflite"
	"github.com/mattn/go-tflite/delegates/xnnpack"
)

// enableXNNPACK 在 Linux + cgo 环境下启用 XNNPACK delegate 加速（可选优化）。
// 返回清理函数；若 delegate 创建失败则返回空操作。
func enableXNNPACK(options *tflite.InterpreterOptions) func() {
	d := xnnpack.New(options)
	if d == nil {
		return func() {}
	}
	return func() { d.Delete() }
}
