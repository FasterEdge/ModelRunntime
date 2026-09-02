// FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
//go:build !linux || !cgo

package main

import tflite "github.com/mattn/go-tflite"

// enableXNNPACK 在非 Linux 或无 cgo 环境下不可用（XNNPACK 需要 cgo 链接原生库），
// 返回空操作，示例仍可编译运行，仅无加速 delegate。
func enableXNNPACK(*tflite.InterpreterOptions) func() {
	return func() {}
}
