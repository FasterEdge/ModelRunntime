// FasterEdge 开源项目 · https://github.com/FasterEdge · https://gitee.com/FasterEdge
// TensorFlow Lite Go 推理示例（github.com/mattn/go-tflite）
// 动态参数均用注释占位，按实际模型修改

package main

import (
	"fmt"
	"log"
	"time"

	tflite "github.com/mattn/go-tflite"
	"github.com/mattn/go-tflite/delegates/xnnpack"
)

func main() {
	// ===== 动态参数（按实际模型修改）=====
	// MODEL_PATH: .tflite 模型路径
	modelPath := "model.tflite" // TODO: 替换为 .tflite 模型路径

	model := tflite.NewModelFromFile(modelPath)
	if model == nil {
		log.Fatal("cannot load model")
	}
	defer model.Delete()

	options := tflite.NewInterpreterOptions()
	defer options.Delete()
	options.SetNumThread(4) // TODO: 按 CPU 核数调整
	if d := xnnpack.New(options); d != nil { // TODO: XNNPACK 委派可选
		defer d.Delete()
	}

	interpreter := tflite.NewInterpreter(model, options)
	if interpreter == nil {
		log.Fatal("cannot create interpreter")
	}
	defer interpreter.Delete()

	interpreter.AllocateTensors()

	// TODO: 填充输入（注意量化模型需按 scale/zero_point 转换）
	// in := interpreter.GetInputTensor(0)
	// copy(in.Float32s(), data)

	// 预热
	for i := 0; i < 3; i++ {
		interpreter.Invoke()
	}

	start := time.Now()
	interpreter.Invoke()
	fmt.Printf("inference time: %v\n", time.Since(start))

	// TODO: 读取输出 out := interpreter.GetOutputTensor(0)
}
