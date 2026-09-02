// FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
// ONNX Runtime Go 推理示例（github.com/yalue/onnxruntime_go）
// 动态参数均用注释占位，按实际模型修改

package main

import (
	"fmt"
	"log"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	// ===== 动态参数（按实际模型修改）=====
	// MODEL_PATH: .onnx 模型路径
	modelPath := "model.onnx" // TODO: 替换为你的 .onnx 模型路径
	// INPUT_SHAPE: 输入张量 shape
	inputShape := ort.NewShape(1, 3, 640, 640) // TODO: 按模型输入修改
	// INPUT_NAME: 输入名（可用 session.InputNames() 查询）
	inputName := "images" // TODO: 按模型输入名修改

	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment()

	// TODO: 填充实际输入数据
	inputData := make([]float32, 1*3*640*640)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		log.Fatal(err)
	}
	defer inputTensor.Destroy()

	// TODO: 输出名按模型查询（输出 shape 用 nil 自动推断）
	outputTensor, err := ort.NewEmptyTensor[float32](nil)
	if err != nil {
		log.Fatal(err)
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath, []string{inputName}, []string{"output0"}, // TODO: 输出名
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Destroy()

	// 预热
	for i := 0; i < 3; i++ {
		if err := session.Run(); err != nil {
			log.Fatal(err)
		}
	}

	start := time.Now()
	if err := session.Run(); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("inference time: %v\n", time.Since(start))
	// TODO: 读取 outputTensor.GetData() 并后处理
}
