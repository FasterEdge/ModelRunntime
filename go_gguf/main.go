// GGUF 大模型 Go 推理示例（配合 llama.cpp 的 llama-server / llama-cli）
// 动态参数均用注释占位，按实际模型修改

package main

import (
	"fmt"
	"log"
	"os/exec"
)

func main() {
	// ===== 动态参数（按实际模型修改）=====
	// LLAMA_CLI: llama.cpp 可执行文件路径（llama-cli / llama-server）
	llamaCLI := "/usr/local/bin/llama-cli" // TODO: 替换为 llama-cli 路径
	// MODEL_PATH: .gguf 模型路径
	modelPath := "model.gguf" // TODO: 替换为 .gguf 模型路径
	// PROMPT: 输入提示词
	prompt := "Hello, how are you?" // TODO: 替换输入
	// N_THREADS: 线程数
	nThreads := "8" // TODO: 按 CPU 核数修改
	// N_PREDICT: 最大生成 token 数
	nPredict := "256" // TODO: 按需修改

	// 说明：也可以直接调用 llama-server 的 HTTP API（/completion 或 /v1/chat/completions）
	cmd := exec.Command(llamaCLI,
		"-m", modelPath,
		"-p", prompt,
		"-t", nThreads,
		"-n", nPredict,
		"--no-display-prompt")
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(out))
}
