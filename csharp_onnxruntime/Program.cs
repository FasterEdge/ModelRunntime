// ONNX Runtime C# (.NET) 推理示例
// 动态参数均用注释占位，按实际模型修改

using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

class Program
{
    static void Main(string[] args)
    {
        // ===== 动态参数（按实际模型修改）=====
        // MODEL_PATH: .onnx 模型路径
        string modelPath = "model.onnx";   // TODO: 替换为你的 .onnx 模型路径
        // INPUT_SHAPE: 输入张量 shape {N,C,H,W}
        int[] inputShape = { 1, 3, 640, 640 };  // TODO: 按模型输入修改
        // INPUT_NAME: 输入名（可用 session.InputMetadata 查询）
        string inputName = "images";       // TODO: 按模型输入名修改

        using var session = new InferenceSession(modelPath);

        // TODO: 填充实际输入数据（读图、预处理 -> CHW）
        var inputData = new float[1 * 3 * 640 * 640];
        var tensor = new DenseTensor<float>(inputData, inputShape);
        var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) };

        // 预热
        for (int i = 0; i < 3; i++)
        {
            using var _ = session.Run(inputs);
        }

        var sw = Stopwatch.StartNew();
        using var results = session.Run(inputs);
        sw.Stop();
        Console.WriteLine($"inference time: {sw.ElapsedMilliseconds} ms");

        // TODO: 解析 results.First().AsTensor<float>()
        var output = results.First().AsTensor<float>();
        Console.WriteLine($"output shape: [{string.Join(",", output.Dimensions)}]");
    }
}
