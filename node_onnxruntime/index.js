// ONNX Runtime Node.js 推理示例
// 动态参数均用注释占位，按实际模型修改

const ort = require("onnxruntime-node");

async function main() {
  // ===== 动态参数（按实际模型修改）=====
  // MODEL_PATH: .onnx 模型路径
  const modelPath = "model.onnx";       // TODO: 替换为你的 .onnx 模型路径
  // INPUT_SHAPE: 输入张量 shape [N,C,H,W]
  const inputShape = [1, 3, 640, 640];  // TODO: 按模型输入修改
  // INPUT_NAME: 输入名
  const inputName = "images";           // TODO: 按模型输入名修改

  const session = await ort.InferenceSession.create(modelPath);

  // TODO: 填充实际输入数据（读图、预处理 -> CHW，Float32Array）
  const inputData = new Float32Array(1 * 3 * 640 * 640);
  const tensor = new ort.Tensor("float32", inputData, inputShape);

  // 预热
  for (let i = 0; i < 3; i++) {
    await session.run({ [inputName]: tensor });
  }

  const start = Date.now();
  const outputs = await session.run({ [inputName]: tensor });
  const ms = Date.now() - start;
  console.log(`inference time: ${ms} ms`);

  // TODO: 解析 outputs（键为输出名，值为 Tensor）
  for (const [name, t] of Object.entries(outputs)) {
    console.log(`output ${name}: shape=[${t.dims}]`);
  }
}

main().catch(console.error);
