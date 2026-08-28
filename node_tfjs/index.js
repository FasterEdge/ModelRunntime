// TensorFlow.js Node.js 推理示例
// 动态参数均用注释占位，按实际模型修改

const tf = require("@tensorflow/tfjs-node");

async function main() {
  // ===== 动态参数（按实际模型修改）=====
  // MODEL_PATH: 模型路径（SavedModel 目录 或 http(s):// 远程）
  const modelPath = "file://./model";  // TODO: 替换为模型路径（file:// 或 https://）
  // INPUT_SHAPE: 输入张量 shape [N,H,W,C]
  const inputShape = [1, 224, 224, 3]; // TODO: 按模型输入修改

  const model = await tf.loadGraphModel(modelPath);

  // TODO: 填充实际输入数据（读图、预处理，Float32Array）
  const inputData = new Float32Array(1 * 224 * 224 * 3);
  const input = tf.tensor(inputData, inputShape);

  // 预热
  for (let i = 0; i < 3; i++) {
    const _ = model.predict(input);
    tf.dispose(_);
  }

  const start = Date.now();
  const output = model.predict(input);
  await output.data();
  console.log(`inference time: ${Date.now() - start} ms`);

  // TODO: 解析 output（output.argMax(1).dataSync() 等）
  tf.dispose(input);
  tf.dispose(output);
}

main().catch(console.error);
