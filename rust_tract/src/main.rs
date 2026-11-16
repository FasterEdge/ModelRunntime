// FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
// tract (ONNX / TensorFlow) Rust 推理示例
// 动态参数均用注释占位，按实际模型修改

use std::time::Instant;
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    // ===== 动态参数（按实际模型修改）=====
    // MODEL_PATH: .onnx 模型路径
    // TODO: 替换为你的 .onnx 模型路径
    let model_path = "model.onnx";

    // INPUT_SHAPE: 输入张量 shape（1,3,H,W）
    // TODO: 按模型输入修改
    let input_shape: [usize; 4] = [1, 3, 640, 640];

    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .into_optimized()?
        .into_runnable()?;

    // TODO: 填充实际输入数据（读图、预处理 -> CHW）
    let input = tract_ndarray::Array4::<f32>::zeros((
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ));

    // 预热
    for _ in 0..3 {
        let _ = model.run(tvec!(input.clone().into_tensor().into_tvalue()))?;
    }

    let start = Instant::now();
    let _output = model.run(tvec!(input.into_tensor().into_tvalue()))?;
    println!("inference time: {:?}", start.elapsed());

    // TODO: 解析输出 output[0]（tract_core::tensor::Tensor）
    Ok(())
}
