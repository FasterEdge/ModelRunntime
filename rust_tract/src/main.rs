// FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
// tract (ONNX / TensorFlow) Rust 推理示例
// 动态参数均用注释占位，按实际模型修改

use std::time::Instant;

fn main() -> tract_onnx::Result<()> {
    // ===== 动态参数（按实际模型修改）=====
    // MODEL_PATH: .onnx 模型路径
    let model_path = "model.onnx"; // TODO: 替换为你的 .onnx 模型路径
    // INPUT_SHAPE: 输入张量 shape（1,3,H,W）
    let input_shape: &[usize] = &[1, 3, 640, 640]; // TODO: 按模型输入修改

    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .into_optimized()?
        .into_runnable()?;

    // TODO: 填充实际输入数据（读图、预处理 -> CHW）
    let input = tract_ndarray::Array4::<f32>::zeros((1, 3, 640, 640));

    // 预热
    for _ in 0..3 {
        let _ = model.run(tvec!(input.clone().into()))?;
    }

    let start = Instant::now();
    let output = model.run(tvec!(input.into()))?;
    println!("inference time: {:?}", start.elapsed());

    // TODO: 解析输出 output[0]（tract_core::tensor::Tensor）
    Ok(())
}
