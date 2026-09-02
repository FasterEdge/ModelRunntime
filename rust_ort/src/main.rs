// FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
// ort (ONNX Runtime) Rust 推理示例
// 动态参数均用注释占位，按实际模型修改

use std::time::Instant;

fn main() -> ort::Result<()> {
    // ===== 动态参数（按实际模型修改）=====
    // MODEL_PATH: .onnx 模型路径
    let model_path = "model.onnx"; // TODO: 替换为你的 .onnx 模型路径
    // INPUT_SHAPE: 输入张量 shape
    let input_shape: [i64; 4] = [1, 3, 640, 640]; // TODO: 按模型输入修改
    // INPUT_NAME: 输入名
    let input_name = "images"; // TODO: 按模型输入名修改

    let session = ort::Session::builder()?
        .with_intra_threads(4)? // TODO: 按 CPU 核数调整
        .commit_from_file(model_path)?;

    // TODO: 填充实际输入数据
    let input = ndarray::Array4::<f32>::zeros((1, 3, 640, 640));

    // 预热
    for _ in 0..3 {
        let _ = session.run(ort::inputs![input_name => input.view()]?)?;
    }

    let start = Instant::now();
    let outputs = session.run(ort::inputs![input_name => input.view()]?)?;
    println!("inference time: {:?}", start.elapsed());

    // TODO: 解析 outputs（session 输出名可通过模型查询）
    Ok(())
}
