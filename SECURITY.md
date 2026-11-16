# 安全政策

## 支持的版本

| 版本 | 支持状态 |
| ---- | -------- |
| main (开发分支) | ✅ 支持 |

## 报告安全漏洞

FasterEdge 项目重视安全。请**不要在公共渠道（Issue/讨论区/群聊）公开未修复的漏洞细节**。

请通过以下渠道私密报告：

- **GitHub Security Advisory**（推荐）：在本仓库的 *Security* → *Report a vulnerability* 提交，或直接访问
  `https://github.com/FasterEdge/ModelRunntime/security/advisories/new`
- **邮件**：`shun_@outlook.com`（主题请注明 `[FasterEdge Security]`）

### 报告中请包含

1. 漏洞类型与影响组件
2. 触发条件与复现步骤（尽量精简）
3. 影响范围（可读性 / 可用性 / 安全性）
4. 建议修复方向（可选）

### 响应承诺

- **48 小时内**：确认收到报告并评估严重性
- **7 天内**：给出修复计划或缓解措施
- **修复发版后**：在致谢中列入报告者（如同意）

## 致谢

我们感谢所有以负责任方式报告问题的安全研究人员。

## 依赖安全说明

- `python_gguf` 使用当前最新版 `llama-cpp-python 0.3.37`。该版本传递依赖
  `diskcache 5.6.3`，当前上游尚未发布修复 `PYSEC-2026-2447` 的版本。示例只使用
  `Llama`，未使用 `LlamaDiskCache`；请勿把模型缓存目录开放给不可信用户写入。
- `NPURunntime/python_qualcomm` 需要 `qai-hub 0.58.0`，其上游将 `protobuf` 限制为
  `<=6.31.1`，因此示例固定为已修复 CVE-2026-0994 的 `protobuf==5.29.6`。
- `rust_candle` 的 `candle-core 0.11.1 -> gemm 0.19.0` 传递依赖
  `paste 1.0.15`，上游标记为 `RUSTSEC-2024-0436`（停止维护）。该 crate 仅作为
  编译期过程宏参与构建，不进入运行时输入路径；示例应只加载可信模型文件。
- `NPURunntime/python_paddlelite` 的公共 PyPI 包仅发布到 `paddlelite 2.12`，且
  wheel 只覆盖 CPython 2.7/3.5-3.7。现代 Python/NPU 环境必须使用 Paddle Lite
  官方发布页提供的对应构建包；requirements 仅在 Python 3.7 及以下安装 PyPI 包。
- Ascend CANN、Cambricon Neuware、Horizon、MediaTek NeuroPilot、Rockchip RKNN、
  Sophon SDK 等厂商包通常不由公共 PyPI 分发，`pip-audit` 无法覆盖这些外部 SDK；
  请以厂商安全公告和实际安装版本为准。
