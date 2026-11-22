#!/usr/bin/env python3
# FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
"""驱动 matrix.json 中被标记为 verified 的示例：真实构建 + 运行，把证据写回 matrix.json。

设计原则：
- 只有本文件里的用例真跑过，对应示例才会被标成 verified；gen_matrix.py 永远不会
  自动把某个示例标成 verified，因为"跑通"这件事只有本脚本能证明。
- 每个用例都在仓库外的临时目录里执行（系统临时目录），不在示例目录内留产物。
- 失败必须让退出码非零，禁止"失败也算通过"。

用法：
    python tools/verify_matrix.py             # 运行所有用例并更新 matrix.json 的 evidence
    python tools/verify_matrix.py --check     # 只检查，失败退出码 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = REPO_ROOT / "matrix.json"

GO_BIN = Path(os.environ.get("GO_BIN", "")) if os.environ.get("GO_BIN") else None
MODEL_DIR = REPO_ROOT / "python_onnx"
PYTHON_BIN = os.environ.get("PYTHON_BIN", "")


def _venv_python() -> str:
    """优先用 PYTHON_BIN / 仓库内 venv，否则退回 PATH 里的 python。
    Windows 上 mise shim 与 venv 都可能影响解释器选择，因此显式解析。"""
    if PYTHON_BIN:
        return PYTHON_BIN
    for cand in (REPO_ROOT / ".venv" / "Scripts" / "python.exe",
                 REPO_ROOT / ".venv" / "bin" / "python"):
        if cand.is_file():
            return str(cand)
    return sys.executable


def sh(argv: list[str], cwd: Path, timeout: int = 600, env_extra: dict | None = None) -> dict:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout, env=env)
        return {"returncode": proc.returncode,
                "stdout": (proc.stdout or "")[-2000:],
                "stderr": (proc.stderr or "")[-2000:],
                "command": " ".join(str(a) for a in argv)}
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": f"超时（>{timeout}s）",
                "command": " ".join(str(a) for a in argv)}
    except FileNotFoundError as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc),
                "command": " ".join(str(a) for a in argv)}


def go_env() -> dict:
    """Go 不在默认 PATH 时（Windows mise shim 场景）从常见安装位置补上。

    注意：Windows 上 mise shim 常常指向一个不存在的 go，直接 `go build` 会报
    "failed to execute mise"。这里按目录探测 go.exe / go，取第一个真的可用的。
    另外 Windows 的 CreateProcess 用父进程的 PATH 搜索可执行文件，不看新 env，
    所以下面统一返回 go 的绝对路径。
    """
    exe = "go.exe" if os.name == "nt" else "go"
    if GO_BIN:
        p = GO_BIN / exe if GO_BIN.is_dir() else GO_BIN
        if p.is_file():
            return {"PATH": str(p.parent) + os.pathsep + os.environ["PATH"], "GO": str(p)}
    candidates = [
        Path(r"C:\Users\Dlut\AppData\Roaming\CherryStudio\Toolchain\mise\installs\go\1.27.0\bin"),
        Path(r"C:\Program Files\Go\bin"),
        Path.home() / "go" / "bin",
    ]
    for c in candidates:
        if (c / exe).is_file():
            return {"PATH": str(c) + os.pathsep + os.environ["PATH"], "GO": str(c / exe)}
    return {"GO": "go"}


def case_python_onnx(work: Path) -> dict:
    """在临时目录里用仓库自带 yolov5n.onnx + example2.png 真实推理 10 次。"""
    src = REPO_ROOT / "python_onnx"
    for name in ("test.py", "yolov5n.onnx", "example2.png", "requirements.txt"):
        shutil.copy2(src / name, work / name)
    # 依赖由外部 venv 提供；未提供则如实失败
    pip_check = sh([_venv_python(), "-c",
                    "import onnxruntime,cv2,numpy;print(onnxruntime.__version__,cv2.__version__,numpy.__version__)"],
                   work)
    if pip_check["returncode"] != 0:
        return {**pip_check, "note": "缺 onnxruntime/cv2/numpy；先 python -m venv 并安装 requirements.txt"}
    run = sh([_venv_python(), "test.py"], work, timeout=900)
    return {"imports": pip_check, "run": run,
            "expected_marker": "FPS:",
            "observed": run["stdout"].count("FPS:")}


def case_go_onnxruntime(work: Path) -> dict:
    """构建 Go 示例，链接 ONNX Runtime 共享库，用 yolov5n.onnx 实跑。"""
    src = REPO_ROOT / "go_onnxruntime"
    for name in ("main.go", "go.mod", "go.sum"):
        if (src / name).is_file():
            shutil.copy2(src / name, work / name)
    # 共享库：优先示例目录内的 venv，其次 PATH
    dll = _find_onnxruntime_dll()
    if dll is None:
        return {"returncode": -1, "stderr": "找不到 onnxruntime.dll/libonnxruntime.so",
                "note": "需安装 onnxruntime（pip 包内含 capi 共享库）"}
    shutil.copy2(dll, work / dll.name)
    shutil.copy2(MODEL_DIR / "yolov5n.onnx", work / "model.onnx")

    env = go_env()
    # onnxruntime_go 是 cgo 包（onnxruntime_wrapper.c），CGO_ENABLED=0 会报
    # "build constraints exclude all Go files"，这里显式要求 cgo 并在缺 C 编译器时如实失败。
    env["CGO_ENABLED"] = "1"
    if not (shutil.which("gcc") or shutil.which("cc")):
        return {"returncode": -1, "stderr": "找不到 gcc/cc，onnxruntime_go 是 cgo 包，无法构建",
                "note": "需 C 编译器（Windows 上 MinGW-w64 即可）"}
    build = sh([env["GO"], "build", "-o", "run_gort.exe" if os.name == "nt" else "run_gort", "."],
               work, env_extra=env)
    if build["returncode"] != 0:
        return build
    run = sh([str(work / ("run_gort.exe" if os.name == "nt" else "run_gort"))], work,
             env_extra={"PATH": str(work) + os.pathsep + os.environ["PATH"]})
    return {"dll": str(dll), "build": build, "run": run,
            "expected_marker": "inference time:",
            "observed": run["stdout"].count("inference time:")}


def case_node_onnxruntime(work: Path) -> dict:
    """用 node_onnxruntime 自带的 node_modules 与 yolov5n.onnx 实跑。"""
    src = REPO_ROOT / "node_onnxruntime"
    if not (src / "node_modules").is_dir():
        return {"returncode": -1, "stderr": "node_modules 缺失",
                "note": "先 npm ci（示例自带 package-lock.json）"}
    shutil.copytree(src / "node_modules", work / "node_modules", dirs_exist_ok=True)
    shutil.copy2(MODEL_DIR / "yolov5n.onnx", work / "model.onnx")
    script = (
        'const ort=require("onnxruntime-node");\n'
        '(async()=>{const s=await ort.InferenceSession.create("model.onnx");\n'
        'const t=new ort.Tensor("float32",new Float32Array(1*3*640*640),[1,3,640,640]);\n'
        'const st=Date.now();const o=await s.run({[s.inputNames[0]]:t});\n'
        'console.log("inference time:",Date.now()-st,"ms");\n'
        'for(const[k,v]of Object.entries(o))console.log(k,v.dims);})()\n'
        '.catch(e=>{console.error("ERR",e.message);process.exit(1);});\n'
    )
    (work / "probe.js").write_text(script, encoding="utf-8")
    run = sh(["node", "probe.js"], work, timeout=600)
    return {"run": run, "expected_marker": "inference time:",
            "observed": run["stdout"].count("inference time:")}


def _find_onnxruntime_dll() -> Path | None:
    """在示例自带的 venv 或系统里找 ONNX Runtime 共享库。"""
    for root in [REPO_ROOT / "python_onnx", REPO_ROOT / ".venv", REPO_ROOT.parent / ".tmp"]:
        for base in (root, root / "Lib" / "site-packages" / "onnxruntime" / "capi",
                     root / "Scripts" / ".." / "Lib" / "site-packages" / "onnxruntime" / "capi"):
            if not base.is_dir():
                continue
            for pattern in ("onnxruntime.dll", "libonnxruntime.so"):
                hit = next(base.rglob(pattern), None)
                if hit:
                    return hit
    return None


def case_python_pb(work: Path) -> dict:
    """在临时目录里用仓库自带 yolov3_resnet18.pb + example2.png 真实推理（tf.compat.v1 冻结图）。"""
    src = REPO_ROOT / "python_pb"
    for name in ("test.py", "yolov3_resnet18.pb", "example2.png"):
        shutil.copy2(src / name, work / name)
    py = _venv_python()
    pip_check = sh([py, "-c", "import tensorflow,cv2,numpy;print(tensorflow.__version__)"], work)
    if pip_check["returncode"] != 0:
        return {**pip_check, "note": "缺 tensorflow/cv2/numpy；先 python -m venv 并安装 requirements.txt"}
    env = {"TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONIOENCODING": "utf-8"}
    run = sh([py, "test.py"], work, timeout=1800, env_extra=env)
    result_file = work / "result_example2.png"
    return {"run": run, "expected_marker": "FPS:",
            "observed": run["stdout"].count("FPS:"),
            "output_image": result_file.is_file()}


def case_python_pth_pt(work: Path) -> dict:
    """在临时目录里用仓库自带 yolov5s.pt + example.png 真实前向 + NMS 后处理。"""
    src = REPO_ROOT / "python_pth_pt"
    for name in ("test.py", "yolov5s.pt", "example.png"):
        shutil.copy2(src / name, work / name)
    py = _venv_python()
    pip_check = sh([py, "-c", "import torch,cv2,numpy;print(torch.__version__)"], work)
    if pip_check["returncode"] != 0:
        return {**pip_check, "note": "缺 torch/cv2/numpy；先 python -m venv 并安装 requirements.txt"}
    run = sh([py, "test.py"], work, timeout=900)
    return {"run": run, "expected_marker": "检测到",
            "observed": run["stdout"].count("检测到")}


CASES = {
    "python_onnx": case_python_onnx,
    "python_pb": case_python_pb,
    "python_pth_pt": case_python_pth_pt,
    "go_onnxruntime": case_go_onnxruntime,
    "node_onnxruntime": case_node_onnxruntime,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="驱动 verified 示例并写入证据")
    parser.add_argument("--check", action="store_true", help="只检查，失败退出码 1")
    args = parser.parse_args()

    if not MATRIX_PATH.is_file():
        print("matrix.json 不存在，先运行 python tools/gen_matrix.py", file=sys.stderr)
        return 1
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))

    failures = []
    for sample in matrix["samples"]:
        sid = sample["id"]
        if sid not in CASES:
            continue
        case = CASES[sid]
        with tempfile.TemporaryDirectory(prefix=f"fe-verify-{sid}-") as td:
            result = case(Path(td))
        ok = result.get("returncode", 0) == 0 and result.get("observed", 0) > 0
        sample["evidence"] = {
            "verified": ok,
            "command": result.get("command") or result.get("run", {}).get("command", ""),
            "note": result.get("note", ""),
            "returncode": result.get("returncode"),
            "observed": result.get("observed"),
            "expected_marker": result.get("expected_marker", ""),
            "output_tail": (result.get("run") or result).get("stdout", "")[-800:],
        }
        if ok:
            sample["status"] = "verified"
            print(f"[PASS] {sid}")
        else:
            sample["status"] = "placeholder"
            failures.append(sid)
            print(f"[FAIL] {sid}: {(result.get('stderr') or '')[:300]}")

    matrix["summary"] = {
        "total": len(matrix["samples"]),
        "verified": sum(1 for s in matrix["samples"] if s["status"] == "verified"),
        "compilable": sum(1 for s in matrix["samples"] if s["status"] == "compilable"),
        "placeholder": sum(1 for s in matrix["samples"] if s["status"] == "placeholder"),
    }
    MATRIX_PATH.write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8")
    if failures:
        print(f"失败用例: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
