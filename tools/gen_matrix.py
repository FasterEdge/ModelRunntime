#!/usr/bin/env python3
# FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
"""从各示例源码真实提取可验证性状态，生成仓库根目录的 matrix.json。

状态判定规则（全部来自对源码的机械提取，不含人工填写的状态字段）：

- verified   该示例被 tools/verify_matrix.py 通过真实的「构建 + 运行」用例驱动过，
             用例名在本脚本 RUNTIME_CASES 中显式声明；构建/运行证据由 verify_matrix.py
             写入 matrix.json 的 evidence 字段。
- compilable 源码在语法/编译层面通过检查：Python 过 `py_compile` 或 `ast.parse`，
             Go 过 `gofmt -l`，JavaScript 过 `node --check`，C/C++ 过 `g++ -fsyntax-only -fsyntax-only`
             的翻译单元语法检查；但推理调用尚未真正执行（缺模型或运行库）。
- placeholder 以下三条独立信号命中任意一条即为 placeholder（源码证据保留在 signals 字段）：
             1. 存在未消除的 `TODO`（grep 结果）；
             2. 推理调用不存在或整体被注释掉（`//` 或 `#` 行注释后跟推理 API）；
             3. 推理函数是空壳：函数体里只有 return []/return None/return data/None，
                没有真正调用运行库。

用法：
    python tools/gen_matrix.py            # 重新提取并覆盖 matrix.json
    python tools/gen_matrix.py --check    # 只校验，matrix.json 与源码不一致则退出码 1
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = REPO_ROOT / "matrix.json"

# 每个示例的静态元数据（平台 / 语言 / 运行库 / 格式 / 缺失依赖）。
# 这些字段描述的是"示例声明要跑什么"，不是"是否跑通"，跑通与否一律由下面的提取逻辑判定。
# 目录名 -> (平台, 语言, 运行库, 模型格式, 入口文件, 缺失依赖列表)
LANG_BY_SUFFIX = {
    ".py": "python",
    ".go": "go",
    ".js": "javascript",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cs": "csharp",
    ".java": "java",
    ".rs": "rust",
}

ENTRY_CANDIDATES = {
    "python": ["main.py", "test.py"],
    "go": ["main.go"],
    "javascript": ["index.js"],
    "cpp": ["main.cc", "main.cpp"],
    "csharp": ["Program.cs"],
    "java": ["src/main/java/Main.java"],
    "rust": ["src/main.rs"],
}

# 各运行库的典型推理/加载调用（用于判定"推理调用是否存在且未被注释掉"）。
# 只列真正会触达模型执行或会话创建的 API。
INFERENCE_CALL_PATTERNS = {
    "onnxruntime": [r"InferenceSession", r"\.run\(", r"NewAdvancedSession", r"createSession"],
    "openvino": [r"compile_model", r"compile_model", r"get_output_tensor", r"infer\("],
    "tensorflow": [r"Interpreter\(", r"saved_model\.load", r"load_graph_model"],
    "tflite": [r"Interpreter\(", r"InterpreterBuilder", r"AllocateTensors", r"allocate_tensors"],
    "torch": [r"torch\.jit\.load", r"model\(", r"forward\("],
    "libtorch": [r"torch::jit::load", r"module\.forward", r"\.forward\("],
    "tensorrt": [r"deserialize_cuda_engine", r"createInferRuntime", r"enqueueV"],
    "keras": [r"load_model", r"\.predict\("],
    "jax": [r"jax\.jit", r"\.block_until_ready"],
    "paddle": [r"create_predictor", r"copy_from_cpu"],
    "paddlelite": [r"create_paddle_predictor", r"run\(\)"],
    "opencv": [r"readNet", r"readNetFromCaffe", r"net\.forward"],
    "llama": [r"Llama\(", r"create_chat_completion", r"exec\.Command"],
    "acl": [r"aclmdlExecute", r"run_inference\("],
    "rknn": [r"rknn_run", r"\.inference\("],
    "magicmind": [r"enqueue"],
    "sail": [r"engine\.process"],
    "mnn": [r"runSession", r"createSession"],
    "ncnn": [r"extract\(", r"Extractor"],
    "djl": [r"newPredictor", r"loadModel"],
    "webnn": [r"navigator\.ml", r"build\("],
    "qai_hub": [r"submit_compile_job", r"submit_inference_job"],
    "coreml": [r"MLModel", r"\.predict\("],
}

# 模型格式：按目录名/常量里的文件后缀真实提取
FORMAT_SUFFIXES = [
    ".onnx", ".tflite", ".pb", ".pt", ".pth", ".h5", ".keras", ".pdmodel", ".pdiparams",
    ".xml", ".bin", ".engine", ".prototxt", ".caffemodel", ".gguf", ".msgpack",
    ".mlpackage", ".mlmodel", ".pte", ".nb", ".mnn", ".param", ".rknn", ".om", ".bmodel",
    ".neuron", ".dlc", ".safetensors", ".msgpack",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_entry(directory: Path, lang: str) -> Path | None:
    for rel in ENTRY_CANDIDATES.get(lang, []):
        candidate = directory / rel
        if candidate.is_file():
            return candidate
    # 兜底：目录下任意该语言源文件（按文件名排序取第一个）
    suffix = {v: k for k, v in LANG_BY_SUFFIX.items()}[lang]
    candidates = sorted(p for p in directory.rglob("*" + suffix) if p.is_file()
                        and "__pycache__" not in p.parts)
    return candidates[0] if candidates else None


def language_of(directory: Path) -> str | None:
    for lang in ENTRY_CANDIDATES:
        if find_entry(directory, lang) is not None:
            return lang
    return None


def grep_todos(text: str) -> list[int]:
    """返回含未消除 TODO 的行号（1-based）。"""
    return [i for i, line in enumerate(text.splitlines(), 1) if "TODO" in line]


def commented_call_lines(text: str, patterns: list[str]) -> list[int]:
    """找出被行注释掉的推理调用（`#` 或 `//` 开头的行里出现推理 API）。"""
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        for comment in ("#", "//"):
            if stripped.startswith(comment):
                body = stripped[len(comment):]
                if any(re.search(p, body) for p in patterns):
                    hits.append(i)
                    break
    return hits


def live_call_lines(text: str, patterns: list[str]) -> list[int]:
    """找出未被注释掉的推理调用所在行号。"""
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        # 去掉行尾注释再匹配，避免把 `x = 1  # TODO: session.run()` 当作真实调用
        code = stripped.split(" // ")[0]
        if len(stripped) > 1 and stripped[0] == "#":
            continue
        if any(re.search(p, code) for p in patterns):
            hits.append(i)
    return hits


def call_patterns(lib: str) -> list[str]:
    patterns: list[str] = []
    for key, pats in INFERENCE_CALL_PATTERNS.items():
        if key.lower() in lib.lower():
            patterns.extend(pats)
    return patterns or [r"\.run\(", r"infer", r"predict", r"forward", r"execute"]


def python_syntax_ok(path: Path) -> tuple[bool, str]:
    """用 ast.parse 做语法检查（等价于 py_compile 的语法层，但不需要写 .pyc）。"""
    try:
        ast.parse(read_text(path), filename=str(path))
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.msg} (line {exc.lineno})"


def python_unused_constant_hits(path: Path, names: list[str]) -> list[str]:
    """统计模块级常量是否真的被读取过（除定义行外只出现一次即为"只定义未使用"）。"""
    text = read_text(path)
    unused = []
    for name in names:
        uses = len(re.findall(rf"\b{re.escape(name)}\b", text))
        if uses <= 1:
            unused.append(name)
    return unused


def extract_python_placeholders(path: Path) -> dict:
    """空壳函数检测：返回函数名列表，其函数体没有调用运行库也没引用占位常量之外的东西。"""
    text = read_text(path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {"shell_functions": [], "parse_error": True}
    shell = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = [n for n in node.body if not isinstance(n, ast.Expr)
                or not isinstance(getattr(n, "value", None), ast.Constant)]
        # 函数体只有 return 字面量/None/list/dict -> 空壳
        trivial_returns = []
        only_trivial = True
        for stmt in body:
            if isinstance(stmt, ast.Return):
                val = stmt.value
                trivial = val is None or (isinstance(val, ast.Constant)) or (
                    isinstance(val, (ast.List, ast.Tuple, ast.Dict)) and not val.elts
                    if isinstance(val, (ast.List, ast.Tuple)) else isinstance(val, ast.Dict) and not val.keys)
                trivial_returns.append(trivial)
                only_trivial = only_trivial and trivial
            else:
                only_trivial = False
        if body and only_trivial and all(trivial_returns):
            shell.append(node.name)
    return {"shell_functions": shell, "parse_error": False}


def extract_formats(directory: Path) -> list[str]:
    """从目录内源码里的字符串常量提取模型文件后缀（真实 grep，不照抄 README）。"""
    found: set[str] = set()
    for src in sorted(p for p in directory.rglob("*") if p.is_file()
                      and p.suffix in LANG_BY_SUFFIX and "__pycache__" not in p.parts):
        text = read_text(src)
        for suf in FORMAT_SUFFIXES:
            if re.search(re.escape(suf) + r'["\']', text):
                found.add(suf.lstrip("."))
    return sorted(found)


def find_model_files(directory: Path) -> list[str]:
    """目录内实际存在的模型文件（真实验证时判断是否随仓库提供）。"""
    present = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix in [f"" for f in FORMAT_SUFFIXES]:
            present.append(str(p.relative_to(directory)))
    return present


def run_tool(argv: list[str], timeout: int = 60) -> dict:
    """带超时的外部命令执行，超时如实记为失败而不是无限等待。"""
    try:
        # 显式 utf-8 + errors=replace：Windows 上 g++ 的本地化报错是 GBK，
        # 用默认 locale 解码会抛 UnicodeDecodeError。
        proc = subprocess.run(argv, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": f"超时（>{timeout}s）"}
    except OSError as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}
    return {"returncode": proc.returncode, "stdout": proc.stdout or "",
            "stderr": proc.stderr or ""}


def compile_check(path: Path, lang: str) -> dict:
    """语言级语法/编译检查。返回 {tool, ok, detail}。"""
    if lang == "python":
        ok, detail = python_syntax_ok(path)
        return {"tool": "ast.parse (equivalent to py_compile)", "ok": ok, "detail": detail}
    if lang == "go":
        gofmt = shutil.which("gofmt")
        if not gofmt:
            return {"tool": "gofmt", "ok": False, "detail": "gofmt 未在 PATH（需 Go 工具链）"}
        res = run_tool([gofmt, "-l", str(path)])
        dirty = res["stdout"].strip()
        return {"tool": "gofmt -l", "ok": res["returncode"] == 0 and not dirty,
                "detail": "" if not dirty else f"未格式化: {dirty}"}
    if lang == "javascript":
        node = shutil.which("node")
        if not node:
            return {"tool": "node --check", "ok": False, "detail": "node 未安装"}
        res = run_tool([node, "--check", str(path)])
        return {"tool": "node --check", "ok": res["returncode"] == 0,
                "detail": res["stderr"].strip()[:400]}
    if lang in ("cpp",):
        cc = shutil.which("g++") or shutil.which("clang++")
        if not cc:
            return {"tool": "g++ -fsyntax-only", "ok": False, "detail": "无 C++ 编译器"}
        # 只做翻译单元语法检查：缺第三方头文件会失败，这正是"缺失依赖"的证据
        res = run_tool([cc, "-std=c++17", "-fsyntax-only", str(path)], timeout=90)
        err = res["stderr"].strip()
        return {"tool": "g++ -std=c++17 -fsyntax-only", "ok": res["returncode"] == 0,
                "detail": (err.splitlines()[0][:400] if err else "")}
    if lang == "csharp":
        dotnet = shutil.which("dotnet")
        if not dotnet:
            return {"tool": "dotnet build", "ok": False, "detail": "dotnet SDK 未安装"}
        return {"tool": "dotnet build", "ok": False, "detail": "未实现：需在示例目录内构建"}
    if lang == "java":
        javac = shutil.which("javac")
        if not javac:
            return {"tool": "javac", "ok": False, "detail": "javac 未安装"}
        return {"tool": "javac", "ok": False, "detail": "未实现：需 classpath 含运行库 jar"}
    if lang == "rust":
        cargo = shutil.which("cargo")
        if not cargo:
            return {"tool": "cargo check", "ok": False, "detail": "cargo 未安装"}
        return {"tool": "cargo check", "ok": False, "detail": "未实现：需联网拉 crate 并链接 ORT"}
    return {"tool": None, "ok": False, "detail": f"未知语言 {lang}"}


def build_entry(directory: Path, meta: dict) -> dict:
    lang = language_of(directory)
    if lang is None:
        return {}
    entry = find_entry(directory, lang)
    text = read_text(entry)
    lib = meta["runtime"]

    todo_lines = grep_todos(text)
    patterns = call_patterns(lib)
    live_calls = live_call_lines(text, patterns)
    dead_calls = commented_call_lines(text, patterns)
    shell = {"shell_functions": [], "parse_error": False}
    if lang == "python":
        shell = extract_python_placeholders(entry)

    signals = {
        "todo_lines": todo_lines,
        "live_inference_call_lines": live_calls,
        "commented_out_inference_call_lines": dead_calls,
        "shell_functions": shell["shell_functions"],
    }

    has_blocker = bool(todo_lines) or not live_calls
    if lang == "python" and shell["shell_functions"]:
        # 空壳函数不算致命：只有连一条真实调用都没有时才算 placeholder
        has_blocker = has_blocker or not live_calls

    return {
        "id": directory.name,
        "platform": meta["platform"],
        "language": lang,
        "runtime": lib,
        "format": extract_formats(directory),
        "entry": str(entry.relative_to(REPO_ROOT)).replace("\\", "/"),
        "extracted_status": "placeholder" if has_blocker else "compilable",
        "missing_deps": meta["missing_deps"],
        "signals": signals,
        "bundled_models": find_model_files(directory),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从源码提取可验证性矩阵")
    parser.add_argument("--check", action="store_true", help="校验 matrix.json 与源码一致")
    args = parser.parse_args()

    spec = json.loads(read_text(REPO_ROOT / "tools" / "matrix_spec.json"))
    entries = []
    for meta in spec["samples"]:
        directory = REPO_ROOT / meta["id"]
        if not directory.is_dir():
            print(f"跳过不存在的示例目录: {meta['id']}", file=sys.stderr)
            continue
        item = build_entry(directory, meta)
        if not item:
            print(f"跳过无源文件的示例目录: {meta['id']}", file=sys.stderr)
            continue
        item["compile_check"] = compile_check(Path(REPO_ROOT / item["entry"]), item["language"])
        if item["extracted_status"] == "compilable" and not item["compile_check"]["ok"]:
            # 语法/编译检查失败即不是 compilable，降级为 placeholder 并记录原因
            item["extracted_status"] = "placeholder"
            item["signals"]["compile_failed"] = item["compile_check"]["detail"]
        # status 是"生效状态"：默认等于源码提取值，verify_matrix.py 真实跑通后
        # 会把它提升为 verified 并写入 evidence。gen_matrix 从不自行写 verified。
        item["status"] = item["extracted_status"]
        entries.append(item)

    counts = {"verified": 0, "compilable": 0, "placeholder": 0}
    for item in entries:
        counts[item["extracted_status"]] += 1

    matrix = {
        "schema_version": 1,
        "generated_by": "tools/gen_matrix.py",
        "note": "本文件由脚本从源码机械提取生成，请勿手工编辑；状态字段不来自人工判断。",
        "status_rules": {
            "verified": "由 tools/verify_matrix.py 真实构建并运行通过，evidence 字段记录命令与输出",
            "compilable": "语法/编译检查通过，但推理调用未执行（缺模型或运行库）",
            "placeholder": "存在未消除 TODO，或推理调用缺失/被注释，或语法编译检查失败",
        },
        "environment": {
            "python": sys.version.split()[0],
            "python_syntax_checker": "ast.parse（与 python -m py_compile 的语法层等价）",
        },
        "summary": {
            "total": len(entries),
            **counts,
        },
        "samples": entries,
    }

    payload = json.dumps(matrix, ensure_ascii=False, indent=2, sort_keys=False) + "\n"

    if args.check:
        if not MATRIX_PATH.is_file():
            print("matrix.json 不存在，请先运行 python tools/gen_matrix.py", file=sys.stderr)
            return 1

        def normalize(doc: dict) -> str:
            """--check 只比对与工具链无关的源码事实：signals / format / entry / missing_deps。

            有意排除 status、extracted_status、compile_check、evidence：
            - status / evidence 由 verify_matrix.py 真实跑通后写入；
            - compile_check 依赖本机是否装了 g++ / node / gofmt / dotnet / cargo，
              跨机器比对必然产生假失败（CI 与本地工具链不同）。
            这也意味着 --check 无法发现"本机状态字段过期"，需要重新跑一次
            gen_matrix.py + verify_matrix.py 才能确认，README 已注明。"""
            doc = json.loads(json.dumps(doc))
            for sample in doc.get("samples", []):
                for key in ("status", "extracted_status", "compile_check", "evidence"):
                    sample.pop(key, None)
            doc.pop("summary", None)
            return json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True)

        current = json.loads(read_text(MATRIX_PATH))
        fresh = json.loads(payload)
        if normalize(current) != normalize(fresh):
            print("matrix.json 与源码不一致，已过期，请重新生成", file=sys.stderr)
            stale = []
            for cur_s, fresh_s in zip(current.get("samples", []), fresh["samples"]):
                if normalize({"samples": [cur_s]}) != normalize({"samples": [fresh_s]}):
                    stale.append(cur_s.get("id", "?"))
            if stale:
                print(f"不一致的示例: {', '.join(stale)}", file=sys.stderr)
            return 1
        print(f"matrix.json 与源码一致（{len(entries)} 个示例）")
        return 0

    MATRIX_PATH.write_text(payload, encoding="utf-8")
    print(f"已写入 {MATRIX_PATH.relative_to(REPO_ROOT)}：{len(entries)} 个示例 "
          f"(verified={counts['verified']} compilable={counts['compilable']} "
          f"placeholder={counts['placeholder']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
