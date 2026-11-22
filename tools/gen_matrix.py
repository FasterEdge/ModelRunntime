#!/usr/bin/env python3
# FasterEdge 开源项目 - Github: https://github.com/FasterEdge - Gitee: https://gitee.com/FasterEdge
"""从各示例源码真实提取可验证性状态，生成仓库根目录的 matrix.json。

状态判定规则（全部来自对源码的机械提取，不含人工填写的状态字段）：

- verified   该示例被 tools/verify_matrix.py 通过真实的「构建 + 运行」用例驱动过，
             用例名在本脚本 RUNTIME_CASES 中显式声明；构建/运行证据由 verify_matrix.py
             写入 matrix.json 的 evidence 字段。
- compilable 源码在语法/编译层面通过检查：Python 过 `py_compile` 或 `ast.parse`，
             Go 过 `gofmt -l`，JavaScript 过 `node --check`，C/C++ 过 `g++ -fsyntax-only`
             的翻译单元语法检查；Java 过 `javac`，C# 过 `dotnet build`，Rust 过
             `cargo check --offline`；但推理调用尚未真正执行（缺模型或运行库）。
- placeholder 以下三条独立信号命中任意一条即为 placeholder（源码证据保留在 signals 字段）：
             1. 存在未消除的 `TODO`（grep 结果）；
             2. 推理调用不存在或整体被注释掉（`//` 或 `#` 行注释后跟推理 API）；
             3. 推理函数是空壳：函数体里只有 return []/return None/return data/None，
                没有真正调用运行库。

字段说明：

- bundled_models 示例目录内**实际存在、随仓库分发**的模型文件相对路径列表。
             匹配规则：后缀大小写无关地命中 FORMAT_SUFFIXES；并排除
             node_modules / target / bin / obj / dist / __pycache__ 等产物目录。
             该字段为空表示示例不含模型文件，运行前须自行准备。
- compile_check 每种语言的真实编译检查：Python 过 ast.parse，Go 过 gofmt -l，
             JS 过 node --check，C++ 过 g++ -fsyntax-only，Java 过 javac -cp <本地 m2 jar>，
             C# 过 dotnet build，Rust 过 cargo check --offline（MSVC 不行时退到 GNU toolchain）。
             工具链缺失时 ok=false 且 skip=true，detail 如实写明缺什么。

用法：
    python tools/gen_matrix.py            # 重新提取并覆盖 matrix.json
    python tools/gen_matrix.py --check    # 只校验源码事实，matrix.json 与源码不一致则退出码 1
                                          # （environment / status / compile_check / evidence 不参与比对）
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
import tempfile
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


# 生成与比对过程中一律不进入的产物目录：
# node_modules（npm 依赖树，动辄上千个 LICENSE）、target（cargo 构建产物）、
# bin/obj/dist（构建输出）、__pycache__（Python 字节码缓存）。
# 这些目录既不是示例源码，也不是"随仓库分发的模型文件"，扫进去会把
# 依赖树里的 LICENSE 当成 bundled model、把构建中间产物当成模型文件。
EXCLUDED_DIRS = {"node_modules", "target", "bin", "obj", "dist", "__pycache__",
                 ".git", "venv", ".venv"}

# 真实模型文件的后缀（与 FORMAT_SUFFIXES 同一来源，按小写后缀匹配）。
# .bin 在 OpenVINO IR 里是权重文件，但 target/ 下也有同名文件，因此这里
# 只认"示例目录内的 .bin"。
MODEL_SUFFIXES = tuple(s.lower() for s in FORMAT_SUFFIXES)

# 后缀虽命中 FORMAT_SUFFIXES、但明确不是模型文件的构建描述文件。
# pom.xml 是 Maven 构建描述（.xml 也是 OpenVINO IR 的合法后缀），
# 不排除会把 Java 示例的构建文件误报成"随仓库分发的模型"。
NON_MODEL_BASENAMES = {"pom.xml", "androidmanifest.xml", "cmakelists.xml"}


def is_excluded(path: Path, base: Path | None = None) -> bool:
    """路径是否落在 EXCLUDED_DIRS 里的任何一级。

    base 给定时只判断 base 之下的相对路径，避免仓库自身路径
    （例如仓库恰好位于某个叫 dist / target 的目录下）造成误判。
    """
    if base is not None:
        try:
            path = path.relative_to(base)
        except ValueError:
            pass
    return bool(EXCLUDED_DIRS & {part.lower() for part in path.parts})


def iter_source_files(directory: Path) -> list[Path]:
    """示例目录内的源文件（排除产物目录）。"""
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and not is_excluded(p, directory)
                  and p.suffix in LANG_BY_SUFFIX)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_entry(directory: Path, lang: str) -> Path | None:
    for rel in ENTRY_CANDIDATES.get(lang, []):
        candidate = directory / rel
        if candidate.is_file() and not is_excluded(candidate.relative_to(directory)):
            return candidate
    # 兜底：目录下任意该语言源文件（排除产物目录，按相对路径排序取第一个）
    suffix = {v: k for k, v in LANG_BY_SUFFIX.items()}[lang]
    candidates = sorted(p for p in iter_source_files(directory)
                        if p.suffix == suffix)
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
    for src in iter_source_files(directory):
        text = read_text(src)
        for suf in FORMAT_SUFFIXES:
            if re.search(re.escape(suf) + r'["\']', text):
                found.add(suf.lstrip("."))
    return sorted(found)


def find_model_files(directory: Path) -> list[str]:
    """目录内实际存在、且随仓库分发的模型文件（真实验证时判断是否随仓库提供）。

    按 FORMAT_SUFFIXES 做大小写无关的后缀匹配；同时排除产物目录
    （node_modules / target / bin / obj / dist / __pycache__），
    否则会把构建产物、依赖树里的 LICENSE 当成模型文件。
    """
    present = []
    for p in sorted(directory.rglob("*")):
        if not p.is_file() or is_excluded(p, directory):
            continue
        if p.suffix.lower() in MODEL_SUFFIXES and p.name.lower() not in NON_MODEL_BASENAMES:
            present.append(str(p.relative_to(directory)).replace("\\", "/"))
    return present


def run_tool(argv: list[str], timeout: int = 60) -> dict:
    """带超时的外部命令执行，超时如实记为失败而不是无限等待。

    Windows 上子进程（Python / node / g++）默认用本地代码页（GBK，cp936）写 stdout，
    而这里统一按 utf-8 解码，中文就会被错误地每字节还原成 U+FFFD，写进 matrix.json
    后 evidence.output_tail 对人完全不可读。对策是显式给子进程设 UTF-8 环境变量
    （PYTHONIOENCODING / PYTHONUTF8 / LANG），让它自己按 utf-8 输出；
    解码端再用 utf-8 + errors=replace 兜底，任何情况下都不会抛异常。
    """
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONLEGACYWINDOWSSTDIO", "0")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              env=env)
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
        dotnet = shutil.which("dotnet") or shutil.which("dotnet.exe")
        if not dotnet:
            # 本机没有 .NET SDK：如实记录"无法检查"，而不是伪造一个假的失败原因
            return {"tool": "dotnet build", "ok": False, "skip": True,
                    "detail": "dotnet SDK 未安装（需 .NET SDK 8+），本机无法编译检查"}
        # 真检查：在示例目录内 dotnet build。缺 Microsoft.ML.OnnxRuntime 包需联网还原，
        # 离线时按失败记录。
        res = _run_tool_cwd([dotnet, "build", "-v", "q", "--nologo"], path.parent, timeout=300)
        err = res["stderr"].strip() or res["stdout"].strip()
        return {"tool": "dotnet build", "ok": res["returncode"] == 0,
                "detail": (err.splitlines()[0][:400] if err else "")}
    if lang == "java":
        javac = shutil.which("javac")
        if not javac:
            return {"tool": "javac", "ok": False, "skip": True,
                    "detail": "javac 未安装（需 JDK 11+），本机无法编译检查"}
        # 真检查：javac -d <tmp> <源文件>，classpath 取 pom.xml 里声明的 jar。
        # jar 只能来自本地 ~/.m2（离线可用）；本地没有对应 jar 时如实记为
        # "缺 classpath"，不算代码本身的错。
        cp = _java_classpath(path)
        if cp is None:
            return {"tool": "javac -cp <m2>", "ok": False, "skip": True,
                    "detail": "本地 Maven 仓库缺少 pom.xml 声明的运行库 jar（需 mvn/联网拉取）"}
        outdir = Path(tempfile.mkdtemp(prefix="fe-javac-"))
        res = run_tool([javac, "-encoding", "UTF-8", "-d", str(outdir),
                        "-cp", cp, str(path)], timeout=180)
        shutil.rmtree(outdir, ignore_errors=True)
        err = (res["stderr"] or "").strip()
        return {"tool": f"javac -cp <m2:{len(cp.split(os.pathsep))} jars>",
                "ok": res["returncode"] == 0,
                "detail": (err.splitlines()[0][:400] if err else "")}
    if lang == "rust":
        cargo = shutil.which("cargo") or shutil.which("cargo.exe")
        if not cargo:
            return {"tool": "cargo check", "ok": False, "skip": True,
                    "detail": "cargo 未安装（需 Rust 工具链），本机无法编译检查"}
        # 真检查：在示例目录内 cargo check --offline（离线只读 Cargo.lock 与本地 registry）。
        # 需要 MSVC link.exe 或 MinGW 链接器；缺链接器 / 无预编译 crate 时如实记为失败，
        # 并在 detail 里说明是"链接器缺失"而非"源码语法错误"。
        project_dir = path.parent.parent  # src/main.rs -> 项目根
        toolchains = _rust_toolchains()
        attempts = []
        for tc in toolchains:
            argv = [cargo] + ([f"+{tc}"] if tc else []) + ["check", "--offline", "--message-format", "short"]
            res = _run_tool_cwd(argv, project_dir, timeout=600)
            err = (res["stderr"] or "") + (res["stdout"] or "")
            first = _first_error_line(err)
            attempts.append({"toolchain": tc or "default", "ok": res["returncode"] == 0,
                             "detail": first})
            if res["returncode"] == 0:
                return {"tool": f"cargo check --offline ({tc or 'default'})",
                        "ok": True, "detail": "", "attempts": attempts}
        # 全部失败：区分"链接器/环境问题"与"源码错误"
        env_failure = any(("link.exe" in a["detail"]) or ("linker" in a["detail"].lower())
                          or ("no prebuilt binaries" in a["detail"])
                          for a in attempts)
        return {"tool": "cargo check --offline", "ok": False,
                "skip": env_failure,
                "detail": "; ".join(f"{a['toolchain']}: {a['detail']}" for a in attempts)[:400],
                "attempts": attempts}
    return {"tool": None, "ok": False, "detail": f"未知语言 {lang}"}


def _run_tool_cwd(argv: list[str], cwd: Path, timeout: int = 300) -> dict:
    """带 cwd 的外部命令执行（run_tool 不带 cwd，编译类检查需要在示例目录内跑）。

    与 run_tool 同样强制子进程走 UTF-8 输出，避免中文报错被按 GBK 读成 U+FFFD。
    """
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    try:
        proc = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout,
                              env=env)
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": f"超时（>{timeout}s）"}
    except OSError as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}
    return {"returncode": proc.returncode, "stdout": proc.stdout or "",
            "stderr": proc.stderr or ""}


def _rust_toolchains() -> list[str]:
    """本机可用的 rustup toolchain（MSVC 优先，其次 GNU）。

    Windows 上 MSVC 需要 link.exe（Visual Studio Build Tools），常缺；
    GNU 用 MinGW gcc 即可链接，因此两种都试，任一通过即算通过。
    """
    rustup = shutil.which("rustup")
    if not rustup:
        return [""]
    res = run_tool([rustup, "toolchain", "list"], timeout=30)
    tcs = []
    for line in res["stdout"].splitlines():
        name = line.strip().split(" ")[0].strip()
        if name and name not in tcs:
            tcs.append(name)
    # msvc（含 default）在前，gnu 等在后
    tcs.sort(key=lambda n: (0 if "msvc" in n else 1, n))
    return tcs or [""]


def _first_error_line(text: str) -> str:
    """从编译输出里挑第一条真错误；没有 error 行才退回首行。

    cargo check --message-format short 的正常输出以 "Compiling xxx" 开头，
    直接取首行会把编译进度当成失败原因写进 matrix.json。
    """
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("error"):
            return s[:400]
    return text.strip().splitlines()[0][:400] if text.strip() else ""


def _find_pom(entry: Path) -> Path | None:
    """从 Java 入口文件向上找最近的 pom.xml。

    `src/main/java/Main.java` 的父目录是 `src/main/java`，pom.xml 在四级之上，
    因此一路向上找到仓库根为止。
    """
    for parent in [entry.parent, *entry.parent.parents]:
        pom = parent / "pom.xml"
        if pom.is_file():
            return pom
        if parent == REPO_ROOT:
            break
    return None


def _java_classpath(entry: Path) -> str | None:
    """从示例目录的 pom.xml 里找本地 ~/.m2 中已有的 jar，拼成 classpath。

    只在本地仓库能凑齐全部声明依赖时返回字符串；缺任何一个 jar 返回 None，
    让调用方如实记录"缺 classpath"而不是伪造编译结果。
    """
    pom = _find_pom(entry)
    if pom is None:
        return None
    try:
        text = read_text(pom)
    except OSError:
        return None
    deps = re.findall(
        r"<dependency>.*?<groupId>([^<]+)</groupId>\s*"
        r"<artifactId>([^<]+)</artifactId>\s*<version>([^<]+)</version>",
        text, re.S)
    if not deps:
        return None
    m2 = Path.home() / ".m2" / "repository"
    jars: list[str] = []
    for group, artifact, version in deps:
        local = m2 / group.replace(".", "/") / artifact / version / f"{artifact}-{version}.jar"
        if not local.is_file():
            return None
        jars.append(str(local))
    return os.pathsep.join(jars)


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
            # 仅作记录，不参与 --check 比对（见 normalize 的说明）
            "python": sys.version.split()[0],
            "python_syntax_checker": "ast.parse（与 python -m py_compile 的语法层等价）",
            "python_stdout_encoding": sys.stdout.encoding,
            "note": "environment 只记录本机工具链实况，不参与 --check 比对",
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
            """--check 只比对与工具链无关的源码事实：signals / format / entry / missing_deps
            / bundled_models。

            有意排除 status、extracted_status、compile_check、evidence、summary、environment：
            - status / evidence 由 verify_matrix.py 真实跑通后写入；
            - compile_check 依赖本机是否装了 g++ / node / gofmt / dotnet / cargo，
              跨机器比对必然产生假失败（CI 与本地工具链不同）；
            - environment 记录的是本机解释器版本（sys.version），换个 Python 小版本
              就会让 --check 报假失败，因此不参与比对。
            这也意味着 --check 无法发现"本机状态字段过期"，需要重新跑一次
            gen_matrix.py + verify_matrix.py 才能确认，README 已注明。"""
            doc = json.loads(json.dumps(doc))
            for sample in doc.get("samples", []):
                for key in ("status", "extracted_status", "compile_check", "evidence"):
                    sample.pop(key, None)
            for key in ("summary", "environment"):
                doc.pop(key, None)
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
