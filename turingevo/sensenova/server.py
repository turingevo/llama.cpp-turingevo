#!/usr/bin/env python3
"""启动 llama-sensenova-server,加载本机的 SenseNova-U1.5 权重。

  python3 server.py                     # 127.0.0.1:8888
  python3 server.py --port 9000 -ngl 39
  python3 server.py --model /path/lm.gguf --fm /path/fm.gguf
  python3 server.py --dry-run           # 只打印命令,不起服务

路径优先级: 命令行参数 > 环境变量 > 默认值。
  SENSENOVA_SERVER_BIN   服务端二进制
  SENSENOVA_MODEL_DIR    权重目录
  SENSENOVA_MODEL        语言模型 GGUF 文件名
  SENSENOVA_FM           FM sidecar 文件名
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

DEFAULT_MODEL_DIR = Path("/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp")
DEFAULT_MODEL = "SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf"
DEFAULT_FM = "F16-fm.gguf"

BIN_NAME = "llama-sensenova-server"
BUILD_DIRS = ("build-alldev/bin", "build/bin")


def find_binary():
    env = os.environ.get("SENSENOVA_SERVER_BIN")
    if env:
        return Path(env)
    names = [BIN_NAME, BIN_NAME + ".exe"]
    for sub in BUILD_DIRS:
        for name in names:
            cand = ROOT / sub / name
            if cand.is_file():
                return cand
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def main():
    ap = argparse.ArgumentParser(description="启动 llama-sensenova-server",
                                 add_help=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--ngl", type=int, default=999)
    ap.add_argument("--model", default=os.environ.get("SENSENOVA_MODEL", ""),
                    help="语言模型 GGUF 路径")
    ap.add_argument("--fm", default=os.environ.get("SENSENOVA_FM", ""),
                    help="FM sidecar 路径")
    ap.add_argument("--model-dir",
                    default=os.environ.get("SENSENOVA_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    ap.add_argument("--output-dir", default="",
                    help="url 响应的落盘目录，默认系统临时目录下的 llama-sensenova-server")
    ap.add_argument("--no-allow-output-dir", action="store_true",
                    help="不允许请求里指定 output_dir")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    args, extra = ap.parse_known_args()

    binary = find_binary()
    if binary is None:
        sys.exit(f"error: 找不到 {BIN_NAME}，先编译，"
                 f"或用 SENSENOVA_SERVER_BIN 指定路径")

    model_dir = Path(args.model_dir)
    model = Path(args.model) if args.model else model_dir / DEFAULT_MODEL
    fm = Path(args.fm) if args.fm else model_dir / DEFAULT_FM

    for label, path in (("语言模型", model), ("FM sidecar", fm)):
        if not path.is_file():
            sys.exit(f"error: {label}不存在: {path}")

    output_dir = args.output_dir
    if not output_dir:
        import tempfile
        output_dir = str(Path(tempfile.gettempdir()) / "llama-sensenova-server")

    # 用户自己透传的同名参数优先,避免出现两个 -ngl 这种重复项
    def given(*flags):
        return any(a in flags for a in extra)

    cmd = [str(binary)]
    if not given("-m", "--model"):
        cmd += ["-m", str(model)]
    if not given("--fm"):
        cmd += ["--fm", str(fm)]
    if not given("--host"):
        cmd += ["--host", args.host]
    if not given("--port"):
        cmd += ["--port", str(args.port)]
    if not given("-ngl"):
        cmd += ["-ngl", str(args.ngl)]
    if not given("--output-dir"):
        cmd += ["--output-dir", output_dir]
    if not given("--allow-output-dir") and not args.no_allow_output_dir:
        cmd.append("--allow-output-dir")
    cmd += [a for a in extra if a != "--"]

    print("启动: " + " ".join(cmd))
    if args.dry_run:
        return 0

    proc = subprocess.Popen(cmd)

    def stop(signum, frame):
        proc.send_signal(signal.SIGINT)

    try:
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
    except ValueError:
        # 不是主线程,交给默认处理
        pass

    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.wait()
        return 0


if __name__ == "__main__":
    sys.exit(main())
