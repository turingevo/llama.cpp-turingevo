#!/usr/bin/env python3
"""启动 llama-server 提供 Qwen3-TTS 语音合成服务。

  python3 server.py                          # 127.0.0.1:8080
  python3 server.py --port 9002 -n 512
  python3 server.py --model-dir /path/to/gguf
  python3 server.py --model a.gguf --mmproj mmproj.gguf
  python3 server.py --dry-run                # 只打印命令

路径优先级: 命令行参数 > 环境变量 > 默认值。
  LLAMA_SERVER_BIN   llama-server 可执行文件
  TTS_MODEL_DIR      模型目录
  TTS_MODEL          主模型 GGUF
  TTS_MMPROJ         mmproj GGUF

服务就绪后:
  curl -X POST http://127.0.0.1:8080/tts \
      -F input=你好 -F lang=zh -F speaker_ref=@ref.wav -o out.wav
"""

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

BIN_NAME = "llama-server"
BUILD_DIRS = ("build-alldev/bin", "build/bin")

DEFAULT_MODEL_DIR = Path("/media/wmx/soft2/huggingface_cache/models/audio-tts/Qwen3-TTS-12Hz-1.7B-Base-GGUF")


def find_binary(explicit):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            sys.exit(f"error: 指定的 binary 不存在: {path}")
        return path
    env = os.environ.get("LLAMA_SERVER_BIN")
    if env:
        return Path(env)
    for sub in BUILD_DIRS:
        for name in (BIN_NAME, BIN_NAME + ".exe"):
            cand = ROOT / sub / name
            if cand.is_file():
                return cand
    found = shutil.which(BIN_NAME)
    if found:
        return Path(found)
    sys.exit(f"error: 找不到 {BIN_NAME}，先编译，或用 --bin / LLAMA_SERVER_BIN 指定")


def pick_gguf(model_dir):
    # 声码器和说话人编码器都在 mmproj 里，必须成对加载
    files = sorted(p for p in model_dir.glob("*.gguf") if p.is_file())
    mmproj = [p for p in files if p.name.lower().startswith("mmproj")]
    mains = [p for p in files if not p.name.lower().startswith("mmproj")]
    if not mains:
        sys.exit(f"error: {model_dir} 下没有主模型 gguf")
    if len(mains) > 1:
        names = "\n  ".join(p.name for p in mains)
        sys.exit(f"error: 主模型有多个，用 --model 指定:\n  {names}")
    if not mmproj:
        sys.exit(f"error: {model_dir} 下没有 mmproj-*.gguf，声码器在 mmproj 里")
    if len(mmproj) > 1:
        names = "\n  ".join(p.name for p in mmproj)
        sys.exit(f"error: mmproj 有多个，用 --mmproj 指定:\n  {names}")
    return mains[0], mmproj[0]


def url_host(host):
    return "127.0.0.1" if host in ("0.0.0.0", "::", "") else host


def port_busy(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((url_host(host), port)) == 0


def wait_ready(proc, host, port, timeout):
    url = f"http://{url_host(host)}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return f"llama-server 已退出，返回码 {proc.returncode}"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                body = resp.read().decode("utf-8", "replace")
            if '"ok"' in body:
                return ""
        except Exception:
            pass
        time.sleep(0.5)
    return f"等待 {timeout}s 仍未就绪"


def main():
    ap = argparse.ArgumentParser(description="启动 llama-server 跑 Qwen3-TTS")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bin", default="", help="llama-server 路径")
    ap.add_argument("--model-dir", default=os.environ.get("TTS_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    ap.add_argument("--model", default=os.environ.get("TTS_MODEL", ""), help="主模型 GGUF")
    ap.add_argument("--mmproj", default=os.environ.get("TTS_MMPROJ", ""), help="mmproj GGUF")
    ap.add_argument("-ngl", type=int, default=99, help="卸载到 GPU 的层数")
    ap.add_argument("-c", "--ctx", type=int, default=4096,
                    help="上下文长度，够装下文本和 512 帧音频码")
    ap.add_argument("-np", type=int, default=1, help="并行槽数，测试够用且省显存")
    ap.add_argument("-n", "--n-predict", type=int, default=0,
                    help="默认音频帧上限，请求里的 n_predict 优先；0 表示用内置 512 帧（约 41s）")
    ap.add_argument("-t", "--threads", type=int, default=0, help="CPU 线程数，0 为自动")
    ap.add_argument("--wait", type=float, default=180.0, help="等待就绪的秒数，0 表示不等待")
    ap.add_argument("--ui", action="store_true", help="启用内置 WebUI")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    args, extra = ap.parse_known_args()
    extra = [a for a in extra if a != "--"]

    binary = find_binary(args.bin)
    model_dir = Path(args.model_dir)
    if args.model and args.mmproj:
        model, mmproj = Path(args.model), Path(args.mmproj)
    elif args.model or args.mmproj:
        auto_model, auto_mmproj = pick_gguf(model_dir)
        model = Path(args.model) if args.model else auto_model
        mmproj = Path(args.mmproj) if args.mmproj else auto_mmproj
    else:
        model, mmproj = pick_gguf(model_dir)

    for label, path in (("主模型", model), ("mmproj", mmproj)):
        if not path.is_file():
            sys.exit(f"error: {label}不存在: {path}")

    if not args.dry_run and port_busy(args.host, args.port):
        sys.exit(f"error: {args.host}:{args.port} 已被占用，换一个 --port")

    # 用户透传的同名参数优先，避免出现两个 -ngl
    def given(*flags):
        return any(a in flags for a in extra)

    cmd = [str(binary)]
    if not given("-m", "--model"):
        cmd += ["-m", str(model)]
    if not given("--mmproj", "-mm"):
        cmd += ["--mmproj", str(mmproj)]
    if not given("--host", "-H"):
        cmd += ["--host", args.host]
    if not given("--port", "-p"):
        cmd += ["--port", str(args.port)]
    if not given("-ngl", "--gpu-layers", "-n-gpu-layers"):
        cmd += ["-ngl", str(args.ngl)]
    if not given("-c", "--ctx", "--ctx-size"):
        cmd += ["-c", str(args.ctx)]
    if not given("-np", "--parallel"):
        cmd += ["-np", str(args.np)]
    if args.n_predict > 0 and not given("-n", "--n-predict"):
        cmd += ["-n", str(args.n_predict)]
    if args.threads > 0 and not given("-t", "--threads"):
        cmd += ["-t", str(args.threads)]
    if not args.ui and not given("--ui", "--no-webui", "--webui"):
        cmd.append("--no-webui")
    cmd += extra

    print("模型: " + model.name)
    print("声码器: " + mmproj.name)
    print("启动: " + " ".join(cmd), flush=True)
    if args.dry_run:
        return 0

    proc = subprocess.Popen(cmd)

    def stop(signum, frame):
        proc.send_signal(signal.SIGINT)

    try:
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
    except ValueError:
        pass

    if args.wait > 0:
        err = wait_ready(proc, args.host, args.port, args.wait)
        if err:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            sys.exit(f"error: {err}")
        base = f"http://{url_host(args.host)}:{args.port}"
        print(f"就绪: {base}", flush=True)
        print(f"示例: curl -X POST {base}/tts "
              f"-F input=你好 -F lang=zh -F speaker_ref=@ref.wav -o out.wav", flush=True)

    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.wait()
        return 0


if __name__ == "__main__":
    sys.exit(main())
