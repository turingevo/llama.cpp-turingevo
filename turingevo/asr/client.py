#!/usr/bin/env python3
"""调用 llama-server 的语音转写接口，仅用标准库。

  python3 client.py -f test.wav
  python3 client.py -f a.wav -f b.wav --language zh
  python3 client.py -f test.wav --json
  python3 client.py -f test.wav --raw -o out.txt
  python3 client.py -f test.wav --url http://127.0.0.1:9001

默认会去掉 Qwen3-ASR 输出里的 "language Chinese<asr_text>" 前缀，
加 --raw 可保留模型原始输出。
"""

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

MARKER = "<asr_text>"


def build_multipart(fields, file_field, file_path):
    boundary = uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.append(
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n".encode("utf-8")
        )
    ctype = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    chunks.append(
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{Path(file_path).name}\"\r\n"
        f"Content-Type: {ctype}\r\n\r\n".encode("utf-8")
    )
    chunks.append(Path(file_path).read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def clean_text(text):
    if MARKER in text:
        text = text.split(MARKER, 1)[1]
    return text.strip()


def audio_duration(path):
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else None
    except Exception:
        return None


def transcribe(base_url, args, path):
    fields = {}
    if args.language:
        fields["language"] = args.language
    if args.prompt:
        fields["prompt"] = args.prompt
    body, ctype = build_multipart(fields, "file", path)

    req = urllib.request.Request(
        base_url + "/v1/audio/transcriptions",
        data=body,
        headers={"Content-Type": ctype},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    elapsed = time.perf_counter() - start

    payload = json.loads(raw)
    text = payload.get("text", "")
    if not args.raw:
        text = clean_text(text)
    return payload, text, elapsed


def main():
    ap = argparse.ArgumentParser(description="调用 llama-server 做语音转写")
    ap.add_argument("-f", "--file", action="append", required=True,
                    help="音频文件，可重复（wav/mp3/flac）")
    ap.add_argument("--language", default="", help="语种，如 zh/en/ja；留空自动识别")
    ap.add_argument("--prompt", default="", help="附加提示词或上下文")
    ap.add_argument("--url", default="", help="服务地址，优先于 host/port")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--raw", action="store_true", help="保留 language xxx<asr_text> 前缀")
    ap.add_argument("--json", action="store_true", help="打印服务端原始 JSON")
    ap.add_argument("--timing", action="store_true", help="输出耗时和实时倍率")
    ap.add_argument("-o", "--output", default="", help="把转写文本写入文件")
    args = ap.parse_args()

    base_url = (args.url or f"http://{args.host}:{args.port}").rstrip("/")
    multi = len(args.file) > 1
    lines = []
    failed = 0

    for path in args.file:
        if not Path(path).is_file():
            print(f"error: 文件不存在: {path}", file=sys.stderr)
            failed += 1
            continue
        try:
            payload, text, elapsed = transcribe(base_url, args, path)
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace").strip()
            print(f"error: {path}: HTTP {err.code} {detail}", file=sys.stderr)
            failed += 1
            continue
        except urllib.error.URLError as err:
            print(f"error: 连不上 {base_url}: {err.reason}", file=sys.stderr)
            failed += 1
            continue
        except json.JSONDecodeError:
            print(f"error: {path}: 服务端返回的不是 JSON", file=sys.stderr)
            failed += 1
            continue

        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        elif multi:
            print(f"{path}\t{text}")
        else:
            print(text)
        lines.append(f"{path}\t{text}" if multi else text)

        if args.timing:
            dur = audio_duration(path)
            extra = f", {dur:.2f}s 音频, {dur / elapsed:.1f}x" if dur else ""
            print(f"[timing] {path}: {elapsed:.3f}s{extra}", file=sys.stderr)

    if args.output and lines:
        Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已写入: {args.output}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
