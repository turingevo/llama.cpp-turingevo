#!/usr/bin/env python3
"""调用 llama-server 的 /tts 接口做语音合成，仅用标准库。

  python3 client.py -p "你好，世界" --lang zh -o out.wav
  python3 client.py -p "Hello world" --speaker ref.wav -o out.wav
  python3 client.py -f text.txt --stream -o out.wav
  python3 client.py -f long.txt --split 150 -o out.wav
  python3 client.py -p "测试" --format pcm -o out.pcm
  python3 client.py -p "测试" --url http://127.0.0.1:9002

有参考音频时用 multipart 上传，没有则用 JSON；
--stream 边收边写，收完再修正 WAV 头里的 0xFFFFFFFF 长度字段；
--split N 按标点切成不超过 N 字的片段，逐段合成后拼接，用于超过单次帧上限的长文本。
"""

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SENTENCE_END = "。！？!?；;…"
COMMA_END = "，,、 "


def build_multipart(fields, file_field, file_path):
    boundary = uuid.uuid4().hex
    chunks = []
    for name, value in fields.items():
        chunks.append(
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n".encode("utf-8")
        )
    ctype = mimetypes.guess_type(file_path)[0] or "audio/wav"
    chunks.append(
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"{file_field}\"; "
        f"filename=\"{Path(file_path).name}\"\r\n"
        f"Content-Type: {ctype}\r\n\r\n".encode("utf-8")
    )
    chunks.append(Path(file_path).read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def join_sep(a, b):
    if not a or not b:
        return ""
    if "\u4e00" <= a[-1] <= "\u9fff" or "\u4e00" <= b[0] <= "\u9fff":
        return ""
    return " "


def split_text(text, limit):
    sentences = []
    buf = ""
    for ch in text:
        if ch in "\r\n":
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
            continue
        buf += ch
        if ch in SENTENCE_END:
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
    if buf.strip():
        sentences.append(buf.strip())

    chunks = []
    cur = ""
    for sent in sentences:
        while len(sent) > limit:
            cut = max(sent.rfind(c, 0, limit) for c in COMMA_END) + 1
            if cut < limit // 2:
                cut = limit
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(sent[:cut])
            sent = sent[cut:]
        sep = join_sep(cur, sent)
        if len(cur) + len(sep) + len(sent) <= limit:
            cur += sep + sent
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return chunks


def wav_info(path):
    size = Path(path).stat().st_size
    with open(path, "rb") as fh:
        head = fh.read(min(size, 4096))
    if len(head) < 44 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return None
    channels = int.from_bytes(head[22:24], "little")
    rate = int.from_bytes(head[24:28], "little")
    bits = int.from_bytes(head[34:36], "little")
    if not rate or not channels or not bits:
        return None
    data_bytes = max(size - 44, 0)  # 流式响应已回填长度，头部固定 44 字节
    return data_bytes / (rate * channels * bits // 8), rate


def patch_wav_sizes(path):
    """流式 WAV 的长度字段是 0xFFFFFFFF，按实际文件大小回填。"""
    size = Path(path).stat().st_size
    with open(path, "r+b") as fh:
        head = fh.read(min(size, 4096))
        if len(head) < 44 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            return
        pos = 12
        while pos + 8 <= len(head):
            chunk_id = head[pos:pos + 4]
            chunk_size = int.from_bytes(head[pos + 4:pos + 8], "little")
            if chunk_id == b"data":
                if chunk_size == 0xFFFFFFFF:
                    fh.seek(4)
                    fh.write((size - 8).to_bytes(4, "little"))
                    fh.seek(pos + 4)
                    fh.write((size - pos - 8).to_bytes(4, "little"))
                return
            pos += 8 + chunk_size + (chunk_size & 1)


def pcm_info(path, content_type):
    rate, bits, channels = 0, 32, 1
    for part in (content_type or "").split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key == "rate":
            rate = int(value)
        elif key == "bits":
            bits = int(value)
        elif key == "channels":
            channels = int(value)
    if not rate:
        return None
    size = Path(path).stat().st_size
    return size / (rate * channels * bits // 8), rate


def wav_fmt_data(blob):
    if len(blob) < 44 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        return None
    pos = 12
    fmt = None
    while pos + 8 <= len(blob):
        chunk_id = blob[pos:pos + 4]
        chunk_size = int.from_bytes(blob[pos + 4:pos + 8], "little")
        if chunk_id == b"fmt ":
            fmt = blob[pos + 8:pos + 8 + chunk_size]
        elif chunk_id == b"data":
            end = len(blob) if chunk_size == 0xFFFFFFFF else pos + 8 + chunk_size
            return fmt, blob[pos + 8:end]
        pos += 8 + chunk_size + (chunk_size & 1)
    return None


def write_wav(path, fmt, data):
    header = (
        b"RIFF" + (36 + len(data)).to_bytes(4, "little") + b"WAVE"
        + b"fmt " + len(fmt).to_bytes(4, "little") + fmt
        + b"data" + len(data).to_bytes(4, "little")
    )
    Path(path).write_bytes(header + data)


def post_tts(base_url, payload, speaker, timeout, sink):
    if speaker:
        form = {k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in payload.items()}
        body, ctype = build_multipart(form, "speaker_ref", speaker)
    else:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ctype = "application/json"

    req = urllib.request.Request(base_url + "/tts", data=body,
                                 headers={"Content-Type": ctype})
    start = time.perf_counter()
    first_byte = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content_type = resp.headers.get("Content-Type", "")
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            if first_byte is None:
                first_byte = time.perf_counter() - start
            sink(chunk)
    return content_type, time.perf_counter() - start, first_byte


def main():
    ap = argparse.ArgumentParser(description="调用 llama-server 合成语音")
    ap.add_argument("-p", "--prompt", default="", help="要合成的文本")
    ap.add_argument("-f", "--prompt-file", default="", help="文本文件，-p 优先")
    ap.add_argument("--lang", default="", help="语种，如 zh/en/ja；留空用模型默认")
    ap.add_argument("--speaker", default="", help="参考音频，用于克隆音色（wav/mp3）")
    ap.add_argument("-o", "--output", default="", help="输出文件，默认 out.wav / out.pcm")
    ap.add_argument("--format", choices=["wav", "pcm"], default="wav")
    ap.add_argument("--stream", action="store_true", help="边生成边写文件")
    ap.add_argument("--split", type=int, default=0,
                    help="按标点切成每段最多 N 字，逐段合成后拼接，0 表示不切")
    ap.add_argument("-n", "--n-predict", type=int, default=0,
                    help="音频帧上限，0 用服务端默认（512 帧约 41s）")
    ap.add_argument("--top-k", type=int, default=0, help="声学码采样 top-k，0 用默认")
    ap.add_argument("--top-p", type=float, default=0.0, help="nucleus 采样，0 用默认")
    ap.add_argument("--repeat-penalty", type=float, default=0.0, help="重复惩罚，0 用默认")
    ap.add_argument("--seed", type=int, default=None, help="随机种子，固定后结果可复现")
    ap.add_argument("--url", default="", help="服务地址，优先于 host/port")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    prompt = args.prompt
    if not prompt and args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").rstrip()
    if not prompt:
        sys.exit("error: 需要 -p 文本或 -f 文本文件")
    if args.speaker and not Path(args.speaker).is_file():
        sys.exit(f"error: 参考音频不存在: {args.speaker}")

    output = args.output or ("out.wav" if args.format == "wav" else "out.pcm")
    base_url = (args.url or f"http://{args.host}:{args.port}").rstrip("/")

    payload = {
        "input": prompt,
        "response_format": args.format,
        "stream": args.stream,
    }
    if args.lang:
        payload["lang"] = args.lang
    if args.n_predict > 0:
        payload["n_predict"] = args.n_predict
    if args.top_k > 0:
        payload["top_k"] = args.top_k
    if args.top_p > 0:
        payload["top_p"] = args.top_p
    if args.repeat_penalty > 0:
        payload["repeat_penalty"] = args.repeat_penalty
    if args.seed is not None:
        payload["seed"] = args.seed

    chunks = split_text(prompt, args.split) if args.split > 0 else [prompt]
    start = time.perf_counter()
    first_byte = None
    content_type = ""

    def call(seg, sink, label):
        nonlocal first_byte, content_type
        seg_payload = dict(payload)
        seg_payload["input"] = seg
        try:
            ct, elapsed, first = post_tts(base_url, seg_payload, args.speaker,
                                          args.timeout, sink)
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace").strip()
            sys.exit(f"error: {label}HTTP {err.code} {detail}")
        except urllib.error.URLError as err:
            sys.exit(f"error: {label}连不上 {base_url}: {err.reason}")
        content_type = ct
        if first_byte is None:
            first_byte = first
        return elapsed

    if len(chunks) == 1:
        with open(output, "wb") as out:
            call(chunks[0], out.write, "")
        total_bytes = Path(output).stat().st_size
        if args.stream and args.format == "wav":
            patch_wav_sizes(output)
    else:
        blobs = []
        for idx, seg in enumerate(chunks, 1):
            buf = []
            call(seg, buf.append, f"第 {idx}/{len(chunks)} 段: ")
            blobs.append(b"".join(buf))
            print(f"分段 {idx}/{len(chunks)}: {len(seg)} 字 -> {len(blobs[-1])} 字节",
                  file=sys.stderr)
        if args.format == "wav":
            parts = [wav_fmt_data(b) for b in blobs]
            if any(p is None for p in parts):
                sys.exit("error: 服务端返回的不是 WAV")
            fmt = parts[0][0]
            if any(p[0] != fmt for p in parts):
                sys.exit("error: 各段音频格式不一致")
            data = b"".join(p[1] for p in parts)
            write_wav(output, fmt, data)
        else:
            Path(output).write_bytes(b"".join(blobs))
        total_bytes = Path(output).stat().st_size

    elapsed = time.perf_counter() - start
    info = wav_info(output) if args.format == "wav" else pcm_info(output, content_type)

    print(f"已保存: {output}")
    if info:
        duration, rate = info
        speed = f", {duration / elapsed:.1f}x 实时" if elapsed > 0 else ""
        print(f"音频: {duration:.2f}s, {rate} Hz, {total_bytes} 字节")
        if len(chunks) > 1:
            print(f"耗时: {elapsed:.2f}s, {len(chunks)} 段{speed}")
        elif args.stream:
            print(f"耗时: {elapsed:.2f}s, 首包 {first_byte:.2f}s{speed}")
        else:
            print(f"耗时: {elapsed:.2f}s{speed}")
        if args.n_predict == 0 and duration > 40.9 and len(chunks) == 1:
            sys.stdout.flush()
            print("提示: 已到服务端默认 512 帧上限（约 41s），更长请加 -n 或用 --split",
                  file=sys.stderr)
    else:
        print(f"音频: {total_bytes} 字节, {content_type}")
        print(f"耗时: {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
