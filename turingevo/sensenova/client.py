#!/usr/bin/env python3
"""llama-sensenova-server 测试客户端，仅用标准库。

支持六种协议:
  openai       POST /v1/images/generations   （默认，纯文生图）
  edits        POST /v1/images/edits         （带输入图的编辑，multipart）
  native       POST /generate                （私有协议，支持 think 和输入图）
  infographic  POST /v1/infographic/generate  （omni-note 私有协议，支持 think 和输入图）
  vqa          POST /v1/chat/completions     （视觉理解问答，-p 是问题）
  interleave   POST /v1/interleave/generate  （图文交错，边写文字边出图）
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPT_FILE = os.path.join(HERE, "prompts", "菜谱制作.txt")


def parse_args():
    ap = argparse.ArgumentParser(description="调用 llama-sensenova-server 生成图片")
    ap.add_argument("--api", choices=["openai", "edits", "native", "infographic", "vqa", "interleave"],
                    default="openai", help="协议，默认 openai")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--model", default="sensenova-u1.5", help="OpenAI 协议里的 model 字段")
    ap.add_argument("--n", type=int, default=1, help="OpenAI 协议生成数量，仅保存第一张")
    ap.add_argument("-p", "--prompt", help="提示词文本，优先于 --prompt-file")
    ap.add_argument("-f", "--prompt-file", default=DEFAULT_PROMPT_FILE,
                    help="提示词文件（默认 prompts/菜谱制作.txt）")
    ap.add_argument("-o", "--output", default="sensenova.png", help="输出 PNG 路径")
    ap.add_argument("-i", "--image", action="append", default=[],
                    help="输入图，可重复；edits/native/infographic/interleave 协议生效")
    ap.add_argument("--width", type=int, default=0, help="宽度，32 的倍数；0 表示按协议默认")
    ap.add_argument("--height", type=int, default=0, help="高度，32 的倍数；0 表示按协议默认")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42, help="-1 表示随机")
    ap.add_argument("--cfg", type=float, default=1.0, help="文本引导强度")
    ap.add_argument("--img-cfg", type=float, default=1.0, help="图像引导强度，编辑时有效")
    ap.add_argument("--cfg-norm", choices=["none", "global", "channel"], default="none",
                    help="CFG 结果的范数约束方式")
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--think", action="store_true",
                    help="先跑 think 推理再出图；interleave 默认就开")
    ap.add_argument("--no-think", action="store_true",
                    help="关掉 interleave 的 think（只用于 interleave 协议）")
    ap.add_argument("--max-think-tokens", type=int, default=1024, help="think 预算（interleave 不用）")
    ap.add_argument("--input-max-pixels", type=int, default=0, help="每张输入图的像素上限，0 表示自动")
    ap.add_argument("--auto-pixels", type=int, default=1024*1024, help="自动尺寸的输出像素")
    ap.add_argument("--think-output", default="", help="把 think 文本写到文件")
    ap.add_argument("--format", default=None,
                    help="openai/edits/interleave: b64_json/url; native: png/b64_json; 默认按协议选择")
    ap.add_argument("--server-output-dir", default="",
                    help="infographic/interleave 协议的服务端输出目录（需 --allow-output-dir）")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="vqa 回答上限 / interleave 文字预算")
    ap.add_argument("--max-images", type=int, default=3, help="interleave 出图上限")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="vqa/interleave 采样温度，0 表示贪心")
    ap.add_argument("--top-p", type=float, default=0.95, help="vqa/interleave nucleus 采样")
    ap.add_argument("--top-k", type=int, default=0, help="vqa/interleave top-k 采样，0 表示关闭")
    ap.add_argument("--repeat-penalty", type=float, default=1.0, help="vqa/interleave 重复惩罚")
    ap.add_argument("--timeout", type=float, default=600.0, help="请求超时秒数")
    return ap.parse_args()


def load_prompt(args):
    if args.prompt is not None:
        return args.prompt
    try:
        with open(args.prompt_file, encoding="utf-8") as f:
            # 去掉结尾换行，服务端按原样使用字符串
            return f.read().rstrip()
    except OSError as e:
        sys.exit(f"无法读取提示词文件 {args.prompt_file}: {e}")


def resolve_format(args):
    if args.api in ("openai", "edits", "interleave"):
        fmt = args.format or "b64_json"
        if fmt not in ("b64_json", "url"):
            sys.exit(f"{args.api} 协议 --format 仅支持 b64_json / url")
    elif args.api == "native":
        fmt = args.format or "png"
        if fmt not in ("png", "b64_json"):
            sys.exit("native 协议 --format 仅支持 png / b64_json")
    else:
        fmt = "png"
    return fmt


def resolve_size(args):
    """返回 (width, height)；0 表示由服务端推导。"""
    if args.width and args.height:
        return args.width, args.height
    if args.width or args.height:
        sys.exit("--width 和 --height 必须一起给")
    if args.api in ("edits", "interleave"):
        return 0, 0
    return 1024, 1024


def interleave_think(args):
    if args.think and args.no_think:
        sys.exit("--think 和 --no-think 不能同时给")
    if args.api != "interleave" and args.no_think:
        sys.exit("--no-think 只用于 interleave 协议")
    return not args.no_think


def check_health(base, timeout):
    try:
        with urllib.request.urlopen(base + "/health", timeout=min(timeout, 5.0)) as r:
            body = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as e:
        sys.exit(f"无法连接 {base}: {e}\n"
                 f"先启动服务: ./build-alldev/bin/llama-sensenova-server "
                 f"-m MODEL.gguf --fm FM.gguf --port {base.rsplit(':', 1)[1]}")
    if body.get("status") != "ok":
        sys.exit(f"服务状态异常: {body}")


def read_images(paths):
    """读取输入图，返回 (文件名, 内容类型, 数据) 列表。"""
    out = []
    for path in paths:
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            sys.exit(f"无法读取输入图 {path}: {e}")
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        out.append((os.path.basename(path), ctype, data))
    return out


def build_multipart(fields, files):
    """files: (字段名, 文件名, 内容类型, 数据) 列表。"""
    boundary = "----sensenova" + os.urandom(8).hex()
    body = b""
    for name, value in fields:
        body += (f"--{boundary}\r\n"
                 f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").encode("utf-8")
        body += f"{value}\r\n".encode("utf-8")
    for name, filename, ctype, data in files:
        body += (f"--{boundary}\r\n"
                 f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                 f"Content-Type: {ctype}\r\n\r\n").encode("utf-8")
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def build_request(args, prompt, fmt):
    """返回 (路径, JSON 载荷 或 multipart 正文, 内容类型)。"""
    width, height = resolve_size(args)

    if args.api == "openai":
        if args.image:
            sys.exit("openai 协议不支持输入图，改用 --api edits")
        payload = {
            "prompt": prompt,
            "model": args.model,
            "n": args.n,
            "size": f"{width}x{height}",
            "response_format": fmt,
            "output_format": "png",
            "cfg_scale": args.cfg,
            "num_steps": args.steps,
            "timestep_shift": args.shift,
            "seed": args.seed,
            "think": args.think,
            "max_think_tokens": args.max_think_tokens,
            "cfg_norm": args.cfg_norm,
        }
        return "/v1/images/generations", payload, "application/json"

    if args.api == "edits":
        if not args.image:
            sys.exit("edits 协议需要至少一张输入图")
        size = "auto" if width == 0 else f"{width}x{height}"
        fields = [
            ("prompt", prompt),
            ("model", args.model),
            ("n", str(args.n)),
            ("size", size),
            ("response_format", fmt),
            ("output_format", "png"),
            ("cfg_scale", str(args.cfg)),
            ("img_cfg_scale", str(args.img_cfg)),
            ("cfg_norm", args.cfg_norm),
            ("num_steps", str(args.steps)),
            ("timestep_shift", str(args.shift)),
            ("seed", str(args.seed)),
            ("think", "true" if args.think else "false"),
            ("max_think_tokens", str(args.max_think_tokens)),
            ("input_max_pixels", str(args.input_max_pixels)),
            ("auto_pixels", str(args.auto_pixels)),
        ]
        files = [("image", name, ctype, data)
                 for name, ctype, data in read_images(args.image)]
        body, ctype = build_multipart(fields, files)
        return "/v1/images/edits", body, ctype

    if args.api == "native":
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": args.steps,
            "seed": args.seed,
            "cfg": args.cfg,
            "img_cfg": args.img_cfg,
            "cfg_norm": args.cfg_norm,
            "shift": args.shift,
            "think": args.think,
            "max_think_tokens": args.max_think_tokens,
            "input_max_pixels": args.input_max_pixels,
            "auto_size_pixels": args.auto_pixels,
            "response_format": fmt,
        }
        if args.image:
            payload["images"] = list(args.image)
        return "/generate", payload, "application/json"

    if args.api == "interleave":
        payload = {
            "prompt": prompt,
            "model": args.model,
            "width": width,
            "height": height,
            "auto_pixels": args.auto_pixels,
            "num_steps": args.steps,
            "timestep_shift": args.shift,
            "seed": args.seed,
            "max_images": args.max_images,
            "max_new_tokens": args.max_new_tokens,
            "think": interleave_think(args),
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repeat_penalty,
            "input_max_pixels": args.input_max_pixels,
            "response_format": fmt,
        }
        # send data URLs so the request also works when client and server are on different hosts
        images = [
            f"data:{ctype};base64,{base64.b64encode(data).decode('ascii')}"
            for _, ctype, data in read_images(args.image)
        ]
        if images:
            payload["images"] = images
        if args.server_output_dir:
            payload["output_dir"] = args.server_output_dir
        return "/v1/interleave/generate", payload, "application/json"

    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "cfg_scale": args.cfg,
        "img_cfg_scale": args.img_cfg,
        "cfg_norm": args.cfg_norm,
        "num_steps": args.steps,
        "timestep_shift": args.shift,
        "seed": args.seed,
        "think": args.think,
        "max_think_tokens": args.max_think_tokens,
        "input_max_pixels": args.input_max_pixels,
        "auto_pixels": args.auto_pixels,
    }
    if args.image:
        payload["images"] = list(args.image)
    if args.server_output_dir:
        payload["output_dir"] = args.server_output_dir
    return "/v1/infographic/generate", payload, "application/json"


def fetch(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def save_think(args, text):
    if not text:
        return
    if args.think_output:
        with open(args.think_output, "w", encoding="utf-8") as f:
            f.write(text)
    print("--- think ---")
    print(text)
    print("--- think end ---")


def save_image(args, base, fmt, body, server_ms, headers):
    think_text = ""
    if args.api in ("openai", "edits"):
        try:
            j = json.loads(body.decode("utf-8"))
            data = j["data"]
        except (ValueError, KeyError) as e:
            sys.exit(f"无法解析 JSON 响应: {e}")
        if len(data) != args.n:
            sys.exit(f"返回 {len(data)} 张图片，期望 {args.n} 张")
        item = data[0]
        if fmt == "b64_json":
            png = base64.b64decode(item["b64_json"])
        else:
            png = fetch(item["url"], args.timeout)
        think_text = item.get("revised_prompt", "")
    elif args.api == "native":
        if fmt == "b64_json":
            try:
                j = json.loads(body.decode("utf-8"))
                png = base64.b64decode(j["image"])
            except (ValueError, KeyError) as e:
                sys.exit(f"无法解析 JSON 响应: {e}")
            server_ms = j.get("elapsed_ms", server_ms)
            think_text = j.get("think", "")
        else:
            png = body
            raw = headers.get("x-think-text")
            if raw:
                think_text = base64.b64decode(raw).decode("utf-8", "replace")
    else:
        try:
            j = json.loads(body.decode("utf-8"))
        except ValueError as e:
            sys.exit(f"无法解析 JSON 响应: {e}")
        if j.get("status") != "success":
            sys.exit(f"生成失败: {j.get('message', '未知错误')}")
        path = j.get("image_path") or ""
        url = j.get("image_url") or ""
        if path and os.access(path, os.R_OK):
            with open(path, "rb") as f:
                png = f.read()
        elif url:
            if not url.startswith("http"):
                url = base + url
            png = fetch(url, args.timeout)
        else:
            sys.exit("响应里没有可读取的 image_path 或 image_url")
        server_ms = server_ms or f"{j.get('generation_time', 0) * 1000:.0f}"
        think_text = j.get("think", "")

    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        sys.exit("响应不是 PNG 数据")

    with open(args.output, "wb") as f:
        f.write(png)
    save_think(args, think_text)
    return png, server_ms


def numbered_path(path, idx):
    if idx == 0:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_{idx}{ext}" if ext else f"{path}_{idx}"


def read_segment_image(args, base, fmt, seg):
    if fmt == "url":
        path = seg.get("image_path") or ""
        url = seg.get("url") or ""
        if path and os.access(path, os.R_OK):
            with open(path, "rb") as f:
                return f.read()
        if url:
            if not url.startswith("http"):
                url = base + url
            return fetch(url, args.timeout)
        sys.exit("图片段里没有可读取的 image_path 或 url")
    try:
        return base64.b64decode(seg["b64_json"])
    except (KeyError, ValueError) as e:
        sys.exit(f"无法解析图片段: {e}")


def run_interleave(args, base):
    if args.max_images < 1:
        sys.exit("--max-images 必须大于 0")
    fmt = resolve_format(args)
    prompt = load_prompt(args)
    if not prompt:
        sys.exit("提示词为空")
    think = interleave_think(args)

    check_health(base, args.timeout)
    path, payload, ctype = build_request(args, prompt, fmt)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": ctype},
        method="POST",
    )

    width, height = resolve_size(args)
    size_desc = f"{width}x{height}" if width else "auto"
    print(f"请求 {base}{path} [interleave]: {size_desc}, {args.steps} 步, "
          f"seed {args.seed}, 最多 {args.max_images} 张图, think {'on' if think else 'off'}"
          + (f", 输入图 {len(args.image)} 张" if args.image else ""))

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            body = r.read()
            server_ms = r.headers.get("X-Elapsed-Ms")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        sys.exit(f"服务返回 HTTP {e.code}: {detail}")
    except (urllib.error.URLError, TimeoutError) as e:
        sys.exit(f"请求失败: {e}")
    wall_ms = (time.time() - t0)*1000.0

    try:
        j = json.loads(body.decode("utf-8"))
        segments = j.get("segments") or []
    except (ValueError, AttributeError) as e:
        sys.exit(f"无法解析 JSON 响应: {e}")

    think_text = j.get("think") or ""
    if args.think_output and think_text:
        with open(args.think_output, "w", encoding="utf-8") as f:
            f.write(think_text)
    if think_text:
        print("--- think ---")
        print(think_text)
        print("--- think end ---")

    answer = j.get("answer") or ""
    if not answer and not think_text:
        answer = j.get("text") or ""
    if answer:
        print(answer)

    saved = []
    for seg in segments:
        if not isinstance(seg, dict) or seg.get("type") != "image":
            continue
        png = read_segment_image(args, base, fmt, seg)
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            sys.exit("图片段不是 PNG 数据")
        out_path = numbered_path(args.output, len(saved))
        with open(out_path, "wb") as f:
            f.write(png)
        saved.append(out_path)

    if not saved:
        print("warning: 没有生成图片，加大 --max-new-tokens 或关掉 think", file=sys.stderr)

    usage = j.get("usage") or {}
    print(f"finish: {j.get('finish_reason', '?')}, "
          f"prompt {usage.get('prompt_tokens', '?')} tokens, "
          f"text {usage.get('text_tokens', '?')} tokens, images {len(saved)}, "
          f"服务端耗时: {server_ms or '未知'} ms, 客户端往返: {wall_ms:.0f} ms")
    if saved:
        print("图片: " + ", ".join(os.path.abspath(p) for p in saved))


def build_vqa_payload(args, question):
    content = [{"type": "text", "text": question}]
    for _, ctype, data in read_images(args.image):
        b64 = base64.b64encode(data).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{ctype};base64,{b64}"},
        })
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repeat_penalty,
        "seed": args.seed,
        "input_max_pixels": args.input_max_pixels,
    }


def run_vqa(args, base):
    if args.prompt is not None:
        question = args.prompt
    elif args.prompt_file != DEFAULT_PROMPT_FILE:
        try:
            with open(args.prompt_file, encoding="utf-8") as f:
                question = f.read().rstrip()
        except OSError as e:
            sys.exit(f"无法读取问题文件 {args.prompt_file}: {e}")
    else:
        sys.exit("vqa 模式需要 -p 问题（或用 -f 指定问题文件）")
    if not question:
        sys.exit("问题为空")

    check_health(base, args.timeout)

    payload = build_vqa_payload(args, question)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    sampling = "贪心" if args.temperature == 0 else f"temperature {args.temperature}, top_p {args.top_p}, top_k {args.top_k}"
    print(f"请求 {base}/v1/chat/completions [vqa]: 输入图 {len(args.image)} 张, "
          f"max_new_tokens {args.max_new_tokens}, {sampling}")

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            body = r.read()
            server_ms = r.headers.get("X-Elapsed-Ms")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        sys.exit(f"服务返回 HTTP {e.code}: {detail}")
    except (urllib.error.URLError, TimeoutError) as e:
        sys.exit(f"请求失败: {e}")
    wall_ms = (time.time() - t0)*1000.0

    try:
        j = json.loads(body.decode("utf-8"))
        choice = j["choices"][0]
        message = choice["message"]
    except (ValueError, KeyError, IndexError) as e:
        sys.exit(f"无法解析响应: {e}")

    reasoning = message.get("reasoning_content") or ""
    answer = message.get("content") or ""
    if args.think_output and reasoning:
        with open(args.think_output, "w", encoding="utf-8") as f:
            f.write(reasoning)
    if reasoning:
        print("--- think ---")
        print(reasoning)
        print("--- think end ---")
    print(answer)

    usage = j.get("usage") or {}
    print(f"finish: {choice.get('finish_reason')}, "
          f"prompt {usage.get('prompt_tokens', '?')} tokens, "
          f"completion {usage.get('completion_tokens', '?')} tokens, "
          f"服务端耗时: {server_ms or '未知'} ms, 客户端往返: {wall_ms:.0f} ms")


def main():
    args = parse_args()
    if args.no_think and args.api != "interleave":
        sys.exit("--no-think 只用于 interleave 协议")
    base = f"http://{args.host}:{args.port}"
    if args.api == "vqa":
        run_vqa(args, base)
        return
    if args.api == "interleave":
        run_interleave(args, base)
        return
    fmt = resolve_format(args)
    prompt = load_prompt(args)
    if not prompt:
        sys.exit("提示词为空")

    check_health(base, args.timeout)

    path, payload, ctype = build_request(args, prompt, fmt)
    if isinstance(payload, bytes):
        data = payload
    else:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": ctype},
        method="POST",
    )

    width, height = resolve_size(args)
    size_desc = f"{width}x{height}" if width else "auto"
    print(f"请求 {base}{path} [{args.api}]: {size_desc}, {args.steps} 步, "
          f"seed {args.seed}, prompt {len(prompt)} 字"
          + (f", 输入图 {len(args.image)} 张" if args.image else "")
          + (", think" if args.think else ""))

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            body = r.read()
            content_type = r.headers.get("Content-Type", "")
            headers = {k.lower(): v for k, v in r.headers.items()}
            server_ms = headers.get("x-elapsed-ms")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        sys.exit(f"服务返回 HTTP {e.code}: {detail}")
    except (urllib.error.URLError, TimeoutError) as e:
        sys.exit(f"请求失败: {e}")
    wall_ms = (time.time() - t0)*1000.0

    if content_type.startswith("application/json"):
        png, server_ms = save_image(args, base, fmt, body, server_ms, headers)
    else:
        png = body
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            sys.exit("响应不是 PNG 数据")
        with open(args.output, "wb") as f:
            f.write(png)
        raw = headers.get("x-think-text")
        save_think(args, base64.b64decode(raw).decode("utf-8", "replace") if raw else "")

    print(f"服务端耗时: {server_ms if server_ms is not None else '未知'} ms, "
          f"客户端往返: {wall_ms:.0f} ms")
    print(f"图片: {len(png)} 字节, 已保存到 {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
