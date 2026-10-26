#!/usr/bin/env python3
"""Compact pipeline JSON prompt into a t2i prompt.

The input JSON contains pipeline metadata (style_sources, _assemble,
style_reference_directive) and the same content restated in several
sections. Only the fields an image model can use are kept.

usage: compress_prompt.py in.json [-o out.txt] [--stats]

 proj_root=/media/wmx/ws1/workspace/ai/llama.cpp
 dir=${proj_root}/turingevo/sensenova/prompts
 python3 ${proj_root}/turingevo/sensenova/compress_prompt.py ${dir}/瑞士旅行攻略.json   --level metadata -o ${dir}/瑞士旅行攻略_metadata.txt --stats
 python3 ${proj_root}/turingevo/sensenova/compress_prompt.py ${dir}/瑞士旅行攻略.json   --level compact -o ${dir}/瑞士旅行攻略_compact.txt --stats

"""

import argparse
import json
import re

HEX = r"#[0-9a-fA-F]{6}"

# constraints.negative entries that describe the reference template are
# meaningless here: the CLI gets no reference image.
NO_REF_KEYS = ("reference template", "template composition grammar")

# long English policy lines -> short Chinese, matched by substring
NEG_MAP = [
    ("no watermark", "无水印"),
    ("no gibberish", "无乱码"),
    ("correct spelling", "拼写正确"),
    ("no invented numbers", "不生成未给出的数字/统计/名称"),
    ("no bracketed internal tags", "不把内部标签当正文"),
    ("no layout-role words", "不把版式角色词当正文"),
    ("no separate bottom reading-order", "不重复画阅读顺序条"),
    ("no style-descriptor words", "不把风格名当标题"),
    ("no placeholder graphics", "无占位灰图/空图框"),
]


def find(pattern, text, default=""):
    m = re.search(pattern, text)
    return m.group(1).strip() if m else default


def luma(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def split_hexes(text):
    return re.findall(HEX, text)


def palette(style_text):
    ink = find(rf"ink=({HEX})", style_text)
    key = split_hexes(find(rf"key hex:\s*([^|]+)", style_text))
    accent = split_hexes(find(rf"accent hex[^:]*:\s*([^|]+)", style_text))
    bg = find(rf"【近纯白\s*({HEX})】", style_text) \
        or find(rf"PIL 客观底色=({HEX})", style_text)

    # the key list is polluted by near-white/cream entries: the 底色纠偏
    # note says those are a misread of the paper, not a real palette
    key = [h for h in key if h.lower() not in (bg.lower(), ink.lower()) and luma(h) < 0.85]

    parts = []
    if bg:
        parts.append(f"底色近纯白 {bg}，不要米黄/牛皮纸底")
    if ink:
        parts.append(f"墨线 {ink}")
    if key:
        parts.append("主色 " + " ".join(key))
    if accent:
        parts.append("点睛 " + " ".join(accent))
    return "；".join(parts)


def trim_sentence(text, keep=1):
    # drop trailing clauses that repeat the layout already stated elsewhere
    parts = re.split(r"[；。]", text)
    return "。".join(p for p in parts[:keep] if p.strip()) + "。"


def compact(d):
    style = d.get("style", {})
    subject = d.get("subject", {})
    details = d.get("details", {})
    canvas = d.get("canvas", {})
    comp = details.get("composition", {})
    sections = details.get("content_sections", [])
    out = []

    title = subject.get("title", {}).get("text", "")
    subtitle = sections[0]["items"][0] if sections and sections[0].get("items") else ""

    head = d.get("task", "")
    if title:
        head = f"{head}。主标题「{title}」"
        if subtitle:
            head += f"，副标题「{subtitle}」"
    out.append(head + "。")

    size = canvas.get("aspect") or canvas.get("orientation", "")
    if size:
        out.append(f"画布 {size}，{canvas.get('safe_margins', '')}。")

    pal = palette(style.get("color_value", ""))
    if pal:
        out.append("配色：" + pal + "。")

    look = [style.get("style_family", ""), style.get("illustration_craft", ""),
            style.get("color_grading", ""), style.get("tone", ""),
            style.get("spatial_rhythm", ""), style.get("image_treatment", "")]
    look = [s.strip().rstrip("。；") for s in look if s.strip()]
    if look:
        out.append("风格：" + "；".join(look) + "。")

    motif = comp.get("signature_motif") or style.get("signature_motif", "")
    if motif:
        out.append("构图：" + motif)
    for field in ("containment", "connectors", "regularity"):
        val = comp.get(field, "")
        if val:
            out.append(val)

    body = []
    for sec in sections[1:]:
        body.append("｜".join([sec.get("heading", "")] + sec.get("items", [])))
    if body:
        out.append("图中文字（逐字照抄，不要加编号）：")
        out.extend(body)

    deco = [s for s in subject.get("subjects_must_include", []) if "涂鸦" in s]
    if deco:
        out.append("点缀涂鸦：" + "；".join(deco) + "。")

    negs = []
    for n in d.get("constraints", {}).get("negative", []):
        if any(k in n for k in NO_REF_KEYS):
            continue
        for needle, short in NEG_MAP:
            if needle in n:
                negs.append(short)
                break
    if negs:
        out.append("禁止：图内文字一律简体中文，" + "；".join(negs) + "。")

    return "\n".join(out) + "\n"


def metadata_text(d):
    """Drop pipeline-only metadata, keep every content-bearing field."""
    x = json.loads(json.dumps(d))
    for key in ("style_sources", "_assemble", "style_reference_directive"):
        x.pop(key, None)
    details = x.get("details", {})
    details.pop("data_verbatim", None)  # repeats content_sections
    details.pop("layout", None)         # zones repeat content_sections
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out")
    ap.add_argument("--level", choices=("compact", "metadata"), default="compact")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    raw = open(args.src, encoding="utf-8").read()
    data = json.loads(raw)
    text = metadata_text(data) if args.level == "metadata" else compact(data)
    if not text.endswith("\n"):
        text += "\n"
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
    else:
        print(text, end="")
    if args.stats:
        print(f"chars: {len(raw)} -> {len(text)} ({100*len(text)/len(raw):.1f}%)",
              file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
