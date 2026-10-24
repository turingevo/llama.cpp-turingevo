#!/usr/bin/env python3
"""
Convert SenseNova-U1 safetensors into F16 GGUF files for llama.cpp.

Shares the naming table with retag.py, so both paths produce the same layout:
LM GGUF with both MoT branches, plus the FM sidecar for generation. The output
can be quantized with llama-quantize like any other F16 GGUF.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gguf-py"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gguf    # noqa: E402
import retag   # noqa: E402
import torch   # noqa: E402
from safetensors import safe_open  # noqa: E402

# torch shapes of the FM tensors that fix the sidecar dims
FM_SHAPES = {
    "vision.patch_embd.weight": (retag.VISION_N_EMBD, 3, 16, 16),
    "vision.dense_embd.weight": (retag.N_EMBD, retag.VISION_N_EMBD, 2, 2),
    "head.conv1.weight":        (retag.N_EMBD//4, retag.N_EMBD//4, 3, 3),
    "head.conv2.weight":        (3*64, retag.N_EMBD//16, 3, 3),
    "head.fc0.weight":          (retag.N_EMBD, retag.N_EMBD),
    "head.fc2.weight":          (3*32*32, retag.N_EMBD),
}


class Shards:
    def __init__(self, src: Path):
        if src.is_dir():
            index = src / "model.safetensors.index.json"
            if index.exists():
                self.weight_map = json.loads(index.read_text())["weight_map"]
            else:
                files = sorted(src.glob("*.safetensors"))
                if len(files) != 1:
                    raise SystemExit(f"cannot choose a shard in {src}, use the index file")
                self.weight_map = {k: files[0].name for k in safe_open(str(files[0]), framework="pt").keys()}
            self.root = src
        else:
            self.weight_map = {k: src.name for k in safe_open(str(src), framework="pt").keys()}
            self.root = src.parent
        self.handles = {}

    def names(self):
        return set(self.weight_map)

    def get(self, name: str):
        shard = self.weight_map[name]
        handle = self.handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.root / shard), framework="pt", device="cpu")
            self.handles[shard] = handle
        return handle.get_tensor(name)


def to_numpy(t, f32=False):
    # norm weights are applied with ggml_mul, which requires F32
    return np.ascontiguousarray((t.to(torch.float32) if f32 else t.to(torch.float16)).numpy())


def lm_groups(src: Shards):
    names = src.names()
    partners = set()
    for name in names:
        _, partner = retag.map_lm_tensor(name)
        if partner is not None:
            partners.add(partner)

    groups = []
    used = set()
    for name in sorted(names):
        if name in used or name in partners:
            continue
        dst, partner = retag.map_lm_tensor(name)
        if dst is None:
            if name.startswith(retag.LM_IGNORED_PREFIX):
                continue
            raise SystemExit(f"unmapped tensor: {name}")
        used.add(name)
        if partner is None:
            groups.append((dst, [name]))
        else:
            if partner not in names:
                raise SystemExit(f"missing tensor: {partner}")
            used.add(partner)
            groups.append((dst, [name, partner]))
    return groups


def fm_groups(src: Shards):
    names = src.names()
    has_conv = "fm_modules.fm_head.conv1.weight" in names
    has_linear = "fm_modules.fm_head.0.weight" in names
    if has_conv == has_linear:
        raise SystemExit("cannot determine pixel head type")

    groups = [(retag.FM_MAP[n], n) for n in sorted(names) if n in retag.FM_MAP]
    if len(groups) != 16:
        raise SystemExit(f"unexpected FM tensor count: {len(groups)}")
    return groups, "conv" if has_conv else "linear"


def add_lm_tensor(writer, src: Shards, dst: str, names):
    if len(names) == 1:
        writer.add_tensor(dst, to_numpy(src.get(names[0]), "norm" in dst))
    else:
        parts = [to_numpy(src.get(n), True).reshape(-1) for n in names]
        writer.add_tensor(dst, np.concatenate(parts))


def add_fm_tensor(writer, src: Shards, dst: str, name: str):
    t = src.get(name)
    shape = FM_SHAPES.get(dst)
    if shape is not None and tuple(t.shape) != shape:
        raise SystemExit(f"unexpected shape for {name}: {tuple(t.shape)}")
    # CUDA add of a F16 bias fails the broadcast size assert, keep biases F32
    writer.add_tensor(dst, to_numpy(t, name.endswith(".bias")))


def main():
    ap = argparse.ArgumentParser(description="Convert SenseNova-U1 safetensors to F16 GGUF")
    ap.add_argument("--src", required=True, type=Path, help="model dir or single safetensors file")
    ap.add_argument("--config", type=Path, help="dir with config.json and tokenizer files (default: --src)")
    ap.add_argument("--out", type=Path, help="output LM GGUF")
    ap.add_argument("--out-fm", type=Path, help="output FM sidecar GGUF")
    ap.add_argument("--name", help="general.name (default: source dir name)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out is None and args.out_fm is None:
        raise SystemExit("nothing to write, pass --out and/or --out-fm")
    if args.config is None:
        args.config = args.src if args.src.is_dir() else args.src.parent
    if args.name is None:
        args.name = args.src.stem if args.src.is_file() else args.src.name

    for p in (args.out, args.out_fm):
        if p is not None and p.exists() and not args.overwrite:
            raise SystemExit(f"output exists: {p}")

    print(f"reading {args.src}", flush=True)
    src = Shards(args.src)

    if args.out is not None:
        lm = lm_groups(src)
        if len(lm) != 4 + retag.N_LAYER*22:
            raise SystemExit(f"unexpected LM tensor count: {len(lm)}")

        writer = gguf.GGUFWriter(args.out, "sensenova_u1", use_temp_file=True)
        retag.add_lm_kv(writer, args.name, args.config, gguf.LlamaFileType.MOSTLY_F16)
        # set after the tokenizer import above, so its caches do not land in the output dir
        tempfile.tempdir = str(args.out.parent)

        print(f"writing {args.out}", flush=True)
        for i, (dst, names) in enumerate(lm):
            add_lm_tensor(writer, src, dst, names)
            if (i + 1) % 100 == 0:
                print(f"  lm: {i + 1} tensors", flush=True)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

    if args.out_fm is not None:
        groups, head_type = fm_groups(src)
        writer_fm = gguf.GGUFWriter(args.out_fm, "sensenova_u1_fm", use_temp_file=True)
        retag.add_fm_kv(writer_fm, args.name, retag.load_fm_config(args.config), head_type)
        tempfile.tempdir = str(args.out_fm.parent)

        print(f"writing {args.out_fm}", flush=True)
        for dst, name in groups:
            add_fm_tensor(writer_fm, src, dst, name)
        writer_fm.write_header_to_file()
        writer_fm.write_kv_data_to_file()
        writer_fm.write_tensors_to_file(progress=True)
        writer_fm.close()

    print("done", flush=True)


if __name__ == "__main__":
    main()
