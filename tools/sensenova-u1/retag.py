#!/usr/bin/env python3
"""
Retag SenseNova-U1 vendor GGUF files into the llama.cpp naming layout.

The vendor files store torch-style tensor names (up to 65 chars, so they cannot
be loaded as-is: GGML_MAX_NAME is 64), name the architecture "qwen3" and carry no
tokenizer metadata. This tool rewrites both without touching the quantized
bytes: tensors are copied verbatim, so quantization provenance is preserved.

Output:
  - LM GGUF: understanding branch plus generation branch ("_mot_gen" weights),
    readable by llama.cpp as arch "sensenova_u1".
  - FM sidecar GGUF: vision-gen patch embedder, timestep / noise-scale embedders
    and the pixel head, used by the generation graph.
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gguf-py"))

import gguf  # noqa: E402

N_LAYER        = 42
N_EMBD         = 4096
N_FF           = 12288
N_HEAD         = 32
N_HEAD_KV      = 8
N_CTX_TRAIN    = 16384
RMS_EPS        = 1e-6
ROPE_FREQ_BASE = 5e6
ROPE_FREQ_BASE_SPATIAL = 1e4

LM_ROOT = {
    "language_model.model.embed_tokens.weight": "token_embd.weight",
    "language_model.lm_head.weight":            "output.weight",
    "language_model.model.norm.weight":         "output_norm.weight",
    "language_model.model.norm_mot_gen.weight": "output_norm.gen.weight",
}

# layer-local torch suffix -> llama.cpp suffix
LM_LAYER = {
    "input_layernorm":                  "attn_norm.weight",
    "input_layernorm_mot_gen":          "attn_norm.gen.weight",
    "post_attention_layernorm":         "ffn_norm.weight",
    "post_attention_layernorm_mot_gen": "ffn_norm.gen.weight",

    "self_attn.q_proj":          "attn_q.weight",
    "self_attn.q_proj_mot_gen":  "attn_q.gen.weight",
    "self_attn.k_proj":          "attn_k.weight",
    "self_attn.k_proj_mot_gen":  "attn_k.gen.weight",
    "self_attn.v_proj":          "attn_v.weight",
    "self_attn.v_proj_mot_gen":  "attn_v.gen.weight",
    "self_attn.o_proj":          "attn_output.weight",
    "self_attn.o_proj_mot_gen":  "attn_output.gen.weight",

    "mlp.gate_proj":         "ffn_gate.weight",
    "mlp.down_proj":         "ffn_down.weight",
    "mlp.up_proj":           "ffn_up.weight",
    "mlp_mot_gen.gate_proj": "ffn_gate.gen.weight",
    "mlp_mot_gen.down_proj": "ffn_down.gen.weight",
    "mlp_mot_gen.up_proj":   "ffn_up.gen.weight",
}

# torch suffix -> (second torch suffix, destination suffix); stored as one tensor
LM_LAYER_CONCAT = {
    "self_attn.q_norm":         ("self_attn.q_norm_hw.weight",          "attn_q_norm.weight"),
    "self_attn.q_norm_mot_gen": ("self_attn.q_norm_hw_mot_gen.weight",  "attn_q_norm.gen.weight"),
    "self_attn.k_norm":         ("self_attn.k_norm_hw.weight",          "attn_k_norm.weight"),
    "self_attn.k_norm_mot_gen": ("self_attn.k_norm_hw_mot_gen.weight",  "attn_k_norm.gen.weight"),
}

FM_MAP = {
    "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight":  "vision.patch_embd.weight",
    "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.bias":    "vision.patch_embd.bias",
    "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight":  "vision.dense_embd.weight",
    "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.bias":    "vision.dense_embd.bias",

    "fm_modules.timestep_embedder.mlp.0.weight":    "timestep.mlp.0.weight",
    "fm_modules.timestep_embedder.mlp.0.bias":      "timestep.mlp.0.bias",
    "fm_modules.timestep_embedder.mlp.2.weight":    "timestep.mlp.2.weight",
    "fm_modules.timestep_embedder.mlp.2.bias":      "timestep.mlp.2.bias",

    "fm_modules.noise_scale_embedder.mlp.0.weight": "noise_scale.mlp.0.weight",
    "fm_modules.noise_scale_embedder.mlp.0.bias":   "noise_scale.mlp.0.bias",
    "fm_modules.noise_scale_embedder.mlp.2.weight": "noise_scale.mlp.2.weight",
    "fm_modules.noise_scale_embedder.mlp.2.bias":   "noise_scale.mlp.2.bias",

    # U1.5 uses a convolutional pixel head
    "fm_modules.fm_head.conv1.weight": "head.conv1.weight",
    "fm_modules.fm_head.conv1.bias":   "head.conv1.bias",
    "fm_modules.fm_head.conv2.weight": "head.conv2.weight",
    "fm_modules.fm_head.conv2.bias":   "head.conv2.bias",

    # V3 uses a two-layer linear pixel head
    "fm_modules.fm_head.0.weight": "head.fc0.weight",
    "fm_modules.fm_head.0.bias":   "head.fc0.bias",
    "fm_modules.fm_head.2.weight": "head.fc2.weight",
    "fm_modules.fm_head.2.bias":   "head.fc2.bias",
}

# not part of text-to-image generation
LM_IGNORED_PREFIX = (
    "vision_model.",  # understanding-branch vision encoder
    "fm_modules.",    # written to the FM sidecar
)


def load_tokenizer_vocab(config_dir: Path):
    # the config dir declares custom code in config.json, which makes
    # AutoTokenizer ask for confirmation on stdin. The tokenizer itself is a
    # plain slow Qwen2 BPE, so bypass AutoTokenizer's remote-code check.
    from transformers import AutoTokenizer, Qwen2Tokenizer
    real_from_pretrained = Qwen2Tokenizer.from_pretrained
    AutoTokenizer.from_pretrained = staticmethod(lambda *a, **kw: real_from_pretrained(*a, **kw))

    from conversion.base import TextModel

    class VocabHelper(TextModel):
        model_arch = gguf.MODEL_ARCH.QWEN3

        def __init__(self, dir_model):
            self.dir_model = dir_model
            self.hparams = {"vocab_size": 151936}

    tokens, toktypes, tokpre = VocabHelper(config_dir).get_vocab_base()
    if tokpre != "qwen2":
        raise ValueError(f"unexpected pre-tokenizer: {tokpre}")
    return tokens, toktypes


def map_lm_tensor(name: str):
    if name in LM_ROOT:
        return LM_ROOT[name], None

    m = re.fullmatch(r"language_model\.model\.layers\.(\d+)\.(.+)", name)
    if m is None:
        return None, None
    bid  = int(m.group(1))
    rest = m.group(2)
    if not rest.endswith(".weight"):
        return None, None
    rest = rest[:-len(".weight")]

    if rest in LM_LAYER:
        return f"blk.{bid}.{LM_LAYER[rest]}", None
    if rest in LM_LAYER_CONCAT:
        partner, dst = LM_LAYER_CONCAT[rest]
        return f"blk.{bid}.{dst}", f"language_model.model.layers.{bid}.{partner}"

    return None, None


def add_tensor_preserving_bytes(writer, name, tensor, data=None):
    # norm weights are applied with ggml_mul, which requires F32
    if "norm" in name:
        if data is None:
            data = np.asarray(tensor.data)
        writer.add_tensor(name, data.astype(np.float32))
    else:
        assert data is None
        writer.add_tensor(name, np.asarray(tensor.data), raw_dtype=tensor.tensor_type)


def detect_file_type(src: Path):
    for tag in sorted((t.name for t in gguf.LlamaFileType), key=len, reverse=True):
        if tag.startswith("MOSTLY_") and tag[len("MOSTLY_"):] in src.name:
            return getattr(gguf.LlamaFileType, tag)
    return gguf.LlamaFileType.MOSTLY_F16


def write_lm(args, reader, writer):
    names = {t.name: t for t in reader.tensors}
    tokens, toktypes = load_tokenizer_vocab(args.config)
    # set after the torch import above, so its caches do not land in the output dir
    tempfile.tempdir = str(args.out.parent)

    writer.add_name(args.name)
    writer.add_context_length(N_CTX_TRAIN)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_HEAD_KV)
    writer.add_layer_norm_rms_eps(RMS_EPS)
    writer.add_rope_freq_base(ROPE_FREQ_BASE)
    writer.add_rope_freq_base_spatial(ROPE_FREQ_BASE_SPATIAL)
    writer.add_file_type(detect_file_type(args.src))
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(args.config, load_merges=True)
    special_vocab.add_to_gguf(writer)

    # concatenated tensors are read together with their primary name, which may
    # come later in the file
    partner_names = set()
    for t in reader.tensors:
        _, partner = map_lm_tensor(t.name)
        if partner is not None:
            partner_names.add(partner)

    done = set()
    n = 0
    for t in reader.tensors:
        if t.name in done or t.name in partner_names:
            continue
        dst, partner = map_lm_tensor(t.name)
        if dst is None:
            if t.name.startswith(LM_IGNORED_PREFIX):
                continue
            raise ValueError(f"unmapped tensor: {t.name}")

        if partner is None:
            add_tensor_preserving_bytes(writer, dst, t)
        else:
            other = names[partner]
            if t.data.dtype != other.data.dtype:
                raise ValueError(f"{t.name} and {partner} have different dtypes")
            data = np.concatenate([np.reshape(t.data, -1), np.reshape(other.data, -1)])
            add_tensor_preserving_bytes(writer, dst, t, data)
            done.add(partner)

        done.add(t.name)
        n += 1
        if n % 100 == 0:
            print(f"  lm: {n} tensors", flush=True)

    if n != 4 + N_LAYER * 22:
        raise ValueError(f"unexpected tensor count: {n}")


def write_fm(args, reader, writer):
    has_conv = "fm_modules.fm_head.conv1.weight" in {t.name for t in reader.tensors}
    has_linear = "fm_modules.fm_head.0.weight" in {t.name for t in reader.tensors}
    if has_conv == has_linear:
        raise ValueError("cannot determine pixel head type")

    writer.add_name(args.name)
    writer.add_string("sensenova_u1_fm.head_type", "conv" if has_conv else "linear")
    writer.add_uint32("sensenova_u1_fm.patch_size", 16)
    writer.add_uint32("sensenova_u1_fm.merge_size", 2)
    writer.add_uint32("sensenova_u1_fm.n_embd", N_EMBD)
    writer.add_uint32("sensenova_u1_fm.vision_n_embd", 1024)
    writer.add_float32("sensenova_u1_fm.rope_freq_base", 10000.0)
    writer.add_float32("sensenova_u1_fm.noise_scale", 1.0)
    writer.add_uint32("sensenova_u1_fm.noise_scale_base_seq_len", 64)
    writer.add_float32("sensenova_u1_fm.noise_scale_max_value", 16.0)
    writer.add_bool("sensenova_u1_fm.add_noise_scale_embedding", True)
    writer.add_string("sensenova_u1_fm.noise_scale_mode", "resolution")

    n = 0
    for t in reader.tensors:
        dst = FM_MAP.get(t.name)
        if dst is None:
            continue
        add_tensor_preserving_bytes(writer, dst, t)
        n += 1

    if n != 16:
        raise ValueError(f"unexpected FM tensor count: {n}")


def main():
    ap = argparse.ArgumentParser(description="Retag SenseNova-U1 GGUF files for llama.cpp")
    ap.add_argument("--src", required=True, type=Path, help="vendor GGUF")
    ap.add_argument("--config", required=True, type=Path, help="config dir with tokenizer files")
    ap.add_argument("--out", type=Path, help="output LM GGUF")
    ap.add_argument("--out-fm", type=Path, help="output FM sidecar GGUF")
    ap.add_argument("--name", help="general.name (default: source file stem)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.name is None:
        args.name = args.src.stem

    for p in (args.out, args.out_fm):
        if p is not None and p.exists() and not args.overwrite:
            raise SystemExit(f"output exists: {p}")

    print(f"reading {args.src}", flush=True)
    reader = gguf.GGUFReader(args.src)

    if args.out is not None:
        print(f"writing {args.out}", flush=True)
        writer = gguf.GGUFWriter(args.out, "sensenova_u1", use_temp_file=True)
        write_lm(args, reader, writer)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()

    if args.out_fm is not None:
        print(f"writing {args.out_fm}", flush=True)
        writer_fm = gguf.GGUFWriter(args.out_fm, "sensenova_u1_fm", use_temp_file=True)
        write_fm(args, reader, writer_fm)
        writer_fm.write_header_to_file()
        writer_fm.write_kv_data_to_file()
        writer_fm.write_tensors_to_file(progress=True)
        writer_fm.close()

    print("done", flush=True)


if __name__ == "__main__":
    main()
