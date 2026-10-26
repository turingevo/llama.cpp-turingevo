# SenseNova-U1 量化方法

从 HF safetensors 自行产出 llama.cpp 可用 GGUF 的完整流程，以及如何精确复现
原厂 Q5_K_M 的量化配方。2026-10-25 在 i9-13900K + RTX 4070 Ti SUPER 上实测。

---

## 1. 结论

- 新增 `tools/sensenova-u1/convert_sensenova.py`：safetensors -> F16 LM GGUF +
  FM sidecar，命名表、q/k norm 拼接、FM 张量划分全部复用 `retag.py`，两条路
  径产出的张量名和 KV 完全一致。
- F16 LM：928 张量、34.9 GB；FM sidecar：16 张量、121 MiB。转换总耗时 11 分 23 秒。
- `llama-quantize` 配 4 个 override 可产出与 原厂 文件**逐张量逐字节一致**的
  Q5_K_M（12.70 GiB），出图 sha256 与现有基线相同。
- 不加 override 的 stock Q5_K_M（11.66 GiB）也能用，但配方与 原厂 不同，
  出图会变。差异集中在 86 个张量，明细见第 4 节。

---

## 2. 两条路线

| | 路线 A：safetensors 自行量化 | 路线 B：原厂 GGUF 重标定 |
|---|---|---|
| 输入 | HF safetensors（33 GB，BF16） | 原厂 已量化 GGUF |
| 工具 | `convert_sensenova.py` + `llama-quantize` | `retag.py` |
| 量化自由度 | 任意 ftype、逐张量 override、后续可加 imatrix | 无，只重命名和拆 sidecar |
| 依赖 | torch / safetensors / transformers / gguf-py | gguf-py |
| 适用 | 自定义精度、自己训练后的权重；base / V3（仅代码路径支持，未实测） | 复现 原厂 结果 |

两条路线最终都需要一对文件：LM GGUF（arch `sensenova_u1`）+ FM sidecar
（arch `sensenova_u1_fm`），推理时分别传给 `-m` 和 `--fm`。

路线 B 的命令：

```bash
python3 tools/sensenova-u1/retag.py --src 原厂.gguf --config "$SRC" \
    --out retag.gguf --out-fm retag-fm.gguf
```

### 2.1 FM sidecar 与量化等级无关

- **名字来源**：FM 是 原厂 自己的命名（Flow Matching 模块组，代码里是
  `fm_modules` / `FlowMatchingHead` / `fm_head`）；sidecar 是本项目加的概念，
  指从主权重里拆出来、单独传给 `--fm` 的那个 GGUF。
- **内容固定**：sidecar 在转换时一次写定，8 个权重 F16 + 8 个 bias F32，共 121 MiB。
  之后 LM 量化成 F16 / Q5_K_M / Q6_K / Q8_0 都不影响它。`llama-quantize`
  只有一个模型输入，加载的是 LM 的 arch，全程不碰 sidecar。
- **配对方式**是一份 LM 配一份 sidecar，任意量化等级共用同一个 sidecar：

  | LM 文件 | `--fm` |
  |---|---|
  | 原厂 原版 Q5_K_M | `F16-fm.gguf` |
  | stock Q5_K_M（不加 override） | 同上 |
  | Q6_K 或将来自己量化的任何 ftype | 同上 |

- 盘上现在只保留一个 sidecar：`F16-fm.gguf`，由转换器从 safetensors 生成。
  `retag.py` 从 原厂 GGUF 拆出的那份张量和 KV 与它逐字节相同，只有文件整体的
  写入顺序和对齐不同，实测可互换（用转换器产出的 sidecar 配 原厂 LM，出图
  sha256 与基线相同），验证后未保留。
- **不能换的是模型版本**：8step-V2、base、V3 的 FM 权重和 `head_type` 都不同，
  sidecar 必须和 LM 同版本。量化等级随便换，版本不能混。

---

## 3. 路线 A：safetensors -> F16 GGUF -> llama-quantize

### 3.1 环境

需要带 `torch`、`safetensors`、`transformers`、`numpy` 的 Python 环境。
`gguf-py` 由脚本自己插入 `sys.path`，不用安装。当前用的环境：

```
/media/wmx/soft1/AI-model/1-ART/SenseNova-U1/.venv/bin/python
```

tokenizer 走 `transformers` 的 `Qwen2Tokenizer`（slow），配置目录里要有
`config.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt`。

### 3.2 转换命令

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp

SRC=/media/wmx/ws3/huggingface_cache/models/image/SenseNova-U1/SenseNova-U1.5-8B-MoT-SFT-8step-V2
OUT=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp

PY=/media/wmx/soft1/AI-model/1-ART/SenseNova-U1/.venv/bin/python

$PY tools/sensenova-u1/convert_sensenova.py --src "$SRC" --config "$SRC" \
    --out    "$OUT/SenseNova-U1.5-8B-MoT-SFT-8step-V2-F16.gguf" \
    --out-fm "$OUT/F16-fm.gguf"
```

`--src` 可以是目录、单个 safetensors 文件，或没有 index 的单分片目录。
`--out` 和 `--out-fm` 可以只给一个（例如只重做 FM sidecar）。

### 3.3 输出内容

LM GGUF（928 张量）：

| 内容 | 类型 |
|---|---|
| RMSNorm（含 q/k norm、output norm、每层 und + gen 各一份） | F32，338 个 |
| 其余权重（attn、FFN、embedding、lm_head，und + gen 两套） | F16，590 个 |
| tokenizer / hparams / chat template | KV，27 项 |

FM sidecar（16 张量）：

| 内容 | 类型 |
|---|---|
| patch embed / dense merge conv、timestep MLP、noise-scale MLP、像素头 | 权重 F16，bias F32，各 8 个 |
| sidecar 参数 | KV，16 项（`sensenova_u1_fm.*`） |

几个必须保持的细节：

- **norm 必须 F32**：图里用 `ggml_mul` 乘 norm 权重，F16 会触发类型断言。
- **q/k norm 合并**：HF 每层 `q_norm` + `q_norm_hw` 各 64 维，输出里拼成
  128 维的 `attn_q_norm.weight`；`*_mot_gen` 同理。拼接用 F32，结果与 原厂
  逐字节相同。
- **FM bias 必须 F32**：CUDA 的 broadcast add 不支持 F16 src1，
  `ggml-cuda/binbcast.cu` 会断言失败。
- **BF16 -> F16 精度**：BF16 尾数 7 位、F16 尾数 10 位，权重幅值范围内不丢位。
  实测：stock 配方下 842 个同类型张量与 原厂 逐字节一致；用 4.3 的 override
  后 928 个全部一致。

### 3.4 磁盘与耗时

- 输出 34.9 GB；GGUFWriter 的临时 spool 另外占约 35 GB，所以输出目录要有
  ~70 GB 空闲。脚本把 `tempfile.tempdir` 指向输出目录，避免写到 /tmp。
- 实测：**11 分 23 秒**（CPU 时间 4 分 17 秒，瓶颈在磁盘 I/O，约 3 GB/分钟）。
- 转换本身不需要 GPU。

---

## 4. llama-quantize 在本模型上的实际行为

### 4.1 ftype 是配方，不是统一类型

`Q5_K_M` 对每个张量单独选型：类别、层号、张量形状都会影响结果。
本模型是 42 层 dense（无 MoE），每层有两套 MoT 分支（`blk.N.*` 和
`blk.N.*.gen.weight`）：FFN 的 gate/up/down 和 attention 投影都各存两份。
所以 `ffn_down` 共 84 个，`attn_v` 也有 84 个，但分类器只命中 und 的 42 个
（见 4.2）。

### 4.2 默认（不加 override）的 Q5_K_M

| 张量 | stock Q5_K_M 结果 | 数量 |
|---|---|---|
| token_embd | Q5_K | 1 |
| output.weight | **Q6_K** | 1 |
| ffn_down（und + gen） | **Q6_K 63 个**，其余 Q5_K | 84 |
| attn_v（仅 und 命中） | **Q6_K 21 个**，其余 Q5_K | 42 |
| 其它线性层 | Q5_K | 462 |
| RMSNorm | F32（不量化） | 338 |

合计：F32 338 + Q5_K 505 + Q6_K 85 = 928，文件 11.66 GiB（5.74 BPW）。

为什么不是所有张量都 Q5_K：

1. **分类是子串匹配**（`src/llama-quant.cpp`）：

   | 匹配串 | 命中的张量 |
   |---|---|
   | `attn_v.weight` | 只有 und 的 `blk.N.attn_v.weight` |
   | `ffn_down` | und 和 gen 都命中（`ffn_down.gen.weight` 里含 `ffn_down`） |

2. **`use_more_bits(i, n) = i < n/8 || i >= 7*n/8 || (i - n/8) % 3 == 2`**
   会给部分 FFN_DOWN / ATTENTION_V 升到 Q6_K。对应本模型：
   - `i_ffn_down` 从 0 数到 83（84 个都命中），`n_ffn_down = 42`（层数），
     所以 `i >= 36` 全部为真，再加前 5 个和中间的 10 个，共 **63 个**；
   - `i_attention_wv` 只数到 41（gen 不命中），`n = 42`，为真的是
     0-4、7、10、13、16、19、22、25、28、31、34、36-41，共 **21 个**。
3. **embedding 和 lm_head 按 ftype 默认值处理**：Q5_K 和 Q6_K。

### 4.3 复现 原厂 配方

原厂 的 12.70 GiB 文件里没有任何 Q6_K：

| 张量 | 原厂 结果 |
|---|---|
| token_embd | F16 |
| output.weight | F16 |
| ffn_down / attn_v / 其它线性层 | 全部 Q5_K（588 个） |
| RMSNorm | F32（338 个） |

要复现，必须关掉两处默认行为：

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp
OUT=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp

./build/bin/llama-quantize \
    --token-embedding-type f16 \
    --output-tensor-type f16 \
    --tensor-type ffn_down=q5_K \
    --tensor-type attn_v=q5_K \
    "$OUT/SenseNova-U1.5-8B-MoT-SFT-8step-V2-F16.gguf" \
    "$OUT/mine-Q5_K_M.gguf" Q5_K_M
```

override 的语义（`llama_tensor_get_type`）：

- `--token-embedding-type` / `--output-tensor-type` 是最早生效的专用开关；
- `--tensor-type PATTERN=TYPE` 用 `std::regex_search` 匹配完整张量名，
  **第一个命中生效**；命中后 `manual = true`，完全跳过
  `llama_tensor_get_type_impl`，所以 `use_more_bits` 的升位不会再发生；
- 模式 `attn_v` 和 `ffn_down` 都不带 `.weight`，und 和 gen 两个分支一起命中。

耗时：60-83 秒（纯 CPU）。

### 4.4 Q6_K 没有 `_M` 变体

ftype 表（`tools/quantize/quantize.cpp`）里只有 `Q6_K`，传 `Q6_K_M` 会直接报
`invalid ftype`。带 `_S/_M/_L` 后缀的只有 Q2_K、Q3_K、Q4_K、Q5_K 四档。

```bash
./build/bin/llama-quantize \
    "$OUT/SenseNova-U1.5-8B-MoT-SFT-8step-V2-F16.gguf" \
    "$OUT/SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q6_K.gguf" \
    Q6_K
```

结果是 590 个权重全部 Q6_K、338 个 norm 保持 F32，共 13.34 GiB（6.56 BPW），
纯 CPU 约 90 秒。这是 stock Q5_K_M 的精度超集：所有张量的位宽都不低于它。

显存要注意：每层 Q6_K 恰好 302 MiB，42 层加 embd/output 共 13.34 GiB；2720x1536
下 KV 约 868 MiB，桌面常驻约 1 GiB，全 offload 会 OOM。用 `-ngl 39`（3 层留
CPU）可省约 0.95 GiB，`-ngl 38` 更稳；每层落 CPU 在 2720x1536 / 8 步下约多 28 秒。

### 4.5 如果目标是出图质量，而不是复现 原厂

出图分支不跑 lm_head（`src/models/sensenova-u1.cpp` 只在非 gen pass 时构建
logits），所以 原厂 把 `output.weight` 保成 F16（约 1.16 GiB）对 t2i 没有收益。
把这份预算挪给 FFN 更划算。一个定向配方（**未实测，仅建议**）：

```bash
./build/bin/llama-quantize \
    --token-embedding-type f16 \
    --output-tensor-type q5_K \
    --tensor-type ffn_down=q6_K \
    --tensor-type attn_v=q6_K \
    F16.gguf out.gguf Q5_K_M
```

估算体积约 12.5 GiB，与 原厂 文件同档：embedding 保 F16，FFN 和 attn_v
全升 Q6_K，output 降回 Q5_K。

同一 prompt / seed 下，stock 与 原厂 配方出图平均像素差 51/255（8 步蒸馏
模型对权重扰动很敏感，构图会变）。单样本文字保真各有胜负：stock 的成员列表
和底部横幅更干净，原厂 的批次规格框更干净。样本量不足以判断系统性优劣，
要做质量结论需要多 seed 对比。

---

## 5. 验证结果

### 5.1 tensor / KV 一致性

对照对象：`Q5_K_M-厂家方案.gguf`（原厂）。

```python
# 需要 gguf-py：PYTHONPATH=/media/wmx/ws1/workspace/ai/llama.cpp/gguf-py
import sys
import numpy as np
from gguf import GGUFReader

a, b = (GGUFReader(p) for p in sys.argv[1:3])
ta = {t.name: t for t in a.tensors}
tb = {t.name: t for t in b.tensors}
print("names only in one side:", sorted(set(ta) ^ set(tb)))

bad = []
for n in sorted(ta):
    if ta[n].tensor_type != tb[n].tensor_type:
        bad.append((n, "type", ta[n].tensor_type.name, tb[n].tensor_type.name))
    elif not np.array_equal(ta[n].data, tb[n].data):
        bad.append((n, "bytes"))
print("differing tensors:", len(bad), bad[:10])

for k in sorted(set(a.fields) & set(b.fields)):
    if not np.array_equal(a.fields[k].data, b.fields[k].data):
        print("differing kv:", k)
```

实测结果：

| 项目 | 结果 |
|---|---|
| 张量名 | 928 vs 928，无差异 |
| 张量内容 | **928/928 逐字节相同**（override 配方） |
| KV | **27/27 相同** |
| 文件大小 | 均为 13,640,347,136 字节 |
| 出图 | `sha256 be8f000b...`，与基线 `u15_8step.png` 相同 |

stock Q5_K_M 与 原厂 的 86 个差异张量（4.2 节明细）：
`63 ffn_down + 21 attn_v + output.weight + token_embd`。

### 5.2 FM sidecar 一致性

新转换器产出的 sidecar（现盘上的 `F16-fm.gguf`）与 `retag.py` 从 原厂 文件
拆出的 sidecar：

- 16 个张量内容逐字节相同，16 个 KV 相同；
- 用新 sidecar 配 原厂 LM 出图，与基线 PNG **逐字节相同**；
- 两个文件的整体字节不同（写入顺序与对齐差异），但张量和 KV 一致，
  不影响加载和结果。retag 产出的那份验证后未保留。

### 5.3 端到端命令

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp
M=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp

./build-alldev/bin/llama-sensenova-cli \
    -m $M/SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
    --fm $M/F16-fm.gguf \
    -p "$(cat /media/wmx/soft1/AI-model/1-ART/SenseNova-U1/wmx_test/infographic_cmp/prompt.txt)" \
    --width 2720 --height 1536 --steps 8 --cfg 1.0 --seed 42 \
    -o check.png

sha256sum check.png u15_8step.png
```

2720x1536 / 8 步单次出图约 90 秒（含模型加载）。复现基线时用的是自量化的
`mine-Q5_K_M.gguf`（与 原厂 文件逐张量逐字节相同，验证后已删）；换成 原厂
原文件同样得到该 sha256。

**注意用 `build-alldev/` 的二进制。** `build/bin/llama-sensenova-cli` 是
`GGML_CUDA=OFF` 的构建，会静默回落到 CPU，一次出图要几十分钟。
量化工具 `llama-quantize` 用哪个构建都一样，它只跑 CPU。

---

## 6. 常见坑

1. **不要量化 FM sidecar。** 它只有 16 个张量，卷积走 `conv_2d_direct`，
   量化没有收益；而且 arch 是 `sensenova_u1_fm`，`llama-quantize` 走
   `llama_model_loader` 加载时根本不认这个架构。它本身与 LM 的量化等级无关，
   一份可以配任何 ftype，见 2.1 节。
2. **路线 B 不做量化。** `retag.py` 只重命名和拆 sidecar，精度由 原厂 决定。
   想要自定义配方必须走路线 A。
3. **`--tensor-type` 用的是正则，不是前缀。** `ffn_down` 会同时命中 gen 分支；
   想只改 und 需要写成 `ffn_down\.weight$` 这类带锚点的模式，但注意
   `std::regex` 默认 ECMAScript 语法。
4. **stock Q5_K_M 的 embedding 是 Q5_K。** 对 8 步蒸馏模型，embedding 精度
   影响比常规 LLM 大；要稳一点就保 F16。
5. **改完配方不要拿旧图对拍。** 同一 prompt / seed，配方变化会改变构图和文字
   排布；要做质量对比必须重新出图，最好多跑几个 seed。
6. **磁盘**：F16 34.9 GB + spool 35 GB + 量化产物 12-13 GB。空间不够时
   先只写 LM（不加 `--out-fm`），量化完再单独产 sidecar。
7. **tokenizer 缓存**：脚本在 `add_lm_kv` 之后才把 `tempfile.tempdir` 指向输出目录，
   避免 transformers 的缓存落到模型目录。

---

## 7. 产物清单

默认目录 `/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp/`：

| 文件 | 大小 | 内容 | 去留 |
|---|---|---|---|
| `Q5_K_M-厂家方案.gguf` | 13.64 GB | 原厂 原版 LM，F16 x 2 + F32 x 338 + Q5_K x 588 | 保留，对照用 |
| `F16-fm.gguf` | 121 MiB | FM sidecar，转换器从 safetensors 产出 | 保留，出图必需，配任何 LM |
| `...-F16.gguf` | 34.91 GB | 路线 A 的 F16 LM（928 张量） | 再量化才需要，可从 safetensors 重建（约 12 分钟） |
| `...-Q5_K_M.gguf` | 12.52 GB | stock Q5_K_M（无 override），文字比 原厂 清楚 | 保留，当前质量基线 |
| `...-Q6_K.gguf` | 14.32 GB | 全量 Q6_K（590 权重），A 的精度超集 | 出图对比中，2720x1536 需 `-ngl 39` |

验证厂商配方时产出的 `mine-Q5_K_M.gguf` 与 原厂 原文件逐张量逐字节相同，
验证完成后已删除，不占空间。

---

## 8. 实测环境

### 8.1 机器与构建

| 项目 | 值 |
|---|---|
| CPU | i9-13900K |
| GPU | RTX 4070 Ti SUPER 16 GB（sm_8.9） |
| CUDA | 13.1 |
| 构建目录 | `build-alldev/`，Release，`BUILD_SHARED_LIBS=OFF` |
| CUDA 选项 | `GGML_CUDA=ON`、`GGML_CUDA_FA=ON`、`GGML_CUDA_FA_ALL_QUANTS=ON` |
| CPU 选项 | `GGML_BLAS=ON`（OpenBLAS）、`GGML_OPENMP=ON`、`GGML_NATIVE=ON` |
| LM | `SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf`（12.52 GB，stock 配方，11.66 GiB） |
| FM | `F16-fm.gguf`（121 MiB） |
| 转换 | 11 分 23 秒 |
| 量化 Q5_K_M | 60-83 秒 |

### 8.2 不同分辨率的出图耗时

2026-10-25 实测。固定参数：`--steps 8 --cfg 1.0 --seed 42 -ngl 999`，
prompt 为 `a red apple on a wooden table`（258 prompt token）。

| 分辨率 | 图像 token | 总耗时 (s) | 其中加载+前缀 (s) | 每步 (ms) | 8 步纯采样 (s) | 峰值显存 (GiB) |
|---|---|---|---|---|---|---|
| 256x256 | 64 | 5.4 | 5.0 | 27 | 0.22 | 12.6 |
| 512x512 | 256 | 5.8 | 4.9 | 85 | 0.68 | 12.6 |
| 768x768 | 576 | 6.6 | 5.0 | 184 | 1.47 | 12.8 |
| 1024x1024 | 1024 | 7.9 | 5.0 | 335 | 2.68 | 13.0 |
| 1280x768 | 960 | 7.6 | 5.0 | 298 | 2.38 | 12.9 |
| 1536x1024 | 1536 | 9.7 | 5.1 | 546 | 4.37 | 13.2 |
| 2048x1152 | 2304 | 12.3 | 5.1 | 864 | 6.91 | 13.6 |
| 2720x1536 | 4080 | 17.7 | 5.3 | 1512 | 12.10 | 14.5 |

列定义：

- **总耗时**：进程墙钟，含模型加载、前缀解码、8 步采样、PNG 写出。
- **加载+前缀**：第一步完成时刻减去一个步长，即进入采样循环之前的耗时
  （加载 11.66 GiB 权重 + 258 token 条件前缀 + 上下文创建）。
- **每步**：相邻两次 `step N/8` 打印的时间间隔平均值（第一个到第八个共 7 个间隔），
  不含加载与前缀。
- **8 步纯采样** = 每步 x 8，是真正随分辨率变化的部分。
- **峰值显存**：`nvidia-smi` 0.15 s 轮询的最大值，含桌面基线约 0.8 GiB；
  扣掉基线后进程占用约 11.8-13.7 GiB。

读法：

- 固定成本约 5 s，小图（<=512）几乎全耗在加载上，总时间看不出差别。
- 每步耗时与图像 token 近似线性，约 0.33-0.42 ms/token/step；
  1024x1024 的 335 ms/步是常用参考点。
- 2720x1536 在 16 GB 卡上不需要降 `-ngl`（峰值 14.5 GiB，还剩约 1.5 GiB）；
  只有 Q6_K 权重更大时才需要 `-ngl 39`。**前提是短 prompt**：prompt 变长后
  `n_ctx` 和 KV 线性增长，4457 token 的 prompt 在 2720x1536 会 OOM，
  需 `-ngl 41`（约 197 s），或降到 2048x1152 保持全 GPU（约 14 s）。
- 每个尺寸只跑一次；重复跑 1024x1024 得到 7.87 s / 8.04 s，抖动在 0.2 s 内。
- 首次运行（CUDA 上下文冷启动）会多约 6 s：256x256 首跑 11.7 s，热跑 5.4 s。

> 本节旧版记录的"2720x1536 / 8 步出图 约 90 秒"在当前构建下无法复现
> （本轮实测 17.7 s，相邻的 2048x1152 为 12.3 s），以上表数据为准。
> 旧数字的测量条件已不可考。
