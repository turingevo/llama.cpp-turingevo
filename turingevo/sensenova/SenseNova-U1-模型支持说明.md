# SenseNova-U1 接入 llama.cpp 说明

在 llama.cpp 里新增 SenseNova-U1（NEO-Unify MoT）文生图推理支持的实现记录。
内容包括：改了哪些文件、模型文件怎么准备、量化方案的区别、推理链路与关键坑位。

---

## 1. 结论先说

- 新增独立工具 `tools/sensenova-u1/`，编出 `llama-sensenova-cli`，不依赖 Python。
- 模型权重走 llama.cpp 原有 GGUF 加载路径（`src/models/` 新增 gen 分支张量），
  视觉编码器 / timestep MLP / 像素头由工具自带的小型 ggml 图实现（FM sidecar）。
- 1024x1024 / 8 步在 RTX 4070 Ti SUPER 上 8 秒出图，CPU 约 5 分钟。
- 出图质量与供应商 Python 参考实现同档（信息图类 prompt 结构、配色、文字布局一致）。
- 另支持视觉问答（VQA）：`--vqa` / `POST /v1/chat/completions`，复用理解分支，不跑去噪。
- 另支持图文交错：模型在回复里边写文字边发 `<img>`，每个标签就地出图并回灌上下文，
  然后继续写。CLI（`--interleave`）和 HTTP 服务端（`POST /v1/interleave/generate`）都支持。

---

## 2. 改动清单

### 新增

| 文件 | 行数 | 作用 |
|---|---|---|
| `tools/sensenova-u1/main.cpp` | 392 | CLI 参数解析、think 文本输出 |
| `tools/sensenova-u1/pipeline.cpp` / `pipeline.h` | 1747 / 158 | 共享推理管线：prompt 模板、M-RoPE 位置、think、CFG、三序列 prefix decode、去噪主循环、VQA、interleave |
| `tools/sensenova-u1/image.cpp` / `image.h` | 271 / 27 | 输入图解码、缩放、归一化（编辑用） |
| `tools/sensenova-u1/fm.cpp` / `fm.h` | 558 / 140 | FM sidecar 加载；视觉编码、timestep 嵌入、像素头三个 ggml 图，另含理解侧 vision 编码 |
| `tools/sensenova-u1/png.cpp` / `png.h` | 133 / 21 | 极简 PNG 写出（stored deflate，无外部依赖） |
| `tools/sensenova-u1/retag.py` | 350 | 把 HF 命名 / 未拆分的 GGUF 重标定为 llama.cpp 可加载格式，并生成 FM sidecar |
| `tools/sensenova-u1/CMakeLists.txt` | 13 | target `llama-sensenova-cli`，静态库 `sensenova-core` |
| `tools/sensenova-server/main.cpp` | 1351 | HTTP 服务：OpenAI `generations`/`edits`/`chat` 与私有 `generate`/`infographic`/`interleave` 协议 |

### 修改

| 文件 | 改动 |
|---|---|
| `src/models/sensenova-u1.cpp` | 加载 MoT 生成分支张量（`*.gen.weight`）、4 段位置输入、`normalize_and_rotate` |
| `src/models/models.h` | 新增 gen 分支的图输入结构体声明 |
| `tools/CMakeLists.txt` | 加入 `sensenova-u1` 子目录 |
| `.gitignore` | 忽略本地 build/venv 目录 |

生成分支张量用自定义后缀命名（`tn(LLM_TENSOR_ATTN_NORM, "gen.weight", i)`），
没有往 `llama-arch` 的枚举里加新张量类型，避免影响其它模型。

---

## 3. 模型文件准备

HF 原始权重的命名是 `language_model.model.layers.N...`，且视觉编码器、FM head
和 LM 混在一个文件里，llama.cpp 无法直接用。`retag.py` 做三件事：

1. **重命名**为标准 `blk.N.*` 布局，`arch` 设为 `sensenova_u1`，写入 hparams 与 tokenizer。
2. **合并 q/k norm**：HF 每层把 `q_norm` 和 `q_norm_hw` 分开存 64+64，这里拼成 128。
3. **拆出 FM sidecar**：视觉 patch/dense 卷积、timestep MLP、noise-scale MLP、
   像素头，单独写成一个 GGUF（`general.architecture = sensenova_u1_fm`），
   并在 KV 里记录（键名统一带 `sensenova_u1_fm.` 前缀）：

   ```
   head_type                       "conv" | "linear"
   patch_size                      16
   merge_size                      2
   n_embd                          4096
   vision_n_embd                   1024
   rope_freq_base                  10000.0
   noise_scale                     1.0
   noise_scale_base_seq_len        64
   noise_scale_max_value           16.0
   add_noise_scale_embedding       true
   noise_scale_mode                "resolution"
   ```

输出（当前使用）：

```
/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp/
  SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf   11.66 GiB  LM（stock 配方）
  F16-fm.gguf                                        154 MiB  FM sidecar
```

sidecar 共 20 个张量：生成侧 16 个是 patch_embedding(w,b)、dense_embedding(w,b)、
timestep mlp.0/mlp.2(w,b)、noise_scale mlp.0/mlp.2(w,b)、fm_head conv1/conv2(w,b)；
另 4 个 `vision_und.*` 是理解侧视觉编码器，编辑和 VQA 用。权重 F16、bias F32。
源权重里没有理解侧那组时只能拆出 16 个（121 MiB），文生图不受影响。

`head_type` 用来区分两种像素头：`conv` 是 V1.5 的 `ConvDecoder`
（conv1 -> GELU -> conv2 -> pixel shuffle），`linear` 是 V3 的线性头
（直接 3072 维 / token）。工具按该标记选路径。

也可以绕过 vendor GGUF：用 `tools/sensenova-u1/convert_sensenova.py` 把
safetensors 直接转成 F16 GGUF，再用 `llama-quantize` 自行量化。完整命令、
Q5_K_M 的实际选型规则、vendor 配方的 override 和一致性验证结果见
`SenseNova-U1-量化方法.md`。

---

## 4. 量化方案的区别

对比 `SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M`（少步蒸馏版）与
`SenseNova-U1.5-8B-MoT-Q5_K_M`（base 版）：

| 项目 | 8step-V2-Q5_K_M | base-Q5_K_M |
|---|---|---|
| 文件 | 12.7 GiB (LM) + 154 MiB (FM) | 14.07 GiB（单文件全量） |
| 张量数 | 928 | 1116 |
| 命名 | retag 后（`blk.N.*`） | HF 原始命名 |
| token_embd | F16，1187 MiB | F16，1187 MiB |
| lm_head | **F16**，1187 MiB | **F32**，2374 MiB |
| 主干线性层 | 588 x Q5_K，10.38 GiB | 550 x Q5_K，10.29 GiB |
| v_proj | Q5_K | **Q6_K**（42 个） |
| RMSNorm | F32，合计约 3.2 MB | F32 |
| 视觉编码 + 像素头 | 拆到 sidecar，**F16** | 内嵌，**F32** |
| 每层 norm 张量 | 2 个（q/k 各一，und + gen 合并） | 6 个（64+64 分开，另有 `_mot_gen` 变体） |

要点：

- 上表是 原厂 配方。本地现在用的是 stock 配方（`llama-quantize` 默认）：
  Q5_K_M 11.66 GiB、Q6_K 13.34 GiB、Q8_0 17.28 GiB，三者都是 590 权重 + 338 F32 norm，
  共用同一个 `F16-fm.gguf`。
- 两边都**没有**量化 embedding 和 lm_head，都保留 RMSNorm 为 F32。
- base 版大 1.35 GiB，主要就是 `lm_head` 用 F32（2.37 GiB）而 8step 用 F16。
- base 版按 Q5_K_M 惯例把 `v_proj` 升到 Q6_K；8step 版全部 Q5_K。
- base 版视觉 / 像素头保持 F32 内嵌；8step 版拆出来并转 F16。

**速度**：单次前向两者基本相同（同 42 层 / 4096 / 12288，lm_head 在出图时被跳过）。
总耗时差别在步数与 CFG：8step 版 8 步 x 1 次前向，base 版 50 步 x 2 次（cfg 4.0），
约 12.5 倍差距。

---

## 5. 推理链路

主干参数：42 层 / n_embd 4096 / n_ff 12288 / 32 头 / 8 KV 头 / head_dim 128 /
RMS eps 1e-6 / 时间 RoPE base 5e6 / 空间 RoPE base 1e4 / n_ctx_train 16384 /
vocab 151936。图像侧 patch_size 16、merge_size 2，即 1 个图像 token 覆盖 32x32 像素。

与供应商 `modeling_neo_chat.py::t2i_generate` 逐步对应：

### 5.1 prompt

条件分支（`SYSTEM_MESSAGE_FOR_GEN` 已内嵌进 pipeline.cpp）：

```
<|im_start|>system\n{SYSTEM}<|im_end|>\n
<|im_start|>user\n{prompt}<|im_end|>\n
<|im_start|>assistant\n<think>\n
```

`<think>` 用的是特殊 token 151667。think 模式下前缀只到 `<think>\n`，
后面的推理文本由模型自回归生成，遇到 `</think>` 停止；非 think 模式直接接
`<think>\n\n</think>\n\n<img>`。

编辑时 prompt 里的 `<image>` 占位会展开成 `<img><IMG_CONTEXT>...</img>`，
占位数量按输入图张数补齐。

无条件分支（cfg > 1 时才用）：

```
<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<img>
```

9 个 token。`add_bos = add_eos = false`。

### 5.2 prefix decode

最多三条序列各跑一次普通因果 decode，KV 留在 seq 0 / 1 / 2：条件分支、
图像条件分支（编辑时用）、无条件分支。文本位置 0..n-1。

### 5.3 去噪循环

每步：

1. 当前图像（`[3][H][W]`，归一化到 [-1,1]）转 `[W][H][C]`，过视觉编码：
   `conv3x3x16x16 stride16` -> GELU -> `conv1024->4096 k2 s2` -> 每 token 4096 维。
2. 加上 timestep 嵌入（sinusoidal 128 -> cat[cos,sin] -> 256 -> SiLU -> 4096），
   以及 noise-scale 嵌入（`noise_scale/16`，同结构）。
3. 生成 pass：把 1024 个图像 token 的 embedding 作为一批输入，`pos` 全等于 prefix 长度，
   M-RoPE 的 h/w 段分别填 `i/token_w`、`i%token_w`；**关闭因果注意力**
   （`llama_set_causal_attn(ctx, false)`，等价于参考实现里 `attention_mask=None`
   走 flash 的 `causal=False`）。
4. 取最后隐状态，过像素头得到 `x_pred`（像素空间）。
5. `v = (x_pred - img) / max(1-t, 0.05)`，`img += (t_next - t) * v`。
   patchify/unpatchify 只是纯置换，所以在像素空间做与参考实现的 latent 空间等价。
6. 有 cfg 时对 uncond 序列重复 3-5，再
   `v = v_uncond + cfg * (v_cond - v_uncond)`（`cfg_norm='none'`）。

### 5.4 时间表

`t_i = linspace(0,1,steps+1)`，然后 `sigma = 1-t`，`sigma = shift*sigma/(1+(shift-1)*sigma)`，
`t = 1-sigma`。默认 `shift = 3.0`。

### 5.5 噪声尺度

`noise_scale = min(sqrt((H/16 * W/16) / merge_size^2 / 64), 16)`：

| 分辨率 | noise_scale |
|---|---|
| 1024x1024 | 4.0 |
| 2720x1536 | 7.98 |

初始图像 `img = noise_scale * N(0,1)`。RNG 用自带的 splitmix64 + Box-Muller，
保证跨平台确定，但与 torch 的 RNG 不同源。

### 5.6 像素头（V1.5）

隐状态 `[token][4096]` -> `[4096][h][w]` -> pixel_shuffle(2) -> `conv1 1024->1024 k3 p1`
-> GELU -> pixel_shuffle(2) -> `conv2 256->192 k3 p1` -> pixel_shuffle(8) -> `[3][H][W]`。

输出 `clamp(img*0.5+0.5, 0, 1) * 255` 写 PNG。

### 5.7 视觉理解问答（VQA）

和出图是两条路：VQA 只用理解侧 vision embedder 和 LM 的理解分支，不进去噪循环。

1. 输入图按 `min(2048², 4096²/图数)` 缩放，过 sidecar 里的 `vision_und.*`
   patch / dense embedder，得到每个图像 token 的 embedding。
2. prompt 是单轮 chat，与厂商 `neo1_0` 模板等价（无 system）：
   `<|im_start|>user\n{问题}<|im_end|>\n<|im_start|>assistant\n`。
3. 前缀 decode 时开 causal attention、关生成分支，取最后位置 logits 逐 token 生成，
   遇到 `<|im_end|>` 停止；默认贪心，可开采样。
4. 回复里若带 `<think>...</think>`，按标签拆成 think 与 answer 两段返回。
   这段 think 是模型自己写的，VQA 没有开关；预算按两段合计。

### 5.8 图文交错（interleave）

对应厂商 `modeling_neo_chat.py::interleave_gen`，只实现 cfg = 1.0 的单条件分支，
因此不需要厂商那三条 cache（条件 / 文本无条件 / 图像无条件），一条 KV 序列就够。

1. prompt 用厂商 `examples/interleave` 的 `DEFAULT_SYSTEM_MESSAGE`（已内嵌为
   `SYSTEM_MESSAGE_FOR_INTERLEAVE`），`--no-think` 时补 `<think>\n\n</think>\n\n`。
   输入图按 `<image>` 占位展开，占位不够时在最前面补 `<image>\n`（厂商不加 `Image-N:`）。
2. 文字自回归：每 token 走普通因果 decode。遇到 `<img>` 进入出图，遇到 `<|im_end|>` 结束。
3. 位置（与厂商逐位一致）：

   | token 块 | 位置 |
   |---|---|
   | 最后一段文字 | T |
   | `<img>` | T+1 |
   | 图像 token | T+2（全部同一时间位置，h/w 按 raster） |
   | `</img>` | T+3 |
   | 下一段文字 | T+4 |

   图像块内部双向注意力；图像 token 看不到 `</img>`（厂商的 mask 就是
   `mask[:N, past+N] = -inf`）。实现上先解 `</img>` 之前的非因果图像块，
   再用因果 decode 补 `</img>`，两者效果等价。
4. 去噪：复用 FM 视觉编码 + timestep 嵌入 + 生成分支，每步开始前
   `llama_memory_seq_rm(mem, seq, t_img, -1)` 把上一步的图像 token KV 滚掉，
   所以图像块只占固定长度，不随步数增长。
5. 收尾：出图后把 `[-1,1]` 图像转回 `[0,1]` 再做理解侧 ImageNet 归一化，过
   `vision_und.*` 重新编码，作为输入图 token 追加到 T+2，然后解 `</img>`（T+3），
   下一段文字从 T+4 继续。这与厂商 `append_image_to_cache` 一致（理解侧 pass，
   即 `gen_pass=false`）。
6. 文字预算只数文字 token（厂商 `current_generated_tokens` 也只在文字循环里加），
   图像 token 不占额度。`n_ctx = prefix + max_new_tokens + max_images*(图像 token + 2) + 8`。
7. 文字里每个生成图的位置输出 `<image>` 标记（厂商同样是 `generated_text += "<image>"`）。
8. 服务端 `POST /v1/interleave/generate` 直接调用同一函数，把全文按 `<image>` 拆成
   有序 `segments`（text/image 交替），另回传 `think`、`answer` 与 token/image 统计；
   请求字段用 `num_steps` / `timestep_shift`，与生成端点一致。

与厂商的差异：

- 厂商默认 `max_images=10`、`max_new_tokens=8192`、`num_steps=30`；本地默认
  `3 / 4096 / 8`（8step 蒸馏权重）。
- 厂商默认 `think_mode=False`，本地 interleave 默认开 think（官方示例也是开），
  因为系统提示要求图像出现在 think 块内。
- 不支持 CFG（`--cfg` 等被忽略），也不支持厂商的多 cache 交错调度。

---

## 6. 实现中的关键坑

1. **KV 必须 `kv_unified = true`。** `n_ctx` 会被向上取整（例如 333 -> 512），
   默认非 unified 时 `n_ctx_seq = n_ctx / n_seq_max`，两条序列各分一半，
   装不下 prefix + 图像 token。
2. **`n_batch` 不能只按图像 token 设。** 文本 prefix 可能比图像 token 还长，
   必须取两者最大值，否则 `n_tokens_all <= cparams.n_batch` 断言失败。
3. **非因果注意力要求 `n_ubatch >= n_tokens`。** 所以 `n_ubatch = n_batch`。
4. **embedding 输入的 batch 位置数组是 4 段、section-major**，`llama_batch_init`
   只分配一份，得手工构造 `llama_batch`（自备 embd / pos / n_seq_id / seq_id / logits）。
5. **每步生成完要清掉图像 token 的 KV**：`llama_memory_seq_rm(mem, seq, prefix_len, -1)`。
6. **`ggml_permute` 语义**：`result->ne[axis_i] = a->ne[i]`，即 a 的第 i 维放到
   `axis_i` 位置。最初按相反语义写，FM 图全错。
7. **bias 广播**：`ggml_add` 要求 `ggml_can_repeat`，1 维 bias 要
   `ggml_reshape_3d(bias, 1, 1, C)` 才能加到 `[W][H][C]` 上。
8. **conv1 形状**是 `[3,3,n_embd/4,n_embd/4]` = `[3,3,1024,1024]`，
   不是 `[3,3,n_embd,n_embd/4]`（`ConvDecoder` 的 hidden_dim 是 1024）。
9. **q/k norm 与 RoPE 的配合**：`normalize_and_rotate` 先把 `[head_dim]` 的
   norm 权重 reshape 成 `[head_dim/2, 2]`，对时间 / 空间两半分别归一化，
   再用 3 次 `ggml_rope_ext` 分别旋转时间（dim = head_dim/2）、高、宽
   （各 dim = head_dim/4，靠 `ggml_rope_set_offset` 错开位置）。
   不要试图先 `ggml_concat` 成完整向量再一次性旋转：拼出来的 strided view
   过不了 `ggml_is_contiguous_rows` 断言，且各轴 freq base 不同
   （时间 5e6，空间 1e4）。
10. **`n_outputs_max` 要按上下文类型分。** 默认它等于 `n_batch`，reserve 图会按
    `n_batch` 个 token 分配 lm_head 输出，即 `n_vocab x n_batch` 个 float；
    4096 token 前缀时约 151936 x 4096 x 4 B ≈ 2.3 GiB，加上其它缓冲直接
    `graph_reserve: failed to allocate compute buffers`。纯文本上下文（VQA / think）
    每次 decode 只要最后一个 token 的 logits，设 1 即可。但 embedding 上下文
    （`embeddings=true`：prefix decode、去噪、interleave 追加图像）每次输出所有 token，
    必须设成 `n_batch`，否则 `GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)`。
    两边不冲突：embedding 模式的图不建 lm_head（见 `src/models/sensenova-u1.cpp`
    里 `if (!gen_pass && !cparams.embeddings)`），不会重新引入那 2.3 GiB。

---

## 7. 验证情况

已验证：

- prompt token 数与参考实现一致（同一 prompt 253 / 258 / 1196 等）。
- 1024x1024 / 8 步 / cfg 1.0 出图结构正确：信息图 prompt 得到 SENSETIME 复古漫画风、
  深蓝星空、双角色、分栏布局，与 prompt 和供应商参考图同档。
- cfg > 1 双序列路径可跑通（含 seq 1 的 KV 复用与释放）。
- 256x256 / 2 步冒烟通过，无 NaN。
- CPU 与 CUDA 两个构建都能跑，CUDA 约 40 倍加速。
- 2720x1536 / 8 步在 CUDA 上跑通（Q5_K_M，43/43 层在 GPU，计算缓冲 885 MiB）。
- Q6_K 同分辨率全 GPU 跑通：`--fm-cpu` 加 `-ot '^token_embd\.weight$=cpu'`
  `-ot '^output\.weight$=cpu'`，34.5 秒，峰值 15,831 MiB（见量化方法 4.4）。
  Q8_0 权重 17.28 GiB，16 GB 卡放不下，需 `-ngl 20` 上下或纯 CPU。
- VQA 在 CPU 上跑通：1024x512 测试图正确读出 `ZEBRA-7391`；
  2720x1536 的 4094 token 前缀预填充通过（计算缓冲 930 MiB）。
- VQA 在 GPU 上端到端跑通：2720x1536 图 + 一句问题，prompt 4099 tokens，
  回答 109 tokens，服务端约 3 s。
- 图文交错在 CPU 上跑通（256x256 / 2 步 / 上限 1 张图 / `--no-think`）：
  文字 48 token 后命中预算，出图 1 张，`<image>` 标记位置正确，日志里能看到
  去噪两步和追加图像后的继续生成。2 步的图只用于验证链路，8step 权重按 8 步出图。
- 服务端 `POST /v1/interleave/generate` 在 CPU 上跑通：`client.py --api interleave`
  的 b64_json 与 url 两种格式都验过，`segments` 里 text/image 顺序正确，
  url 模式落盘文件能通过 `/images/` 取回；参数越界（`max_images`、`max_new_tokens`、
  尺寸、`response_format`）都返回 400。
- 同一二进制上单图生成（256x256 / 2 步）回归通过，`n_outputs_max` 的修正
  没有破坏原有的文生图路径。

未验证 / 已知差异：

- interleave 只在 CPU 上验证过，GPU 还没跑（生产服务占用显存，等停机窗口）；
  也没有和厂商 `examples/interleave` 做过逐图对比。
- 没有和 Python 参考实现做逐像素对比。原因：两边 RNG 不同源，且量化矩阵乘实现不同
  （融合 `Q5_K x Q8_K` / BLAS 反量化 + F32 sgemm / CUDA kernel）。实测同一 seed 下
  三种后端出的图互不相同、各自可复现；prefix KV 在 layer 0 就差约 3%（relL2），
  到 layer 41 约 7%，8step 蒸馏模型会把这点扰动放大。
  若要严格对比，需要给工具加 `--noise PATH` 喂同一份初始噪声，再比视觉相似度。
- PNG 未压缩，2720x1536 约 12.5 MB。

---

## 8. 后续可做

1. 加 `--noise PATH` / `--dump-noise PATH`，与参考实现共用初始噪声，做量化级别的对齐验证。
2. PNG 换 zlib 压缩（当前 stored deflate，体积大 5-10 倍）。
3. 支持 base 版（50 步 / cfg 4.0）模型：需要一个把 HF 命名直接重标定的入口
   （`retag.py` 目前按 8step 版的张量布局写死）。
4. 清理 `llama_graph_n_input_tensors` 的日志刷屏。
5. 若要进上游：`retag.py` 这类转换脚本、FM 侧载和 gen 分支命名都需要先和
   maintainer 讨论设计，不建议直接提 PR。
