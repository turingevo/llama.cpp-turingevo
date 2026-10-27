# llama-sensenova-cli 使用说明

SenseNova-U1（NEO-Unify MoT）文生图（t2i）命令行工具。纯 C++ 实现，走 llama.cpp 的
ggml/llama 运行时，CPU 与 CUDA 均可跑。


## 1. 构建目标与产物

### 1.1 目标定义

源文件 `tools/sensenova-u1/CMakeLists.txt`：

```cmake
set(TARGET sensenova-core)
add_library(${TARGET} STATIC pipeline.cpp fm.cpp image.cpp png.cpp)
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama vendor::stb ${CMAKE_THREAD_LIBS_INIT})

set(TARGET llama-sensenova-cli)
add_executable(${TARGET} main.cpp)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries(${TARGET} PRIVATE sensenova-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
```

| 项目 | 内容 |
|---|---|
| 目标名 | `llama-sensenova-cli`（二进制同名），推理代码在静态库 `sensenova-core` |
| 源文件 | `main.cpp`（CLI）、`pipeline.cpp`（共享推理管线：文生图 / 编辑 / VQA / interleave）、`fm.cpp`（视觉编码器、timestep、像素头）、`image.cpp`（输入图解码与缩放）、`png.cpp`（PNG 编码） |
| 链接 | `sensenova-core`（内含 `llama-common`、`llama`、`vendor::stb`、线程库） |
| C++ 标准 | C++17 |
| 安装 | `LLAMA_TOOLS_INSTALL=ON` 时安装到 `<prefix>/bin/` |
| 构建条件 | `tools/CMakeLists.txt` 无条件 `add_subdirectory(sensenova-u1)`；只要 `LLAMA_BUILD_TOOLS=ON` 且不是 Emscripten 构建，就存在该目标 |

### 1.2 两个构建目录的产物

同一份源码编出两个二进制，参数完全相同，区别只在编译选项：

| 路径 | 后端 | 大小 | 编译选项 |
|---|---|---|---|
| `build/bin/llama-sensenova-cli` | CPU | 约 77 KB | `BUILD_SHARED_LIBS=ON`，`GGML_CUDA=OFF`，`GGML_BLAS=OFF` |
| `build-alldev/bin/llama-sensenova-cli` | CUDA (GPU) | 约 219 MB | `BUILD_SHARED_LIBS=OFF`，`GGML_CUDA=ON`，`GGML_BLAS=ON`（OpenBLAS） |

链接方式不同，运行时依赖也不同：

- **CPU 版是动态链接**，依赖同目录下的 `libllama*.so`、`libllama-common*.so`、`libggml*.so`。
  把二进制拷到别处时要连这些库一起拷，或设置 `LD_LIBRARY_PATH`。
- **CUDA 版静态链接** llama/ggml，但仍依赖系统库：`libopenblas`、`libgomp`、`libcudart`、
  `libcublas`、`libcuda`、`libnccl`。

想用 GPU 就跑 `build-alldev/bin/` 那个。该目标已加入 `turingevo/shared/build-all-dev.sh` 的 `TARGETS` 列表。

### 1.3 构建命令

CPU 构建（对应 `build/`）：

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp

cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON

cmake --build build --config Release -j"$(nproc)" --target llama-sensenova-cli
```

CUDA + OpenBLAS 构建（对应 `build-alldev/`）：

```bash
cmake -B build-alldev -S . \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS \
  -DGGML_OPENMP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build build-alldev --config Release -j"$(nproc)" --target llama-sensenova-cli
```

只编这一个目标时不要用 `cmake --install`：它会安装所有已启用的 install 目标，未编译的目标会报
`file INSTALL cannot find ...`。需要安装就手工拷：

```bash
cp build/bin/llama-sensenova-cli /your/prefix/bin/
```

---

## 2. 运行前需要的两个 GGUF

必须同时提供两个 GGUF：

| 参数 | 文件 | 内容 |
|---|---|---|
| `-m` | `SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf` | 语言模型主干（42 层，含 MoT 生成分支） |
| `--fm` | `F16-fm.gguf` | FM sidecar：视觉 patch 编码器、timestep MLP、像素头，另含理解侧 vision 权重 |

- LM 可以是任意量化档（Q5_K_M、Q6_K、Q8_0 等），sidecar 一份 F16 通用，与量化等级无关。
- 程序会检查 sidecar 的 `sensenova_u1_fm.n_embd` 与模型 embedding 维度是否一致，不一致直接报错退出。
- `Q6_K` 权重 13.34 GiB，2720x1536 想全 GPU 要按下面「显存不够时」加
  `-ot` 和 `--fm-cpu`，实测 34.5 秒；不加就得降 `-ngl`。
- `Q8_0` 权重 17.28 GiB，即使把 embd/output 放 CPU 仍超过整卡可用显存，
  需 `-ngl 20` 上下或纯 CPU。

默认位置：

```
/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp/
```

### 2.1 文件从哪来

只能用 `tools/sensenova-u1/` 下两个脚本产出的 GGUF。HF 原始命名、且没拆 sidecar 的文件
（例如 `SenseNova-U1.5-8B-MoT-Q5_K_M.gguf`）**不能**直接加载。

**方式一：从厂商量化 GGUF 重标定**（保留原有量化数据，最快）：

```bash
python3 tools/sensenova-u1/retag.py \
  --src <vendor.gguf> \
  --config <含 config.json 与 tokenizer 文件的目录> \
  --out <lm.gguf> \
  --out-fm <fm.gguf>
```

**方式二：从 safetensors 转 F16，再自行量化**：

```bash
python3 tools/sensenova-u1/convert_sensenova.py \
  --src <HF 模型目录或单个 safetensors> \
  --config <含 config.json 与 tokenizer 文件的目录> \
  --out <lm-f16.gguf> \
  --out-fm <fm.gguf>
```

两个脚本都要求 `--config` 指向含 `config.json` 与 tokenizer 文件的目录；`--name` 可覆盖模型名，
`--overwrite` 允许覆盖已存在的输出。

sidecar 的张量个数决定能力：拆出 16 个张量（没有理解侧 vision 权重）只能文生图；
拆出 20 个张量（多出 `vision_und.*`）才能做图像编辑。
`retag.py` 会打印实际拆出的个数，缺权重时也会打印一行提示。
`convert_sensenova.py` 同理：HF 权重里有 `vision_model.embeddings.*` 时产出 20 个，
没有时产出 16 个。

注意：上游 `conversion/sensenova_u1.py` 会过滤 `_mot_gen` 张量
（`if not name.startswith("language_model.") or "_mot_gen" in name: return None`），
它转出来的 GGUF 没有生成分支，不能给这个工具用。必须用上面两个脚本的产物，
或自行给上游转换器扩展生成分支导出。

---

## 3. 基本用法

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp

M=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp

./build-alldev/bin/llama-sensenova-cli \
  -m $M/SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm $M/F16-fm.gguf \
  -p "a red apple on a wooden table" \
  --width 1024 --height 1024 \
  --steps 8 --cfg 1.0 --seed 42 \
  -o out.png
```

跑完输出 `out.png`，stderr 打印每一步的时间点：

```
prompt tokens: 258, image tokens: 1024 (1024x1024), steps: 8, cfg: 1.00
step 1/8: t=0.0000 -> 0.0455
...
wrote out.png
```

---

## 4. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-m, --model PATH` | 必填 | 语言模型 GGUF |
| `--fm PATH` | 必填 | FM sidecar GGUF。编辑需要带理解侧 vision 权重的 sidecar（`F16-fm.gguf`） |
| `-p, --prompt TEXT` | 必填 | 图像描述。**不要**自己加 im_start 等模板标记 |
| `-i, --image PATH` | 无 | 输入图，做图像编辑（it2i）或 interleave 的参考图。可重复，多图时 prompt 里用 `<image>` 占位 |
| `-o, --output PATH` | `sensenova-u1.png` | 输出 PNG。interleave 多图时依次编号：`out.png`、`out_1.png`、`out_2.png` |
| `--width N` | 0（自动） | 宽，必须是 32 的正整数倍。0 表示按协议推导：无输入图取 1024，有输入图按首图宽高比 |
| `--height N` | 0（自动） | 高，必须是 32 的正整数倍 |
| `--steps N` | 8 | 去噪步数。8step 蒸馏模型用 8 |
| `--cfg F` | 1.0 | 文本引导强度 CFG。1.0 = 只跑条件分支；>1 会额外跑无条件分支，耗时约翻倍 |
| `--img-cfg F` | 1.0 | 图像引导强度，只在编辑时有效。必须配 `--image` |
| `--cfg-norm MODE` | none | CFG 结果的范数约束：`none` / `global` / `channel` |
| `--shift F` | 3.0 | timestep shift，与供应商参考实现一致 |
| `--seed N` | 42 | 随机种子（自带 splitmix64 + Box-Muller，跨平台确定） |
| `--think` | 关 | 先自回归生成 `<think>...</think>` 推理文本，再去噪出图（仅文生图 / 编辑） |
| `--no-think` | 关 | 关掉 interleave 的 think（interleave 默认开 think） |
| `--max-think-tokens N` | 1024 | think 的 token 预算 |
| `--print-think` | 关 | 把 think 文本打到 stdout |
| `--think-output PATH` | 无 | 把 think 文本写到文件 |
| `--input-max-pixels N` | 0（自动） | 每张输入图的像素上限。0 表示 `min(2048², 4096²/图数)`；VQA 同理 |
| `--auto-pixels N` | 1048576 | 自动尺寸时的输出像素。参考实现用 2048²，16 GB 卡放不下 |
| `--vqa` | 关 | 问答模式：`-p` 当问题，`-i` 当待理解图，只输出文字、不出图 |
| `--interleave` | 关 | 图文交错：模型边写文字边发 `<img>`，每个 `<img>` 就地生成一张图，文字打到 stdout |
| `--max-images N` | 3 | interleave 的图像预算，也是 `n_ctx` 的估算依据 |
| `--max-new-tokens N` | 1024 | VQA 的回答预算 / interleave 的文字预算（图像 token 不占这个额度） |
| `--temperature F` | 0.0 | VQA / interleave 采样温度，0 表示贪心 |
| `--top-k N` | 0 | VQA / interleave top-k 采样，0 表示关闭 |
| `--top-p F` | 0.95 | VQA / interleave nucleus 采样 |
| `--repeat-penalty F` | 1.0 | VQA / interleave 重复惩罚 |
| `--answer-output PATH` | 无 | 把 VQA 答案 / interleave 全文（含 `<image>` 标记）写到文件 |
| `-ngl N` | 999 | offload 到 GPU 的层数，**含 output 层**：42 层模型要 `-ngl 43` 才是全部，`-ngl 42` 会留 layer 0 在 CPU；CPU 构建下无效 |
| `-ot, --override-tensor PATTERN=cpu` | 无 | 把匹配的权重放到 CPU，可重复。PATTERN 是正则、子串匹配，建议写成 `'^token_embd\.weight$=cpu'` 这种锚定形式 |
| `--fm-cpu` | 关 | FM sidecar（视觉编码器、像素头）跑在 CPU。2720x1536 省约 760 MiB 显存，代价是每张图多约 16 秒 |
| `-t, --threads N` | 0（由 llama.cpp 取核心数） | CPU 线程数 |
| `-h, --help` | | 打印用法 |

图像 token 数 = `(width/32) * (height/32)`：

| 分辨率 | 图像 token |
|---|---|
| 256x256 | 64 |
| 512x512 | 256 |
| 1024x1024 | 1024 |
| 2720x1536（参考分辨率） | 4080 |

`n_ctx` 由程序自动算，不用手填：

```
n_ctx = 条件前缀 + 图像条件前缀 + 无条件前缀 + 图像 token + 16
```

图像条件前缀和无条件前缀只在需要 CFG 时实际使用，但公式里始终计入；
`--think` 时再加上 `--max-think-tokens` 的余量。2720x1536、1196 token 前缀时
`n_ctx` 是 5376，KV 882 MiB。

### 4.1 显存不够时

三个开关，按性价比排序：

| 开关 | 省显存 | 代价 |
|---|---|---|
| `-ot '^token_embd\.weight$=cpu' -ot '^output\.weight$=cpu'` | 974 MiB 权重，另加约 2.6 GiB 的 logits 缓冲 | 基本为零：embedding 只查约 1200 个 prompt token，出图不走 lm_head |
| `--fm-cpu` | 约 760 MiB（121 MiB 权重 + 约 640 MiB 图缓冲） | 2720x1536 / 8 步多约 16 秒 |
| 降 `-ngl` | 每层 302 MiB（Q6_K）+ 21 MiB KV | 每层多约 28 秒 |

Q6_K 在 2720x1536 全 GPU 的最小配置：

```bash
M=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp

./build-alldev/bin/llama-sensenova-cli \
  -m $M/SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q6_K.gguf \
  --fm $M/F16-fm.gguf \
  -p "a red apple on a wooden table" \
  --width 2720 --height 1536 --steps 8 --cfg 1.0 --seed 42 \
  --fm-cpu -ngl 43 \
  -ot '^token_embd\.weight$=cpu' -ot '^output\.weight$=cpu' \
  -o q6k.png
```

实测 34.5 秒，峰值 15,831 / 16,376 MiB（43/43 层在 CUDA0，GPU 权重 12,681 MiB），
只剩 107 MiB，跑图时别再开占显存的程序。注意 `-ot` 必须锚定，写
`output\.weight` 会把 42 个 `blk.N.attn_output.weight` 一起挪到 CPU。

---

## 5. 运行流程与输出

### 5.1 文生图 / 图像编辑

1. 按内置模板拼 prompt：system 提示词 + 用户描述 + `assistant` 前缀 + `<think>\n\n</think>\n\n<img>`。
   模板取自厂商的 `SYSTEM_MESSAGE_FOR_GEN`，用户只传自然语言描述。
   `--think` 时前缀只到 `<think>\n`，后面的推理文本由模型自回归生成。
2. 有输入图时，用理解侧的 vision embedder 编码每张图，得到图像 token 的 embedding；
   prompt 里的 `<image>` 占位按张数展开成 `<img><IMG_CONTEXT>...</img>`，
   占位不够时自动补在最前面（多图补 `Image-1:<image>`）。
3. 在序列 0 上解码条件前缀；`--cfg > 1`（编辑时为 `--img-cfg > 1` 或两者都 >1）
   再在序列 1 / 2 上解码图像条件前缀和无条件前缀。
4. `--think` 时在序列 0 上自回归生成推理文本，遇到 `</think>` 或用完
   `--max-think-tokens` 停止，然后接上 `\n\n<img>` 进入生成分支。
5. 关闭 causal attention（图像块内部是双向注意力）。
6. 从随机噪声开始，按 timestep 循环 `--steps` 次：
   - FM sidecar 编码当前图像 latent；
   - 加 timestep / noise scale 嵌入；
   - 走 LM 的生成分支得到 hidden state；
   - FM 像素头预测像素；
   - 计算 velocity 并更新图像；
   - 需要 CFG 时再跑其他分支并做混合。
7. 把 `[-1, 1]` 的浮点图像转成 8-bit RGB，写 PNG。

运行时上下文参数：`n_seq_max=3`、`kv_unified=true`、`embeddings=true`。
生成分支的开关由 `llama_set_gen_pass()` 控制（见 `src/models/sensenova-u1.cpp`）。

PNG 采用 stored deflate（不压缩）：1024x1024 约 3.1 MB，2720x1536 约 12.5 MB。

### 5.2 图文交错（interleave）

一次响应里既写文字又出图，对应厂商 `interleave_gen`。只用一条 KV 序列，CFG 固定 1.0
（`--cfg` / `--img-cfg` / `--cfg-norm` 给了也只会打印一条警告）。

1. prompt 用厂商 interleave 的系统提示词（要求模型把推理和图像放进 `<think>` 块），
   用户输入接在后面；默认开 think，`--no-think` 时补 `<think>\n\n</think>\n\n`。
2. prefix 解码后逐 token 生成文字。遇到 `<img>` 时开始出图，文字里该位置记一个
   `<image>` 标记。
3. 位置安排（与厂商一致）：最后一段文字在 T，`<img>` 在 T+1，图像 token 全在 T+2
   （时间位置相同，h/w 按 raster 展开，块内双向注意力），`</img>` 在 T+3，
   下一段文字从 T+4 继续。
4. 出图：FM 视觉编码 + timestep 嵌入 -> 生成分支，去噪 `--steps` 步。每步开始前
   把上一步写的图像 token KV 滚掉，所以图像块只占固定长度。
5. 出图后把图像按理解侧归一化重新编码，作为输入图 token 追加进上下文，
   再解 `</img>`，模型据此继续写。最多出 `--max-images` 张。
6. 写完 `</think>` 后继续写答案，答案里也可以再出图。`--max-new-tokens` 只数
   文字 token，图像 token 不占额度。

`n_ctx` 估算：prefix + `--max-new-tokens` + `--max-images * (图像 token 数 + 2)` + 8。
1024x1024 每张图占 1024 个位置，加 4096 文字预算、3 张图，KV 约 1.3 GiB。
别在多图时用 2720x1536：每张图 4080 个位置，3 张图加文字预算会超过
`n_ctx_train`（16384），16 GB 卡也会先显存不足。

输出：全文写 stdout（带 `<image>` 标记），`--answer-output` 存文件；图按
`-o` 的名字编号写盘（`out.png`、`out_1.png`、`out_2.png`），think 文本仍走
`--print-think` / `--think-output`。

---

## 6. 常用示例

GPU，参考分辨率（供应商对比图用的就是这组参数）：

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp
M=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp

./build-alldev/bin/llama-sensenova-cli \
  -m $M/SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm $M/F16-fm.gguf \
  -p "$(cat /media/wmx/soft1/AI-model/1-ART/SenseNova-U1/wmx_test/infographic_cmp/prompt.txt)" \
  --width 2720 --height 1536 --steps 8 --cfg 1.0 --seed 42 \
  -o u15_8step.png
```

快速冒烟（CPU 也能几秒出图）：

```bash
M=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp

./build/bin/llama-sensenova-cli \
  -m $M/SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm $M/F16-fm.gguf \
  -p "a red apple" \
  --width 256 --height 256 --steps 2 \
  -o smoke.png
```

开 CFG（质量更高但慢一倍，8step 模型一般不需要）：

```bash
./build-alldev/bin/llama-sensenova-cli ... --cfg 2.0 --steps 8
```

think 模式（先推理再出图，推理文本单独存一份）：

```bash
M=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp

./build-alldev/bin/llama-sensenova-cli \
  -m $M/SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm $M/F16-fm.gguf \
  -p "一张介绍宋代点茶工艺的信息图" \
  --width 1024 --height 1024 --steps 8 \
  --think --print-think --think-output think.txt \
  -o think.png
```

图像编辑（it2i）。必须用带理解侧 vision 权重的 FM，即 `F16-fm.gguf`：

```bash
M=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp

./build-alldev/bin/llama-sensenova-cli \
  -m $M/SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm $M/F16-fm.gguf \
  -p "把背景换成雪山" \
  -i input.png \
  --width 1024 --height 1024 --steps 8 \
  -o edited.png
```

`--width/--height` 都不给时，输出尺寸按第一张输入图的宽高比推导。
多图编辑时 prompt 里用 `<image>` 指定每张图的顺序：

```bash
-p "把 <image>1 的风格套到 <image>2 上" -i a.png -i b.png
```

编辑时 `--img-cfg` 控制保留原图的程度，1.0 表示完全跟随原图条件：

```bash
./build-alldev/bin/llama-sensenova-cli ... -i input.png --img-cfg 1.5 --cfg 1.0
```

视觉问答（VQA）。`--vqa` 时 `-p` 是问题、`-i` 是输入图，只出文字：

```bash
M=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp
IMG=/media/wmx/soft1/AI-model/1-ART/SenseNova-U1/examples/vqa/data/images/menu.jpg

./build-alldev/bin/llama-sensenova-cli \
  -m $M/SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm $M/F16-fm.gguf \
  -i $IMG \
  -p "菜单上最便宜的汤多少钱？" \
  --vqa --max-new-tokens 1024
```

stdout 只打印答案；8step 权重爱先写一段 think，想看推理过程加 `--print-think`，
想存文件加 `--think-output`。VQA 没有 `--think` 开关，think 是模型自己写的，
`--max-new-tokens` 按 think 与答案合计。采样与多图示例：

```bash
# 采样，避免 think 绕圈
./build-alldev/bin/llama-sensenova-cli ... -i $IMG -p "转成 JSON" \
  --vqa --temperature 0.7 --top-p 0.9 --max-new-tokens 2048

# 多图
./build-alldev/bin/llama-sensenova-cli ... -i a.jpg -i b.jpg \
  -p "两张图的编号分别是多少？" --vqa
```

输入图分辨率直接影响读字质量：512² 时菜单价格会编，按默认
`min(2048², 4096²/图数)` 才能读准，代价是图像 token 数和耗时上升。
VQA 不需要生成分支，但需要 FM sidecar 里的理解侧 vision 权重（`F16-fm.gguf`）。

图文交错（`--interleave`）。模型自己决定在哪出图，默认开 think：

```bash
M=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp

./build-alldev/bin/llama-sensenova-cli \
  -m $M/SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm $M/F16-fm.gguf \
  -p "画一张红苹果放在木桌上的图，再画一张切开后的样子，分别说明口感" \
  --interleave --max-images 2 --max-new-tokens 2048 \
  --steps 8 -o interleave.png
```

跑完 `interleave.png` 与 `interleave_1.png` 是两张图，stdout 是带 `<image>` 标记的全文。
不用 think、只想要纯答案时：

```bash
./build-alldev/bin/llama-sensenova-cli ... --interleave --no-think --max-images 1
```

参考图用 `-i` 传（走理解侧 vision 权重），输出尺寸默认跟随第一张输入图的宽高比
（`--auto-pixels` 控制像素预算），没有输入图时是 1024x1024，都可用
`--width/--height` 覆盖。每张生成图都按同一尺寸出，不支持逐图不同尺寸。

---

## 7. 错误与退出条件

以下情况程序以退出码 1 结束，并在 stderr 打印原因：

- 缺少 `-m`、`--fm`、`-p`，或出现未知参数；
- `--width` / `--height` 不是 32 的正整数倍；
- `--steps <= 0`（`--vqa` 模式不检查）；
- `--vqa` 时 `--max-new-tokens <= 0`，或采样参数越界；
- `--vqa` 与 `--interleave` 同时给；
- `--interleave` 时 `--max-new-tokens` / `--max-images` 不是正数，或 prompt 里的
  `<image>` 占位比输入图多；
- `--img-cfg != 1.0` 但没有输入图；
- `--cfg-norm` 不是 none/global/channel；
- 输入图读不了、像素预算不足（小于 512x512）；
- FM sidecar 没有理解侧 vision 权重，却带了输入图；
- 模型加载失败、上下文创建失败；
- 条件或无条件前缀解码失败；
- 没有可用后端设备，或 FM backend 创建失败；
- FM sidecar 加载失败（缺 metadata 或张量）；
- LM 与 FM 的 `n_embd` 不一致；
- 视觉编码、timestep 嵌入、生成分支、像素头任一阶段失败；
- PNG 写入失败。

模型加载阶段还会检查生成分支是否完整：

- LM GGUF 缺少任意一层 `*.gen.weight`：抛 `SenseNova U1: incomplete generation branch in layer N`；
- 完全没有生成分支：构图时 `GGML_ABORT`，提示 `generation branch weights are missing`。

---

## 8. 性能参考

测试机：i9-13900K + RTX 4070 Ti SUPER 16 GB。

| 配置 | 时间 |
|---|---|
| CUDA 1024x1024 / 8 步 | **8.1 - 8.4 s**（含模型加载） |
| CPU 1024x1024 / 8 步 | 约 5 分钟 |
| CPU 256x256 / 2 步 | 9.2 s |
| CUDA VQA 2720x1536（4099 token 前缀 + 109 token 回答） | 3.0 s |
| CUDA 2720x1536 / 8 步，Q6_K 全 GPU（`--fm-cpu` + 两个 `-ot`） | 34.5 s，峰值 15.8 GiB |

显存占用（1024x1024）：权重 12.7 GiB + KV 252 MiB + compute buffer 211 MiB，
峰值约 13.5 GiB。2720x1536 的 KV 是 882 MiB（`n_ctx` 5376）。
VQA 的前缀带上图像 token 会更长：2720x1536 时约 4100 token，compute buffer 约 0.9 GiB。

---

## 9. 注意事项与限制

1. **不同后端的出图不完全相同。** 同一 seed、同一 prompt，CPU（无 BLAS）、
   CPU（有 BLAS）、CUDA 三种组合出的图互不相同，且各自可复现。原因是量化
   矩阵乘实现不同（融合 `Q5_K x Q8_K` / 反量化 + F32 sgemm / CUDA 自家 kernel），
   prefix KV 在 layer 0 就差约 3%，8step 蒸馏模型会把这扰动放大。
   三者的图都连贯、符合 prompt，所以不是 bug；但别指望跨后端逐像素对齐。

2. **PNG 目前不压缩**（stored deflate），1024x1024 约 3.1 MB，
   2720x1536 约 12.5 MB。

3. `--cfg > 1` 会为无条件分支占用第二个 KV 序列，每步多跑一次 42 层前向。

4. prompt 模板已内置，用户只传自然语言描述。

5. 图像编辑依赖 FM sidecar 里的理解侧 vision 权重。`F16-fm.gguf` 拆了 20 个张量
   （16 个生成侧 + 4 个理解侧 `vision_und.*`），文生图和编辑都能用。
   如果 `retag.py` 只拆出 16 个张量，说明源 GGUF 里没有理解侧那组权重，
   此时编辑会直接报错，文生图不受影响。

6. think 文本是贪心解码（argmax）出来的，不做采样；VQA 默认也是贪心，
   想要采样显式给 `--temperature > 0`。

7. VQA 只支持单轮问答，没有多轮历史。图文交错（`--interleave`）在 CLI 与 HTTP 服务端
   都有入口（服务端是 `POST /v1/interleave/generate`，见 `server-client-使用说明.md`
   3.8 节）。interleave 只用一条 KV 序列，CFG 固定 1.0；去噪每步都会滚掉上一轮的
   图像 token KV，最后只保留重新编码的理解侧图像 token，所以上下文不会按去噪步数膨胀。

8. **VQA 的 `<think>` 是模型自发输出的**，没有开关。`--max-new-tokens` 按
   think 与答案合计，预算不够会停在 think 里并打印
   `reply ended inside the think block`，加大预算即可。

---

## 10. 相关文件

| 路径 | 作用 |
|---|---|
| `tools/sensenova-u1/CMakeLists.txt` | 目标定义 |
| `tools/sensenova-u1/main.cpp` | CLI 参数解析、think 输出 |
| `tools/sensenova-u1/pipeline.cpp` / `pipeline.h` | 共享推理管线：prompt 模板、M-RoPE 位置、think、CFG、去噪循环、VQA、interleave |
| `tools/sensenova-u1/image.cpp` / `image.h` | 输入图解码、缩放、归一化 |
| `tools/sensenova-u1/fm.cpp` / `fm.h` | FM sidecar 加载、视觉编码、timestep 嵌入、像素头 |
| `tools/sensenova-u1/png.cpp` / `png.h` | PNG 编码（stored deflate） |
| `tools/sensenova-u1/retag.py` | 厂商 GGUF 重标定 + 拆 sidecar |
| `tools/sensenova-u1/convert_sensenova.py` | safetensors 转 F16 GGUF + FM sidecar |
| `tools/sensenova-server/main.cpp` | HTTP 服务：OpenAI（generations / edits / chat completions）与私有协议（generate / infographic） |
| `src/models/sensenova-u1.cpp` | LM 图构建与生成分支 |
| `src/models/models.h` | `sensenova_u1` 架构结构体（含 `gen_layers`） |
| `turingevo/shared/build-all-dev.sh` | CUDA 构建脚本，`TARGETS` 含 `llama-sensenova-cli` |
| `turingevo/sensenova/client.py` | 服务端测试客户端（openai / edits / native / infographic / vqa / interleave 六种协议） |
| `turingevo/sensenova/cli.sh` | CLI 测试脚本 |
| `turingevo/sensenova/server.py` | 服务启动脚本（跨平台） |
| `turingevo/docs/usage.md` | 全部编译目标与二进制说明 |


##  11 prompt 模板

sensenova studio 模板
https://unify.light-ai.top/gallery?model=u15-lite

1 用prompt压缩 ：
 `turingevo/sensenova/compress_prompt.py`

2 再生成图片 ：
 `turingevo/sensenova/cli.sh`