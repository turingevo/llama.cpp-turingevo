# SenseNova-U1.5 8B (8步蒸馏) llama.cpp 量化版

**SenseNova-U1.5** 原生统一多模态模型（视觉理解 + 图像生成 + 图像编辑）的 **llama.cpp 兼容** GGUF 量化权重，面向单卡部署。

把官方 `SenseNova-U1.5-8B-MoT-SFT` 与官方 8 步蒸馏 LoRA（`SenseNova-U1.5-8B-MoT-LoRA-8step-V2`）合并后的权重量化成 GGUF，配合 TuringEvo 的 llama.cpp 分支，可在消费级显卡上跑 **文生图 / 图像编辑 / 视觉问答 / 图文交错**。

> 本仓库是 **llama.cpp 原生格式**（`general.architecture = sensenova_u1`）。
> 另一个仓库 [TuringEvo/SenseNova-U1.5-8B-MoT-GGUF](https://modelscope.cn/models/TuringEvo/SenseNova-U1.5-8B-MoT-GGUF) 是 **SenseNova官方量化工具输出的单文件版**（架构标为 `qwen3`），**llama.cpp 加载不了**，别下错。

## 模型来源与许可

| 项目 | 说明 |
|---|---|
| 基座 | [SenseNova/SenseNova-U1.5-8B-MoT-SFT](https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT-SFT)（Apache-2.0） |
| 蒸馏 LoRA | `SenseNova-U1.5-8B-MoT-LoRA-8step-V2`（0.4B） |
| 合并方式 | `W += strength * (lora_alpha / rank) * (lora_up @ lora_down)`，strength = 1.0 |
| 量化 | `llama-quantize` stock 配方（未使用原厂 override） |
| 推理框架 | [turingevo/llama.cpp-turingevo](https://github.com/turingevo/llama.cpp-turingevo) 的 `turingevo` 分支 |

上游 ggml-org/llama.cpp 只合入了 `sensenova_u1` 的**理解侧**架构；扩散生成分支（`llama_set_gen_pass`）、FM 侧载、`llama-sensenova-cli`、`llama-sensenova-server` 都在本分支，需要从 [turingevo/llama.cpp-turingevo](https://github.com/turingevo/llama.cpp-turingevo)  构建。

## 文件清单（两个文件必须成对使用）

| 文件 | 大小 | 说明 |
|---|---|---|
| `SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf` | 11.66 GiB (5.74 BPW) | 主权重（`-m`）。stock Q5_K_M：338 个 norm 为 F32 + 505 个 Q5_K + 85 个 Q6_K |
| `F16-fm.gguf` | 154 MB | 侧载（`--fm`）。20 个张量 = 16 个生成侧 FM（patch embed / dense merge conv / timestep MLP / noise-scale MLP / 像素头）+ 4 个理解侧 `vision_und.*`，全 F16/F32，未量化 |

- 主权重与侧载**必须同版本**（8step-V2 / base / V3 的 FM 权重互不通用），量化等级可以随便换。
- 侧载里的 4 个理解侧 vision 权重是**图像编辑和视觉问答的必需项**，缺了会直接报错。
- 侧载沿用 `F16-fm.gguf` 这个名字：它指代的是权重精度（F16），与主权重的量化等级无关，换 Q6_K 也不改这个名字。它内部的 `general.name` 仍是原厂遗留的 `...-Q5_K_M`，不影响加载。

参数口径：官方标称 8B；GGUF 内同时含理解侧与生成侧两套分支，逐张量统计约 17.45B。上下文 16384，42 层，embedding 4096。

## 快速开始

### 构建

```bash
git clone -b turingevo https://github.com/turingevo/llama.cpp-turingevo.git
cd llama.cpp-turingevo
cmake -B build -S . -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target llama-sensenova-cli llama-sensenova-server
```

### 1 文生图

```bash
./build/bin/llama-sensenova-cli \
  -m SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm F16-fm.gguf \
  -p "一只橘猫坐在窗台上看雪" \
  --width 1024 --height 1024 --steps 8 --seed 42 -ngl 999 -o out.png
```

### 2 图像编辑

编辑必须用带理解侧 vision 权重的侧载，也就是本仓库的 `F16-fm.gguf`：

```bash
./build/bin/llama-sensenova-cli \
  -m SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm F16-fm.gguf \
  -p "把背景换成雪山" \
  -i input.png \
  --img-cfg 1.0 \
  --width 1024 --height 1024 --steps 8 --seed 42 \
  -ngl 999 -o edited.png
```

- 不给 `--width/--height` 时，输出尺寸按第一张输入图的宽高比推导。
- `--img-cfg` 是图像引导强度，1.0 表示完全跟随原图条件；调到 1.5 会更强地保留原图内容。
- 多图编辑时用 `<image>` 指定每张图的顺序：

```bash
./build/bin/llama-sensenova-cli \
  -m SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm F16-fm.gguf \
  -p "把 <image>1 的风格套到 <image>2 上" \
  -i a.png -i b.png \
  --steps 8 --seed 42 -ngl 999 -o edited.png
```

### 3 视觉问答

```bash
./build/bin/llama-sensenova-cli \
  -m SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm F16-fm.gguf \
  -i menu.jpg --vqa -p "这张图里有什么菜？" -ngl 999
```

### 4 图文交错

```bash
./build/bin/llama-sensenova-cli \
  -m SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm F16-fm.gguf \
  -p "写一篇露营笔记并配图" --interleave --max-images 3 -ngl 999
```

### 5 服务端

```bash
./build/bin/llama-sensenova-server \
  -m SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm F16-fm.gguf --host 0.0.0.0 --port 8080
```

| 端点 | 用途 |
|---|---|
| `POST /v1/images/generations` | 文生图（OpenAI 兼容） |
| `POST /v1/images/edits` | 图像编辑（OpenAI 兼容） |
| `POST /v1/chat/completions` | 视觉问答（OpenAI 兼容） |
| `POST /generate` | 私有协议：文生图 |
| `POST /v1/infographic/generate` | 私有协议：信息图 |
| `POST /v1/interleave/generate` | 私有协议：图文交错 |
| `GET /v1/models`、`GET /health`、`GET /props` | 元信息 |

## 下载

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download
model_dir = snapshot_download('TuringEvo/SenseNova-U1.5-8B-MoT-SFT-8step-V2-llama.cpp-GGUF')
```

```bash
git clone https://www.modelscope.cn/TuringEvo/SenseNova-U1.5-8B-MoT-SFT-8step-V2-llama.cpp-GGUF.git
```

## 实测性能

测试机：i9-13900K + RTX 4070 Ti SUPER 16GB，CUDA，8 步。

| 场景 | 结果 |
|---|---|
| 文生图 1024x1024 | 8.1 - 8.4 s（含模型加载） |
| 文生图 2720x1536 | 单卡可行，KV 约 0.9 GiB |
| 视觉问答 2720x1536 | 3.0 s（前缀约 4100 token） |
| 峰值显存 | 1024x1024 约 13.5 GiB（权重 12.7 + KV 0.25 + compute 0.21） |

## 已知限制

1. **不同后端出的图不相同**：CPU / CPU+BLAS / CUDA 的量化矩阵乘实现不同，prefix KV 在 layer 0 就差约 3%，8 步蒸馏会放大这点扰动。各后端各自可复现，但别指望跨后端逐像素对齐。
2. 本仓库 Q5_K_M 用的是 stock 配方，与原厂 Q5_K_M 配方不同（差异集中在 86 个张量的量化类型），出图会有差别，文字清晰度反而略优。
3. PNG 目前不压缩（stored deflate），1024x1024 约 3.1 MB，2720x1536 约 12.5 MB。
4. `--cfg > 1` 会为无条件分支多占一条 KV 序列，每步多跑一次前向；本模型推荐 cfg 1.0。
5. 视觉问答只支持单轮，没有多轮历史。
6. 图文交错的 GPU 路径已经实机验证。

## 引用与致谢

- 官方仓库 [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)
- 技术报告 [SenseNova-U1.5 论文](https://arxiv.org/abs/2605.12500)
- 官方权重 [ModelScope](https://modelscope.cn/models/SenseNova/SenseNova-U1.5-8B-MoT) / [Hugging Face](https://huggingface.co/collections/sensenova/sensenova-u15)

## 许可

本仓库权重是官方 Apache-2.0 权重的衍生与量化版本，同样按 Apache-2.0 发布。量化与合并由 TuringEvo 完成。

## 联系方式

- 官网 https://turingevo.com
- API https://api.turingevo.com
- 代码 https://github.com/turingevo/llama.cpp-turingevo

