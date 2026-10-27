# server.py / client.py 使用说明

SenseNova-U1.5 本地服务的一对起停脚本:

| 脚本 | 作用 |
|---|---|
| `server.py` | 找权重、找二进制，把 `llama-sensenova-server` 拉起来 |
| `client.py` | 调服务出图或问答，支持文生图、图像编辑（it2i）、think 推理、VQA、图文交错 |

两个脚本都只用 Python 标准库，跨平台（Linux / macOS / Windows）都能跑。

---

## 1. 前置条件

| 项 | 说明 |
|---|---|
| 服务端二进制 | `build-alldev/bin/llama-sensenova-server`（CUDA），或 `build/bin/...`（CPU），或用 `SENSENOVA_SERVER_BIN` 指定 |
| 语言模型 | `SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf` |
| FM sidecar | `F16-fm.gguf`，20 个张量（16 生成侧 + 4 理解侧）。**图像编辑必须有理解侧那 4 个** |
| 显存 | 1024x1024 约 13.5 GiB；2720x1536 的 KV 多约 0.7 GiB。Q5 权重 12.7 GiB 是大头 |

权重默认在：

```
/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp/
```

---

## 2. server.py

### 2.1 最快启动

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp/turingevo/sensenova
python3 server.py
```

等价于：

```bash
llama-sensenova-server \
  -m .../SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf \
  --fm .../F16-fm.gguf \
  --host 127.0.0.1 --port 8888 -ngl 999 \
  --output-dir <临时目录>/llama-sensenova-server \
  --allow-output-dir
```

模型加载要十几秒，`/health` 返回 `{"status":"ok"}` 才算就绪。

### 2.2 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host ADDR` | 127.0.0.1 | 监听地址 |
| `--port N` | 8888 | 监听端口 |
| `--ngl N` | 999 | offload 到 GPU 的层数，**含 output 层**：42 层模型要 43 才是全部 |
| `-ot PATTERN=cpu` | 无 | 把匹配的权重放到 CPU，可重复。模式是正则，建议锚定：`'^token_embd\.weight$=cpu'` |
| `--fm-cpu` | 关 | FM sidecar 跑在 CPU，省约 760 MiB 显存；2720x1536 每张图慢约 16 秒 |
| `--model PATH` | 权重目录下的 Q5_K_M | 语言模型 GGUF 全路径 |
| `--fm PATH` | 权重目录下的 `F16-fm.gguf` | FM sidecar 全路径 |
| `--model-dir PATH` | 见上文 | 权重目录，给 `--model` / `--fm` 拼默认名用 |
| `--output-dir PATH` | 系统临时目录下的 `llama-sensenova-server` | `response_format=url` 时图片落盘位置 |
| `--no-allow-output-dir` | 关 | 不允许请求里指定 `output_dir` |
| `--dry-run` | 关 | 只打印要执行的命令，不起服务 |

其余参数原样透传给服务端二进制，例如：

```bash
python3 server.py --port 9000 -ngl 39 --model-name my-u1
```

同名参数以你传的为准，不会出现两个 `-ngl`。

### 2.3 环境变量

| 变量 | 作用 |
|---|---|
| `SENSENOVA_SERVER_BIN` | 服务端二进制路径 |
| `SENSENOVA_MODEL_DIR` | 权重目录 |
| `SENSENOVA_MODEL` | 语言模型 GGUF 文件名或路径 |
| `SENSENOVA_FM` | FM sidecar 文件名或路径 |

优先级：命令行 > 环境变量 > 默认值。

### 2.4 常用变体

```bash
# 换端口
python3 server.py --port 9000

# 显存不够，只上 39 层
python3 server.py -ngl 39

# 省显存：embd 和 output 放 CPU，速度、出图都不变
python3 server.py -ngl 43 \
  -ot '^token_embd\.weight$=cpu' -ot '^output\.weight$=cpu'

# 换权重：Q6_K 全 GPU（需要再省约 1.6 GiB，所以加 --fm-cpu）
python3 server.py --model .../SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q6_K.gguf \
  -ngl 43 -ot '^token_embd\.weight$=cpu' -ot '^output\.weight$=cpu' --fm-cpu

# 先看命令再起
python3 server.py --dry-run
```

---

## 3. client.py

### 3.1 六种协议

`--api` 决定打哪个端点：

| `--api` | 端点 | 用途 | 支持输入图 | think 文本位置 |
|---|---|---|---|---|
| `openai`（默认） | `POST /v1/images/generations` | OpenAI 文生图 | 否 | `data[0].revised_prompt` |
| `edits` | `POST /v1/images/edits` | OpenAI 图像编辑（multipart） | 是 | `data[0].revised_prompt` |
| `native` | `POST /generate` | 私有协议，omni-note 在用 | 是 | JSON 的 `think` 字段（b64_json）或 `X-Think-Text` 头（png） |
| `infographic` | `POST /v1/infographic/generate` | 私有协议，omni-note 在用 | 是 | JSON 的 `think` 字段 |
| `vqa` | `POST /v1/chat/completions` | OpenAI 视觉问答 | 是 | `choices[0].message.reasoning_content` |
| `interleave` | `POST /v1/interleave/generate` | 图文交错，边写文字边出图 | 是 | JSON 的 `think` 字段 |

服务端还提供 `GET /health`、`GET /v1/models`、`GET /props`、`GET /images/{file}`。

### 3.2 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--api NAME` | openai | `openai` / `edits` / `native` / `infographic` / `vqa` / `interleave` |
| `--host ADDR` | 127.0.0.1 | 服务地址 |
| `--port N` | 8888 | 服务端口 |
| `--model NAME` | sensenova-u1.5 | OpenAI 协议里的 `model` 字段，只是回显 |
| `--n N` | 1 | 生成张数，最大 4。只保存第一张 |
| `-p, --prompt TEXT` | 无 | 提示词文本，优先于 `--prompt-file` |
| `-f, --prompt-file PATH` | `prompts/菜谱制作.txt` | 提示词文件 |
| `-o, --output PATH` | `sensenova.png` | 输出 PNG |
| `-i, --image PATH` | 无 | 输入图，可重复。`edits` / `native` / `infographic` / `interleave` 生效 |
| `--width N` | 0 | 宽，32 的倍数。0 表示按协议默认 |
| `--height N` | 0 | 高，32 的倍数。0 表示按协议默认 |
| `--steps N` | 8 | 去噪步数。8step 蒸馏模型用 8 |
| `--seed N` | 42 | 随机种子，-1 表示随机 |
| `--cfg F` | 1.0 | 文本引导强度 |
| `--img-cfg F` | 1.0 | 图像引导强度，只在编辑时有效 |
| `--cfg-norm MODE` | none | `none` / `global` / `channel` |
| `--shift F` | 3.0 | timestep shift |
| `--think` | 关 | 先跑 think 推理再出图（vqa 没有这个开关；interleave 默认就开） |
| `--no-think` | 关 | 关掉 interleave 的 think，只用于 `--api interleave` |
| `--max-think-tokens N` | 1024 | think 的 token 预算（interleave 不用这个字段，预算看 `--max-new-tokens`） |
| `--input-max-pixels N` | 0 | 每张输入图的像素上限，0 表示自动 |
| `--auto-pixels N` | 1048576 | 自动尺寸时的输出像素 |
| `--think-output PATH` | 无 | 把 think 文本写到文件 |
| `--format FMT` | 按协议 | openai/edits/interleave: `b64_json`/`url`；native: `png`/`b64_json` |
| `--server-output-dir PATH` | 无 | infographic/interleave 协议的服务端输出目录，服务端要带 `--allow-output-dir` |
| `--max-new-tokens N` | 1024 | vqa 的回答预算（think 也占预算）/ interleave 的文字预算 |
| `--max-images N` | 3 | interleave 的出图上限，1-8 |
| `--temperature F` | 0.0 | vqa/interleave 采样温度，0 表示贪心 |
| `--top-p F` | 0.95 | vqa/interleave nucleus 采样 |
| `--top-k N` | 0 | vqa/interleave top-k 采样，0 表示关闭 |
| `--repeat-penalty F` | 1.0 | vqa/interleave 重复惩罚 |
| `--timeout N` | 600 | 请求超时秒数 |

尺寸规则：`--width`/`--height` 都不给时，`edits` 按第一张输入图的宽高比推导；`interleave` 有输入图时按首图宽高比推到 `--auto-pixels`，没有输入图就是 1024x1024；其余协议用 1024x1024。

### 3.3 文生图

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp/turingevo/sensenova

# 用提示词文件
python3 client.py -f prompts/菜谱制作.txt -o /tmp/dish.png

# 直接给提示词
python3 client.py -p "a red apple on a wooden table" -o /tmp/apple.png

# 指定尺寸、步数、种子
python3 client.py -p "a red apple" --width 1024 --height 1024 --steps 8 --seed 42 -o /tmp/apple.png
```

### 3.4 图像编辑（it2i）

```bash
# 单图编辑，尺寸按输入图宽高比自动推导
python3 client.py --api edits -i /tmp/sn_t2i.png -p "把背景换成雪山" -o /tmp/edited.png

# 指定输出尺寸
python3 client.py --api edits -i /tmp/sn_t2i.png -p "把背景换成雪山" \
    --width 1024 --height 1024 -o /tmp/edited.png

# 多图：prompt 里用 <image> 占位，按出现顺序对应输入图
python3 client.py --api edits -i a.png -i b.png \
    -p "Image-1:<image>
Image-2:<image>
把第一张的风格套到第二张上" -o /tmp/merged.png

# 保留原图的程度靠 --img-cfg
python3 client.py --api edits -i /tmp/sn_t2i.png -p "只换颜色" --img-cfg 1.5 -o /tmp/recolor.png
```

约束：`openai` 协议不接受输入图，传了会直接报错；编辑时服务端必须有带理解侧 vision 权重的 FM，否则报
`the FM sidecar has no vision weights for input images`。

### 3.5 think 模式

```bash
# think 文本打到 stdout
python3 client.py -p "一张介绍宋代点茶工艺的信息图" --think -o /tmp/infographic.png

# think 文本另外存一份
python3 client.py -p "一张介绍宋代点茶工艺的信息图" --think \
    --think-output /tmp/think.txt --max-think-tokens 2048 -o /tmp/infographic.png
```

think 文本按协议从不同位置取，客户端已经统一成 stdout 里的 `--- think ---` 区块。

### 3.6 视觉问答（VQA）

`-p` 是问题，`-i` 是输入图，可重复；输出是文字，`-o` 不起作用。

```bash
IMG=/media/wmx/soft1/AI-model/1-ART/SenseNova-U1/examples/vqa/data/images/menu.jpg

# 单图问答
python3 client.py --api vqa -i $IMG -p "菜单上最便宜的汤多少钱？"

# 多图问答
python3 client.py --api vqa -i a.jpg -i b.jpg -p "这两张图里的编号分别是多少？"

# 采样 + 放宽预算，think 单独存一份
python3 client.py --api vqa -i $IMG -p "把菜单转成 JSON" \
    --temperature 0.7 --top-p 0.9 --max-new-tokens 2048 --think-output /tmp/think.txt
```

响应格式（OpenAI chat completions，非流式）：

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "sensenova-u1.5",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "答案", "reasoning_content": "think 文本"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 533, "completion_tokens": 102, "total_tokens": 635}
}
```

说明：

- 只支持一轮 `user` 消息，`system` 消息会被忽略；多轮和 `stream=true` 会返回 400。
- 图片用 data URL（`data:image/jpeg;base64,...`）传，也接受裸 base64 和本机路径；`http(s)` 远端 URL 不支持。
- VQA 没有 think 开关（`--think` 只对出图协议有效）。8step 蒸馏权重做 VQA 时经常自己先输出一段 `<think>`，这是模型行为；`content` 里已经去掉 think，原始 think 在 `reasoning_content`，客户端见到就打印。
- `max_tokens` 是 think 与答案的合计预算，输出撞上它时 `finish_reason` 是 `length`，答案可能被截断，加大预算即可。
- 输入图默认按 `min(2048^2, 4096^2/n)` 缩放；要省内存/时间可以传 `input_max_pixels` 字段。

### 3.7 走私有协议（omni-note 那条路）

```bash
python3 client.py --api native -p "a red apple" -o /tmp/apple.png
python3 client.py --api infographic -p "a red apple" -o /tmp/apple.png
python3 client.py --api infographic -p "a red apple" \
    --server-output-dir /tmp/llama-sensenova-server -o /tmp/apple.png
```

### 3.8 图文交错（interleave）

模型在一条回复里边写文字边出图：文字流里遇到 `<img>` 就地跑一次扩散，把生成的图重新编码塞回上下文，再接着往下写。服务端和 CLI 两种入口都支持。

#### 服务端

`POST /v1/interleave/generate`，`client.py --api interleave`：

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp/turingevo/sensenova

# 默认 think 开，最多 3 张图，无输入图时 1024x1024
python3 client.py --api interleave \
    -p "画一张红苹果放在木桌上的图，再画一张切开后的样子，分别说明口感" \
    --max-images 2 --max-new-tokens 2048 -o /tmp/il.png

# 关 think，文字更短、出图更快
python3 client.py --api interleave -p "画一张红苹果放在木桌上" --no-think \
    --max-images 1 -o /tmp/il_fast.png

# 带参考图（客户端转成 data URL 发过去），尺寸按首图宽高比自动推导
python3 client.py --api interleave -i /tmp/apple.png \
    -p "把这张图扩成两格制作步骤，每格配一段说明" --max-images 2 -o /tmp/il_steps.png

# 图片走 url，服务端落盘；--server-output-dir 需要服务端带 --allow-output-dir
python3 client.py --api interleave -p "画一张红苹果" --format url \
    --server-output-dir /tmp/llama-sensenova-server -o /tmp/il_url.png
```

输出：

- stdout 打印 think（`--- think ---` 区块）和 think 之外的正文；`--think-output PATH` 另存 think。
- 图片按 `-o` 编号：`il.png`、`il_1.png`、`il_2.png`，数量不超过 `--max-images`。
- 末尾一行统计：`finish`、prompt / text token 数、图片数、服务端耗时。
- 一张图都没出时提示加大 `--max-new-tokens` 或关 think。

请求字段（JSON；`width`/`height` 都为 0 或缺省表示自动）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `prompt` | 必填 | 提示词 |
| `model` | `sensenova-u1.5` | 只做回显 |
| `width` / `height` | 0 | 输出尺寸，32 的倍数；都给 0 时自动 |
| `auto_pixels` | 1048576 | 自动尺寸时的输出像素 |
| `num_steps` | 8 | 每张图的去噪步数，1-100 |
| `seed` | 42 | -1 表示随机 |
| `timestep_shift` | 3.0 | timestep shift |
| `max_images` | 3 | 出图上限，1-8 |
| `max_new_tokens` | 4096 | 文字预算，think 也占；1-16384 |
| `think` | true | 交错默认开 |
| `temperature` / `top_k` / `top_p` / `repetition_penalty` | 0 / 0 / 0.95 / 1.0 | 文字采样 |
| `input_max_pixels` | 0 | 每张输入图的像素上限 |
| `image` / `images` | 无 | 路径、裸 base64 或 data URL；`client.py` 传 data URL |
| `response_format` | `b64_json` | `b64_json` / `url` |
| `output_dir` | 无 | 仅 `url` 模式，需服务端 `--allow-output-dir` |

响应（`b64_json`）：

```json
{
  "id": "interleave-...",
  "created": 1792953004,
  "model": "sensenova-u1.5",
  "think": "【模型内心独白】...",
  "text": "全文，含 think 与 <image> 标记",
  "answer": "去掉 think 的正文，保留 <image> 标记",
  "segments": [
    {"type": "text", "text": "..."},
    {"type": "image", "index": 0, "width": 1024, "height": 1024, "b64_json": "..."},
    {"type": "text", "text": "..."}
  ],
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 186, "text_tokens": 96, "images": 1, "total_tokens": 282}
}
```

- `segments` 有序，文字和图片按模型输出顺序排列；文字段取自 `text`，所以开了 think 时也可能包含 think 内容。
- `url` 模式把 `b64_json` 换成 `url`；请求带 `output_dir` 时换成 `image_path`（此时不再走 `/images/`）。
- `finish_reason=length` 表示文字预算用完，最后一段可能被截断。

**为什么单开私有端点**：OpenAI 的 images 响应只有 `data[]` 图片加一个 `revised_prompt`，装不下 text -> image -> text 的顺序；塞进 chat completions 又要客户端支持 assistant 消息里的图片。所以新增一个端点，旧协议不动。

#### CLI（不走服务）

同样的能力也可以直接跑 `llama-sensenova-cli`，适合本机调试：

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp
M=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp

./build-alldev/bin/llama-sensenova-cli \
  -m $M/SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf --fm $M/F16-fm.gguf \
  -p "画一张红苹果放在木桌上的图，再画一张切开后的样子，分别说明口感" \
  --interleave --max-images 2 --max-new-tokens 2048 --steps 8 \
  -o turingevo/out/interleave.png
```

`cli.sh` 的对应写法：

```bash
cd turingevo/SenseNova
./cli.sh -f prompts/omni-note.txt --interleave --max-images 2
./cli.sh -f prompts/omni-note.txt --interleave --no-think
./cli.sh -f prompts/omni-note.txt --interleave -i /tmp/apple.png
```

CLI 参数与原理见 `llama-sensenova-cli-使用说明.md` 5.2 节。注意 `cli.sh` 的 2720x1536 默认尺寸只用于单图模式，交错模式不给 `--width/--height` 时按首图宽高比或 1024x1024。

显存提醒：一张 1024x1024 图约 1024 个图像 token，KV 约 172 KB/token，三张图加几千文字 token 约 1.3 GiB KV，16 GB 卡够用；2720x1536 的多图会撞 `n_ctx_train = 16384`，别开。

---

## 4. 端到端联调

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp/turingevo/sensenova

# 1. 起服务（另开一个终端，或加 &）
python3 server.py &

# 2. 等就绪
until curl -sf http://127.0.0.1:8888/health >/dev/null; do sleep 1; done

# 3. 打一次文生图
python3 client.py -p "a red apple" -o /tmp/apple.png

# 4. 打一次编辑
python3 client.py --api edits -i /tmp/apple.png -p "换成雪山背景" --think -o /tmp/edited.png

# 5. 打一次问答
python3 client.py --api vqa -i /tmp/apple.png -p "图里是什么水果？"

# 6. 打一次图文交错（think 等得久，先用小预算试链路）
python3 client.py --api interleave -p "画一张红苹果，再画一张切开后的样子" \
    --no-think --max-images 2 -o /tmp/il.png

# 7. 停服务（Ctrl-C 也会干净退出）
kill %1
```

---

## 5. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `无法连接 http://127.0.0.1:8888` | 服务没起来或端口不对 | 先跑 `python3 server.py`，用 `--port` 对齐 |
| `服务返回 HTTP 400: ...` | 参数不合法 | 按错误信息改，比如步数范围 1-100、尺寸要 32 的倍数 |
| `服务返回 HTTP 500: ... no vision weights` | FM 缺理解侧 vision 权重 | 换 `F16-fm.gguf`（20 个张量） |
| `出来的图是花的或全灰` | 显存不够导致部分层溢出到 CPU，或步数太少 | 降 `-ngl`，或把 `--steps` 提到 8 |
| `edits 协议需要至少一张输入图` | 忘了 `-i` | 补上输入图 |
| `openai 协议不支持输入图` | 用错协议 | 换成 `--api edits` |
| vqa 返回 `remote image URLs are not supported` | 传了 http(s) 图片地址 | 改成本机路径或 data URL |
| vqa 返回 `only a single user message is supported` | 发了多轮 messages | 只保留最后一条 user 消息 |
| vqa 的 `finish_reason` 是 `length`、答案残缺 | 预算被 think 吃掉 | 加 `--max-new-tokens`，或用 `--temperature` 采样减少绕圈 |
| `cannot create context`，服务日志里 `cudaMalloc failed: out of memory` | 显存不够：VQA 的前缀（文字 + 图像 token）越长，prefill 计算缓冲越大 | 降 `-ngl`；或给 `--input-max-pixels` 限制输入图像素（如 1048576），减少图像 token |
| 客户端一直等 | 服务端还在加载模型 | 看服务终端日志，加载完 `/health` 才会返回 ok |

交错模式（3.8 节，服务端与 CLI 通用）的常见问题：

| 现象 | 原因 | 处理 |
|---|---|---|
| `finish_reason=length` / `(truncated)`，最后一段文字断了 | 文字预算用完 | 加大 `--max-new-tokens`（如 2048），或加 `--no-think` |
| 一张图都没出（服务端 `usage.images=0` / CLI no image 警告） | think 或正文把文字预算吃光，没走到 `<img>` | 同上；think 模式建议预算 2048 以上 |
| `--vqa and --interleave are mutually exclusive` | 两个模式一起传了 | 只留一个 |
| CLI 交错模式下 cfg 相关参数没反应 | 该模式固定 cfg = 1.0 | 正常现象，启动时会打 warning |
| `output_dir is disabled, start server with --allow-output-dir` | 请求里带了 `output_dir` | 去掉该字段，或服务端加 `--allow-output-dir` |
| `max_images must be between 1 and 8` | 请求的出图上限越界 | 改成 1-8 |

---

## 6. 与厂商 examples 的对应关系

厂商仓库 `SenseNova-U1/examples/` 下的五个示例与本仓库的对应：

| 厂商示例 | 本仓库对应 | 说明 |
|---|---|---|
| `t2i` | `client.py`（默认 `--api openai`） | 文生图，参数名按 llama.cpp 习惯，见 3.3 |
| `editing` | `client.py --api edits` | 图像编辑（it2i），见 3.4 |
| `vqa` | `client.py --api vqa` | 视觉问答，见 3.6 |
| `serving` | `server.py` + `client.py` | 常驻服务；本仓库多一个私有 `/generate` |
| `interleave` | `client.py --api interleave` / `llama-sensenova-cli --interleave` | 图文交错；服务端走私有 `/v1/interleave/generate`，见 3.8 |

### 6.1 interleave 的状态

`examples/interleave` 是图文交错生成：先自回归出文字，遇到 `<img>` 就地跑扩散生成一张图，把图嵌入塞回上下文再继续往下写。

两个入口都已支持（命令见 3.8）：

- 服务端：`POST /v1/interleave/generate`，响应用有序的 `segments` 数组表达 text/image 交错。
- CLI：`--interleave`，本机直跑，原理见 `llama-sensenova-cli-使用说明.md` 5.2 节。

两边共用同一条管线：cfg = 1.0 时厂商的三份 cache 退化成一条 KV 序列，图像 token 每步前滚、出图后按理解侧重新编码回灌，位置序列与厂商 `interleave_gen` 逐位一致。

服务端没有把它塞进 OpenAI 协议，是因为 `data[]` + 单个 `revised_prompt` 装不下 text -> image -> text 的顺序；单开私有端点可以保持旧协议完全不动。只要多图排版、不需要交错的场景，仍然建议用一次文生图加提示词描述版式。

---

## 7. 相关文件

| 路径 | 作用 |
|---|---|
| `server.py` | 启动服务端 |
| `client.py` | 调用服务端出图 |
| `cli.sh` | 直接调 CLI 的测试脚本，不走服务 |
| `prompts/` | 提示词样本 |
| `llama-sensenova-cli-使用说明.md` | CLI 的完整参数与原理 |
| `SenseNova-U1-模型支持说明.md` | 模型结构与参考实现对照 |
| `tools/sensenova-server/main.cpp` | 服务端实现（端点、协议、参数校验） |
| `tools/sensenova-u1/pipeline.cpp` | 推理管线（prompt 模板、think、CFG、去噪循环、VQA、interleave） |
