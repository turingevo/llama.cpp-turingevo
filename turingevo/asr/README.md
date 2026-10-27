# Qwen3-ASR 服务使用说明

用 `llama-server` 跑 Qwen3-ASR 做语音转写的一对脚本：

| 脚本 | 作用 |
|-|-|
| `server.py` | 找模型、找二进制，启动 `llama-server` 并等 `/health` 就绪 |
| `client.py` | 调 `POST /v1/audio/transcriptions` 转写音频，可多文件、可存文本 |

两个脚本只用 Python 标准库，Linux / macOS / Windows 都能跑。

---

## 1. 前置条件

| 项 | 说明 |
|-|-|
| 服务端二进制 | `build-alldev/bin/llama-server`（CUDA）或 `build/bin/llama-server`（CPU），也可用 `LLAMA_SERVER_BIN` 指定 |
| 语言模型 | `Qwen3-ASR-0.6B-Q8_0.gguf`，走 `libllama` 正常加载 |
| 音频编码器 | `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf`，**必须成对加载**，没有它就没有音频输入能力 |
| 显存 | `-np 1 -c 8192` 约 2.6 GiB；默认 4 槽会到 6.8 GiB 左右 |

权重默认在：

```
/media/wmx/soft2/huggingface_cache/models/audio-asr/Qwen3-ASR-0.6B-GGUF/
```

模型目录里必须有且各只有一个主模型和 `mmproj-*.gguf`，脚本才能自动识别；多个同名文件时用 `--model` / `--mmproj` 指定。

---

## 2. server.py

### 2.1 最快启动

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp
python3 turingevo/asr/server.py
```

等价于：

```bash
build-alldev/bin/llama-server \
  -m .../Qwen3-ASR-0.6B-Q8_0.gguf \
  --mmproj .../mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
  --host 127.0.0.1 --port 8081 -ngl 99 -c 8192 -np 1 --no-webui
```

看到这两行才算就绪：

```
就绪: http://127.0.0.1:8081
示例: curl -X POST http://127.0.0.1:8081/v1/audio/transcriptions -F file=@test.wav -F language=zh
```

脚本前台运行，`Ctrl-C` 会把信号转给 `llama-server` 一起退出。

### 2.2 参数

| 参数 | 默认 | 说明 |
|-|-|-|
| `--host ADDR` | 127.0.0.1 | 监听地址 |
| `--port N` | 8081 | 监听端口，被占用时直接报错退出 |
| `--bin PATH` | 自动查找 | `llama-server` 路径 |
| `--model-dir PATH` | 见上文 | 模型目录 |
| `--model PATH` | 目录下的主模型 | 主模型 GGUF 全路径 |
| `--mmproj PATH` | 目录下的 `mmproj-*.gguf` | 音频编码器 GGUF 全路径 |
| `-ngl N` | 99 | offload 到 GPU 的层数 |
| `-c N` | 8192 | 总上下文，单槽为 `N / np`。约 15.5 token/秒音频，8192 够 8 分钟左右 |
| `-np N` | 1 | 并行槽数。1 最省显存，压测并发时再调大 |
| `-t N` | 0 | CPU 线程数，0 表示自动 |
| `--wait N` | 180 | 等待就绪的秒数，0 表示不等 |
| `--ui` | 关 | 启用内置 WebUI，默认 `--no-webui` |
| `--dry-run` | 关 | 只打印要执行的命令 |

其余参数原样透传给 `llama-server`，例如 `--flash-attn`、`-fa`、`--verbose`。同名参数以你传的为准，不会出现两个 `-ngl`。

### 2.3 环境变量

| 变量 | 作用 |
|-|-|
| `LLAMA_SERVER_BIN` | `llama-server` 路径 |
| `ASR_MODEL_DIR` | 模型目录 |
| `ASR_MODEL` | 主模型路径 |
| `ASR_MMPROJ` | mmproj 路径 |

优先级：命令行 > 环境变量 > 默认值。

### 2.4 常用变体

```bash
# 换端口，两个槽
python3 turingevo/asr/server.py --port 9001 -np 2

# 换模型目录
python3 turingevo/asr/server.py --model-dir /path/to/models

# 目录里文件多，手动指定
python3 turingevo/asr/server.py \
  --model /path/Qwen3-ASR-0.6B-Q4_K_M.gguf \
  --mmproj /path/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf

# 先看命令再起
python3 turingevo/asr/server.py --dry-run
```

---

## 3. client.py

### 3.1 参数

| 参数 | 默认 | 说明 |
|-|-|-|
| `-f, --file PATH` | 无（必填） | 音频文件，可重复。wav / mp3 / flac |
| `--language LANG` | 空 | 语种提示，如 `zh` / `en` / `ja`；留空自动识别 |
| `--prompt TEXT` | 空 | 附加提示词或上下文，服务端会拼进识别提示里 |
| `--url URL` | 空 | 完整服务地址，优先于 host/port |
| `--host ADDR` | 127.0.0.1 | 服务地址 |
| `--port N` | 8081 | 服务端口 |
| `--timeout N` | 120 | 单次请求超时秒数 |
| `--raw` | 关 | 保留 `language Chinese<asr_text>` 前缀 |
| `--json` | 关 | 打印服务端原始 JSON |
| `--timing` | 关 | 额外输出耗时和实时倍率（stderr） |
| `-o, --output PATH` | 空 | 把转写文本写入文件 |

### 3.2 基本用法

```bash
# 单文件
python3 turingevo/asr/client.py -f test.wav

# 看耗时和倍率
python3 turingevo/asr/client.py -f test.wav --timing
# [timing] test.wav: 0.113s, 5.55s 音频, 49.1x
# 欢迎大家来体验达摩院推出的语音识别模型。

# 指定语种
python3 turingevo/asr/client.py -f test.wav --language en
```

默认输出是去掉前缀后的纯文本。Qwen3-ASR 原始输出形如 `language Chinese<asr_text>...`，服务端不处理这个前缀，客户端在 `<asr_text>` 处切开。

### 3.3 批量与输出

```bash
# 多文件：每行 path<TAB>text
python3 turingevo/asr/client.py -f a.wav -f b.wav

# 多文件并保存
python3 turingevo/asr/client.py -f a.wav -f b.wav -o result.txt

# 单文件保存
python3 turingevo/asr/client.py -f a.wav -o a.txt

# 服务端原始 JSON（含 usage）
python3 turingevo/asr/client.py -f a.wav --json

# 保留语言前缀
python3 turingevo/asr/client.py -f a.wav --raw
```

任意一个文件失败时继续处理其余文件，最后返回码为 1。

### 3.4 语言与提示词

`--language` 不是解码约束，而是拼进提示词的一段文本（服务端追加 `(language: zh)`）。口音重或语种混杂时，自动识别通常更好，强制指定反而可能误判。

`--prompt` 适合给领域词、人名、专有名词提供上下文，例如：

```bash
python3 turingevo/asr/client.py -f meeting.wav --prompt "会议涉及：昌学卡、换位思考、活动课"
```

---

## 4. HTTP 接口

### 4.1 请求

`POST /v1/audio/transcriptions`，multipart/form-data：

| 字段 | 必填 | 说明 |
|-|-|-|
| `file` | 是 | 音频文件内容 |
| `prompt` | 否 | 识别提示，留空用默认 "Transcribe audio to text" |
| `language` | 否 | 语种提示 |
| `response_format` | 否 | 只支持 `json`，传别的直接报错 |
| `stream` | 否 | `true` 返回 SSE，`transcript.text.delta` 增量事件 |
| `max_tokens` | 否 | 转写文本的 token 上限 |
| `temperature` | 否 | 采样温度 |

```bash
curl -X POST http://127.0.0.1:8081/v1/audio/transcriptions \
  -F file=@test.wav -F language=zh
```

### 4.2 响应

```json
{
  "type": "transcript.text.done",
  "text": "language Chinese<asr_text>欢迎大家来体验达摩院推出的语音识别模型。",
  "usage": {
    "type": "tokens",
    "input_tokens": 171,
    "output_tokens": 30,
    "total_tokens": 201,
    "input_tokens_details": {"cached_tokens": 170}
  }
}
```

兼容 OpenAI 的转写路径，但响应体是 OpenAI 新版的 `transcript.text.done` 形状，**没有** `segments`、时间戳、`duration` 字段。需要分段时间戳得用带时间戳的 ASR 模型，当前 GGUF 不提供。

### 4.3 限制

- 只有带音频编码器的服务能转写：没加载 mmproj 的服务返回 501 `The current model does not support audio input.`
- 同一份音频走 `/v1/chat/completions` + `input_audio` 也能转写，但返回的是未清洗的原始文本，带 `language ...<asr_text>` 前缀。
- Qwen3-ASR 只有转写能力，没有 `--mmproj` 的纯文本模型服务无法加载音频。

---

## 5. 长音频与显存

- 每秒音频约占 15.5 个 token。`-c 8192` 单槽约 8 分钟，`-c 16384` 约 17 分钟。
- `-c` 是总上下文，单槽为 `c / np`。开 2 槽而总上下文不变时，每槽能处理的音频长度减半。
- 更长的音频建议自己按静音或固定时长切片，逐段转写后拼接，避免单次请求吃满上下文。
- 显存：`-np 1 -c 8192` 实测 2.6 GiB 左右；用默认 4 槽会到 6.8 GiB。与 TTS 服务同开时建议两边都用 `-np 1`。

---

## 6. 本机实测

RTX 4070 Ti SUPER 16 GB，CUDA 构建，`-ngl 99 -c 8192 -np 1`：

| 音频 | 时长 | 转写耗时 | 倍率 |
|-|-|-|-|
| `jfk.wav`（英文） | 11.00 s | 0.092 s | 119.8x |
| `asr_example.wav`（中文） | 5.55 s | 0.074 - 0.113 s | 49 - 75x |
| 200 s 合成音频 | 199.84 s | 2.902 s | 68.9x |

模型加载约 1.2 s。

---

## 7. 排错

| 现象 | 原因与处理 |
|-|-|
| `error: 找不到 llama-server` | 先编译，或用 `--bin` / `LLAMA_SERVER_BIN` 指定 |
| `error: ... 已被占用` | 端口被别的服务占了，换 `--port` |
| `error: ... 下没有 mmproj-*.gguf` | 只拷了主模型，音频编码器缺失 |
| `error: 连不上 http://...` | 服务没起或端口不对，先 `curl http://127.0.0.1:8081/health` |
| `HTTP 501 this server does not support audio input` | 服务是用纯文本模型或 TTS 模型起的 |
| 转写结果多出 `language ...<asr_text>` | 用了 `--raw` 或直接调 HTTP；客户端默认会剥掉 |
| 长音频只出来一段 | 单槽上下文不够，抬高 `-c` 或切段 |
