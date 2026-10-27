# Qwen3-TTS 服务使用说明

用 `llama-server` 跑 Qwen3-TTS 做语音合成的一对脚本：

| 脚本 | 作用 |
|-|-|
| `server.py` | 找模型、找二进制，启动 `llama-server` 并等 `/health` 就绪 |
| `client.py` | 调 `POST /tts` 合成语音，支持音色克隆、流式输出、长文本分段拼接 |

两个脚本只用 Python 标准库，Linux / macOS / Windows 都能跑。

---

## 1. 前置条件

| 项 | 说明 |
|-|-|
| 服务端二进制 | `build-alldev/bin/llama-server`（CUDA）或 `build/bin/llama-server`（CPU），也可用 `LLAMA_SERVER_BIN` 指定 |
| 语言模型 | `Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf`，走 `libllama` 加载 |
| 声码器 | `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf`，**必须成对加载**。波形重建（code2wav）和说话人编码器都在里面 |
| 显存 | `-np 1 -c 4096` 约 2.1 GiB；默认 4 槽会到 6.8 GiB 左右 |

权重默认在：

```
/media/wmx/soft2/huggingface_cache/models/audio-tts/Qwen3-TTS-12Hz-1.7B-Base-GGUF/
```

模型是 12.5 帧/秒，即 1 帧 = 80 ms 音频，后文的时长换算都按这个来。

---

## 2. server.py

### 2.1 最快启动

```bash
cd /media/wmx/ws1/workspace/ai/llama.cpp
python3 turingevo/tts/server.py
```

等价于：

```bash
build-alldev/bin/llama-server \
  -m .../Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf \
  --mmproj .../mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99 -c 4096 -np 1 --no-webui
```

看到这两行才算就绪：

```
就绪: http://127.0.0.1:8080
示例: curl -X POST http://127.0.0.1:8080/tts -F input=你好 -F lang=zh -F speaker_ref=@ref.wav -o out.wav
```

脚本前台运行，`Ctrl-C` 会把信号转给 `llama-server` 一起退出。

### 2.2 参数

| 参数 | 默认 | 说明 |
|-|-|-|
| `--host ADDR` | 127.0.0.1 | 监听地址 |
| `--port N` | 8080 | 监听端口，被占用时直接报错退出 |
| `--bin PATH` | 自动查找 | `llama-server` 路径 |
| `--model-dir PATH` | 见上文 | 模型目录 |
| `--model PATH` | 目录下的主模型 | 主模型 GGUF 全路径 |
| `--mmproj PATH` | 目录下的 `mmproj-*.gguf` | 声码器 GGUF 全路径 |
| `-ngl N` | 99 | offload 到 GPU 的层数 |
| `-c N` | 4096 | 总上下文，单槽为 `N / np`。要同时装下文本和每帧 1 个 token |
| `-np N` | 1 | 并行槽数。1 最省显存 |
| `-n N` | 0 | 默认音频帧上限，请求里的 `n_predict` 优先；0 表示用内置 512 帧 |
| `-t N` | 0 | CPU 线程数，0 表示自动 |
| `--wait N` | 180 | 等待就绪的秒数，0 表示不等 |
| `--ui` | 关 | 启用内置 WebUI，默认 `--no-webui` |
| `--dry-run` | 关 | 只打印要执行的命令 |

其余参数原样透传给 `llama-server`，例如 `--temp 0.9`、`--seed 42`、`-fa`。同名参数以你传的为准。

### 2.3 环境变量

| 变量 | 作用 |
|-|-|
| `LLAMA_SERVER_BIN` | `llama-server` 路径 |
| `TTS_MODEL_DIR` | 模型目录 |
| `TTS_MODEL` | 主模型路径 |
| `TTS_MMPROJ` | mmproj 路径 |

优先级：命令行 > 环境变量 > 默认值。

### 2.4 常用变体

```bash
# 换端口，默认帧上限放到 2000（约 160 s）
python3 turingevo/tts/server.py --port 9002 -n 2000

# 长文本配大上下文
python3 turingevo/tts/server.py -c 8192

# 手动指定权重
python3 turingevo/tts/server.py \
  --model /path/Qwen3-TTS-12Hz-1.7B-Base-Q6_K.gguf \
  --mmproj /path/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf

# 先看命令再起
python3 turingevo/tts/server.py --dry-run
```

---

## 3. client.py

### 3.1 参数

| 参数 | 默认 | 说明 |
|-|-|-|
| `-p, --prompt TEXT` | 空 | 要合成的文本，优先于 `-f` |
| `-f, --prompt-file PATH` | 空 | 文本文件 |
| `--lang LANG` | 空 | `zh` / `en` / `de` / `it` / `pt` / `es` / `ja` / `ko` / `fr` / `ru`，留空用模型默认（en） |
| `--speaker PATH` | 空 | 参考音频，克隆音色（wav / mp3） |
| `-o, --output PATH` | `out.wav` / `out.pcm` | 输出文件 |
| `--format {wav,pcm}` | wav | wav 是 24 kHz 单声道 s16le；pcm 是 24 kHz 单声道 float32，无头 |
| `--stream` | 关 | 边生成边写文件，首包时间进入输出 |
| `--split N` | 0 | 按标点切成每段最多 N 字，逐段合成后拼接；0 表示不切 |
| `-n, --n-predict N` | 0 | 音频帧上限，0 用服务端默认（512 帧约 41 s） |
| `--top-k N` | 0 | 声学码采样 top-k，0 用模型默认 |
| `--top-p F` | 0 | nucleus 采样，0 用模型默认 |
| `--repeat-penalty F` | 0 | 重复惩罚，0 用服务端默认（1.05） |
| `--seed N` | 无 | 固定后结果可复现 |
| `--url URL` | 空 | 完整服务地址，优先于 host/port |
| `--host ADDR` | 127.0.0.1 | 服务地址 |
| `--port N` | 8080 | 服务端口 |
| `--timeout N` | 300 | 请求超时秒数 |

### 3.2 基础合成

```bash
# 中文
python3 turingevo/tts/client.py -p "你好，世界。" --lang zh -o out.wav

# 英文
python3 turingevo/tts/client.py -p "Hello world, this is a test." --lang en -o out.wav

# 从文件读
python3 turingevo/tts/client.py -f script.txt --lang zh -o out.wav

# 固定种子，便于复现和比对
python3 turingevo/tts/client.py -p "同一段文本" --lang zh --seed 42 -o out.wav
```

输出示例：

```
已保存: out.wav
音频: 2.96s, 24000 Hz, 142124 字节
耗时: 1.20s, 2.5x 实时
```

不给 `--speaker` 时音色随机，服务端日志会警告 `no speaker reference provided`。

### 3.3 音色克隆

```bash
python3 turingevo/tts/client.py -p "这段话用参考音频的音色读。" \
  --lang zh --speaker ref.wav -o clone.wav
```

参考音频建议 3-15 秒、单人、无背景音乐。给 `--speaker` 时客户端用 multipart 上传 `speaker_ref`；不给则走 JSON。

### 3.4 流式与 PCM

```bash
# 流式：边生成边落盘，输出会带首包时间
python3 turingevo/tts/client.py -p "流式输出测试。" --lang zh --stream -o stream.wav

# PCM：无头 float32，交给下游播放器或拼接
python3 turingevo/tts/client.py -p "PCM 输出。" --lang zh --format pcm -o out.pcm
```

流式 WAV 的服务端头里长度字段是 `0xFFFFFFFF`，客户端收完会按实际大小回填，ffprobe 能正常读出时长。直接用 `curl` 落盘的流式 WAV 头仍是占位值，需要自己修。

### 3.5 长文本

单次请求默认最多 512 帧，即 40.96 s，超出的部分会被截断。两种解法：

```bash
# 1) 抬高单次上限：-n 是帧数，2000 帧约 160 s
python3 turingevo/tts/client.py -f long.txt --lang zh -n 2000 -o out.wav

# 2) 按句切段、逐段合成、拼接成一个文件，每段 150 字（约 30 s）
python3 turingevo/tts/client.py -f long.txt --lang zh --split 150 -o out.wav

# 音色克隆同样支持
python3 turingevo/tts/client.py -f long.txt --lang zh --speaker ref.wav --split 150 -o out.wav
```

选择建议：

- 一两分钟的稿子用 `-n`，整段一次生成，语调连续；服务端上下文建议同时抬到 `-c 8192`。
- 再长用 `--split`。除上限之外，模型在超长文本上还会自己提前收尾（实测 1119 字单请求只说了约 2.4 遍就 EOS），分段能绕开这个问题；代价是段与段之间语调不完全连续。
- 输出贴着 40.9 s 时客户端会提示「已到 512 帧上限」。

`--split` 按 `。！？!?；…` 和换行切句，超长句再按逗号/空格硬切，每段用同一个 `--seed`。

---

## 4. HTTP 接口

### 4.1 请求

`POST /tts`，JSON 或 multipart/form-data：

| 字段 | 必填 | 说明 |
|-|-|-|
| `input` | 是 | 要合成的文本，别名 `prompt` |
| `lang` | 否 | 语种代码，留空用模型默认 |
| `speaker_ref` | 否 | multipart 文件字段，参考音频 |
| `speaker_ref_b64` | 否 | JSON 字段，参考音频 base64 |
| `response_format` | 否 | `wav`（默认）或 `pcm` |
| `stream` | 否 | 流式返回音频数据 |
| `n_predict` | 否 | 音频帧上限，别名 `max_tokens`；不传用服务端值，再没有就是 512 |
| `top_k` | 否 | 声学码采样 top-k |
| `top_p` | 否 | nucleus 采样 |
| `repeat_penalty` | 否 | 重复惩罚，默认 1.05 |
| `seed` | 否 | 随机种子，-1 随机 |

```bash
# JSON
curl -X POST http://127.0.0.1:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"input":"你好，世界。","lang":"zh"}' -o out.wav

# multipart + 参考音频
curl -X POST http://127.0.0.1:8080/tts \
  -F input="你好，世界。" -F lang=zh -F speaker_ref=@ref.wav -o out.wav
```

### 4.2 响应

| `response_format` | Content-Type | 说明 |
|-|-|-|
| `wav` | `audio/wav` | 24 kHz 单声道 s16le，标准 44 字节头 |
| `pcm` | `audio/pcm;rate=24000;encoding=float;bits=32` | 无头 float32 |
| `pcm` + `stream` | 同上，chunked | 边生成边下发 |

`stream=true` 时 WAV 头长度为 `0xFFFFFFFF`（chunked 传输），需要调用方自己回填。

注意：`temperature` 不是 `/tts` 的请求级参数，传了会被忽略。温度只能通过服务端启动参数 `--temp` 调整。

### 4.3 能力边界

- 这个服务只提供 `/tts`。转写接口 `/v1/audio/transcriptions` 需要音频编码器，这里是生成侧模型，会返回 501。
- `/v1/chat/completions` 对 TTS 模型不可用（501 `this server does not support chat/completions`）。
- 语种和文本都在请求里给，不需要 chat template。

---

## 5. 本机实测

RTX 4070 Ti SUPER 16 GB，CUDA 构建，`-ngl 99 -c 4096 -np 1`：

| 文本 | 音频时长 | 生成耗时 | 倍率 |
|-|-|-|-|
| 英文短句 | 2.96 s | 1.20 s | 2.5x |
| 372 字中文（`-n 2000`） | 71.68 s | 14.0 s | 5.1x |
| 1119 字中文（`--split 150`） | 199.84 s | 38.7 s | 5.2x |

其他：

- 模型加载约 1.2 s。
- 固定 `--seed` 后两次输出字节完全一致；加 `--speaker` 波形明显不同。
- 与 ASR 服务同开，两边都 `-np 1`，合计显存约 4.7 GiB（含桌面约 1 GiB）。

---

## 6. 排错

| 现象 | 原因与处理 |
|-|-|
| 音频正好 40.96 s，话没说完 | 服务端默认 512 帧上限，加 `-n` 或用 `--split` |
| `error: 找不到 llama-server` | 先编译，或用 `--bin` / `LLAMA_SERVER_BIN` 指定 |
| `error: ... 已被占用` | 端口被占了，换 `--port` |
| `error: ... 下没有 mmproj-*.gguf` | 只拷了主模型，声码器缺失 |
| `HTTP 501 this server does not support audio generation` | 服务是用 ASR 或纯文本模型起的 |
| 音色每次都不同 | 没给 `--speaker`，参考音频决定音色 |
| 中文读出英文口音 | `--lang` 与文本不一致，或参考音频语种不同 |
| `--temp` 无效 | 请求级不支持温度，用服务端 `--temp` |
| 流式 WAV 时长显示异常 | 直接 `curl` 落盘没回填头；用 `client.py` 或自行修改 RIFF/data 长度 |
| 长文本中途断掉 | 单请求遇到模型 EOS 或上下文不足，改用 `--split`，或抬高服务端 `-c` |
