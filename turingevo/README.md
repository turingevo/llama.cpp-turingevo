# TuringEvo 发行版配套目录

本目录存放 TuringEvo 发行版新增的文档、脚本与样例。上游 llama.cpp 的代码保持原样，这里不参与构建。

## 目录结构

| 路径 | 内容 |
|-|-|
| [sensenova/](sensenova/) | SenseNova 系列模型的文档、CLI 与客户端脚本、prompt 样例 |
| [asr/](asr/) | Qwen3-ASR 语音转写：llama-server 启动脚本、客户端与[使用说明](asr/README.md) |
| [tts/](tts/) | Qwen3-TTS 语音合成：llama-server 启动脚本、客户端与[使用说明](tts/README.md) |
| [shared/](shared/) | 跨模型脚本：编译、量化、压测、部署、embedding / reranker 服务 |
| [docs/](docs/) | 跨模型文档：编译目标说明、同步上游 |
| [android/](android/) | Android 构建与部署说明 |
| `out/` | 本地测试输出，不入库 |

## 新增模型

1. 架构实现按上游布局放 `src/models/<model>.cpp`，转换器放 `conversion/`
2. 需要命令行或服务端时，加 `tools/<model>/`
3. 在本目录建 `<family>/`，放该系列的文档、脚本与 prompt 样例，并在下面登记
4. 跨模型脚本改 [shared/](shared/) 里的那一份，用参数传模型路径，不复制

## 模型

| 模型 | 目录 | 说明 |
|-|-|-|
| SenseNova-U1 | [sensenova/](sensenova/) | 文生图、图像编辑、视觉问答、图文交错 |
| Qwen3-ASR-0.6B | [asr/](asr/) | 语音转写，`/v1/audio/transcriptions` |
| Qwen3-TTS-1.7B | [tts/](tts/) | 语音合成与音色克隆，`/tts` |

## 语音服务

两个目录各有一对脚本：`server.py` 启动 llama-server，`client.py` 调用接口，只用 Python 标准库。

```sh
# 转写（默认 127.0.0.1:8081）
python3 turingevo/asr/server.py &
python3 turingevo/asr/client.py -f test.wav --timing

# 合成（默认 127.0.0.1:8080）
python3 turingevo/tts/server.py &
python3 turingevo/tts/client.py -p "你好，世界" --lang zh -o out.wav

# 长文本：单次默认上限 512 帧（约 41s），两种解法
python3 turingevo/tts/client.py -f long.txt --lang zh -n 2000 -o out.wav
python3 turingevo/tts/client.py -f long.txt --lang zh --split 150 -o out.wav
```

`-n` 直接抬高单次生成上限，配合服务端 `-c 8192` 更稳；`--split N` 按标点切成每段不超过 N 字，逐段合成后拼接，规避模型在超长文本上提前收尾，代价是段与段之间的语调不连续。

参数、接口、实测数据和排错见 [asr/README.md](asr/README.md) 与 [tts/README.md](tts/README.md)。

启动脚本默认 `-ngl 99 -np 1`，ASR 用 `-c 8192`、TTS 用 `-c 4096`；模型目录用 `--model-dir` 或环境变量 `ASR_MODEL_DIR` / `TTS_MODEL_DIR` 覆盖。
