# SenseNova

SenseNova 系列模型的文档、脚本与样例。U1 的模型实现与工具链仍在 llama.cpp 标准目录：

| 路径 | 内容 |
|-|-|
| `src/models/sensenova-u1.cpp` | 模型架构 |
| `conversion/sensenova_u1.py` | HF safetensors 转 GGUF |
| `tools/sensenova-u1/` | `llama-sensenova-cli` |
| `tools/sensenova-server/` | `llama-sensenova-server` |

## 文档

| 文件 | 内容 |
|-|-|
| [SenseNova-U1-模型支持说明.md](SenseNova-U1-模型支持说明.md) | 架构、张量、RoPE、图文交错流程 |
| [SenseNova-U1-量化方法.md](SenseNova-U1-量化方法.md) | 量化类型选择与实测 |
| [llama-sensenova-cli-使用说明.md](llama-sensenova-cli-使用说明.md) | CLI 参数、模式与示例 |
| [server-client-使用说明.md](server-client-使用说明.md) | 服务端端点与客户端用法 |

## 脚本与样例

| 路径 | 内容 |
|-|-|
| `cli.sh` | CLI 调用封装：文生图、图像编辑、问答、图文交错 |
| `server.py` | 服务启动脚本 |
| `client.py` | 六种协议的服务端测试客户端 |
| `compress_prompt.py` | prompt 压缩 |
| `prompts/` | 测试用 prompt 与参数样例 |
| [modelscope/](modelscope/) | ModelScope 发布用的模型卡（`README.md` 直接作为仓库根模型卡） |

以后新增 U2 等型号：在本目录下按型号建子目录（例如 `u2/`），系列公共内容留在本层。
