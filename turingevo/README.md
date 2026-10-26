# TuringEvo 发行版配套目录

本目录存放 TuringEvo 发行版新增的文档、脚本与样例。上游 llama.cpp 的代码保持原样，这里不参与构建。

## 目录结构

| 路径 | 内容 |
|-|-|
| [sensenova/](sensenova/) | SenseNova 系列模型的文档、CLI 与客户端脚本、prompt 样例 |
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
