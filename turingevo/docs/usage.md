# llama.cpp 编译目标与可执行文件说明

本文说明 llama.cpp 编译后 `bin/` 目录里的产物：CMake 目标名、二进制名、构建开关、安装位置。本仓库新增的 `llama-sensenova-cli` 的构建与用法见 [llama-sensenova-cli-使用说明.md](../sensenova/llama-sensenova-cli-使用说明.md)。

> **验证状态（2026-10-25）**
>
> - 目标清单来自当前 `turingevo` 分支的 CMake 定义和 `cmake --build build --target help`。
> - **没有执行完整的 `all` 目标。** `build/` 目前只有 3 个可执行文件（其余是共享库），`build-alldev/` 有 14 个可执行文件。本文对未编译目标的描述是“定义上会生成什么”，不代表本机已经构建通过。
> - `llama-sensenova-cli` 已单独编译，并在 CPU、CUDA 两个构建目录里运行过。
> - `turingevo/docs/targets` 是旧版 `make help` 快照，缺少 `llama-sensenova-cli` 等新目标，以本文为准。

---

## 1. 编译目标、二进制与安装位置

### 1.1 命名规则

- 工具目标名通常就是二进制名，例如目标 `llama-cli` 生成 `bin/llama-cli`。
- `*-impl` 是内部库，不是可执行文件，例如 `llama-cli-impl`、`llama-server-impl`、`llama-perplexity-impl`、`llama-bench-impl`。
- 库目标：`ggml`、`ggml-base`、`ggml-cpu`、`llama`、`llama-common`、`llama-common-base`、`llama-diffusion`、`mtmd`、`cpp-httplib`、`server-context`、`vendor-hash`。
- 特殊目标 `llama-app` 的二进制名是 `llama`，由 `set_target_properties(... OUTPUT_NAME llama)` 指定。
- `llama-ui`、`llama-ui-assets` 是 Web UI 的构建内部目标，不单独生成命令行工具。
- 元目标：`all`（默认）、`install`、`clean`、`test`、`rebuild_cache`、`edit_cache`。

### 1.2 输出与安装

| 项目 | 位置 |
|---|---|
| 构建产物 | `<build-dir>/bin/` |
| 安装位置 | `<CMAKE_INSTALL_PREFIX>/bin/` |
| 当前 CPU 构建 | `build/bin/` |
| 当前 CUDA 构建 | `build-alldev/bin/` |

相关 CMake 开关：

| 开关 | 默认值 | 作用 |
|---|---|---|
| `LLAMA_BUILD_TOOLS` | standalone 时 ON | 是否编译 `tools/` 下的工具 |
| `LLAMA_TOOLS_INSTALL` | standalone 时 ON，iOS 为 OFF | 工具是否带 `install(TARGETS ...)` |
| `LLAMA_BUILD_EXAMPLES` | standalone 时 ON | 是否编译 `examples/` |
| `LLAMA_BUILD_TESTS` | standalone 时 ON | 是否编译 `test-*` |
| `LLAMA_BUILD_SERVER` | standalone 时 ON | 是否编译 `llama-server` 及 UI 相关目标 |
| `LLAMA_BUILD_APP` | standalone 时 ON | 是否编译统一入口 `llama` |
| `LLAMA_BUILD_UI` | OFF | 是否构建内嵌 Web UI |
| `GGML_RPC` | OFF | 是否编译 RPC 工具 |
| `LLAMA_CONNECT` | OFF | 是否编译 `llama-connect` |

安装注意：`cmake --install` 会安装所有已启用的 install 目标。如果只编译了部分目标，安装阶段会报 `file INSTALL cannot find ...`，因为未构建的目标在 `bin/` 里不存在。`turingevo/shared/build-all-dev.sh` 因此没有使用 `cmake --install`，而是按 `TARGETS` 列表把实际生成的二进制 `cp` 到安装目录。

### 1.3 当前两个构建目录的实测配置

| 配置 | `build/` | `build-alldev/` |
|---|---|---|
| `BUILD_SHARED_LIBS` | ON | OFF |
| `GGML_CUDA` | OFF | ON |
| `GGML_BLAS` | OFF | ON（OpenBLAS） |
| `GGML_NATIVE` | ON | ON |
| `GGML_OPENMP` | ON | ON |
| `CMAKE_BUILD_TYPE` | Release | Release |
| `LLAMA_BUILD_TOOLS` | ON | ON |
| `LLAMA_BUILD_TESTS` | ON | OFF |
| `LLAMA_BUILD_UI` | ON | OFF |
| `LLAMA_TOOLS_INSTALL` | ON | ON |

当前 `build/bin/` 的实际内容（19 个文件，含共享库）：

```
libggml-base.so*  libggml-cpu.so*  libggml.so*  libllama-common.so*
libllama-quantize-impl.so  libllama.so*
llama-quantize  llama-sensenova-cli  llama-tokenize
```

当前 `build-alldev/bin/` 的实际内容（14 个可执行文件）：

```
llama-batched  llama-batched-bench  llama-bench  llama-cli
llama-diffusion-cli  llama-gguf  llama-gguf-hash  llama-gguf-split
llama-mtmd-cli  llama-perplexity  llama-quantize  llama-sensenova-cli
llama-server  llama-tokenize
```

CUDA 版是静态链接 llama/ggml，但仍然依赖系统库：`libopenblas`、`libgomp`、`libcudart`、`libcublas`、`libcuda`、`libnccl`。CPU 版 `llama-sensenova-cli` 是动态链接，运行时需要 `build/bin/` 下的 `libllama*.so`、`libggml*.so`；把二进制拷到别处时要连同这些库一起拷，或设置 `LD_LIBRARY_PATH`。

---

## 2. 构建命令

### 2.1 配置

CPU 构建（对应 `build/`）：

```bash
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON
```

CUDA + OpenBLAS 构建（对应 `build-all-dev.sh` 的 `build-alldev/`）：

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
```

### 2.2 单目标构建（已执行）

```bash
cmake --build build --config Release -j"$(nproc)" --target llama-sensenova-cli
cmake --build build-alldev --config Release -j"$(nproc)" --target llama-sensenova-cli
```

多个目标可以重复写 `--target`，`turingevo/shared/build-all-dev.sh` 就是这样做的。

### 2.3 完整构建（未执行）

```bash
# 等价于 --target all
cmake --build build --config Release -j"$(nproc)"
```

注意：这里的“完整”只是 `all` 目标。它会编译 tests、examples、server 等所有已启用目标，耗时和磁盘占用都远大于单目标构建。

### 2.4 安装

```bash
cmake --install build --prefix /your/prefix --config Release
```

只安装了部分目标时不要直接用 `cmake --install`，按需要拷贝：

```bash
cp build/bin/llama-sensenova-cli /your/prefix/bin/
```

---

## 3. 常用可执行文件

### 3.1 推理与对话

- `llama-cli`：主要命令行推理工具，支持聊天、补全、指定模板。
- `llama-server`：HTTP 服务器，提供 OpenAI 兼容接口。
- `llama`（目标 `llama-app`）：统一入口二进制。
- `llama-simple`：最小推理示例。
- `llama-simple-chat`：简单聊天示例。
- `llama-completion`：纯文本补全工具，参数比 `llama-cli` 简单。
- `llama-batched`：批量推理。
- `llama-parallel`：并行解码多个独立序列。
- `llama-speculative`：推测解码。
- `llama-speculative-simple`：推测解码简化版。
- `llama-lookahead`：lookahead decoding。
- `llama-lookup`：n-gram 缓存推测解码。
- `llama-lookup-create`：创建 n-gram 查找缓存。
- `llama-lookup-merge`：合并查找缓存。
- `llama-lookup-stats`：查看查找缓存统计。
- `llama-retrieval`：检索增强生成示例。
- `llama-passkey`：长上下文 passkey 检索测试。
- `llama-eval-callback`：推理回调机制示例。

### 3.2 多模态与生成

- `llama-mtmd-cli`：多模态理解（视觉、音频输入）。
- `llama-mtmd-debug`：多模态调试工具。
- `llama-gemma3-cli`：Gemma 3 多模态示例。
- `llama-llava-cli`：LLaVA 多模态示例。
- `llama-qwen2vl-cli`：Qwen2-VL 多模态示例。
- `llama-minicpmv-cli`：MiniCPM-V 多模态示例。
- `llama-diffusion-cli`：扩散模型推理工具。
- `llama-sensenova-cli`：SenseNova-U1 文生图，详见 [llama-sensenova-cli-使用说明.md](../sensenova/llama-sensenova-cli-使用说明.md)。
- `llama-tts`：文本转语音。

### 3.3 嵌入与评估

- `llama-embedding`：提取文本嵌入向量。
- `llama-perplexity`：计算困惑度。
- `llama-bench`：推理性能基准。
- `llama-batched-bench`：批量推理性能基准。
- `llama-tokenize`：分词器测试，输出 token 序列。

### 3.4 模型处理与量化

- `llama-quantize`：模型量化。
- `llama-imatrix`：计算重要性矩阵。
- `llama-finetune`：微调工具（LoRA / 全参）。
- `llama-export-lora`：导出或合并 LoRA。
- `llama-gguf`：查看、转换 GGUF 信息。
- `llama-gguf-hash`：计算 GGUF 哈希。
- `llama-gguf-split`：分割 GGUF。
- `llama-convert-llama2c-to-ggml`：llama2.c 权重转 ggml。
- `llama-fit-params`：参数拟合工具。
- `llama-cvector-generator`：生成 control vectors。
- `llama-results`：评测结果汇总。

`llama-cvector-generator` 和 `llama-export-lora` 只在 `NOT GGML_BACKEND_DL AND GGML_CPU` 时编译，当前两个构建目录都满足。

### 3.5 调试与分析

- `llama-debug`：显示推理内部状态。
- `llama-gen-docs`：从工具帮助生成文档。
- `llama-idle`：空闲电源管理。
- `llama-q8dot`：Q8 点积性能测试。
- `llama-vdot`：向量点积性能测试。

### 3.6 历史文档中的目标名错误

- `llama-save-load-state`：当前目标清单里不存在这个目标。旧快照里对应的是 `test-save-load-state`，当前分支也已经没有。
- `llama-export-graph-ops`：实际目标为 `test-export-graph-ops`，属于测试程序。
- `llama-debug-template-parser`、`llama-template-analysis`：旧版本本文档列过，当前分支的目标清单里已经没有这两个目标。

---

## 4. 测试与验证程序（`test-*`）

这些是 llama.cpp 的单元/集成测试，日常使用一般不需要。只有 `LLAMA_BUILD_TESTS=ON` 时才会生成，`build-alldev/` 当前设置为 OFF，所以那里没有这些二进制。

当前 `turingevo` 分支的目标清单（以 `cmake --build build --target help` 为准）：

```
test-alloc                  test-arg-parser
test-autorelease            test-backend-ops
test-backend-sampler        test-barrier
test-batch-alloc            test-c
test-chat                   test-chat-analysis
test-chat-auto-parser       test-chat-peg-parser
test-chat-template          test-col2im-1d
test-conv-1d                test-export-graph-ops
test-fusion                 test-gbnf-validator
test-gelu-quick             test-gguf
test-grammar-integration    test-grammar-parser
test-jinja                  test-json-schema
test-json-schema-to-grammar test-llama-archs
test-llama-context          test-llama-grammar
test-log                    test-model-load-cancel
test-model-resolution       test-mtmd-c-api
test-mtmd-impl              test-opt
test-peg-parser             test-quantize-fns
test-quantize-perf          test-quantize-stats
test-reasoning-budget       test-rope
test-sampling               test-server-component
test-thread-safety          test-tiled-mulmat
test-tokenizer-0            test-tokenizer-1-bpe
test-tokenizer-1-spm        test-unicode
```

几个常用项：

- `test-backend-ops`：后端算子一致性，改动 ggml 后必跑。
- `test-llama-archs`：模型架构加载测试，改 `src/llama-arch.*` 或模型注册后必跑。
- `test-tokenizer-0` / `test-tokenizer-1-*`：分词器测试。
- `test-gguf`：GGUF 读写测试。

---

## 5. 维护本文档

目标清单会随上游变化。重新生成：

```bash
cmake --build build --target help > turingevo/docs/targets
```

`turingevo/docs/targets` 是旧快照，新增或删除目标后要同时更新本文第 3、4 节。`llama-sensenova-cli` 的细节维护在 [llama-sensenova-cli-使用说明.md](../sensenova/llama-sensenova-cli-使用说明.md)。
