# llama.cpp

> **TuringEvo 发行版**：在官方 llama.cpp 基础上新增 SenseNova-U1 工具链（权重转换、命令行推理、OpenAI 兼容服务端、图文交错生成）。官方模型与后端未做裁剪，可随上游持续合并。
>
> [官网](https://turingevo.com/) · [API](https://api.turingevo.com/) · [博客](https://turingevo.com/blog/) · [GitHub](https://github.com/turingevo) · [Hugging Face](https://huggingface.co/turingevo) · [ModelScope](https://www.modelscope.cn/profile/TuringEvo)

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## TuringEvo 发行版说明

本仓库是 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 的增量分支，默认分支 `turingevo`。上游代码保持原样，新增内容全部放在独立目录，合并上游时无需改动已有模型。

| 组件 | 产物 | 作用 |
|-|-|-|
| `tools/sensenova-u1/` | `llama-sensenova-cli` | 文生图、图像编辑、单轮视觉问答、图文交错生成 |
| `tools/sensenova-u1/convert_sensenova.py` | - | safetensors 转 GGUF，输出 LM 主模型与 FM sidecar |
| `tools/sensenova-u1/retag.py` | - | 把厂商 GGUF 的张量名与架构名改写为 llama.cpp 布局 |
| `tools/sensenova-server/` | `llama-sensenova-server` | OpenAI 兼容协议与私有协议服务端 |

构建：

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j"$(nproc)" --target llama-sensenova-cli llama-sensenova-server
```

转换与运行：

```bash
# safetensors -> GGUF
python tools/sensenova-u1/convert_sensenova.py --src /path/to/SenseNova-U1 --out sn-u1.gguf --out-fm sn-u1-fm.gguf

# 命令行：文生图
./build/bin/llama-sensenova-cli -m sn-u1.gguf --fm sn-u1-fm.gguf -p "a cat on the moon" -o out.png

# 服务端
./build/bin/llama-sensenova-server -m sn-u1.gguf --fm sn-u1-fm.gguf --host 0.0.0.0 --port 8024
```

服务端端点：`GET /v1/models`、`POST /generate`、`POST /v1/chat/completions`、`POST /v1/images/generations`、`POST /v1/images/edits`、`POST /v1/infographic/generate`、`POST /v1/interleave/generate`、`GET /images/{file}`、`GET /health`、`GET /props`。

同步上游：

```bash
git fetch upstream
git switch turingevo
git merge upstream/master
git push
```

## Quick start

A few options to get `llama.cpp` installed on your machine:

```bash
# curl
curl -LsSf https://llama.app/install.sh | sh

# powershell
irm https://llama.app/install.ps1 | iex
```

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

----

## 关于

| | | |
|-|-|-|
| API | https://api.turingevo.com/ | 全球顶级旗舰模型，抵至1.4折 |
| 网站 | https://turingevo.com/ | google NotebookLM 国产版本 |
| 博客 | https://turingevo.com/blog/ | 日常技术分享 |
| GitHub 组织 | https://github.com/turingevo | 项目分享 |
| Hugging Face | https://huggingface.co/turingevo | 模型分享 |
| Model Scope | https://www.modelscope.cn/profile/TuringEvo | 模型分享 |

## 联系方式

| | |
|-|-|
| QQ群 | 873673497 |
| 邮箱 | wmx@turingevo.com |

Artificial intelligence, large language models, deep learning, reinforcement learning, machine vision, speech recognition, agents

## About

| | | |
|-|-|-|
| API | https://api.turingevo.com/ | World-class flagship models, as low as 14% of the original price (up to 86% off) |
| Website | https://turingevo.com/ | A China-local version of Google NotebookLM |
| Blog | https://turingevo.com/blog/ | Daily tech sharing |
| GitHub Org | https://github.com/turingevo | Project sharing |
| Hugging Face | https://huggingface.co/turingevo | Model sharing |
| ModelScope | https://www.modelscope.cn/profile/TuringEvo | Model sharing |

## Contact

| | |
|-|-|
| QQ Group | 873673497 |
| Email | wmx@turingevo.com |
