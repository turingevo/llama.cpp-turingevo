# shared

跨模型脚本，所有模型共用，不按模型复制。

这些脚本按开发机的绝对路径编写，换机器或换目录时先改脚本顶部的 `src_dir`、`llama_prefix` 和模型路径。

| 脚本 | 作用 |
|-|-|
| `build-all-dev.sh` | 配置并编译常用目标，按 TARGETS 列表安装到指定目录 |
| `quantize.sh` | 遍历所有量化类型批量量化（示例模型为 Qwen2-VL） |
| `bench.sh` | `llama-batched-bench` 压测 |
| `test_server.sh` | `llama-server` 冒烟测试 |
| `deploy_app.sh` | 单个可执行文件加依赖库打包，patchelf 改 RPATH |
| `deploy_batch.sh` | 整个目录的可执行文件批量打包 |
| `embeddings_server.sh` / `embeddings_client.py` | embedding 服务与调用示例 |
| `reranker_server.sh` / `reranker_client.py` | rerank 服务与调用示例 |
