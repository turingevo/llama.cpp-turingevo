
#!/bin/bash

# netstat -lnp |grep 8000

# default port=8080 is used by ollama
PORT=8000

# cpu cuda openblas
backend=cuda
llama_prefix=/media/wmx/ws1/workspace/ai/llama.cpp/bin/$backend/
llama_path=$llama_prefix/bin/llama-server
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${llama_prefix}/lib


modelpath=/media/wmx/soft2/huggingface_cache/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf

# modelpath=/media/wmx/soft1/huggingface_cache/Qwen1.5-1.8B-Chat/ggml-model-f16.gguf

$llama_path \
-m $modelpath \
--port $PORT 