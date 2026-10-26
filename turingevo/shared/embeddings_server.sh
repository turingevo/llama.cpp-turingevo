
#!/bin/bash

# sudo netstat -lnp |grep 8000

ebd_prefix=/media/wmx/soft2/huggingface_cache/Qwen/Qwen3-Embedding-0.6B-GGUF
ebd_model_path=$ebd_prefix/Qwen3-Embedding-0.6B-Q8_0.gguf
rk_prefix=/media/wmx/soft2/huggingface_cache/BAAI/bge-reranker-base
rk_model_path=$rk_prefix/bge-reranker-base-q4_k_m.gguf

llama_prefix=/media/wmx/ws1/workspace/ai/llama.cpp/bin/cuda/
llama_path=$llama_prefix/bin/llama-server


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${llama_prefix}/lib

LLAMA_ARG_EMBEDDINGS=1

$llama_path --embeddings -m $ebd_model_path -a EBD --host 0.0.0.0 --port 8000




