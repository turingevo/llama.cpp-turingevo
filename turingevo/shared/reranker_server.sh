
#!/bin/bash

# sudo netstat -lnp |grep 8001

prefix=/media/wmx/soft2/huggingface_cache/BAAI/bge-reranker-base
model_path=$prefix/bge-reranker-base-q4_k_m.gguf

llama_prefix=${LLAMA_PREFIX:-$(cd "$(dirname "$0")/../.." && pwd)/build-alldev}
llama_path=$llama_prefix/bin/llama-server

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${llama_prefix}/lib

LLAMA_ARG_RERANKING=1

$llama_path --rerank -m $model_path -a RNK --host 0.0.0.0 --port 8001




