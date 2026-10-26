
# cpu cuda openblas
backend=cpu
cmd=../bin/$backend/llama-batched-bench

modelpath=/media/wmx/soft1/huggingface_cache/Qwen2-1.5B-Q4_K_M-GGUF/qwen2-1.5b-q4_k_m.gguf
# modelpath=/media/wmx/soft1/huggingface_cache/Qwen1.5-1.8B-Chat/ggml-model-f16.gguf


$cmd \
-m $modelpath \
-c 16384 \
-b 2048 \
-ub 512 \
-ngl 99 -pps -v