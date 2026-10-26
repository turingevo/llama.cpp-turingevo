#!/bin/bash



# usage: ./llama-quantize [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--include-weights] [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--override-kv] model-f32.gguf [model-quant.gguf] type [nthreads]

#   --allow-requantize: Allows requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit
#   --leave-output-tensor: Will leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing
#   --pure: Disable k-quant mixtures and quantize all tensors to the same type
#   --imatrix file_name: use data in file_name as importance matrix for quant optimizations
#   --include-weights tensor_name: use importance matrix for this/these tensor(s)
#   --exclude-weights tensor_name: use importance matrix for this/these tensor(s)
#   --output-tensor-type ggml_type: use this ggml_type for the output.weight tensor
#   --token-embedding-type ggml_type: use this ggml_type for the token embeddings tensor
#   --keep-split: will generate quantized model in the same shards as input
#   --override-kv KEY=TYPE:VALUE
#       Advanced option to override model metadata by key in the quantized model. May be specified multiple times.
# Note: --include-weights and --exclude-weights cannot be used together

# Allowed quantization types:
#    2  or  Q4_0    :  4.34G, +0.4685 ppl @ Llama-3-8B
#    3  or  Q4_1    :  4.78G, +0.4511 ppl @ Llama-3-8B
#    8  or  Q5_0    :  5.21G, +0.1316 ppl @ Llama-3-8B
#    9  or  Q5_1    :  5.65G, +0.1062 ppl @ Llama-3-8B
#   19  or  IQ2_XXS :  2.06 bpw quantization
#   20  or  IQ2_XS  :  2.31 bpw quantization
#   28  or  IQ2_S   :  2.5  bpw quantization
#   29  or  IQ2_M   :  2.7  bpw quantization
#   24  or  IQ1_S   :  1.56 bpw quantization
#   31  or  IQ1_M   :  1.75 bpw quantization
#   36  or  TQ1_0   :  1.69 bpw ternarization
#   37  or  TQ2_0   :  2.06 bpw ternarization
#   10  or  Q2_K    :  2.96G, +3.5199 ppl @ Llama-3-8B
#   21  or  Q2_K_S  :  2.96G, +3.1836 ppl @ Llama-3-8B
#   23  or  IQ3_XXS :  3.06 bpw quantization
#   26  or  IQ3_S   :  3.44 bpw quantization
#   27  or  IQ3_M   :  3.66 bpw quantization mix
#   12  or  Q3_K    : alias for Q3_K_M
#   22  or  IQ3_XS  :  3.3 bpw quantization
#   11  or  Q3_K_S  :  3.41G, +1.6321 ppl @ Llama-3-8B
#   12  or  Q3_K_M  :  3.74G, +0.6569 ppl @ Llama-3-8B
#   13  or  Q3_K_L  :  4.03G, +0.5562 ppl @ Llama-3-8B
#   25  or  IQ4_NL  :  4.50 bpw non-linear quantization
#   30  or  IQ4_XS  :  4.25 bpw non-linear quantization
#   15  or  Q4_K    : alias for Q4_K_M
#   14  or  Q4_K_S  :  4.37G, +0.2689 ppl @ Llama-3-8B
#   15  or  Q4_K_M  :  4.58G, +0.1754 ppl @ Llama-3-8B
#   17  or  Q5_K    : alias for Q5_K_M
#   16  or  Q5_K_S  :  5.21G, +0.1049 ppl @ Llama-3-8B
#   17  or  Q5_K_M  :  5.33G, +0.0569 ppl @ Llama-3-8B
#   18  or  Q6_K    :  6.14G, +0.0217 ppl @ Llama-3-8B
#    7  or  Q8_0    :  7.96G, +0.0026 ppl @ Llama-3-8B
#    1  or  F16     : 14.00G, +0.0020 ppl @ Mistral-7B
#   32  or  BF16    : 14.00G, -0.0050 ppl @ Mistral-7B
#    0  or  F32     : 26.00G              @ 7B
#           COPY    : only copy tensors, no quantizing



# 定义量化类型列表
quantization_types=(
    "Q4_0"
    "Q4_1"
    "Q5_0"
    "Q5_1"
    "IQ2_XXS"
    "IQ2_XS"
    "IQ2_S"
    "IQ2_M"
    "IQ1_S"
    "IQ1_M"
    "TQ1_0"
    "TQ2_0"
    "Q2_K"
    "Q2_K_S"
    "IQ3_XXS"
    "IQ3_S"
    "IQ3_M"
    "Q3_K"
    "IQ3_XS"
    "Q3_K_S"
    "Q3_K_M"
    "Q3_K_L"
    "IQ4_NL"
    "IQ4_XS"
    "Q4_K"
    "Q4_K_S"
    "Q4_K_M"
    "Q5_K"
    "Q5_K_S"
    "Q5_K_M"
    "Q6_K"
    "Q8_0"
    "F16"
    "BF16"
    "F32"
    "COPY"
)




# cpu cuda openblas
backend=cuda
cmd=../bin/$backend/llama-quantize

model=/media/wmx/soft1/huggingface_cache/Qwen2-VL-2B-Instruct/gguf/Qwen2-VL-2B-Instruct-F16.gguf


# 遍历列表中的元素
for qtype in "${quantization_types[@]}"; do
    echo "quantize $model with $qtype"

    out_dir=Qwen2-VL-2B-Instruct-${qtype}.gguf

    $cmd \
    $model \
    ${out_dir} \
    ${qtype}

done



