#!/bin/bash

# ============================================================================
# llama.cpp 编译脚本 - Intel i9-13900K 优化版本
# ============================================================================
# 本脚本为 Intel 第 13 代酷睿处理器 (i9-13900K) 编译 llama.cpp
# 
# CPU 特性:
#   - AVX: 256 位 SIMD 浮点运算
#   - AVX2: 256 位 SIMD 整数运算增强
#   - AVX_VNNI: AI 推理优化的 INT8 乘累加指令
#   
# 加速后端:
#   - CUDA: NVIDIA GPU 加速
#   - OpenBLAS: CPU BLAS 加速
#   - OpenMP: 多线程并行
#
# 指令集策略: 使用 GGML_NATIVE=ON，由编译器自动探测本机支持的全部
#             SIMD 指令集 (AVX2/FMA/F16C/AVX_VNNI 等)，无需手动指定。
# ============================================================================

set -e  # 遇到错误立即退出

# ========== 配置参数 ==========
src_dir=/media/wmx/ws1/workspace/ai/llama.cpp
build_dir=${src_dir}/build-alldev
install_dir=/home/wmx/software/llamacpp

# 指定要编译和安装的目标
TARGETS="llama-cli llama-sensenova-cli llama-sensenova-server llama-mtmd-cli llama-server llama-diffusion llama-diffusion-cli llama-gguf llama-gguf-hash llama-gguf-split llama-batched llama-batched-bench llama-bench llama-quantize llama-perplexity"


# ========== 清理之前的构建 ==========
echo "============================================"
echo "llama.cpp 编译脚本"
echo "============================================"
echo ""
echo "步骤 1/6: 清理之前的构建..."
rm -rf ${build_dir}
mkdir -p ${build_dir}

# ========== 指令集说明 ==========
echo ""
echo "步骤 2/6: 指令集配置..."
echo "  使用 GGML_NATIVE=ON，编译器将自动探测本机 (i9-13900K) 支持的全部指令集"
echo "  （AVX2 / FMA / F16C / AVX_VNNI 等），无需手动指定 -DGGML_AVX2 等选项。"

# ========== 配置 OpenBLAS ==========
echo ""
echo "步骤 3/6: 检查 BLAS 库..."
if dpkg -l | grep -q "libopenblas-dev"; then
    echo "  使用 OpenBLAS 作为 BLAS 后端"
    BLAS_FLAGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
else
    echo "  ⚠ 警告：未找到 libopenblas-dev，将安装..."
    BLAS_FLAGS="-DGGML_BLAS=OFF"
fi

# ========== 执行 CMake 配置 ==========
echo ""
echo "步骤 4/6: 运行 CMake 配置..."
# 静态链接 -DBUILD_SHARED_LIBS=OFF
cmake \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    $BLAS_FLAGS \
    -DGGML_OPENMP=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -B ${build_dir} \
    -S ${src_dir}

# ========== 编译项目 ==========
echo ""
echo "步骤 5/6: 编译项目..."
echo "  使用 $(nproc) 个并发线程进行编译..."
TARGET_ARGS=""
for t in $TARGETS; do
    TARGET_ARGS="$TARGET_ARGS --target $t"
done
cmake --build ${build_dir} --config Release -j$(nproc) $TARGET_ARGS

# ========== 安装 ==========
echo ""
echo "步骤 6/6: 安装到 ${install_dir}/bin/"
mkdir -p ${install_dir}/bin

# 注意：不使用 `cmake --install`，因为它会安装整个项目所有已构建的目标，
# 无法按 target 名精确挑选，且 llama.cpp 未将各工具拆分成独立 install component，
# 容易导致未编译的目标（如 llama-batched）触发 "file INSTALL cannot find" 错误。
# 因此改用下面的 cp 循环，仅安装 TARGETS 列表中编译出的二进制文件。
# cmake --install ${build_dir} --prefix ${install_dir} --config Release

# 仅安装 TARGETS 中实际编译出的二进制文件
for t in $TARGETS; do
    if [ -f ${build_dir}/bin/${t} ]; then
        cp -v ${build_dir}/bin/${t} ${install_dir}/bin/
    else
        echo "  跳过（未编译）: $t"
    fi
done

# ========== 完成 ==========
echo ""
echo "============================================"
echo "编译完成！"
echo "============================================"
echo ""
echo "可执行文件位置:"
ls -la ${install_dir}/bin/
echo ""
echo "测试示例:"
echo "  ${install_dir}/bin/llama-cli -m <模型文件> -t 32 -n 256"
echo ""