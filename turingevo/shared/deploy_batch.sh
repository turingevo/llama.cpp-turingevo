#!/bin/bash

# 检查是否提供了可执行文件路径所在的目录
if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-executables-directory>"
    exit 1
fi

EXECUTABLES_DIR=$1
LIB_DIR="./lib"

# 创建目标目录
mkdir -p "$LIB_DIR"

# 函数：拷贝库到LIB_DIR并修补RPATH
copy_and_patch_lib() {
    local lib_path=$1
    local target_path="$LIB_DIR/$(basename $lib_path)"

    if [[ "$lib_path" =~ ^/ ]] && [ ! -f "$target_path" ]; then
        echo "Copying $lib_path to $LIB_DIR"
        cp --dereference "$lib_path" "$LIB_DIR"
        # 如果库是一个符号链接，也拷贝它指向的实际文件
        if [ -L "$lib_path" ]; then
            actual_file=$(readlink -f "$lib_path")
            cp --dereference "$actual_file" "$LIB_DIR"
        fi

        # 修补库的RPATH
        patchelf --set-rpath '$ORIGIN' "$target_path"
    fi
}

# 遍历目录中的所有可执行文件
find "$EXECUTABLES_DIR" -maxdepth 1 -type f -executable | while read -r executable; do
    echo "Processing dependencies for $executable..."

    # 使用 ldd 查找可执行文件的依赖库，并忽略非文件路径和系统标准路径的库
    ldd "$executable" 2>/dev/null | grep -v "^$" | awk '/=>/{print $3}' | while read -r lib; do
        copy_and_patch_lib "$lib"
    done

    # 拷贝可执行文件到当前目录（如果需要）
    cp "$executable" .

    # 修补可执行文件的RPATH
    patchelf --set-rpath '$ORIGIN/lib' "$(basename $executable)"
done

echo "Deployment completed. Check the current directory and $LIB_DIR."