#!/bin/bash

# 检查是否提供了可执行文件路径
if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-executable>"
    exit 1
fi

EXECUTABLE=$1
DEST_DIR="deploy"
LIB_DIR="$DEST_DIR/libs"

# 创建目标目录
mkdir -p "$LIB_DIR"

# 使用 ldd 查找可执行文件的依赖库
ldd $EXECUTABLE | grep "=> /" | awk '{print $3}' | while read -r lib; do
    echo "Copying $lib to $LIB_DIR"
    cp "$lib" "$LIB_DIR"
done

# 拷贝可执行文件到目标目录
cp "$EXECUTABLE" "$DEST_DIR"

# 使用 patchelf 修补可执行文件和库的路径
echo "Patching executable and libraries..."
patchelf --set-rpath '$ORIGIN/libs' "$DEST_DIR/$(basename $EXECUTABLE)"

find "$LIB_DIR" -type f -name "*.so*" | while read -r lib; do
    patchelf --set-rpath '$ORIGIN' "$lib"
done

echo "Deployment completed. Check the $DEST_DIR directory."

