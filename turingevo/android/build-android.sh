
#!/bin/bash

ANDROID_NDK_PATH=/media/wmx/ws1/software/qtAndroid/Sdk/ndk/23.1.7779620
build_dir=../../build-android
src_dir=../../
install_dir=../../bin/android

cmake \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_C_FLAGS="-march=armv8.7a" \
  -DCMAKE_CXX_FLAGS="-march=armv8.7a" \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -B ${build_dir} \
  -S ${src_dir}


cmake --build ${build_dir} --config Release -j48


cmake --install ${build_dir} --prefix ${install_dir} --config Release