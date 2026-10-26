#!/bin/bash
# llama-sensenova-cli 本地测试脚本
#
# F16-fm.gguf 含 20 个张量：16 个生成侧 + 4 个理解侧 vision 权重
# 图像编辑（-i/--image）需要那 4 个理解侧张量，缺了会直接报错
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/media/wmx/ws1/workspace/ai/llama.cpp
MODEL_DIR=/media/wmx/soft2/huggingface_cache/models/image/SenseNova/llamacpp
MODEL=SenseNova-U1.5-8B-MoT-SFT-8step-V2-Q5_K_M.gguf
FM=F16-fm.gguf
CLI="$ROOT/build-alldev/bin/llama-sensenova-cli"
PROMPT_FILE="$HERE/prompts/omni-note.txt"
OUT_DIR="$ROOT/turingevo/out"

WIDTH=2720
HEIGHT=1536
STEPS=8
SEED=42
CFG=1.0
IMG_CFG=1.0
CFG_NORM=none
THINK=0
THINK_SET=0
INPUT_IMAGES=()
VQA=0
QUESTION=""
INTERLEAVE=0
MAX_IMAGES=3
SIZE_SET=0

usage() {
    cat <<EOF
用法: $(basename "$0") [options]

  -f, --prompt-file P  提示词文件 (默认 prompts/omni-note.txt)
  -q, --question TEXT  视觉问答的问题，给了就切到 --vqa 模式，只用 -i 的图
  -i, --image P        输入图，可重复，用于图像编辑、问答或交错参考图
      --fm NAME        FM sidecar 文件名 (默认 $FM)
      --width N        宽度，32 的倍数 (默认 $WIDTH)
      --height N       高度，32 的倍数 (默认 $HEIGHT)
      --steps N        去噪步数 (默认 $STEPS)
      --seed N         随机种子 (默认 $SEED)
      --cfg F          文本引导强度 (默认 $CFG)
      --img-cfg F      图像引导强度，编辑时有效 (默认 $IMG_CFG)
      --cfg-norm MODE  none / global / channel (默认 $CFG_NORM)
      --think          先跑 think 推理再出图
      --interleave     图文交错，模型边写边出图；尺寸默认跟随首图或 1024x1024
      --max-images N   交错模式的出图上限 (默认 $MAX_IMAGES)
      --no-think       关掉交错模式的 think（交错默认开）
  -h, --help           显示帮助
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -f|--prompt-file) PROMPT_FILE=$2; shift 2 ;;
        -q|--question)    VQA=1; QUESTION=$2; shift 2 ;;
        -i|--image)       INPUT_IMAGES+=("$2"); shift 2 ;;
        --fm)             FM=$2; shift 2 ;;
        --width)          WIDTH=$2; SIZE_SET=1; shift 2 ;;
        --height)         HEIGHT=$2; SIZE_SET=1; shift 2 ;;
        --steps)          STEPS=$2; shift 2 ;;
        --seed)           SEED=$2; shift 2 ;;
        --cfg)            CFG=$2; shift 2 ;;
        --img-cfg)        IMG_CFG=$2; shift 2 ;;
        --cfg-norm)       CFG_NORM=$2; shift 2 ;;
        --think)          THINK=1; THINK_SET=1; shift ;;
        --interleave)     INTERLEAVE=1; shift ;;
        --max-images)     MAX_IMAGES=$2; shift 2 ;;
        --no-think)       THINK=0; THINK_SET=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "error: 未知参数: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[ -x "$CLI" ] || { echo "error: 找不到 $CLI，先跑 turingevo/shared/build-all-dev.sh" >&2; exit 1; }

if [ "$VQA" = "1" ]; then
    [ -n "$QUESTION" ] || { echo "error: 问题为空" >&2; exit 1; }
    ARGS=(-m "$MODEL_DIR/$MODEL" --fm "$MODEL_DIR/$FM" -p "$QUESTION" --vqa -ngl 999)
    for img in ${INPUT_IMAGES+"${INPUT_IMAGES[@]}"}; do
        ARGS+=(-i "$img")
    done
    echo "问答: 输入图 ${#INPUT_IMAGES[@]} 张"
    "$CLI" "${ARGS[@]}"
    exit 0
fi

[ -r "$PROMPT_FILE" ] || { echo "error: 无法读取 $PROMPT_FILE" >&2; exit 1; }

PROMPT=$(cat "$PROMPT_FILE")
[ -n "$PROMPT" ] || { echo "error: 提示词为空" >&2; exit 1; }

# 输出图片名取自 prompt 文件名的 basename（去掉扩展名）
PROMPT_NAME=$(basename "$PROMPT_FILE")
PROMPT_NAME=${PROMPT_NAME%.*}
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/${PROMPT_NAME}.png"

if [ "$INTERLEAVE" = "1" ]; then
    [ "$THINK_SET" = "1" ] || THINK=1
    ARGS=(
        -m "$MODEL_DIR/$MODEL"
        --fm "$MODEL_DIR/$FM"
        -p "$PROMPT"
        --interleave --max-images "$MAX_IMAGES"
        --steps "$STEPS" --seed "$SEED"
        -ngl 999
        -o "$OUT"
    )
    if [ "$SIZE_SET" = "1" ]; then
        ARGS+=(--width "$WIDTH" --height "$HEIGHT")
    fi
    for img in ${INPUT_IMAGES+"${INPUT_IMAGES[@]}"}; do
        ARGS+=(-i "$img")
    done
    if [ "$THINK" = "1" ]; then
        ARGS+=(--print-think --think-output "$OUT_DIR/${PROMPT_NAME}.think.txt")
    else
        ARGS+=(--no-think)
    fi
    echo "交错: $STEPS 步, seed $SEED, 最多 $MAX_IMAGES 张, think $THINK, 输入图 ${#INPUT_IMAGES[@]} 张"
    "$CLI" "${ARGS[@]}"
    echo "图片: $OUT, ${OUT_DIR}/${PROMPT_NAME}_1.png ..."
    exit 0
fi

ARGS=(
    -m "$MODEL_DIR/$MODEL"
    --fm "$MODEL_DIR/$FM"
    -p "$PROMPT"
    --width "$WIDTH" --height "$HEIGHT"
    --steps "$STEPS" --seed "$SEED"
    --cfg "$CFG" --img-cfg "$IMG_CFG" --cfg-norm "$CFG_NORM"
    -ngl 999
    -o "$OUT"
)

for img in ${INPUT_IMAGES+"${INPUT_IMAGES[@]}"}; do
    ARGS+=(-i "$img")
done

if [ "$THINK" = "1" ]; then
    ARGS+=(--think --print-think --think-output "$OUT_DIR/${PROMPT_NAME}.think.txt")
fi

echo "请求: ${WIDTH}x${HEIGHT}, $STEPS 步, seed $SEED, cfg $CFG, img_cfg $IMG_CFG, think $THINK"
"$CLI" "${ARGS[@]}"
echo "图片: $OUT"
