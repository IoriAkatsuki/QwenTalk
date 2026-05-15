#!/bin/bash
# SmolVLM2 视觉理解测试脚本
# 用法: bash smolvlm_test.sh [500m|2.2b] [image_path]

set -e
source /opt/intel/oneapi/setvars.sh 2>/dev/null

LLAMA_DIR="/home/intel/llama.cpp/build_sycl/bin"
MODEL_DIR="/home/intel/models"

SIZE="${1:-500m}"
IMG="${2:-/tmp/test_scene.jpg}"

# 生成测试图片（如果没有提供）
if [ ! -f "$IMG" ]; then
    echo "生成测试图片..."
    python3 -c "
import numpy as np
try:
    import cv2
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (200, 100), (400, 350), (0, 255, 0), 2)
    cv2.putText(img, 'person', (200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imwrite('$IMG', img)
    print('测试图片已生成')
except ImportError:
    from PIL import Image
    img = Image.new('RGB', (640, 480), color=(128, 128, 128))
    img.save('$IMG')
    print('测试图片已生成 (PIL)')
"
fi

if [ "$SIZE" = "500m" ]; then
    MODEL="$MODEL_DIR/smolvlm2-500m-q8_0.gguf"
    MMPROJ="$MODEL_DIR/smolvlm2-500m-mmproj-f16.gguf"
    echo "=== SmolVLM2-500M Q8_0 ==="
elif [ "$SIZE" = "2.2b" ]; then
    MODEL="$MODEL_DIR/smolvlm2-2.2b-q4km.gguf"
    MMPROJ="$MODEL_DIR/smolvlm2-2.2b-mmproj-f16.gguf"
    echo "=== SmolVLM2-2.2B Q4_K_M ==="
else
    echo "用法: $0 [500m|2.2b] [image_path]"
    exit 1
fi

echo "模型: $MODEL"
echo "MMProj: $MMPROJ"
echo "图片: $IMG"
echo ""

# 检查文件
for f in "$MODEL" "$MMPROJ" "$IMG"; do
    if [ ! -f "$f" ]; then
        echo "错误: 文件不存在 $f"
        exit 1
    fi
done

# 方案 A: llama-cli SYCL (iGPU)
echo "=== 测试 1: SYCL iGPU ==="
time $LLAMA_DIR/llama-cli \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    -ngl 99 \
    -fa 0 \
    --image "$IMG" \
    -p "Describe what you see in this image in detail." \
    -n 128 \
    -t 4 \
    2>&1 | tail -20

echo ""
echo "=== 测试 2: CPU only ==="
time $LLAMA_DIR/llama-cli \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    -ngl 0 \
    --image "$IMG" \
    -p "Describe what you see in this image in detail." \
    -n 128 \
    -t 8 \
    2>&1 | tail -20
