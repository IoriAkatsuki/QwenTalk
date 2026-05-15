#!/bin/bash
# 下载 head-pose-estimation-adas-0001 FP16 模型到板卡
# 用法: scp 到板卡后执行，或���接 ssh intel@192.168.1.8 'bash -s' < download_headpose.sh

set -e

MODEL_DIR="/home/intel/models/head-pose-estimation-adas-0001/FP16"
BASE_URL="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/head-pose-estimation-adas-0001/FP16"

mkdir -p "$MODEL_DIR"

for f in head-pose-estimation-adas-0001.xml head-pose-estimation-adas-0001.bin; do
    if [ ! -f "$MODEL_DIR/$f" ]; then
        echo "下载 $f ..."
        curl -L -o "$MODEL_DIR/$f" "$BASE_URL/$f"
    else
        echo "$f 已���在，跳过"
    fi
done

echo "模型大小:"
ls -lh "$MODEL_DIR/"
echo "下载完成: $MODEL_DIR"
