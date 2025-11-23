#!/bin/bash
#JSUB -q gpu
#JSUB -gpgpu 8
#JSUB -J basicts_MyModel_8gpu
#JSUB -o output/output.%J
#JSUB -e error/error.%J
#JSUB -cwd /home/25181214631/BasicTS

GPU_NUM=8

# 建议：脚本出错就立即退出
set -e

# conda 环境
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate BasicTS-hp
export SETUPTOOLS_USE_DISTUTILS=stdlib
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/25181214631/BasicTS

# 确保日志目录存在（防止 JSUB 找不到目录直接报错）
mkdir -p output error

# 根据 GPU_NUM 自动生成 "0,1,2,3" 这种格式
GPUS=""
for ((i=0; i<GPU_NUM; i++)); do
  if [[ -z "$GPUS" ]]; then
    GPUS="$i"
  else
    GPUS="$GPUS,$i"
  fi
done

# 传给 BasicTS，用于 config 里如果需要 GPU_NUM
export BASICTS_GPU_NUM="$GPU_NUM"

echo "[INFO] Using GPUs: $GPUS (total: $GPU_NUM)"

python experiments/train.py --cfg baselines/MyModel/PEMS04.py --gpus "$GPUS"
