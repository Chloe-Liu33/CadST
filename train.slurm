#!/bin/bash
#SBATCH --job-name=STGCN_ood2
#SBATCH --output=logs/mask_STGCN_ood2.out     # 标准输出日志 (%j 会被 Job ID 替换)
#SBATCH --error=logs/mask_STGCN_ood2.err      # 错误日志
#SBATCH --partition=gpu                # gpu, desktop,comp*, fitc, genomics
#SBATCH --gres=gpu:L40S:1                   # 请求 n 块 GPU
#SBATCH --cpus-per-task=8             # 根据需求请求 CPU
#SBATCH --mem=64G                     # 请求内存
#SBATCH --time=3-00:00:00                # max 7 days
#SBATCH --account=ek04              # 账户名
#SBATCH --qos=normal                   # desktopq,irq,m3h,normal,rtq,shortq

#killall -u limeil python || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/ek04_scratch/miniconda3/etc/profile.d/conda.sh
conda activate torch2
python main.py