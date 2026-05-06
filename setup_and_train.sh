#!/bin/bash
############################################################
# SFT + GRPO 训练自动化脚本
# 用于在 tmux 中后台运行完整的训练和配置流程
############################################################

set -e  # 遇到错误立即退出

# 项目根目录（脚本所在目录）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${PROJECT_ROOT}/training.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "SFT + GRPO 训练自动化脚本"
echo "开始时间: $(date)"
echo "=========================================="

#############################################
# 步骤1: 激活 conda 环境
#############################################
echo -e "\n[1/6] 激活 conda 环境..."
# 兼容：通用 conda/mamba 环境（不写死任何个人路径）
if [ -n "${SFT_GRPO_CONDA_ENV:-}" ]; then
    echo "使用 SFT_GRPO_CONDA_ENV=${SFT_GRPO_CONDA_ENV}"
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "${SFT_GRPO_CONDA_ENV}" 2>/dev/null || true
else
    # 不强制 conda：允许在 venv / 系统 python 中直接运行
    echo "⚠ 未检测到可用 conda 环境，继续使用当前 Python: $(command -v python)"
fi
python -V

#############################################
# 步骤2: 安装依赖
#############################################
echo -e "\n[2/6] 检查并安装依赖..."

# 检查 PyTorch 是否已安装
if python -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch 已安装"
else
    echo "正在安装 PyTorch (CUDA 12.1)..."
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
fi

# 检查 TRL 是否已安装
if python -c "import trl" 2>/dev/null; then
    echo "✓ TRL 已安装"
else
    echo "正在安装 TRL 和依赖..."
    # 与 README / 代码保持一致：GRPO 需要 trl>=0.14
    pip install 'transformers==4.48.0' 'trl==0.15.2' 'accelerate==0.34.2'
    pip install peft datasets pyyaml tensorboard numpy pandas wandb
fi

echo "✓ 依赖安装完成"

#############################################
# 步骤3: 下载模型
#############################################
echo -e "\n[3/6] 检查模型..."

MODEL_DIR="${PROJECT_ROOT}/models/Qwen2.5-3B-Instruct"

if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
    echo "✓ 模型已存在于: $MODEL_DIR"
else
    echo "正在下载 Qwen2.5-3B-Instruct 模型..."
    mkdir -p "$MODEL_DIR"
    
    # 设置 Hugging Face 镜像（国内加速）
    export HF_ENDPOINT=https://hf-mirror.com
    
    # 下载模型
    huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
    
    if [ $? -eq 0 ]; then
        echo "✓ 模型下载成功"
    else
        echo "✗ 模型下载失败，请检查网络连接"
        exit 1
    fi
fi

#############################################
# 步骤4: 运行 SFT 训练
#############################################
echo -e "\n[4/6] 开始 SFT 训练..."
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0
python scripts/run_sft.py --config configs/sft_config.yaml

if [ $? -eq 0 ]; then
    echo "✓ SFT 训练完成"
else
    echo "✗ SFT 训练失败"
    exit 1
fi

#############################################
# 步骤5: 运行 GRPO 训练
#############################################
echo -e "\n[5/6] 开始 GRPO 训练..."
export CUDA_VISIBLE_DEVICES=0
python scripts/run_grpo.py --config configs/grpo_config.yaml

if [ $? -eq 0 ]; then
    echo "✓ GRPO 训练完成"
else
    echo "✗ GRPO 训练失败"
    exit 1
fi

#############################################
# 步骤6: 完成
#############################################
echo -e "\n[6/6] 所有任务完成!"
echo "结束时间: $(date)"
echo "=========================================="
echo "输出目录:"
echo "  - SFT 模型: ${PROJECT_ROOT}/outputs/sft/best_model"
echo "  - GRPO 模型: ${PROJECT_ROOT}/outputs/grpo"
echo "=========================================="
echo ""
echo "查看训练日志:"
echo "  tmux attach -t sft_grpo_training"
echo ""
echo "查看 TensorBoard:"
echo "  tensorboard --logdir=${PROJECT_ROOT}/outputs --port=6006"
