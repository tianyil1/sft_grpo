#!/usr/bin/env python3
"""
开源数据集准备脚本
下载并转换开源数据集用于 SFT 和 GRPO 训练
"""
import argparse
import json
import os
from datasets import load_dataset, Dataset
from typing import List, Dict
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def prepare_alpaca_data(output_path: str, num_samples: int = 1000):
    """
    准备 Alpaca 数据集用于 SFT 训练
    Alpaca 是斯坦福大学发布的指令遵循数据集
    
    Args:
        output_path: 输出文件路径
        num_samples: 采样数量（用于 demo，实际使用可以设为 None 使用全部数据）
    """
    logger.info("正在下载 Alpaca 数据集...")
    
    try:
        # 下载 Alpaca 数据集
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        
        # 采样（用于 demo）
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        
        # 转换为项目格式
        formatted_data = []
        for example in dataset:
            formatted_example = {
                "instruction": example["instruction"],
                "input": example.get("input", ""),
                "output": example["output"]
            }
            formatted_data.append(formatted_example)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Alpaca 数据集已保存: {output_path}")
        logger.info(f"  样本数量: {len(formatted_data)}")
        
    except Exception as e:
        logger.error(f"✗ 下载 Alpaca 数据集失败: {e}")
        raise


def prepare_dolly_data(output_path: str, num_samples: int = 1000):
    """
    准备 Dolly 数据集用于 SFT 训练
    Dolly 是 Databricks 发布的开源指令遵循数据集
    
    Args:
        output_path: 输出文件路径
        num_samples: 采样数量
    """
    logger.info("正在下载 Dolly 数据集...")
    
    try:
        # 下载 Dolly 数据集
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        
        # 采样
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        
        # 转换为项目格式
        formatted_data = []
        for example in dataset:
            formatted_example = {
                "instruction": example["instruction"],
                "input": example.get("context", ""),  # Dolly 使用 context 字段
                "output": example["response"]
            }
            formatted_data.append(formatted_example)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Dolly 数据集已保存: {output_path}")
        logger.info(f"  样本数量: {len(formatted_data)}")
        
    except Exception as e:
        logger.error(f"✗ 下载 Dolly 数据集失败: {e}")
        raise


def prepare_hh_rlhf_data(output_path: str, num_samples: int = 500):
    """
    准备 HH-RLHF 数据集用于 GRPO 训练
    HH-RLHF (Helpful and Harmless) 是 Anthropic 发布的人类偏好数据集
    
    Args:
        output_path: 输出文件路径
        num_samples: 采样数量
    """
    logger.info("正在下载 HH-RLHF 数据集...")
    
    try:
        # 下载 HH-RLHF 数据集
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        
        # 采样
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        
        # 转换为项目格式
        formatted_data = []
        for example in dataset:
            # HH-RLHF 的格式是对话历史
            # 提取第一个人类提问作为 prompt
            dialogue = example["chosen"]  # 使用 chosen 回答
            
            # 简单解析：提取第一个 Human 的问题
            lines = dialogue.split("\n")
            prompt = ""
            for line in lines:
                if line.startswith("Human: "):
                    prompt = line.replace("Human: ", "").strip()
                    break
            
            if not prompt:
                continue  # 跳过无法解析的样本
            
            formatted_example = {
                "prompt": prompt,
                "reference": example["chosen"][:500]  # 截断参考回答
            }
            formatted_data.append(formatted_example)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ HH-RLHF 数据集已保存: {output_path}")
        logger.info(f"  样本数量: {len(formatted_data)}")
        
    except Exception as e:
        logger.error(f"✗ 下载 HH-RLHF 数据集失败: {e}")
        raise


def prepare_ultrafeedback_data(output_path: str, num_samples: int = 500):
    """
    准备 Ultrafeedback 数据集用于 GRPO 训练
    Ultrafeedback 是开源的反馈数据集
    
    Args:
        output_path: 输出文件路径
        num_samples: 采样数量
    """
    logger.info("正在下载 Ultrafeedback 数据集...")
    
    try:
        # 下载 Ultrafeedback 数据集
        dataset = load_dataset("openbmb/UltraFeedback", split="train")
        
        # 采样
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        
        # 转换为项目格式
        formatted_data = []
        for example in dataset:
            formatted_example = {
                "prompt": example["instruction"],
                "reference": example.get("completions", [{}])[0].get("response", "")
            }
            formatted_data.append(formatted_example)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Ultrafeedback 数据集已保存: {output_path}")
        logger.info(f"  样本数量: {len(formatted_data)}")
        
    except Exception as e:
        logger.warning(f"⚠ Ultrafeedback 数据集下载失败: {e}")
        logger.info("尝试使用替代数据集...")
        prepare_hh_rlhf_data(output_path, num_samples)


def prepare_chinese_alpaca_data(output_path: str, num_samples: int = 1000):
    """
    准备中文 Alpaca 数据集用于 SFT 训练
    
    Args:
        output_path: 输出文件路径
        num_samples: 采样数量
    """
    logger.info("正在下载中文 Alpaca 数据集...")
    
    try:
        # 尝试下载中文 Alpaca 数据集
        dataset = load_dataset("c-s-ale/alpaca-gpt4-data-cn", split="train")
        
        # 采样
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        
        # 转换为项目格式
        formatted_data = []
        for example in dataset:
            formatted_example = {
                "instruction": example["instruction"],
                "input": example.get("input", ""),
                "output": example["output"]
            }
            formatted_data.append(formatted_example)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 中文 Alpaca 数据集已保存: {output_path}")
        logger.info(f"  样本数量: {len(formatted_data)}")
        
    except Exception as e:
        logger.warning(f"⚠ 中文 Alpaca 数据集下载失败: {e}")
        logger.info("将使用英文 Alpaca 数据集...")
        prepare_alpaca_data(output_path, num_samples)


def main():
    parser = argparse.ArgumentParser(description='准备开源数据集')
    parser.add_argument('--sft_dataset', type=str, default='alpaca',
                        choices=['alpaca', 'dolly', 'chinese_alpaca', 'all'],
                        help='SFT 训练数据集类型')
    parser.add_argument('--grpo_dataset', type=str, default='hh_rlhf',
                        choices=['hh_rlhf', 'ultrafeedback', 'all'],
                        help='GRPO 训练数据集类型')
    parser.add_argument('--num_sft_samples', type=int, default=1000,
                        help='SFT 数据集采样数量（用于 demo）')
    parser.add_argument('--num_grpo_samples', type=int, default=500,
                        help='GRPO 数据集采样数量（用于 demo）')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'),
                        help='输出目录')
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("开始准备开源数据集...")
    logger.info("=" * 50)
    
    # 准备 SFT 数据
    logger.info("\n[1/2] 准备 SFT 训练数据...")
    sft_output_path = os.path.join(args.output_dir, 'sft_data.json')
    
    if args.sft_dataset == 'alpaca':
        prepare_alpaca_data(sft_output_path, args.num_sft_samples)
    elif args.sft_dataset == 'dolly':
        prepare_dolly_data(sft_output_path, args.num_sft_samples)
    elif args.sft_dataset == 'chinese_alpaca':
        prepare_chinese_alpaca_data(sft_output_path, args.num_sft_samples)
    elif args.sft_dataset == 'all':
        # 准备多个数据集并合并
        prepare_alpaca_data(sft_output_path, args.num_sft_samples)
        # 可以添加更多数据集
    
    # 准备 GRPO 数据
    logger.info("\n[2/2] 准备 GRPO 训练数据...")
    grpo_output_path = os.path.join(args.output_dir, 'grpo_data.json')
    
    if args.grpo_dataset == 'hh_rlhf':
        prepare_hh_rlhf_data(grpo_output_path, args.num_grpo_samples)
    elif args.grpo_dataset == 'ultrafeedback':
        prepare_ultrafeedback_data(grpo_output_path, args.num_grpo_samples)
    elif args.grpo_dataset == 'all':
        # 准备多个数据集并合并
        prepare_hh_rlhf_data(grpo_output_path, args.num_grpo_samples)
    
    logger.info("\n" + "=" * 50)
    logger.info("✓ 数据集准备完成！")
    logger.info("=" * 50)
    logger.info(f"\nSFT 数据: {sft_output_path}")
    logger.info(f"GRPO 数据: {grpo_output_path}")
    logger.info("\n下一步：")
    logger.info("1. 检查数据文件格式是否正确")
    logger.info("2. 更新配置文件中的数据路径（如果需要）")
    logger.info("3. 运行训练脚本: bash setup_and_train.sh")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
