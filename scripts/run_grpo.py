#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) 训练脚本
使用 Hugging Face TRL 框架对 SFT 微调后的模型进行强化学习优化
v3: FP16 混合精度 + group_size=8 + 全量 epoch 训练
"""
import argparse
import yaml
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
import logging

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def configure_t4_attention_kernels():
    """
    T4 (sm75) 上禁用需要 sm80/sm90 的 attention kernel，与 SFT 脚本保持一致。
    否则在梯度检查点的 backward 路径可能触发：
    'Expected is_sm80 || is_sm90 to be true'
    """
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    if major < 8:
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            logger.info(f"✓ 已禁用 Flash/MemEfficient SDP（sm{major}{minor}），使用 Math SDP")
        except Exception as e:
            logger.info(f"⚠ 配置 SDP kernel 失败（忽略继续）：{e}")


def reward_func(prompts, completions, **kwargs):
    """
    基于规则的奖励函数（v2：更有区分度）
    目标：让 GRPO 学会生成"更有帮助的、连贯的、长度合适的"回答

    Args:
        prompts: 提示列表
        completions: 生成的回答列表
    Returns:
        奖励列表（范围约 0.0 - 2.0）
    """
    rewards = []
    for completion in completions:
        reward = 0.0
        text = completion.strip()
        words = text.split()
        n_words = len(words)

        # 规则 1：长度奖励（连续函数，避免阶梯效应）
        # 目标长度 50-80 词，用高斯形状给出平滑奖励
        if n_words == 0:
            reward -= 1.0  # 空回答重罚
        else:
            # 以 60 词为中心的高斯奖励，σ=30
            import math
            length_reward = math.exp(-((n_words - 60) ** 2) / (2 * 30 ** 2))
            reward += length_reward  # 0.0 - 1.0

        # 规则 2：完整性奖励（以句号结尾）
        if text.endswith(('.', '!', '?', '。', '！', '？')):
            reward += 0.3

        # 规则 3：多样性奖励（避免重复词，惩罚简单重复）
        if n_words > 0:
            unique_ratio = len(set(w.lower() for w in words)) / n_words
            reward += unique_ratio * 0.5  # 多样性重要，权重加大

            # 惩罚连续重复的短语（常见的退化模式）
            repeat_penalty = 0.0
            for i in range(len(words) - 2):
                bigram = (words[i].lower(), words[i+1].lower())
                next_bigram = (words[i+1].lower(), words[i+2].lower())
                if bigram == next_bigram:
                    repeat_penalty += 0.1
            reward -= min(repeat_penalty, 0.5)

        # 规则 4：句子结构（有多个句子的回答更好）
        n_sentences = sum(1 for c in text if c in '.!?。！？')
        if n_sentences >= 2:
            reward += 0.2

        rewards.append(reward)

    return rewards


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(model_config: dict, hardware_config: dict):
    """加载 SFT 训练后的模型和分词器"""
    model_path = model_config['model_name_or_path']
    logger.info(f"正在加载模型: {model_path}")

    # FP16 精度策略（与 SFT 保持一致）
    use_fp16 = bool(hardware_config.get("fp16", False))
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    logger.info(f"✓ 模型加载精度: {'float16 (FP16)' if use_fp16 else 'float32'}")

    # 检查是否是 LoRA adapter
    adapter_config_path = os.path.join(model_path, 'adapter_config.json')
    is_lora_adapter = os.path.exists(adapter_config_path)

    if is_lora_adapter:
        # 加载 LoRA adapter: 先加载 base model
        import json
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_path = adapter_config['base_model_name_or_path']
        logger.info(f"检测到 LoRA adapter，base model: {base_model_path}")

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"  # GRPO 生成时使用 left padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        base_model = base_model.cuda()

        # 加载并合并 LoRA adapter
        logger.info("正在合并 SFT 的 LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # 合并 LoRA 权重到 base model
        logger.info("✓ LoRA adapter 已合并")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"  # GRPO 生成时使用 left padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        model = model.cuda()

    logger.info(f"✓ 模型已加载到: {next(model.parameters()).device}")

    # 为 GRPO 训练添加新的 LoRA adapter
    if model_config.get('use_lora', False):
        logger.info("正在为 GRPO 配置新的 LoRA...")
        lora_config = LoraConfig(
            r=model_config.get('lora_r', 8),
            lora_alpha=model_config.get('lora_alpha', 16),
            lora_dropout=model_config.get('lora_dropout', 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 关键：梯度检查点 + LoRA 需要显式启用输入梯度
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
        logger.info("✓ 已启用 input_require_grads（梯度检查点兼容）")

    # 启用梯度检查点节省显存
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("✓ 已启用梯度检查点")

    return model, tokenizer


def setup_grpo_config(config: dict) -> GRPOConfig:
    """设置 GRPO 配置"""
    training_config = config['training']
    hardware_config = config['hardware']
    misc_config = config['misc']
    grpo_config = config['grpo']

    use_fp16 = bool(hardware_config.get("fp16", False))

    kwargs = dict(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=float(training_config['learning_rate']),
        warmup_ratio=training_config['warmup_ratio'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        logging_steps=training_config['logging_steps'],
        save_strategy=training_config['save_strategy'],

        # 精度配置
        fp16=use_fp16,
        bf16=False,  # T4 不支持 bf16

        gradient_checkpointing=hardware_config['gradient_checkpointing'],
        dataloader_num_workers=hardware_config['dataloader_num_workers'],
        remove_unused_columns=False,

        # GRPO 特定参数
        num_generations=grpo_config['num_generations'],
        temperature=grpo_config['temperature'],
        max_prompt_length=config['data']['max_prompt_length'],
        max_completion_length=config['data']['max_new_tokens'],

        # 其他
        seed=misc_config['seed'],
        report_to=misc_config.get('report_to', 'none'),
        run_name=misc_config.get('run_name', 'grpo_run'),
        logging_dir=os.path.join(training_config['output_dir'], 'tensorboard'),
    )

    return GRPOConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser(description='GRPO 训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    logger.info(f"正在加载配置文件: {args.config}")
    config = load_config(args.config)

    # T4: 避免触发 sm80/sm90 专用 kernel
    configure_t4_attention_kernels()

    os.makedirs(config['training']['output_dir'], exist_ok=True)

    # 设置模型和分词器
    model, tokenizer = setup_model_and_tokenizer(config['model'], config.get('hardware', {}))

    # 加载数据集
    logger.info(f"正在加载数据集: {config['data']['dataset_path']}")
    dataset = load_dataset('json', data_files=config['data']['dataset_path'], split='train')

    # 只保留 prompt 字段（GRPO 不需要 reference）
    if 'reference' in dataset.column_names:
        dataset = dataset.remove_columns([c for c in dataset.column_names if c != 'prompt'])

    logger.info(f"数据集字段: {dataset.column_names}")
    logger.info(f"训练样本总数: {len(dataset)}")

    # 设置 GRPO 配置
    grpo_config = setup_grpo_config(config)
    logger.info(
        f"✓ GRPO 配置: epochs={grpo_config.num_train_epochs}, "
        f"batch_size={grpo_config.per_device_train_batch_size}, "
        f"grad_accum={grpo_config.gradient_accumulation_steps}, "
        f"group_size(num_generations)={grpo_config.num_generations}, "
        f"fp16={grpo_config.fp16}"
    )

    # 创建 GRPO Trainer
    logger.info("正在创建 GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_func,
        processing_class=tokenizer,
    )

    # 开始训练
    logger.info("开始 GRPO 训练...")
    trainer.train()

    # 保存最终模型
    logger.info("保存模型...")
    final_path = os.path.join(config['training']['output_dir'], 'best_model')
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info(f"✓ 训练完成！模型已保存到: {final_path}")


if __name__ == '__main__':
    main()
