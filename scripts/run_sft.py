#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) 训练脚本
使用 Hugging Face TRL 框架对 Qwen2.5 模型进行监督微调
"""
import argparse
import yaml
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    T4 (sm75) 上禁用需要 sm80/sm90 的 attention kernel。
    否则在 gradient checkpointing 的 backward 路径可能触发：
    'Expected is_sm80 || is_sm90 to be true'
    """
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    # sm80=8.0，sm90=9.0；T4=7.5
    if major < 8:
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            logger.info(f"✓ 已禁用 Flash/MemEfficient SDP（sm{major}{minor}），使用 Math SDP")
        except Exception as e:
            logger.info(f"⚠ 配置 SDP kernel 失败（忽略继续）：{e}")


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(model_config: dict, hardware_config: dict):
    """加载模型和分词器"""
    logger.info(f"正在加载模型: {model_config['model_name_or_path']}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['model_name_or_path'],
        trust_remote_code=model_config['trust_remote_code']
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    # 精度策略：
    # - T4 不支持 bf16，但支持 fp16
    # - fp16 训练使用 AMP + GradScaler 以降低 NaN 风险
    use_fp16 = bool(hardware_config.get("fp16", False))
    torch_dtype = torch.float16 if use_fp16 else torch.float32

    # 加载模型 - 不使用 device_map，手动管理设备
    model = AutoModelForCausalLM.from_pretrained(
        model_config['model_name_or_path'],
        trust_remote_code=model_config['trust_remote_code'],
        torch_dtype=torch_dtype,
    )
    
    # 将模型放到 GPU
    model = model.cuda()
    logger.info(f"✓ 模型已加载到: {next(model.parameters()).device}")
    logger.info(f"✓ 使用精度: {'fp16(AMP)' if use_fp16 else 'fp32'}（T4: bf16 禁用）")
    
    # 如果启用 LoRA，配置 LoRA
    if model_config.get('use_lora', False):
        logger.info("正在配置 LoRA...")
        lora_config = LoraConfig(
            r=model_config.get('lora_r', 16),
            lora_alpha=model_config.get('lora_alpha', 32),
            lora_dropout=model_config.get('lora_dropout', 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("✓ LoRA 已注入（参数 dtype 跟随 base model）")
    
    # 设置为训练模式
    model.train()
    logger.info(f"✓ 模型已设置为训练模式: model.training={model.training}")
    
    # 启用梯度检查点以节省内存
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    
    # 启用梯度检查点
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("✓ 已启用梯度检查点")
    else:
        logger.info("⚠ 模型不支持梯度检查点")
    
    return model, tokenizer


def preprocess_dataset(dataset, tokenizer, max_length: int = 512):
    """预处理数据集 - 添加 text 字段（用于手动训练）"""
    def tokenize_function(examples):
        # 构建文本
        texts = []
        for i in range(len(examples['instruction'])):
            text = f"Instruction: {examples['instruction'][i]}\n"
            if examples['input'][i]:
                text += f"Input: {examples['input'][i]}\n"
            text += f"Response: {examples['output'][i]}"
            texts.append(text)
        
        # Tokenize（不padding，让 collator 处理）
        tokenized = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        
        # 设置 labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # 处理数据集
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['instruction', 'input', 'output'])
    logger.info(f"✓ 数据集已预处理，样本数: {len(dataset)}")
    return dataset


def train_manual(model, train_dataset, eval_dataset, tokenizer, config):
    """手动训练循环 - 绕过 Trainer 的梯度问题"""
    logger.info("开始手动训练循环...")
    
    # TensorBoard writer
    output_dir = config['training']['output_dir']
    tb_dir = os.path.join(output_dir, 'tensorboard')
    writer = SummaryWriter(log_dir=tb_dir)
    logger.info(f"✓ TensorBoard 日志目录: {tb_dir}")
    
    # 数据整理器 - 动态 padding
    # 注意：当 batch_size > 1 时，需要显式把 labels pad 成同长度（并对 pad 位置置 -100）
    def data_collator(features):
        features_wo_labels = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = tokenizer.pad(
            features_wo_labels,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
    
    # DataLoader
    # 注意：batch_size 会显著影响峰值显存（尤其是 backward）。
    # 之前为了保证在 T4 上必定能跑，强制锁为 1；现在改为尊重配置文件。
    batch_size = int(config['training']['per_device_train_batch_size'])
    if batch_size < 1:
        batch_size = 1
    logger.info(f"使用批次大小: {batch_size}（来自 training.per_device_train_batch_size）")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    
    # 优化器
    learning_rate = float(config['training']['learning_rate'])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    # AMP（fp16）: T4 不支持 bf16，这里仅支持 fp16
    use_fp16 = bool(config.get("hardware", {}).get("fp16", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # 梯度累计：用小 micro-batch 控制显存峰值，用累计提高等效 batch 稳定性
    grad_accum_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    if grad_accum_steps < 1:
        grad_accum_steps = 1
    logger.info(
        f"梯度累计步数: {grad_accum_steps}（等效 batch = {batch_size} × {grad_accum_steps} = {batch_size * grad_accum_steps}）"
    )
    
    # 学习率调度器（线性 warmup + 余弦退火）
    total_steps = len(train_loader) * config['training']['num_train_epochs']
    warmup_steps = int(total_steps * config['training'].get('warmup_ratio', 0.03))
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # 余弦退火
        import math
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练循环
    model.train()
    global_step = 0
    total_loss = 0
    logging_steps = config['training']['logging_steps']
    
    for epoch in range(config['training']['num_train_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_train_epochs']}")
        
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            # 将数据移到 GPU
            batch = {k: v.cuda() for k, v in batch.items()}

            # 前向传播（AMP）
            with torch.cuda.amp.autocast(enabled=use_fp16, dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss

            # 反向传播（scaler）
            # 梯度累计：把 loss 按累计步数缩放，保证等效梯度幅度一致
            loss_to_backprop = loss / grad_accum_steps
            if use_fp16:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            do_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(train_loader))
            if do_step:
                # unscale 后再 clip
                if use_fp16:
                    scaler.unscale_(optimizer)

                # 梯度裁剪 & 记录 grad_norm
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )

                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = torch.tensor(0.0, device="cpu")
            
            total_loss += loss.item()
            global_step += 1
            
            # TensorBoard：每步记录 loss/lr/grad_norm
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', current_lr, global_step)
            writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
            writer.add_scalar('train/epoch', epoch + step / len(train_loader), global_step)
            
            if global_step % logging_steps == 0:
                avg_loss = total_loss / logging_steps
                logger.info(
                    f"Step {global_step}/{total_steps}, Loss: {avg_loss:.4f}, "
                    f"LR: {current_lr:.2e}, GradNorm: {grad_norm.item():.3f}"
                )
                # TensorBoard：聚合指标
                writer.add_scalar('train/loss_avg', avg_loss, global_step)
                total_loss = 0
        
        # 每个 epoch 后保存
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(checkpoint_dir)
        logger.info(f"✓ Epoch {epoch + 1} 完成，模型已保存到: {checkpoint_dir}")
    
    # 保存最终模型
    best_model_path = os.path.join(output_dir, 'best_model')
    model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    writer.close()
    logger.info(f"✓ 训练完成！模型已保存到: {best_model_path}")
    logger.info(f"✓ TensorBoard 日志: {tb_dir}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SFT 训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    logger.info(f"正在加载配置文件: {args.config}")
    config = load_config(args.config)

    # T4: 避免触发 sm80/sm90 专用 kernel
    configure_t4_attention_kernels()
    
    # 创建输出目录
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    # 设置模型和分词器
    model, tokenizer = setup_model_and_tokenizer(config['model'], config.get('hardware', {}))
    
    # 加载数据集
    logger.info(f"正在加载数据集: {config['data']['dataset_path']}")
    dataset = load_dataset('json', data_files=config['data']['dataset_path'], split='train')
    
    # 划分训练集和验证集
    dataset = dataset.train_test_split(test_size=config['data']['validation_split_percentage'] / 100)
    
    # 预处理数据集
    logger.info("正在预处理数据集...")
    train_dataset = preprocess_dataset(dataset['train'], tokenizer, config['data']['max_seq_length'])
    eval_dataset = preprocess_dataset(dataset['test'], tokenizer, config['data']['max_seq_length'])
    
    # 手动训练
    train_manual(model, train_dataset, eval_dataset, tokenizer, config)
    
    logger.info(f"训练完成！模型已保存到: {os.path.join(config['training']['output_dir'], 'best_model')}")


if __name__ == '__main__':
    main()
