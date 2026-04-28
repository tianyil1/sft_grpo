"""
模型工具函数
用于模型加载、保存、推理等操作的辅助函数
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import PeftModel, PeftConfig
from typing import List, Optional, Dict
import os


def load_base_model(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    device_map: str = "auto",
    torch_dtype = torch.float16
):
    """
    加载基础模型
    
    Args:
        model_name_or_path: 模型名称或路径
        trust_remote_code: 是否信任远程代码
        device_map: 设备映射
        torch_dtype: 模型精度
        
    Returns:
        模型和分词器
    """
    print(f"正在加载基础模型: {model_name_or_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    
    return model, tokenizer


def load_lora_model(
    base_model_name_or_path: str,
    lora_weights_path: str,
    trust_remote_code: bool = True,
    device_map: str = "auto",
    torch_dtype = torch.float16
):
    """
    加载LoRA微调后的模型
    
    Args:
        base_model_name_or_path: 基础模型名称或路径
        lora_weights_path: LoRA权重路径
        trust_remote_code: 是否信任远程代码
        device_map: 设备映射
        torch_dtype: 模型精度
        
    Returns:
        合并后的模型和分词器
    """
    print(f"正在加载LoRA模型: {lora_weights_path}")
    
    # 加载基础模型
    model, tokenizer = load_base_model(
        base_model_name_or_path,
        trust_remote_code,
        device_map,
        torch_dtype
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(
        model,
        lora_weights_path
    )
    
    # 合并LoRA权重到基础模型（可选，推理时可以提高速度）
    # model = model.merge_and_unload()
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True
) -> str:
    """
    使用模型生成回复
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 输入提示
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_p: nucleus sampling参数
        top_k: top-k sampling参数
        repetition_penalty: 重复惩罚
        do_sample: 是否使用采样
        
    Returns:
        生成的回复
    """
    # 构建输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 配置生成参数
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 只返回生成的部分（去掉输入提示）
    response = response[len(prompt):].strip()
    
    return response


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 4,
    **kwargs
) -> List[str]:
    """
    批量生成回复
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompts: 输入提示列表
        batch_size: 批次大小
        **kwargs: 其他生成参数
        
    Returns:
        生成的回复列表
    """
    responses = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # 编码
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', 256),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=kwargs.get('do_sample', True),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            response = response[len(batch_prompts[j]):].strip()
            responses.append(response)
    
    return responses


def save_model_and_tokenizer(
    model,
    tokenizer,
    output_dir: str,
    save_peft_only: bool = True
):
    """
    保存模型和分词器
    
    Args:
        model: 模型
        tokenizer: 分词器
        output_dir: 输出目录
        save_peft_only: 是否只保存PEFT权重（如果使用了LoRA）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    if save_peft_only and hasattr(model, 'peft_config'):
        # 只保存LoRA权重
        model.save_pretrained(output_dir)
    else:
        # 保存完整模型
        model.save_pretrained(output_dir)
    
    # 保存分词器
    tokenizer.save_pretrained(output_dir)
    
    print(f"模型已保存到: {output_dir}")


def calculate_model_size(model) -> Dict:
    """
    计算模型大小
    
    Args:
        model: 模型
        
    Returns:
        包含模型大小信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "model_size_mb": total_params * 4 / (1024 * 1024)  # 假设float32
    }


def print_model_info(model):
    """
    打印模型信息
    
    Args:
        model: 模型
    """
    info = calculate_model_size(model)
    
    print("=" * 50)
    print("模型信息:")
    print(f"总参数量: {info['total_parameters']:,}")
    print(f"可训练参数量: {info['trainable_parameters']:,}")
    print(f"可训练参数比例: {info['trainable_ratio']:.2%}")
    print(f"模型大小(估算): {info['model_size_mb']:.2f} MB")
    print("=" * 50)


if __name__ == '__main__':
    # 测试代码
    print("测试模型工具函数...")
    
    # 注意：这里需要实际的模型路径才能运行测试
    # model, tokenizer = load_base_model("Qwen/Qwen2.5-3B-Instruct")
    # print_model_info(model)
    # 
    # response = generate_response(model, tokenizer, "你好，请介绍一下自己")
    # print(f"生成回复: {response}")
    
    print("模型工具函数已定义完成")
