"""
数据处理工具函数
用于SFT和GRPO训练数据的加载、预处理和格式化
"""
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import List, Dict, Optional


def load_json_data(file_path: str) -> List[Dict]:
    """
    从JSON文件加载数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json_data(data: List[Dict], file_path: str):
    """
    保存数据到JSON文件
    
    Args:
        data: 数据列表
        file_path: 输出文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_sft_example(example: Dict) -> str:
    """
    格式化SFT训练样本
    
    Args:
        example: 包含instruction, input, output的字典
        
    Returns:
        格式化后的文本
    """
    text = f"Instruction: {example['instruction']}\n"
    if example.get('input'):
        text += f"Input: {example['input']}\n"
    text += f"Response: {example['output']}"
    return text


def format_grpo_example(example: Dict) -> str:
    """
    格式化GRPO训练样本
    
    Args:
        example: 包含prompt的字典
        
    Returns:
        格式化后的提示文本
    """
    return example['prompt']


def preprocess_sft_data(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """
    预处理SFT数据集
    
    Args:
        dataset: 输入数据集
        tokenizer: 分词器
        max_length: 最大序列长度
        
    Returns:
        预处理后的数据集
    """
    def tokenize_function(examples):
        # 格式化输入
        texts = [format_sft_example({'instruction': instr, 'input': inp, 'output': out}) 
                 for instr, inp, out in zip(examples['instruction'], examples['input'], examples['output'])]
        
        # 分词
        tokenized = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # 设置标签（用于计算损失）
        tokenized['labels'] = tokenized['input_ids'].clone()
        
        return tokenized
    
    return dataset.map(tokenize_function, batched=True)


def preprocess_grpo_data(dataset: Dataset, tokenizer, max_prompt_length: int = 256) -> Dataset:
    """
    预处理GRPO数据集
    
    Args:
        dataset: 输入数据集
        tokenizer: 分词器
        max_prompt_length: 最大提示长度
        
    Returns:
        预处理后的数据集
    """
    def tokenize_function(examples):
        # 分词提示
        tokenized = tokenizer(
            examples['prompt'],
            padding='max_length',
            truncation=True,
            max_length=max_prompt_length,
            return_tensors='pt'
        )
        return tokenized
    
    return dataset.map(tokenize_function, batched=True)


def create_sample_data(output_path: str, data_type: str = 'sft', num_samples: int = 10):
    """
    创建示例数据（如果没有真实数据时使用）
    
    Args:
        output_path: 输出文件路径
        data_type: 数据类型 ('sft' 或 'grpo')
        num_samples: 样本数量
    """
    if data_type == 'sft':
        # 创建SFT示例数据
        samples = []
        templates = [
            ("解释{topic}", "", "这是一个关于{topic}的详细解释..."),
            ("如何使用{tool}？", "", "{tool}的使用方法如下..."),
            ("写一个关于{topic}的简短故事", "", "从前..."),
        ]
        topics = ["机器学习", "Python编程", "数据科学", "人工智能", "深度学习"]
        tools = ["Pandas", "NumPy", "Matplotlib", "Scikit-learn", "PyTorch"]
        
        import random
        for i in range(num_samples):
            template = random.choice(templates)
            if "{topic}" in template[0]:
                topic = random.choice(topics)
                sample = {
                    "instruction": template[0].format(topic=topic),
                    "input": template[1],
                    "output": template[2].format(topic=topic)
                }
            else:
                tool = random.choice(tools)
                sample = {
                    "instruction": template[0].format(tool=tool),
                    "input": template[1],
                    "output": template[2].format(tool=tool)
                }
            samples.append(sample)
    
    else:  # grpo
        # 创建GRPO示例数据
        samples = []
        prompts = [
            "写一首关于{topic}的诗",
            "解释什么是{topic}",
            "如何使用{topic}？",
            "写一个关于{topic}的故事",
            "总结{topic}的要点"
        ]
        topics = ["春天", "人工智能", "机器学习", "数据科学", "Python"]
        
        import random
        for i in range(num_samples):
            prompt_template = random.choice(prompts)
            topic = random.choice(topics)
            sample = {
                "prompt": prompt_template.format(topic=topic),
                "reference": f"这是关于{topic}的参考回答..."
            }
            samples.append(sample)
    
    save_json_data(samples, output_path)
    print(f"已创建{num_samples}条{data_type}示例数据，保存到: {output_path}")


def validate_sft_data(data: List[Dict]) -> bool:
    """
    验证SFT数据格式
    
    Args:
        data: SFT数据列表
        
    Returns:
        是否有效
    """
    required_keys = {'instruction', 'input', 'output'}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"样本{i}不是字典格式")
            return False
        if not required_keys.issubset(item.keys()):
            print(f"样本{i}缺少必要字段: {required_keys - set(item.keys())}")
            return False
    return True


def validate_grpo_data(data: List[Dict]) -> bool:
    """
    验证GRPO数据格式
    
    Args:
        data: GRPO数据列表
        
    Returns:
        是否有效
    """
    required_keys = {'prompt'}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"样本{i}不是字典格式")
            return False
        if not required_keys.issubset(item.keys()):
            print(f"样本{i}缺少必要字段: {required_keys - set(item.keys())}")
            return False
    return True


if __name__ == '__main__':
    # 测试代码
    print("测试数据处理工具...")
    
    # 创建示例数据
    create_sample_data('test_sft_data.json', 'sft', 5)
    create_sample_data('test_grpo_data.json', 'grpo', 5)
    
    # 加载和验证数据
    sft_data = load_json_data('test_sft_data.json')
    print(f"\nSFT数据样本数: {len(sft_data)}")
    print(f"SFT数据格式验证: {validate_sft_data(sft_data)}")
    
    grpo_data = load_json_data('test_grpo_data.json')
    print(f"\nGRPO数据样本数: {len(grpo_data)}")
    print(f"GRPO数据格式验证: {validate_grpo_data(grpo_data)}")
