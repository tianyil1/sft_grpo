# SFT + GRPO 训练学习 Demo

> 一个完整的、可跑通的、面向学习的 **SFT + GRPO** 大模型训练 Demo。
>
> 基础模型：**Qwen2.5-3B-Instruct**｜训练框架：**Hugging Face TRL**｜硬件：**Tesla T4 (15GB)**

---

## 目录

1. [项目介绍](#1-项目介绍)
2. [快速开始](#2-快速开始)
3. [项目结构](#3-项目结构)
4. [SFT 原理 + 本项目实践](#4-sft-原理--本项目实践)
5. [GRPO 原理 + 本项目实践](#5-grpo-原理--本项目实践)
6. [完整训练流程对比](#6-完整训练流程对比)
7. [监督与问题排查](#7-监督与问题排查)
8. [踩坑记录（重要）](#8-踩坑记录重要)

---

## 1. 项目介绍

本项目演示了大模型后训练（Post-training）的两个核心阶段：

```
预训练模型 (Qwen2.5-3B-Instruct)
        │
        │  阶段 1：SFT (监督微调)
        │  ─ 让模型学会"跟随指令"
        ▼
SFT 模型 (outputs/sft/best_model)
        │
        │  阶段 2：GRPO (强化学习)
        │  ─ 让模型学会"偏好人类想要的回答"
        ▼
GRPO 模型 (outputs/grpo/best_model)
```

### 本 Demo 的特色

- ✅ **完整可跑通**：每一步都在 Tesla T4 15GB 上实测跑通
- ✅ **开源数据集**：SFT 用 Stanford Alpaca，GRPO 用 Anthropic HH-RLHF
- ✅ **LoRA 微调**：全部使用 LoRA，显存友好
- ✅ **详细注释**：配置、原理、参数说明都在 YAML 和代码注释里
- ✅ **踩坑笔记**：真实训练中遇到的 8+ 个问题都记录在 [踩坑记录](#8-踩坑记录重要)

---

## 2. 快速开始

### 2.1 环境要求

- **GPU**：Tesla T4 (15GB) 或同等/更强（A10/A100/H100 都可）
- **Python**：3.10
- **CUDA**：12.1+
- **磁盘**：≥ 30GB（模型 6GB + 数据 1GB + 输出 2GB + 缓存）

### 2.2 创建 Conda 环境

```bash
# 创建环境（示例：使用环境名；也可用 -p /abs/path/to/env）
conda create -n sft_grpo python=3.10 -y

# 激活环境
conda activate sft_grpo
```

### 2.3 安装依赖

```bash
# PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# 训练相关
pip install 'trl==0.15.2' 'transformers==4.48.0' 'accelerate==0.34.2'
pip install peft datasets pyyaml tensorboard

# 注：GRPO 从 TRL 0.14+ 才引入，0.9.6 及更早版本没有 GRPOTrainer
```

### 2.4 准备数据

```bash
# 进入项目根目录（请按实际路径调整）
cd /path/to/sft_grpo

# 下载并准备开源数据集
python scripts/prepare_data.py \
  --sft_dataset alpaca --num_sft_samples 1000 \
  --grpo_dataset hh_rlhf --num_grpo_samples 500
```

### 2.5 开启训练（推荐用 tmux 后台运行）

```bash
# 创建 tmux 后台会话
tmux new-session -d -s train
tmux attach -t train    # 进入会话

# 阶段 1：SFT 训练（约 27 分钟）
python scripts/run_sft.py --config configs/sft_config.yaml

# 阶段 2：GRPO 训练（demo 版约 15 分钟）
python scripts/run_grpo.py --config configs/grpo_config.yaml

# 退出但不终止：Ctrl+B，然后按 D
```

---

## 3. 项目结构

```
sft_grpo/
├── README.md                  # 本文档
├── configs/
│   ├── sft_config.yaml        # SFT 配置（模型、数据、训练超参）
│   └── grpo_config.yaml       # GRPO 配置（生成、奖励、RL 超参）
├── scripts/
│   ├── prepare_data.py        # 下载开源数据集，转换为项目格式
│   ├── run_sft.py             # SFT 训练主脚本（手动循环，LoRA）
│   └── run_grpo.py            # GRPO 训练主脚本（TRL GRPOTrainer）
├── data/
│   ├── sft_data.json          # Alpaca：instruction/input/output
│   └── grpo_data.json         # HH-RLHF：prompt/reference
├── models/
│   └── Qwen2.5-3B-Instruct/   # base 模型（HuggingFace 下载）
└── outputs/
    ├── sft/best_model/        # SFT 后的 LoRA adapter（~114MB）
    └── grpo/best_model/       # GRPO 后的 LoRA adapter（~57MB）
```

---

## 4. SFT 原理 + 本项目实践

### 4.1 什么是 SFT？

**SFT (Supervised Fine-Tuning)** 是"让预训练模型学会听话"的过程。

预训练模型（如 Qwen2.5-3B）经过海量文本学习后，具有强大的**语言理解和生成能力**，但它的行为是"接龙"——给一段话它就续写。

**问题**：用户希望模型能响应指令（"帮我写一首诗"、"解释量子力学"），而不是瞎续写。

**SFT 的做法**：用 `(指令, 理想回答)` 这样的数据对模型进行**有监督学习**，让它学会：

```
输入: "Instruction: 解释什么是机器学习\nResponse: "
目标: "机器学习是人工智能的一个分支..."
```

### 4.2 SFT 的损失函数

SFT 本质上和语言模型预训练一样，就是 **Next-Token Prediction (因果语言建模)**：

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

- $x_1, x_2, ..., x_T$ 是一条完整的 (指令+回答) 文本
- $P_\theta$ 是模型预测下一个 token 的概率
- 模型参数 $\theta$ 通过梯度下降最大化整段文本的似然

### 4.3 本项目的 SFT 实现

#### 数据格式（Alpaca）

```json
{
  "instruction": "Give three tips for staying healthy.",
  "input": "",
  "output": "1. Eat a balanced diet... 2. Exercise regularly... 3. Get enough sleep..."
}
```

#### 拼接成训练文本

`scripts/run_sft.py` 的 `preprocess_dataset()`：

```python
text = f"Instruction: {example['instruction']}\n"
if example['input']:
    text += f"Input: {example['input']}\n"
text += f"Response: {example['output']}"
# tokenize → 作为 input_ids，同时 labels = input_ids（标准 CLM）
```

#### LoRA 配置

我们用 **LoRA（Low-Rank Adaptation）** 而不是全参数微调，原因：

- 全量微调 3B 模型需要 ~24GB 显存（T4 只有 15GB）
- LoRA 只训练 ~0.96% 的参数（30M / 3B），显存占用小
- LoRA 效果已被工业界广泛验证

LoRA 的核心思想：**冻结原模型权重 $W$，只训练一个低秩增量 $\Delta W = BA$**：

$$
W' = W + \Delta W = W + BA, \quad A \in \mathbb{R}^{r \times d}, \, B \in \mathbb{R}^{d \times r}, \, r \ll d
$$

本项目配置（`sft_config.yaml`）：

```yaml
model:
  use_lora: true
  lora_r: 16           # 秩（rank）
  lora_alpha: 32       # 缩放因子
  lora_dropout: 0.05
  # 应用于注意力和 FFN 的所有线性层
  # target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

#### 关键超参数

```yaml
training:
  num_train_epochs: 3               # 训练 3 轮
  per_device_train_batch_size: 1    # T4 显存限制，批次只能 1
  gradient_accumulation_steps: 8    # 通过梯度累积等效 batch=8
  learning_rate: 2e-4               # LoRA 通常用较大 lr（2e-4）
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"       # 余弦退火
data:
  max_seq_length: 512               # 最大序列长度
```

#### 本项目实际训练曲线

用 Alpaca 1000 条数据训练 3 epochs（共 2700 步）的实际 loss：

| 阶段 | Loss 范围 | 平均 |
|-----|---------|------|
| Step 10（起点） | 1.55 | 1.55 |
| Epoch 1 末（Step 900） | 1.4-1.7 | ~1.50 |
| Epoch 2 末（Step 1800） | 0.9-1.2 | ~1.00 |
| Epoch 3 末（Step 2700） | 0.5-0.8 | **~0.65** |

**Loss 从 1.55 降到 0.65，下降约 58%** ← 这是健康的 SFT 收敛曲线。

#### SFT 后的效果（实测）

```
Q: Explain what machine learning is in one sentence.
A: Machine learning is a branch of artificial intelligence that focuses on
   giving machines the ability to learn and make decisions without human
   intervention. It involves using algorithms to automatically identify
   patterns in data...  ← 学会了按"Instruction→Response"格式回答

Q: What is 15 + 27?
A: 42. The answer is 42. 15 + 27 = 42.  ← 计算正确
```

---

## 5. GRPO 原理 + 本项目实践

### 5.1 为什么还要 GRPO？

SFT 让模型学会"怎么回答"，但无法学到"**什么样的回答更好**"：

| 指令 | 回答 A | 回答 B | SFT 能区分吗？ |
|-----|-------|-------|--------------|
| 介绍下自己 | 我是 AI 助手。 | 您好！我是 Qwen，由阿里云训练的大语言模型，我可以帮您回答问题、写作、编程等等。 | ❌ 不能（两者都"合法"） |

**RLHF（Reinforcement Learning from Human Feedback）** 就是为了解决这个问题：用**奖励信号**告诉模型哪个回答更好。

### 5.2 GRPO vs PPO/DPO

| 方法 | 需要的东西 | 特点 |
|-----|---------|-----|
| **PPO** | Policy + Value 模型 + Reward 模型 + Reference 模型（4 个！） | 经典但资源消耗大 |
| **DPO** | 成对偏好数据 (chosen, rejected) | 简单，不用 RL |
| **GRPO** | Policy + Reward 函数（**不需要 Value 模型**） | DeepSeek-R1 同款，资源节省 |

GRPO 的关键创新：**用同一个 prompt 生成一组 (group) 回答，用组内相对奖励代替 advantage**：

```
对一个 prompt，生成 G 个回答 o_1, o_2, ..., o_G
计算每个回答的奖励 r_1, r_2, ..., r_G
归一化： A_i = (r_i - mean(r)) / std(r)     ← 这是 advantage
用 A_i 作为策略梯度的权重
```

### 5.3 GRPO 的损失函数

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[
\min\!\left(\rho_{i,t} A_i,\ \text{clip}(\rho_{i,t}, 1\!-\!\epsilon, 1\!+\!\epsilon) A_i\right) - \beta\, D_{\text{KL}}[\pi_\theta \parallel \pi_{\text{ref}}]
\right]
$$

其中：

- $\rho_{i,t} = \dfrac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t} \mid q, o_{i,<t})}$ 是新旧策略概率比
- $A_i = \dfrac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$ 是组内相对 advantage
- $D_{\text{KL}}$ 是与参考模型（SFT 模型）的 KL 散度，$\beta$ 是权重
- clip 机制防止策略更新过大（PPO 同款）

**核心思想**：**增大 reward 高的回答的概率，减小 reward 低的回答的概率，但不能偏离 SFT 模型太远（KL 约束）**。

### 5.4 本项目的 GRPO 实现

#### 数据格式（HH-RLHF，只用 prompt）

```json
{
  "prompt": "Why did cells originally combine together to create life?",
  "reference": "..."   // 参考回答，不参与训练
}
```

GRPO **不需要标注的正确答案**，只需要 prompt + 奖励函数。

#### 加载 SFT 模型作为起点

`run_grpo.py` 中关键步骤：

```python
# 1. 加载 base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

# 2. 加载并合并 SFT 的 LoRA adapter
model = PeftModel.from_pretrained(base_model, sft_adapter_path)
model = model.merge_and_unload()   # ← 合并为一个完整模型

# 3. 为 GRPO 添加新的 LoRA adapter（在 SFT 基础上继续训练）
lora_config = LoraConfig(r=8, ...)
model = get_peft_model(model, lora_config)
```

#### 奖励函数（本项目的关键！）

奖励函数决定了 GRPO 会把模型往哪个方向优化。本项目的 v2 版本（`run_grpo.py` 的 `reward_func`）：

```python
def reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        reward = 0.0
        words = completion.strip().split()
        n = len(words)
        
        # 规则 1：高斯长度奖励（目标 60 词，平滑过渡）
        if n == 0:
            reward -= 1.0
        else:
            reward += math.exp(-((n - 60) ** 2) / (2 * 30 ** 2))  # 0 到 1
        
        # 规则 2：完整性（以句号结尾 +0.3）
        if completion.strip().endswith(('.', '!', '?', '。', '！', '？')):
            reward += 0.3
        
        # 规则 3：多样性（唯一词比例 × 0.5）
        if n > 0:
            reward += len(set(w.lower() for w in words)) / n * 0.5
            # 并惩罚连续重复 bigram
        
        # 规则 4：多句奖励（≥2 句话 +0.2）
        if sum(1 for c in completion if c in '.!?') >= 2:
            reward += 0.2
        
        rewards.append(reward)
    return rewards
```

**单元测试的奖励区分度**：

| 回答类型 | Reward |
|---------|-------|
| 空回答 | **-1.000** |
| 重复垃圾（"the the the..."） | +0.026 |
| 一句话（无句号） | +0.825 |
| 正常两句话 | +1.236 |
| 理想三句话 | **+1.526** |

> **关键教训**：奖励函数必须有**足够的区分度**。第一版奖励函数阶梯式给分，所有回答都得 0.65 左右，导致 GRPO 无法学习（详见 [踩坑记录](#8-踩坑记录重要)）。

#### GRPO 关键配置

```yaml
data:
  max_prompt_length: 128
  max_new_tokens: 96          # 生成长度：既不太短被截断，也不太慢

grpo:
  num_generations: 4          # 每个 prompt 生成 4 个回答（用于组内对比）
  temperature: 1.0            # 高温度增加多样性

training:
  per_device_train_batch_size: 4    # 必须 == num_generations（硬约束）
  learning_rate: 2e-5         # GRPO 用比 SFT 小 10 倍的学习率
  max_steps: 20               # demo 快速出结果
```

#### 本项目实际训练（v2）观察

| 指标 | 意义 | 健康值 |
|-----|------|-------|
| `reward` | 平均奖励 | 应稳定上升 |
| `reward_std` | 组内奖励标准差 | ≥ 0.1（太低说明没差异） |
| `kl` | 与 SFT 模型的 KL | 缓慢增长，不能爆炸 |
| `completion_length` | 平均生成长度 | 不应恒等于 max_new_tokens |
| `grad_norm` | 梯度范数 | 0.3-2.0 正常，>10 需警惕 |

---

## 6. 完整训练流程对比

| 维度 | SFT | GRPO |
|-----|-----|------|
| **学习方式** | 监督学习（给答案） | 强化学习（给奖励） |
| **需要的数据** | (指令, 标准答案) | (指令) + 奖励函数 |
| **损失函数** | 交叉熵（Next-Token） | 策略梯度 + KL 惩罚 |
| **训练速度** | 快（每样本 < 1 秒） | 慢（每步要生成 N 个回答，约 30-200 秒） |
| **学习目标** | 学格式、学知识 | 学偏好、学风格 |
| **学习率** | 2e-4（LoRA） | 1e-5 ~ 5e-5（更保守） |
| **轮数** | 2-5 epochs | < 1 epoch（容易过拟合） |
| **评估方式** | Loss 下降 | Reward 上升 + KL 受控 |

---

## 7. 监督与问题排查

### 7.1 训练监控看板（TensorBoard）

本项目已集成 **TensorBoard**，可实时可视化所有关键训练指标。

#### 启动 TensorBoard 服务

```bash
# 激活环境
conda activate sft_grpo

# 启动 TensorBoard（推荐后台运行）
cd /path/to/sft_grpo
tensorboard --logdir outputs/ --host 0.0.0.0 --port 6006 --reload_interval 10

# 或使用 tmux 后台运行
tmux new-session -d -s tensorboard \
  "tensorboard --logdir outputs/ --host 0.0.0.0 --port 6006 --reload_interval 10"
```

#### 在浏览器访问

- 本机：`http://localhost:6006`
- 远程：`http://<机器IP>:6006`

#### 可查看的指标

**SFT 阶段**（写入 `outputs/sft/tensorboard/`）：

| 指标 | 含义 | 健康值 |
|-----|-----|-------|
| `train/loss_step` | 每步原始 loss | 整体下降 |
| `train/loss_avg` | 平滑后的 loss（按 logging_steps 聚合） | 1.5 → 0.6 |
| `train/learning_rate` | 当前学习率（含 warmup + 余弦退火） | 光滑上升→下降 |
| `train/grad_norm` | 梯度范数（已被 clip 到 1.0） | 0.1 - 1.0 |
| `train/epoch` | 训练进度（小数形式） | 0 → num_epochs |

**GRPO 阶段**（写入 `outputs/grpo/tensorboard/`，由 `GRPOTrainer` 自动写入）：

| 指标 | 含义 | 健康值 |
|-----|-----|-------|
| `train/loss` | policy gradient loss | 通常很小（≈0） |
| `train/reward` | 平均奖励 | **应上升** |
| `train/reward_std` | 组内奖励标准差 | **≥ 0.05** 才有学习信号 |
| `train/kl` | 与 SFT 参考模型的 KL 散度 | 缓慢增长，< 0.1 |
| `train/grad_norm` | 梯度范数 | 0.3 - 2.0 |
| `train/completion_length` | 平均生成长度 | **不应恒等于** max_new_tokens |
| `train/learning_rate` | 学习率 | warmup + 衰减 |

#### TensorBoard 截图示意

浏览器打开后，你会看到：

- **SCALARS 面板**：所有数值指标的曲线图，可对比多次 run
- **TIME SERIES 面板**：时间序列视图，支持平滑
- 左侧勾选框可选择性显示 SFT 和 GRPO 的不同指标

### 7.2 如何实时查看训练进度（命令行）

```bash
# 连接 tmux 会话
tmux attach -t train

# 查看训练日志（tail -f 实时刷新）
tail -f /path/to/sft_grpo/training.log
tail -f /path/to/sft_grpo/grpo_training.log

# 监控 GPU
watch -n 1 nvidia-smi
```

### 7.2 判断 SFT 训练是否健康

✅ **健康**：
- Loss 整体下降趋势（从 1.5-2.0 → 0.5-1.0）
- 没有 NaN
- GPU 利用率 > 80%

❌ **异常**：
- Loss = NaN → 学习率过大，或 fp16 溢出 → 降 lr、改 fp32
- Loss 不降 → 数据格式错、lr 过小
- Loss 震荡不收敛 → batch 太小、lr 过大

### 7.3 判断 GRPO 训练是否健康

✅ **健康**：
- `reward` 有上升趋势
- `reward_std` ≥ 0.1（组内有差异）
- `kl` < 0.1（没有严重漂移）
- `completion_length` 有变化（不是恒等值）

❌ **异常**：
- `reward` 横盘或下降 → 奖励函数区分度不够
- `reward_std` ≈ 0 → 同一 prompt 的 N 个回答几乎一样 → 升 temperature 或增加 num_generations
- `kl` 爆炸（> 1） → 学习率过大、`beta` 太小
- `completion_length = max_new_tokens` → 生成长度被截断 → 增大 max_new_tokens

### 7.4 推理测试

训练完成后可以加载 LoRA 做推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_path = "models/Qwen2.5-3B-Instruct"
adapter_path = "outputs/sft/best_model"   # 或 outputs/grpo/best_model

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32).cuda()
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

prompt = "Instruction: Explain quantum computing briefly.\nResponse:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 8. 踩坑记录（重要）

这些是本项目实际训练中真实遇到的问题，以及最终的解决方案。

### 8.1 SFT 阶段

| 问题 | 现象 | 根本原因 | 解决方案 |
|-----|------|---------|---------|
| 梯度不流动 | `RuntimeError: element 0 of tensors does not require grad` | `device_map="auto"` 配合 LoRA+梯度检查点有 bug | 改用 `model.cuda()` 手动放 GPU |
| T4 不支持 bf16 | `Expected is_sm80 \|\| is_sm90 to be true, but got false` | T4 是 Turing 架构，不支持 bf16 的某些 CUDA 内核 | 用 `torch_dtype=float32`（不用 float16 的 AMP） |
| LoRA 梯度为 None | `None of the inputs have requires_grad=True` | 梯度检查点会冻结输入梯度 | 调用 `model.enable_input_require_grads()` |
| OOM（float16 内存不足） | `CUDA out of memory` | float32 的 3B 模型 × 2（激活）超 15GB | 启用 `gradient_checkpointing_enable()` 牺牲速度换内存 |
| Loss 为 NaN（float16 下） | 第 10 步后 loss 全是 NaN | float16 精度不够 + LoRA 参数也是 float16 导致数值不稳定 | 全部改 float32 |

### 8.2 GRPO 阶段

| 问题 | 现象 | 根本原因 | 解决方案 |
|-----|------|---------|---------|
| `GRPOTrainer` 不存在 | `ImportError: cannot import name 'GRPOTrainer'` | TRL 0.9.6 没有 GRPO | 升级到 `trl>=0.14`（本项目用 0.15.2） |
| batch_size 不匹配 | `The global train batch size (1 × 1) must be evenly divisible by the number of generations per prompt (4)` | GRPO 硬约束 | `batch_size` 必须是 `num_generations` 的倍数 |
| **训练"看起来成功"但没学到东西**（最隐蔽！） | reward 一直在 0.65 左右波动，loss ≈ 0，std ≈ 0.01 | 奖励函数用了阶梯式判断（长度 20-200 都给 0.5），所有回答得分几乎一样 | 改为**高斯连续奖励**+惩罚重复+多句奖励，std 从 0.01→0.85 |
| `completion_length` 恒等 64 | 所有 step 的生成长度都等于 `max_new_tokens` | 生成长度设太小，模型总被截断，根本没机会自然结束 | `max_new_tokens` 提高到 96-128 |
| reward_std ≈ 0（组内无差异） | 同一 prompt 生成的 N 个回答几乎一样 | temperature 太低（0.9）+ num_generations 太少（2） | 提高到 temperature=1.0, num_generations=4 |

### 8.3 "成功"的陷阱

**⚠️ 最容易被误导的坑**：GRPO 看起来"训练完成、模型保存、reward 稳定" → 但其实奖励函数无区分度，模型完全没学到东西！

**判断是否真正学习的标准**：
1. `reward_std` **必须 > 0.05**，越大说明对比信号越强
2. `reward` 必须有**上升趋势**，而不是恒定
3. `completion_length` 必须有变化，不能等于 `max_new_tokens`
4. 训练前后对同一 prompt 的生成应有**肉眼可见的差异**

---

## 附录 A：依赖版本对照表

本项目实测可用的版本组合：

```
python==3.10
torch==2.1.2 (cu121)
transformers==4.48.0
trl==0.15.2
accelerate==0.34.2
peft (latest)
datasets (latest)
```

**不兼容的组合**：
- `trl>=1.0` 需要 `torch>=2.5`（有 FSDPModule），旧 PyTorch 用不了
- `trl<0.14` 没有 GRPOTrainer

## 附录 B：硬件参考

| GPU | 显存 | batch_size | 本项目是否可跑 |
|-----|-----|-----------|--------------|
| T4 | 15GB | 1 | ✅ 本项目就是 T4 跑的 |
| V100 | 16/32GB | 2-4 | ✅ |
| A10/A10G | 24GB | 4-8，可用 bf16 | ✅ 更快 |
| A100 | 40/80GB | 8+，可全量微调 | ✅ 最佳 |

---

## 参考资料

- [TRL 官方文档 - GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [DeepSeek-R1 论文（GRPO 重要应用）](https://arxiv.org/abs/2501.12948)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Anthropic HH-RLHF](https://github.com/anthropics/hh-rlhf)

---

*Happy Learning! 🚀*
