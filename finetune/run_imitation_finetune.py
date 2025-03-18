"""
LoRA 模仿微调训练模块

本模块用于实现基于LoRA（Low-Rank Adaptation）的模仿学习微调，通过专家轨迹数据训练语言模型掌握交通信号控制策略。

功能特性：
- 支持LLaMA等因果语言模型的低秩适配
- 支持观察掩码（Observation Masking）训练模式
- 集成梯度累积与分布式训练
- 自动保存最佳模型检查点
- 支持BF16混合精度训练

主要参数：
--base_model         预训练模型路径（必需）
                     示例：./original_models/llama-13b-hf
--data_path          训练数据路径（JSON格式）
                     默认：imitation_fine_tuning_data_jinan_1.json
--output_dir         模型输出目录（必需）
                     示例：./ft_models/ift/llama_ift_13b_jinan_1
--lora_r             LoRA秩维度（默认：8）
--lora_alpha         LoRA缩放系数（默认：16）
--micro_batch_size   单设备批大小（默认：16）
--num_epochs         训练轮次（默认：30）
--cutoff_len         序列截断长度（默认：2048）
--mask               启用观察掩码模式（默认：False）

使用示例：
# 基础训练
python run_imitation_finetune.py \
--base_model ./original_models/llama-13b-hf \
--data_path imitation_data.json \
--output_dir ./ft_models/ift/llama_13b_jinan

注意事项：
1. 需提前安装Peft、Transformers等依赖库
2. 建议在配备24G+显存的GPU上运行13B模型
3. 输出目录会自动创建，需确保有写入权限
4. 使用--mask参数时会对观察状态进行掩码处理
5. 最终保存的为LoRA适配器权重，需使用merge_lora.py合并
"""

import os
import sys
from typing import List
import fire
import torch
from datasets import load_dataset
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
)


def train(
    # model/data params
    base_model: str = "llama_2_13b_chat_hf",  # the only required argument
    data_path: str = "imitation_fine_tuning_data_jinan_1.json",
    output_dir: str = "../../ft_models/ift/llama_ift_13b_jinan_1",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 16,
    num_epochs: int = 30,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,
    val_set_size: int = 0.05,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = True,  # faster, but produces an odd training loss curve
    # other
    mask: bool = False,
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    # 累积多少步的梯度后更新参数
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # 总进程数
    ddp = world_size != 1  # 是否启用分布式数据并行（Distributed Data Parallel）
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps //= world_size

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        # torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast

        if mask:
            print('Masking Obersevation')
            tokenizer.mask_token = "~"

            # Split the input text into lines
            lines = prompt.split('\n')

            # Initialize an empty list to store the modified lines
            masked_lines = []

            # Initialize a flag to indicate if we are between "Observation:" and "Thought:"
            between_observation_and_thought = False

            # Iterate through each line
            for line in lines:
                if "Observation:" in line:
                    between_observation_and_thought = True
                    # split the line and mask all but the first word
                    line = line.split()
                    line[1:] = [tokenizer.mask_token] * len(line[1:])
                    line = " ".join(line)
                    masked_lines.append(line)  # Add the line as-is
                else:
                    masked_lines.append(line)  # Add the line as-is

            # Concatenate the modified lines to form the masked text
            masked_text = '\n'.join(masked_lines)

            prompt = masked_text

        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )

        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        else:
            if len(result["input_ids"]) >= cutoff_len:
                print("WARNING: input too long, truncating")

        masked_token_id = tokenizer.mask_token_id
        ids = [
            -100 if token_id == 3695 else token_id for token_id in result["input_ids"]
        ]

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    data = load_dataset("json", data_files=data_path)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=2024
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=20 if val_set_size > 0 else None,
            save_steps=20,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    return f"""你是一个专业的水库调度专家，请根据当前水库状态和咸潮预测，优化每日放水量。

### 当前状态：
储水量（万立方米）: 
- S1: {data_point["w1"]}
- S2: {data_point["w2"]}
- S3: {data_point["w3"]}

### 咸潮预警：
未来咸潮发生概率：
{data_point["salinity_risk_forecast"]}

### 操作目标：
在满足以下约束条件下，确定今日最优放水量（x1, x2, x3）：
1. S2/S3遭遇咸潮时仅允许放水（x≤0）
2. 总储水量不得低于安全水位
3. 最大化奖励函数值：RF(x1,x2,x3) = {data_point["reward_function"]}

### 响应格式：
放水量（万立方米/日）: 
- S1: [x1值] 
- S2: [x2值]
- S3: [x3值]
奖励值计算：r = [详细计算过程]{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
