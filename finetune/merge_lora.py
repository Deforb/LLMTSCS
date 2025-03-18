"""
LoRA Model Merger

该模块用于将使用LoRA（Low-Rank Adaptation）微调的适配器模型与基础模型合并，
生成可直接部署的完整模型文件。

功能特性：
- 支持因果语言模型（CausalLM）和序列分类模型（SequenceClassification）
- 自动识别PEFT配置任务类型
- 保留原始模型精度（bfloat16）
- 集成Hugging Face模型保存标准

命令行参数：
--adapter_model_name : LoRA适配器路径（必需）
   示例: ./ft_models/ift/llama_13b_hangzhou
--base_model_name    : 基础模型路径（必需）
   示例: ./original_models/llama-13b-hf
--output_name        : 合并输出路径（必需）
   示例: ./merged_models/llama_13b_traffic

使用示例：
python merge_lora.py \
--adapter_model_name ./ft_models/ift/llama_13b_hangzhou \
--base_model_name ./original_models/llama-13b-hf \
--output_name ./merged/llama_13b_hangzhou

注意事项：
1. 需确保基础模型与适配器架构兼容
2. 输出目录需要至少保留2倍基础模型大小的磁盘空间
3. 建议在GPU环境下执行（显存需求与模型大小正相关）
4. 合并后的模型保留原始分词器配置

输出文件：
|- config.json
|- pytorch_model.bin
|- tokenizer_config.json
|- (其他模型相关文件)
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)


@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the
    merged model.
    """

    adapter_model_name: Optional[str] = field(
        default='./ft_models/ift/llama_ift_13b_hangzhou_1',
        metadata={"help": "the adapter name"},
    )
    base_model_name: Optional[str] = field(
        default='./ft_models/merged/llama_ift_13b_hangzhou_1',
        metadata={"help": "the base model name"},
    )
    output_name: Optional[str] = field(
        default='./ft_models/merged/llama_ift_13b_hangzhou_1',
        metadata={"help": "the merged model name"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert (
    script_args.adapter_model_name is not None
), "please provide the name of the Adapter you would like to merge"
assert (
    script_args.base_model_name is not None
), "please provide the name of the Base model"
assert (
    script_args.output_name is not None
), "please provide the output name of the merged model"

peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
if peft_config.task_type == "SEQ_CLS":
    # The sequence classification task is used for the reward model in PPO
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.base_model_name, num_labels=1, torch_dtype=torch.bfloat16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16
    )

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)

# Load the PEFT model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

model = model.merge_and_unload()

model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")
# model.push_to_hub(f"{script_args.output_name}", use_temp_dir=False)
