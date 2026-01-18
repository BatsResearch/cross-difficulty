from typing import Any

from dataclasses import dataclass, field

from trl import SFTConfig


@dataclass
class ExperimentConfig:
  dataset: str = 'arc'
  dataset_quantile_lower_bin: int = 0
  dataset_quantile_upper_bin: int = 5
  quantile_key: str = '1pl_quantile'
  # jinja2 template for the question
  system_prompt: str | None = None

  lora_rank: int = 64
  full_finetuning: bool = False


@dataclass
class PeftConfig:
  target_modules: list[str] = field(default_factory=lambda: [
      'q_proj',
      'k_proj',
      'v_proj',
      'o_proj',
      'gate_proj',
      'up_proj',
      'down_proj',
  ])
  lora_dropout: float = 0
  bias: str = "none"
  layers_to_transform: Any | None = None
  layers_pattern: Any | None = None
  use_rslora: bool = False
  modules_to_save: Any | None = None
  init_lora_weights: bool = True


@dataclass
class UnslothPeftConfig(PeftConfig):
  use_gradient_checkpointing: str = 'unsloth'
  target_modules: list[str] = field(default_factory=lambda: [
      'q_proj',
      'k_proj',
      'v_proj',
      'o_proj',
      'gate_proj',
      'up_proj',
      'down_proj',
  ])


@dataclass
class ModelArguments:
  model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct'
  torch_dtype: str = 'bfloat16'


@dataclass
class QuantConfig:
  load_in_4bit: bool = False
  load_in_8bit: bool = False
  bnb_4bit_use_double_quant: bool = True
  bnb_4bit_quant_type: str = 'nf4'
  bnb_4bit_compute_dtype: str = 'float16'
  quant_method: str = 'bitsandbytes'


@dataclass
class SFTTrainingArguments(SFTConfig):
  output_dir: str = 'output'

  max_seq_length: int | None = None
  seed: int = 42

  bf16: bool = True
  fp16: bool = False

  learning_rate: float = 5e-6
  adam_beta1: float = 0.9
  adam_beta2: float = 0.99
  weight_decay: float = 0.1
  warmup_ratio: float = 0.1
  lr_scheduler_type: str = 'cosine'
  optim: str = 'paged_adamw_8bit'
  logging_steps: int = 1
  per_device_train_batch_size: int = 1
  gradient_accumulation_steps: int = 1

  max_steps: int = -1
  save_steps: int = 100000
  report_to: str = 'wandb'
  max_grad_norm: float = 0.1


@dataclass
class UnslothSFTConfig:
  exp_config: ExperimentConfig
  model_args: ModelArguments
  training_args: SFTTrainingArguments
  peft_config: UnslothPeftConfig
  quant_config: QuantConfig


@dataclass
class TrainSFTConfig:
  exp_config: ExperimentConfig
  model_args: ModelArguments
  training_args: SFTTrainingArguments
  peft_config: PeftConfig
  quant_config: QuantConfig