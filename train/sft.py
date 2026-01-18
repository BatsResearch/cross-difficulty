import os
from dataclasses import asdict

from peft import PeftModel, get_peft_model, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from train.configs import (
    TrainSFTConfig,
    ExperimentConfig,
    ModelArguments,
    SFTTrainingArguments,
    PeftConfig,
    QuantConfig,
)
from train.utils import (
    ChatFormatter,
    DatasetLoader,
    ArgumentParser,
    set_seed,
    get_chat_template,
    get_max_seq_length,
)


class TrainExperiment:

  def __init__(self, config: TrainSFTConfig):
    self.config = config
    if config.training_args.seed is not None:
      set_seed(config.training_args.seed)

    self.model = AutoModelForCausalLM.from_pretrained(
        config.model_args.model_name,
        torch_dtype=config.model_args.torch_dtype,
        #device_map='auto',
        quantization_config=BitsAndBytesConfig(**asdict(config.quant_config))
        if config.quant_config.load_in_4bit or config.quant_config.load_in_8bit
        else None,
    )
    self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        config.model_args.model_name)
    chat_template = get_chat_template(self.tokenizer)
    self.formatter = ChatFormatter(self.tokenizer)

    self.dataset_loader = DatasetLoader(config.exp_config.dataset)
    self.dataset = self.dataset_loader.load_dataset(
        lower_bin=config.exp_config.dataset_quantile_lower_bin,
        upper_bin=config.exp_config.dataset_quantile_upper_bin,
        quantile=config.exp_config.quantile_key,
    )

    if config.exp_config.system_prompt is None:
      config.exp_config.system_prompt = chat_template['system_prompt']
    self.dataset = self.dataset.map(
        lambda x: self.dataset_loader.build_example(
            self.formatter,
            config.exp_config.system_prompt,
            self.dataset_loader.process_example(x),
        ),
        remove_columns=self.dataset.column_names,     )

    if not config.exp_config.full_finetuning:
      self.model: PeftModel = get_peft_model(
          self.model,
          LoraConfig(
              r=config.exp_config.lora_rank,
              lora_alpha=config.exp_config.lora_rank,
              **asdict(config.peft_config),
          ),
      )

    if config.training_args.max_seq_length is None:
      # set max_seq_length to that of the datasets
      config.training_args.max_seq_length = get_max_seq_length(
          self.dataset,
          self.tokenizer,
      )
      print('max_seq_length set to', config.training_args.max_seq_length)
    # NOTE: remove below line if not working
    config.training_args.use_liger = True

    self.trainer = SFTTrainer(
        model=self.model,
        processing_class=self.tokenizer,
        data_collator=DataCollatorForCompletionOnlyLM(
            response_template=chat_template['response_template'],
            instruction_template=chat_template['instruction_template'],
            tokenizer=self.tokenizer,
        ),
        args=config.training_args,
        train_dataset=self.dataset,
    )

  def run(self):
    self.trainer.train()

  def save_model(self):
    ds_name, model_name = (
        self.config.exp_config.dataset.replace('/', '--'),
        self.config.model_args.model_name.replace('/', '--'),
    )
    output_dir = os.path.join(
        self.config.training_args.output_dir,
        f'{model_name}_{ds_name}_{self.config.training_args.seed}',
    )
    self.tokenizer.save_pretrained(output_dir)

    if isinstance(self.model, PeftModel):
      # save adapter first
      try:
        self.model.save_pretrained(output_dir + '_adapter')
      except Exception as e:
        print(f"Error saving LoRA model: {e}")

      # merge and save the model
      model: PreTrainedModel = self.model.merge_and_unload()
    else:
      model: PreTrainedModel = self.model

    # save the model
    try:
      model.save_pretrained(output_dir)
    except Exception as e:
      print(f"Error saving model: {e}")

    try:
      self.trainer.save_model(f'{output_dir}--trainer-saved')
    except Exception as e:
      print(f"Error saving model: {e}")


def main(config: TrainSFTConfig):
  """Train the model with unsloth.

  Args:
      config (Config): Configuration for the training.
  """
  experiment = TrainExperiment(config)
  experiment.run()
  experiment.save_model()


if __name__ == '__main__':
  parser = ArgumentParser((
      ExperimentConfig,
      ModelArguments,
      SFTTrainingArguments,
      PeftConfig,
      QuantConfig,
  ))

  (
      exp_config,
      model_args,
      training_args,
      peft_config,
      quant_config,
  ) = parser.parse_yaml_args_into_dataclasses()

  config = TrainSFTConfig(
      exp_config=exp_config,
      model_args=model_args,
      training_args=training_args,
      peft_config=peft_config,
      quant_config=quant_config,
  )

  main(config)
