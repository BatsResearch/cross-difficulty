import dataclasses
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
import ast

CHAT_TEMPLATES = {
    'llama-3.2': {
        'system_prompt':
            '',
        'response_template':
            '<|start_header_id|>assistant<|end_header_id|>\n\n',
        'instruction_template':
            '<|start_header_id|>user<|end_header_id|>\n\n',
    },
    'qwen-2.5': {
        'system_prompt':
            'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
        'response_template':
            '<|im_start|>assistant\n',
        'instruction_template':
            '<|im_start|>user\n',
    },
}


def set_seed(seed: int):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)


def get_chat_template(tokenizer: PreTrainedTokenizer) -> dict[str, str]:
  for model_type, templates in CHAT_TEMPLATES.items():
    model_template: str = tokenizer.apply_chat_template(
        [{
            'role': 'system',
            'content': templates['system_prompt']
        }, {
            'role': 'user',
            'content': ''
        }, {
            'role': 'assistant',
            'content': ''
        }],
        tokenize=False,
    )

    if templates['response_template'] in model_template and templates[
        'instruction_template'] in model_template:
      if model_type == 'llama-3.2':
        # llama-3.2 needs to set pad token
        tokenizer.pad_token = '<|finetune_right_pad_id|>'

      return templates

  raise ValueError(
      f"Unknown model type for tokenizer {tokenizer.__class__.__name__}.")


def get_max_seq_length(dataset: Dataset, tokenizer: PreTrainedTokenizer) -> int:
  lengths = dataset.map(
      lambda x: {
          'lengths': [
              len(t) for t in tokenizer(
                  x['text'],
                  truncation=False,
              )['input_ids']
          ]
      },
      remove_columns=dataset.column_names,
      batched=True,
  )

  return max(lengths['lengths'])


class ChatFormatter:

  def __init__(self, tokenizer: PreTrainedTokenizer):
    self.tokenizer: PreTrainedTokenizer = tokenizer

  def format_prompt(
      self,
      system_prompt: str,
      user_prompt: str,
      answer: str = '',
      apply_template: bool = True,
      add_generation_prompt: bool = True,
      include_answer: bool = True,
  ) -> str:
    messages = [
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': user_prompt
        },
    ]

    if include_answer:
      messages.append({'role': 'assistant', 'content': answer})

    if apply_template:
      return self.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=add_generation_prompt and not include_answer,
      )
    else:
      return messages


class DatasetLoader:

  def __init__(self, dataset: str):
    self.dataset_name = dataset

  def load_dataset(
      self,
      lower_bin: int = 0,
      upper_bin: int = 5,
      quantile: str = '1pl_quantile',
  ) -> Dataset:
    match self.dataset_name:
      case 'mmlu_pro':
        dataset = self.load_mmlu_pro()
      case 'arc':
        dataset = self.load_arc()
      case 'gsm8k':
        dataset = self.load_gsm8k()
      case 'ifeval':
        dataset = self.load_ifeval()
      case 'bbh':
        dataset = self.load_bbh()
      case 'gpqa_extended':
        dataset = self.load_gpqa_extended()
      case 'musr':
        dataset = self.load_musr()
      case 'math':
        dataset = self.load_math()
      case 'math-solutions-qwen':
        dataset = self.load_math_solutions_qwen()
      case 'math-solutions-llama':
        dataset = self.load_math_solutions_llama()
      case _:
        raise ValueError(f'Unknown dataset: {self.dataset_name}')

    lower_quantile = lower_bin / 10
    upper_quantile = upper_bin / 10
    if upper_quantile == 1:
      upper_quantile += 0.01

    # filter dataset by quantile
    dataset = dataset.filter(
        lambda x: lower_quantile <= x[quantile] < upper_quantile)

    return dataset

  def load_arc(self) -> Dataset:
    dataset = load_dataset(
        'Yeganeh/Cross-Difficulty',
        'arc',
        split='train',
    )

    return dataset

  def load_gsm8k(self) -> Dataset:
    dataset = load_dataset(
        'Yeganeh/Cross-Difficulty',
        'gsm8k',
        split='train',
    )

    return dataset

  def load_mmlu_pro(self) -> Dataset:
    dataset = load_dataset(
        'Yeganeh/Cross-Difficulty',
        'mmlu_pro',
        split='train',
    )

    return dataset

  def load_ifeval(self) -> Dataset:
    dataset = load_dataset(
        'Yeganeh/Cross-Difficulty',
        'ifeval',
        split='train',
    )

    return dataset

  def load_bbh(self) -> Dataset:
    dataset = load_dataset(
        'Yeganeh/Cross-Difficulty',
        'bbh',
        split='train',
    )

    return dataset

  def load_gpqa_extended(self) -> Dataset:
    dataset = load_dataset(
        'Yeganeh/Cross-Difficulty',
        'gpqa_extended',
        split='train',
    )

    return dataset

  def load_musr(self) -> Dataset:
    dataset = load_dataset(
        'Yeganeh/Cross-Difficulty',
        'musr',
        split='train',
    )

    return dataset

  def load_math(self) -> Dataset:
    dataset = load_dataset(
        'Yeganeh/Cross-Difficulty',
        'math',
        split='train',
    )

    return dataset

  def load_math_solutions_qwen(self) -> Dataset:
    dataset = load_dataset(
        'Yeganeh/Model-Format-Consistent-Datasets',
        'math-solutions__Qwen_Qwen2.5-14B-Instruct',
        split='train',
    )

    return dataset

  def load_math_solutions_llama(self) -> Dataset:
    dataset = load_dataset(
        'Yeganeh/Model-Format-Consistent-Datasets',
        'math-solutions__meta-llama_Llama-3.1-8B-Instruct',
        split='train',
    )

    return dataset


  def process_example(self, example: dict) -> dict:
    match self.dataset_name:
      case 'mmlu_pro':
        return self.process_mmlu_pro_example(example)
      case 'arc':
        return self.process_arc_example(example)
      case 'gsm8k':
        return self.process_gsm8k_example(example)
      case 'ifeval':
        return self.process_ifeval_example(example)
      case 'bbh':
        return self.process_bbh_example(example)
      case 'gpqa_extended':
        return self.process_gpqa_extended_example(example)
      case 'musr':
        return self.process_musr_example(example)
      case 'math':
        return self.process_math_example(example)
      case 'math-solutions-qwen':
        return self.process_math_solutions_example(example)
      case 'math-solutions-llama':
        return self.process_math_solutions_example(example)
      case _:
        raise ValueError(f'Unknown dataset: {self.dataset_name}')

  def process_mmlu_pro_example(self, example: dict) -> dict:
    # Process the example to extract the user prompt and response
    question = example['question']
    options = example['options']
    letter_options = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    choices_prompt = '\n'.join(f'{label}) {choice}' for choice, label in zip(
        options,
        letter_options,
    ))

    user_prompt = f"Question: {question}\n{choices_prompt}\nAnswer:\n"
    answer = example['answer']

    return {
        'user_prompt': user_prompt,
        'answer': answer,
    }

  def process_arc_example(self, example: dict) -> dict:
    # Process the example to extract the user prompt and response
    question = example['question']
    choices = example['choices']

    choices_prompt = '\n'.join(f'{label}) {choice}' for label, choice in zip(
        choices['label'],
        choices['text'],
    ))

    user_prompt = f"Question: {question}\n{choices_prompt}\nAnswer:\n"
    answer = example['answerKey']

    return {
        'user_prompt': user_prompt,
        'answer': answer,
    }

  def process_ifeval_example(self, example: dict) -> dict:
    question = example['prompt']
    answer = example['answer']

    return {
        'user_prompt': question,
        'answer': answer,
    }

  def process_gsm8k_example(self, example: dict) -> dict:
    # Process the example to extract the user prompt and response
    question = example['question']
    response, answer = example['answer'].split('#### ')
    response = re.sub(r'<<.*?>>', '', response).strip()

    return {
        'user_prompt': f'Q: {question}\nA:\n',
        'answer': f'{response} So the answer is {answer}',
    }

  def process_bbh_example(self, example: dict) -> dict:
    question = example['question']

    user_prompt = f"Question: {question}\nAnswer:\n"
    answer = example['answer']

    return {
        'user_prompt': user_prompt,
        'answer': answer,
    }

  def process_gpqa_extended_example(self, example: dict) -> dict:
    question = example['question']
    choices = example['options']
    letter_options = 'ABCD'
    choices_prompt = '\n'.join(f'{label}) {choice}' for choice, label in zip(
        choices,
        letter_options,
    ))

    user_prompt = f"Question: {question}\n{choices_prompt}\nAnswer:\n"
    answer = example['answer']

    return {
        'user_prompt': user_prompt,
        'answer': answer,
    }

  def process_musr_example(self, example: dict) -> dict:
    narrative = example['narrative']
    question = example['question']
    choices_list = example['options']

    # If choices_list is a string, parse it as a Python literal
    if isinstance(choices_list, str):
      choices_list = ast.literal_eval(choices_list)

    choices = ""
    for i, choice in enumerate(choices_list):
      choices += f"{i + 1} - {choice}\n"
    user_prompt = f"{narrative}\n\n{question}\n\n{choices}\nAnswer:"
    answer = example['answer']

    return {
        'user_prompt': user_prompt,
        'answer': answer,
    }

  def process_math_example(self, example: dict) -> dict:
    question = example['question']
    answer = example['answer']

    return {
        'user_prompt': f'Problem:\n{question}\n\nAnswer:',
        'answer': answer,
    }

  def process_math_solutions_example(self, example: dict) -> dict:
    question = example['question']

    solution = example['model_solution'] if example.get('model_solution') else example['solution']
    answer = example['answer']

    return {
        'user_prompt': f'Problem:\n{question}\n\nSolution:',
        'answer': solution,
    }


  def build_example(
      self,
      formatter: ChatFormatter,
      system_prompt: str,
      example: dict,
      include_answer: bool = True,
  ) -> dict[str, str]:
    """Builds an example for the dataset.

    Args:
        formatter (ChatFormatter): Chat formatter.
        system_prompt (str): System prompt.
        example (dict): Example from the dataset.

    Returns:
        dict[str, str]: Formatted example.
    """
    return {
        'text':
            formatter.format_prompt(
                system_prompt,
                example['user_prompt'],
                answer=example['answer'],
                apply_template=True,
                add_generation_prompt=False,
                include_answer=include_answer,
            )
    }


class ArgumentParser(transformers.HfArgumentParser):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def parse_yaml_args_into_dataclasses(
      self,
      args=None,
      return_remaining_strings=False,
      args_file_flag: str = '--config',
  ) -> tuple[transformers.hf_argparser.DataClass, ...]:
    """
    Parse command-line args into instances of the specified dataclass types.

    This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
    docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

    Args:
        args:
            List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
        return_remaining_strings:
            If true, also return a list of remaining argument strings.
        look_for_args_file:
            If true, will look for a ".args" file with the same base name as the entry point script for this
            process, and will append its potential content to the command line args.
        args_filename:
            If not None, will uses this file instead of the ".args" file specified in the previous argument.
        args_file_flag:
            If not None, will look for a file in the command-line args specified with this flag. The flag can be
            specified multiple times and precedence is determined by the order (last one wins).

    Returns:
        Tuple consisting of:

            - the dataclass instances in the same order as they were passed to the initializer.abspath
            - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
              after initialization.
            - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
    """

    if args_file_flag:
      args_files = []

      # args files specified via command line flag should overwrite default args files so we add them last
      # Create special parser just to extract the args_file_flag values
      args_file_parser = ArgumentParser()
      args_file_parser.add_argument(args_file_flag, type=str, action="append")

      # Use only remaining args for further parsing (remove the args_file_flag)
      cfg, args = args_file_parser.parse_known_args(args=args)
      cmd_args_file_paths = vars(cfg).get(args_file_flag.lstrip("-"), None)

      if cmd_args_file_paths:
        args_files.extend([Path(p) for p in cmd_args_file_paths])

      file_args = []
      for args_file in args_files:
        if args_file.exists():
          # read as yaml, then convert to args
          with open(args_file, "r") as f:
            file_args += [f"--{k}={v}" for k, v in yaml.safe_load(f).items()]

      # in case of duplicate arguments the last one has precedence
      # args specified via the command line should overwrite args from files, so we add them last
      args = file_args + args if args is not None else file_args + sys.argv[1:]

    namespace, remaining_args = self.parse_known_args(args=args)
    outputs = []
    for dtype in self.dataclass_types:
      keys = {f.name for f in dataclasses.fields(dtype) if f.init}
      inputs = {k: v for k, v in vars(namespace).items() if k in keys}
      for k in keys:
        delattr(namespace, k)
      obj = dtype(**inputs)
      outputs.append(obj)
    if len(namespace.__dict__) > 0:
      # additional namespace.
      outputs.append(namespace)
    if return_remaining_strings:
      return (*outputs, remaining_args)
    else:
      if remaining_args:
        raise ValueError(
            f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}"
        )

      return (*outputs,)
