import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint


def main(model_path: str, model_id: str, save_model_path: str):
  """
  Convert a DeepSpeed Zero checkpoint to a Hugging Face format.

  Args:
      model_path (str): Path to the DeepSpeed Zero checkpoint.
      model_id (str): Identifier for the model.
  """
  model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
  model: PreTrainedModel = load_state_dict_from_zero_checkpoint(model, model_path)
  model.to(torch.bfloat16).save_pretrained(save_model_path)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Convert DeepSpeed Zero checkpoint to Hugging Face format.")
  parser.add_argument("--model_path", type=str, required=True, help="Path to the DeepSpeed Zero checkpoint.")
  parser.add_argument("--model_id", type=str, required=True, help="Identifier for the model.")
  parser.add_argument("--save_model_path", type=str, required=True, help="Path to save the converted model.")

  args = parser.parse_args()
  main(args.model_path, args.model_id, args.save_model_path)
  print(f"Model saved to {args.save_model_path} in Hugging Face format.")