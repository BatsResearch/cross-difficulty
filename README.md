# cross-difficulty
<p align="center">
    <a href="https://huggingface.co/papers/2511.21692"><img src="https://img.shields.io/badge/hf-daily_papers-orange?logo=huggingface" /></a>
    <a href="https://arxiv.org/abs/2511.21692"><img src="https://img.shields.io/badge/arxiv-2511.21692-b31b1b?logo=arxiv" /></a>
    <a href="https://huggingface.co/datasets/BatsResearch/Cross-Difficulty"><img src="https://img.shields.io/badge/datasets-Cross--Difficulty-FFD21E?logo=huggingface" /></a>
</p>

This repository contains code for analyzing LLM generalization across difficulty levels, as described in the paper ["Revisiting Generalization Across Difficulty Levels: It's Not So Easy"](https://arxiv.org/abs/2511.21692).

## Setup

1. Clone the repository:
```bash
git clone https://github.com/BatsResearch/cross-difficulty.git
cd cross-difficulty
```

2. Create and activate the conda environment:
```bash
conda env create -f config/vllm-environment.yml
conda activate easy2hard-vllm
```

3. This project uses the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation.

### Setting Up LM Evaluation Harness

1. Clone and install the LM Evaluation Harness separately:
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

2. Copy the evaluation tasks from this project to the LM Evaluation Harness:
```bash
cp -r /path/to/Cross-Difficulty-Generalization/tasks/* /path/to/lm-evaluation-harness/lm_eval/tasks/
```


## Usage

### Training and Evaluation

The main training script handles training, model conversion, and evaluation in a single pipeline.

1. Configure the scripts by adding your paths and settings in scripts provided in scripts directory.

2. Run the pipeline:
```bash
bash scripts/sft_b200.sh
```

This script will:
- Train models on specified difficulty bins
- Convert checkpoints from DeepSpeed ZeRO format to standard PyTorch format
- Run evaluation using LM Eval Harness

### Zero-Shot Evaluation

To run zero-shot evaluation without training:

1. Configure the script by editing `scripts/run-zero-shot-eval-vllm_b200.sh` to set the appropriate model paths and output directories.

2. Run zero-shot evaluation:
```bash
bash scripts/run-zero-shot-eval-vllm_b200.sh
```

## Citation

```bibtex
@misc{kordi2025revisitinggeneralizationdifficultylevels,
      title={Revisiting Generalization Across Difficulty Levels: It's Not So Easy},
      author={Yeganeh Kordi and Nihal V. Nayak and Max Zuo and Ilana Nguyen and Stephen H. Bach},
      year={2025},
      eprint={2511.21692},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.21692},
}
```

LM Evaluation Harness citation:
```bibtex
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {The Language Model Evaluation Harness},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```
