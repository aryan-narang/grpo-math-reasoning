# GRPO Math Reasoning — Fine-tuning Qwen2.5-1.5B on GSM8K

Fine-tuned Qwen2.5-1.5B-Instruct using **GRPO** (Group Relative Policy
Optimization) — the same RL technique behind DeepSeek-R1 — to improve
mathematical reasoning without any human-labelled preference data.

## Results

| Model | GSM8K Accuracy |
|---|---|
| Qwen2.5-1.5B base | ~22% |
| GRPO fine-tuned | ~41% |

[Training curves on wandb →](https://wandb.ai/profile/aryannarang-code?shareProfileType=copy)

## How it works

For each math problem the model generates 4 candidate solutions. Each is
scored by a reward function that checks answer correctness and reasoning
format. GRPO uses the group mean as a baseline — solutions above it are
reinforced, below it are suppressed. No critic network, no human labels.

## Reward design

- **Correctness** (weight 1.0) — binary: extracted answer vs ground truth  
- **Format** (weight 0.1) — rewards structured `<think>...<answer>` output  
- **Length penalty** (max −0.05) — discourages padding in reasoning chains  

## Stack

Qwen2.5-1.5B · PyTorch · HuggingFace TRL · PEFT/LoRA · bitsandbytes 4-bit · wandb · GSM8K

## Run it yourself

Open the notebook in Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aryan-narang/grpo-math-reasoning/blob/main/grpo_math_reasoning.ipynb)

Requirements: T4 GPU (free Colab tier), ~2 hours for 500 training steps.
