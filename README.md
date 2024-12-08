# GraphRAG Evaluation

This repository contains the code for evaluating **GraphRAG** on tax-related law data in the German language. Note: Data is not included, only the evaluation code.

## Steps to Run the Evaluation

### 0. Your dirs:
```bash
$WORK /home/atuin/b216dc/b216dc15
$HOME /home/hpc/b216dc/b216dc15
$HPCVAULT /home/vault/b216dc/b216dc15

### 1. Activate Conda Environment
```bash
conda activate /apps/jupyterhub/jh3.1.1-py3.11/envs/pytorch-2.3.0
```
### 2. Allocate GPU Resources

```bash
salloc --job-name=interactive_test --gres=gpu:a100:1 -C a100_80 --time=00:30:00 --ntasks=1 --cpus-per-task=4 --partition=a100
```

### 3. Start the VLLM server

```bash
python -m vllm.entrypoints.openai.api_server --model /home/hpc/b216dc/b216dc15/models/facebook_opt-1.3b
```
