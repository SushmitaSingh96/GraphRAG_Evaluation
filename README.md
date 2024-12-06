# GraphRAG_Evaluation
Evaluation of GrpahRAG on tax related law data in German Language. The data is not shared here, only the code used to do the evaluation is.

1. Activate conda env: conda activate /apps/jupyterhub/jh3.1.1-py3.11/envs/pytorch-2.3.0. Here I am using an environment provided in the HPC nodes.
2. Allocate GPU resources using salloc: salloc --job-name=interactive_test --gres=gpu:a100:1 -C a100_80 --time=00:30:00 --ntasks=1 --cpus-per-task=4 --partition=a100.
3. Start vllm server: python -m vllm.entrypoints.openai.api_server --model /home/hpc/b216dc/b216dc15/models/facebook_opt-1.3b
