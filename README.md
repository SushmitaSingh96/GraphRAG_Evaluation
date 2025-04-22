This repository contains the high-level steps and essential commands used to run the LLM and embedding model on an HPC cluster, including the environment setup. These are not full scripts but rather the key commands extracted from more detailed Python and Bash scripts. The final report will be added once it is published by my university. The goal of this project was to evaluate GraphRAG, an open-source Retrieval-Augmented Generation (RAG) framework developed by Microsoft, on German legal tax data. 

### 1. Login to cluster front end

### 2. Job Allocation Request
```bash
salloc --job-name=interactive_test --gres=gpu:a100:1 -C a100_80 --time=04:00:00 --ntasks=1 --cpus-per-task=4 --partition=a100
```

### 3. Activate Env and proxy settings
```bash
conda activate pytorch-2.3.0
```
```bash
  export http_proxy=http://proxy:80 #use these proxy to access the internet
  export https_proxy=http://proxy:80
  export no_proxy=localhost,127.0.0.1 #do not use proxy for localhost
  export NO_PROXY=localhost,127.0.0.1
```
### 4. Start LLM model meta-llama_Llama-3.1-8B-Instruct
```bash
nohup python -m vllm.entrypoints.openai.api_server --model $WORK/models/meta-llama_Llama-3.1-8B-Instruct --port 8000 --gpu_memory_utilization=0.8 --chat-template $HOME/scripts/tool_chat_template_llama3.1_json.jinja > $WORK/logs/llama_8b_8000.log 2>&1 &
```
Other Tested Models:

LLM LlamaFinetuneBase_Mistral-Nemo-12B
```bash
python -m vllm.entrypoints.openai.api_server --model /$WORK/models/LlamaFinetuneBase_Mistral-Nemo-12B --port 8000 --gpu_memory_utilization=0.7
nohup python -m vllm.entrypoints.openai.api_server --model $WORK/models/LlamaFinetuneBase_Mistral-Nemo-12B --port 8000 --dtype half --gpu_memory_utilization=0.8 --max_model_len=128000 --chat-template $HOME/scripts/tool_chat_template_mistral.jinja > $WORK/logs/mistral_nemo_8000.log 2>&1 &
tail -f $WORK/logs/mistral_nemo_8000.log
```

LLM mistralai_Mistral-Nemo-Base-2407
```bash
nohup python -m vllm.entrypoints.openai.api_server --model $WORK/models/mistralai_Mistral-Nemo-Base-2407 --port 8000 --gpu_memory_utilization=0.8 --guided-decoding-backend=lm-format-enforcer --chat-template $HOME/scripts/tool_chat_template_mistral.jinja > $WORK/logs/mistral_nemo_base_8000.log 2>&1 &

```

LLM meta-llama_Llama-3.2-1B
```bash
nohup python -m vllm.entrypoints.openai.api_server --model $WORK/models/meta-llama_Llama-3.2-1B --port=8000 --gpu_memory_utilization=0.8 > $WORK/logs/llama_1b_8000.log 2>&1 &
```

### 5. Start Embedding model intfloat_e5-mistral-7b-instruct
```bash
nohup python -m vllm.entrypoints.openai.api_server --model $WORK/models/intfloat_e5-mistral-7b-instruct --port 8001 --gpu_memory_utilization=0.7> $WORK/logs/mistral_8001.log 2>&1 &
```

### 6. Run Prompt Tune script
```bash
cd legalRAG
python -m graphrag prompt-tune --root . --language German
```

### 7. Run Indexing script
```bash
cd ..
python -m graphrag index --root ./legalRAG
```

### Trouble shooting
Logs of the command runs are present in legalRAG/logs
Logs of the server are present in $WORK/logs/





