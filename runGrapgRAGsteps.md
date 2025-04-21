### FINAL STEPS:

### 1. Login to cluster front end
```bash
ssh
```

### 2. Job Allocation Request
```bash
salloc --job-name=interactive_test --gres=gpu:a100:1 -C a100_80 --time=04:00:00 --ntasks=1 --cpus-per-task=4 --partition=a100
```

### 3. Activate Env and proxy settings
```bash
conda activate pytorch-2.3.0
```
  export http_proxy=http://proxy:80 #use these proxy to access the internet
  export https_proxy=http://proxy:80
  export no_proxy=localhost,127.0.0.1 #do not use proxy for localhost
  export NO_PROXY=localhost,127.0.0.1

### 4. Start LLM model meta-llama_Llama-3.1-8B-Instruct
```bash
nohup python -m vllm.entrypoints.openai.api_server --model $WORK/models/meta-llama_Llama-3.1-8B-Instruct --port 8000 --gpu_memory_utilization=0.8 --chat-template $HOME/scripts/tool_chat_template_llama3.1_json.jinja > $WORK/logs/llama_8b_8000.log 2>&1 &
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





