# GraphRAG Evaluation

This repository contains the code for evaluating **GraphRAG** on tax-related law data in the German language. Note: Data is not included, only the evaluation code.

## Steps to Run the Evaluation

### 0. Your dirs:
```bash
$WORK /home/atuin/b216dc/b216dc15
$HOME /home/hpc/b216dc/b216dc15
$HPCVAULT /home/vault/b216dc/b216dc15
```

### 1. Activate Conda Environment
```bash
conda activate pytorch-2.3.0
```
### 2. Allocate GPU Resources

```bash
salloc --job-name=interactive_test --gres=gpu:a100:1 -C a100_80 --time=00:30:00 --ntasks=1 --cpus-per-task=4 --partition=a100
```

### 3. http and https proxy settings:
```bash
  export http_proxy=http://proxy:80 #use these proxy to access the internet
  export https_proxy=http://proxy:80
  export no_proxy=localhost,127.0.0.1 #do not use proxy for localhost
  export NO_PROXY=localhost,127.0.0.1
```

### 4. Start the VLLM server

IF needed start each server once individually, before starting the script of running them together to load the weights.

LLM LlamaFinetuneBase_Mistral-Nemo-12B
```bash
python -m vllm.entrypoints.openai.api_server --model /$WORK/models/LlamaFinetuneBase_Mistral-Nemo-12B --port 8000 --gpu_memory_utilization=0.7
nohup python -m vllm.entrypoints.openai.api_server --model $WORK/models/LlamaFinetuneBase_Mistral-Nemo-12B --port 8000 --dtype half --gpu_memory_utilization=0.8 --max_model_len=128000 --chat-template $HOME/scripts/tool_chat_template_mistral.jinja > $WORK/logs/mistral_nemo_8000.log 2>&1 &
tail -f $WORK/logs/mistral_nemo_8000.log
```
LLM mistralai_Mistral-Nemo-Base-2407
```bash
nohup python -m vllm.entrypoints.openai.api_server --model $WORK/models/mistralai_Mistral-Nemo-Base-2407 --port 8000 --dtype half --gpu_memory_utilization=0.8 --chat-template $HOME/scripts/tool_chat_template_mistral.jinja > $WORK/logs/mistral_nemo_base_8000.log 2>&1 &
```

```bash
INFO 12-09 23:46:39 launcher.py:19] Available routes are:
INFO 12-09 23:46:39 launcher.py:27] Route: /openapi.json, Methods: GET, HEAD
INFO 12-09 23:46:39 launcher.py:27] Route: /docs, Methods: GET, HEAD
INFO 12-09 23:46:39 launcher.py:27] Route: /docs/oauth2-redirect, Methods: GET, HEAD
INFO 12-09 23:46:39 launcher.py:27] Route: /redoc, Methods: GET, HEAD
INFO 12-09 23:46:39 launcher.py:27] Route: /health, Methods: GET
INFO 12-09 23:46:39 launcher.py:27] Route: /tokenize, Methods: POST
INFO 12-09 23:46:39 launcher.py:27] Route: /detokenize, Methods: POST
INFO 12-09 23:46:39 launcher.py:27] Route: /v1/models, Methods: GET
INFO 12-09 23:46:39 launcher.py:27] Route: /version, Methods: GET
INFO 12-09 23:46:39 launcher.py:27] Route: /v1/chat/completions, Methods: POST
INFO 12-09 23:46:39 launcher.py:27] Route: /v1/completions, Methods: POST
INFO 12-09 23:46:39 launcher.py:27] Route: /v1/embeddings, Methods: POST
```

Embedding
```bash
nohup python -m vllm.entrypoints.openai.api_server --model $WORK/models/intfloat_e5-mistral-7b-instruct --port 8001 --gpu_memory_utilization=0.3 --max-num-seqs 64 > $WORK/logs/mistral_8001.log 2>&1 &
tail -f $WORK/logs/mistral_8001.log
```
prompt tune
```bash
cd graphrag_ollama/taxrag
python -m graphrag prompt-tune --root . --language German --domain tax 
```

iIndexing script
```bash
cd graphrag_ollama
python -m graphrag index --root ./taxrag
```

### 5. Final script

```bash
echo "Load env pytorch-2.3.0"
conda activate pytorch-2.3.0
echo "Load weights of LlamaFinetuneBase_Mistral-Nemo-12B"
#here enter the steps to start serrver of lamaFinetuneBase_Mistral-Nemo-12B model, until the message "INFO:     Uvicorn running on http://0.0.0.0:8000" is received, then shut it down
echo "Load weights of intfloat_e5-mistral-7b-instruct"
##here enter the steps to start serrver of intfloat_e5-mistral-7b-instruct model, until the message "INFO:     Uvicorn running on http://0.0.0.0:8001" is received, then shut it down
echo "set proxies"
export http_proxy=http://proxy:80 #use these proxy to access the internet
export https_proxy=http://proxy:80
export no_proxy=localhost,127.0.0.1 #do not use proxy for localhost
export NO_PROXY=localhost,127.0.0.1
echo "starting prompt tuning script"
#steps here to stat prompt_tune script

### 6. Payload expected by the model:
{
  "model": "/path/to/models/LlamaFinetuneBase_Mistral-Nemo-12B",
  "messages": [
    {"role": "user", "content": "Once upon a time, there was a brave knight."}
  ],
  "temperature": 0.7,
  "max_tokens": 200,
  "stop": ["\n"],
  "top_p": 0.9,
  "top_k": 50
}
