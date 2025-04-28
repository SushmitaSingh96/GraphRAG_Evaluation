# 🚀 GraphRAG Evaluation for Legal Domain

This project systematically evaluates Microsoft's GraphRAG framework on a German tax law dataset, combining LLM-based retrieval with advanced graph search.

🔹 Built full GraphRAG pipelines (Indexing, Prompt Tuning, Baseline, Querying)  
🔹 Used [vLLM](https://github.com/vllm-project/vllm) to serve local LLaMA-3.1-8B-Instruct models as an OpenAI-compatible API  
🔹 Customized large language models (LLaMA-3.1-8B-Instruct) and embeddings (e5-mistral-7b-instruct)  
🔹 Benchmarked retrieval effectiveness using RAGAS and GPT-4o-mini evaluations  
🔹 Scalable to HPC clusters with Slurm job automation and optimized for A100 GPUs

---

## 🚀 Getting Started

### 1. Install Requirements

Clone this repository and install the required packages:

```bash
pip install -r requirements.txt
```

Follow Microsoft's [GraphRAG Installation Guide](https://microsoft.github.io/graphrag/get_started/) to correctly configure GraphRAG.

The GraphRAG workspace used in this project is named **`legalRAG`**.

---

### 2. Important Libraries & Models

| Purpose            | Model Name                          |
| :----------------- | :---------------------------------- |
| Main LLM            | `Llama-3.1-8B-Instruct`             |
| Embedding Model     | `e5-mistral-7b-instruct`            |
| Evaluation LLM (RAGAS) | `gpt-4o-mini`                   |

### 3. Modifications to Default GraphRAG Settings

The following changes were made to improve performance:

| Parameter                  | Original | New  | Description |
| :------------------------- | :------: | :--: | :----------- |
| `LLM_MAX_TOKENS`            | 4000     | 1024 | Reduced maximum context size |
| `LLM_TEMPERATURE`           | 0        | 0.3  | Introduced slight variability in responses |
| `LLM_TOP_P`                 | 1        | 0.8  | Limited token diversity |
| `LLM_CONCURRENT_REQUESTS`   | 25       | 15   | Reduced concurrent requests for stability |

## 4. Job Submission

All tasks (indexing, tuning, baseline creation, querying) are handled using a **single Slurm job script**:  
**`run_graphrag_job.sh`**

#### Usage:

```bash
sbatch run_graphrag_job.sh [index|tune|query|baseline]
```

- To run indexing:
  ```bash
  sbatch run_graphrag_job.sh index
  ```

- To run prompt tuning:
  ```bash
  sbatch run_graphrag_job.sh tune
  ```

- To create baseline (without GraphRAG retrieval):
  ```bash
  sbatch run_graphrag_job.sh baseline
  ```

- To run querying:
  ```bash
  sbatch run_graphrag_job.sh query
  ```

Each job automatically:
- Sets the correct job name
- Redirects logs to appropriate folders (`batch_logs/logs/`)
- Activates the environment
- Executes the corresponding Python script

---

## 📂 Repository Structure

```bash
GraphRAG_Evaluation/
├── run_graphrag_job.sh           # Unified Slurm job submission script
├── scripts/
│    ├── indexing.py              # Indexing script
│    ├── prompt_tune.py           # Prompt tuning script
│    ├── query_graphrag.py        # GraphRAG querying script
│    └── baseline_query.py        # Baseline (LLM-only querying) script
├── requirements.txt              # Python package requirements
└── README.md
```

---

## 📚 Project Details

| Component | Description |
| :--- | :--- |
| **vLLM** | Version 0.6.5 |
| **GraphRAG** | Version 1.0.0 |
| **Requirements** | See `requirements.txt` (all libraries and versions listed) |

### Models Used
| Purpose | Model |  
| :--- | :--- | 
| Main LLM | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 
| Embedding Model | [intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) 
| Evaluation Model | `gpt-4o-mini` 

---

### Evaluation (RAGAS Benchmarking)

Evaluation metrics were computed using RAGAS to assess answer correctness and answer relevancy.

GraphRAG’s retrieval-augmented approach was observed to improve grounding and relevance of generated responses compared to querying the LLM alone.

**Detailed evaluation findings will be made available upon official publication.**

---

## 📢 Notes

- This repository assumes that GraphRAG is properly installed and accessible.
- Ensure your compute environment has sufficient GPU memory.
- The workspace used is named **`legalRAG`**.
- All Python scripts (`indexing.py`, `prompt_tune.py`, `baseline_query.py`, `query_graphrag.py`) should be adapted if applying to different datasets.

---
