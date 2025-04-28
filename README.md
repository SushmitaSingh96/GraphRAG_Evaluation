# GraphRAG Evaluation on German Tax Law Data

This repository evaluates Microsoft's [GraphRAG](https://microsoft.github.io/graphrag/get_started/) framework on a German tax law dataset.

The evaluation focuses on:
- **Prompt tuning** to adapt retrieval for domain-specific needs
- **Indexing** the legal corpus into a GraphRAG workspace
- **Baseline creation** without GraphRAG for comparison
- **Querying** the indexed knowledge base
- **Evaluation** of retrieval performance improvements using [RAGAS](https://docs.ragas.io)

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

## 📈 Key Findings (RAGAS Evaluation)

Using GraphRAG significantly improved both **Answer Correctness** and **Answer Relevancy** compared to using the LLM alone:

| Metric           | Baseline (LLM-only) | GraphRAG | Improvement |
| :--------------- | :-----------------: | :------: | :---------: |
| **Correctness**  | 0.4199               | 0.5018   | **+8.2%** |
| **Relevancy**    | 0.6818               | 0.7617   | **+8.0%** |

GraphRAG improves the overall quality of responses by grounding answers more accurately in the retrieved data.

---

## 📢 Notes

- This repository assumes that GraphRAG is properly installed and accessible.
- Ensure your compute environment has sufficient GPU memory.
- The workspace used is named **`legalRAG`**.
- All Python scripts (`indexing.py`, `prompt_tune.py`, `baseline_query.py`, `query_graphrag.py`) should be adapted if applying to different datasets.

---
