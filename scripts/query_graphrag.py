import json
import subprocess
from tqdm import tqdm
import time
import os

# Define the directory for logs
log_dir = os.path.expandvars("path/to/logs")
os.makedirs(log_dir, exist_ok=True)  # Create the logs directory if it doesn't already exist

# File paths for querying
input_file = os.path.expandvars("path/to/query_data_src.json")  # Input file path
output_file = os.path.expandvars("path/to/graphRAG_answers.json")  # Output file path to save results
graph_rag_root = os.path.expandvars("path/to/legalRAG")  # Root directory for GraphRAG

# Define the models and their ports
models = [
    {
        "name": "meta-llama_Llama-3.1-8B-Instruct",
        "command": [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            os.path.expandvars("path/to/models/meta-llama_Llama-3.1-8B-Instruct"),
            "--port",
            "8000",
            "--gpu_memory_utilization=0.7",
            "--chat-template",
            os.path.expandvars("path/to/tool_chat_template_llama3.1_json.jinja"),
        ],
        "log_file": f"{log_dir}/llama_8b_8000.log",
        "ready_signal": "INFO:     Uvicorn running on http://0.0.0.0:8000",
    },
    {
        "name": "intfloat_e5-mistral-7b-instruct",
        "command": [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            os.path.expandvars("path/to/models/intfloat_e5-mistral-7b-instruct"),
            "--port",
            "8001",
            "--gpu_memory_utilization=0.6",
        ],
        "log_file": f"{log_dir}/mistral_8001.log",
        "ready_signal": "INFO:     Uvicorn running on http://0.0.0.0:8001",
    },
]

# List to keep track of subprocesses
processes = []

def load_query_data():
    """Load the input and existing output data."""
    # Load the input data
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    # Load existing output data if the output file exists
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as outfile:
            output_data = json.load(outfile)
            processed_ids = {item["id"] for item in output_data}
    else:
        output_data = []
        processed_ids = set()

    # Prepare data to process (skip already processed items)
    remaining_data = [item for item in data if item.get("id") not in processed_ids]

    return remaining_data, output_data


def process_queries(remaining_data, output_data):
    """Process queries with progress tracking and periodic saves."""
    for item in tqdm(remaining_data, desc="Processing queries"):
        question = item.get("question")
        if not question:
            continue  # Skip if the question is empty

        try:
            # Run the GRAphRAG query command
            result = subprocess.run(
                [
                    "graphrag",
                    "query",
                    "--root", graph_rag_root,
                    "--method", "local",
                    "--query", question,
                ],
                capture_output=True,
                text=True,
            )

            # Extract the response from the GRAphRAG query
            if result.returncode == 0:
                answer = result.stdout.strip()
            else:
                answer = f"Error: {result.stderr.strip()}"

        except Exception as e:
            answer = f"Error: {str(e)}"

        # Append the result to the output list
        output_data.append({
            "id": item.get("id"),
            "question": question,
            "ground_truth": item.get("ground_truth"),
            "answer": answer,
        })

        # Save the output data periodically
        if len(output_data) % 10 == 0:  # Save every 10 processed items
            with open(output_file, "w", encoding="utf-8") as outfile:
                json.dump(output_data, outfile, ensure_ascii=False, indent=4)

    # Save the final output to the file
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=4)
    print("All queries processed successfully.")


try:
    # Start each model sequentially
    for model in models:
        print(f"Starting model {model['name']}...")
        with open(model["log_file"], "w") as log_file:
            process = subprocess.Popen(
                model["command"],
                stdout=log_file,
                stderr=log_file,
            )
            processes.append(process)

        # Wait for the ready signal in the log file
        log_file_path = model["log_file"]
        print(f"Waiting for {model['name']} to be ready...")
        while True:
            with open(log_file_path, "r") as log_file:
                logs = log_file.read()
                if model["ready_signal"] in logs:
                    print(f"{model['name']} is ready!")
                    break
            time.sleep(5)  # Check logs every 5 seconds

    print("All models started. Processing queries...")

    # Load data and process queries
    remaining_data, output_data = load_query_data()
    process_queries(remaining_data, output_data)

except KeyboardInterrupt:
    print("Interrupted by user. Shutting down processes...")

finally:
    # Shut down all running processes
    print("Shutting down all models...")
    for process in processes:
        process.terminate()  # Gracefully terminate the process
        try:
            process.wait(timeout=10)  # Wait for the process to shut down
        except subprocess.TimeoutExpired:
            print(f"Forcefully killing process {process.pid}...")
            process.kill()  # Force kill if it doesn't shut down gracefully
    print("All processes shut down.")
