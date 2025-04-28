import subprocess
import time
import os

# Define the directory for logs
log_dir = os.path.expandvars("path/to/logs")
os.makedirs(log_dir, exist_ok=True)  # Create the logs directory if it doesn't already exist

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
            os.path.expandvars("path/to/scripts/tool_chat_template_llama3.1_json.jinja"),
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
        ],
        "log_file": f"{log_dir}/mistral_8001.log",
        "ready_signal": "INFO:     Uvicorn running on http://0.0.0.0:8001",
    },
]

# List to keep track of subprocesses
processes = []

def run_prompt_finetune():
    """Run the prompt fine-tuning script."""
    prompt_tune_command = [
        "python",
        "-m",
        "graphrag",
        "prompt-tune",
        "--root",
        ".",
        "--language",
        "German",
    ]
    print("Running the prompt fine-tuning script...")
    try:
        subprocess.run(prompt_tune_command, cwd=os.path.expandvars("path/to/legalRAG"), check=True)
        print("Prompt fine-tuning script completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Prompt fine-tuning script failed: {str(e)}")

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

    print("All models started. Running the prompt fine-tuning script...")
    run_prompt_finetune()

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
