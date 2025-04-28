import json
import requests
import logging
from tqdm import tqdm
import os

# File paths
input_file_path = os.path.expandvars("path/to/query_data_src.json") #Input file path
output_file_path = 'LLM_replies.json' #Output file path to save results
 
# VLLM server URL
vllm_server_url = "http://localhost:8000/v1/chat/completions"

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to send a request to the VLLM server
def query_vllm(question):
    try:
        payload = {
            "model": os.path.expandvars("path/to/models/meta-llama_Llama-3.1-8B-Instruct"),
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 1024
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(vllm_server_url, json=payload, headers=headers)
        response.raise_for_status()
        reply = response.json()
        return reply.get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error querying VLLM for question '{question}': {http_err}")
        return "Error: HTTP error occurred"
    except Exception as e:
        logging.error(f"Error querying VLLM for question '{question}': {e}")
        return "Error: Unable to get response from LLM"

# Load existing results if available
def load_existing_results(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Error loading existing results from {file_path}: {e}")
    return []

# Save intermediate results
def save_intermediate_results(results, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=4)
    except Exception as e:
        logging.error(f"Error saving results to file {file_path}: {e}")

# Main script execution
def main():
    logging.info("Loading input JSON file...")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            input_data = json.load(file)
    except Exception as e:
        logging.error(f"Error reading input JSON file: {e}")
        return

    logging.info("Loading existing results...")
    existing_results = load_existing_results(output_file_path)
    processed_ids = {item['id'] for item in existing_results}  # Set of already processed IDs

    logging.info(f"Skipping {len(processed_ids)} already processed questions.")

    results = existing_results  # Start with existing results
    for idx, item in enumerate(tqdm(input_data, desc="Processing Questions", unit="question")):
        question_id = item.get("id")
        if question_id in processed_ids:
            continue  # Skip already processed IDs

        question = item.get("question", "")
        if not question:
            logging.warning(f"Skipping entry with missing question: {item}")
            continue

        llm_reply = query_vllm(question)
        results.append({
            "id": question_id,
            "question": question,
            "ground_truth": item.get("ground_truth"),
            "answer": llm_reply
        })

        # Save every 10 entries
        if (len(results) - len(existing_results)) % 10 == 0:
            save_intermediate_results(results, output_file_path)

    # Final save
    logging.info("Saving final results to output JSON file...")
    save_intermediate_results(results, output_file_path)

    logging.info(f"Process completed. Results saved to {output_file_path}")

if __name__ == "__main__":
    main()
