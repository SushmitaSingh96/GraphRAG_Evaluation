from datasets import Dataset
import json
from ragas import evaluate
from ragas.metrics import answer_correctness, answer_relevancy
import os
import pandas as pd
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
# File paths
input_file = "LLM_replies.json"
output_file = "evaluation_ragas_LLM.json"
average_file = "overall_ragas_LLM.json"

# Load the dataset
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Load existing results if the output file already exists
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        existing_results = json.load(f)
    processed_ids = {item["id"] for item in existing_results}  # Extract already processed IDs
    results = existing_results
else:
    processed_ids = set()
    results = []

# Map JSON fields to RAGAS-valid columns and keep track of IDs
mapped_data = {
    "id": [item["id"] for item in data],
    "user_input": [item["question"] for item in data],
    "response": [item["answer"] for item in data],
    "reference": [item["ground_truth"] for item in data],
}

# Initialize variables for results
results = []
total_scores = {"answer_correctness": 0, "answer_relevancy": 0}
total_data_points = len(mapped_data["user_input"])

# Process data points
for idx in range(total_data_points):
    # Skip already processed data points
    if mapped_data["id"][idx] in processed_ids:
        continue
    # Prepare data for a single data point
    single_data = {
        "user_input": [mapped_data["user_input"][idx]],
        "response": [mapped_data["response"][idx]],
        "reference": [mapped_data["reference"][idx]],
    }
    dataset = Dataset.from_dict(single_data)

    # Run evaluation
    score = evaluate(dataset, metrics=[answer_correctness, answer_relevancy])
    result = score.to_pandas().to_dict(orient="records")[0]

    # Add the corresponding ID to the result
    result["id"] = mapped_data["id"][idx]
    results.append(result)

    # Update total scores
    total_scores["answer_correctness"] += result["answer_correctness"]
    total_scores["answer_relevancy"] += result["answer_relevancy"]

    # Save results after every 10 data points
    if (idx + 1) % 10 == 0 or (idx + 1) == total_data_points:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Saved results to {output_file} after processing {idx + 1} data points.")

# Calculate averages
average_scores = {
    "answer_correctness": total_scores["answer_correctness"] / total_data_points,
    "answer_relevancy": total_scores["answer_relevancy"] / total_data_points,
}

# Save average scores
with open(average_file, "w", encoding="utf-8") as f:
    json.dump(average_scores, f, indent=4, ensure_ascii=False)

print("Processing complete.")
print("Average Scores saved to:", average_file)
