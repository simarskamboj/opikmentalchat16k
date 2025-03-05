import json
from opik import Opik
from dotenv import load_dotenv
import os

load_dotenv()
OPIK_API = os.getenv("OPIK_API_KEY")
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE")

# Load JSON file
json_file = "data/dev.jsonl"  # Change this to your actual file path
with open(json_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Translate keys to the required format
transformed_data = []
for item in data:
    options_text = " ".join([f"({key}) {value}" for key, value in item["options"].items()])
    user_question = f"{item['question']} Options: {options_text}"
    transformed_data.append({
        "input": user_question,
        "expected_output": item["answer"],
                "reference": item["answer"]
    })

# Get or create a dataset in Opik
client = Opik(api_key=OPIK_API, workspace=OPIK_WORKSPACE)
dataset = client.get_or_create_dataset(name="MedQA-USMLE-dev")

# Insert transformed data into the dataset
dataset.insert(transformed_data)

print("Dataset uploaded successfully!")
