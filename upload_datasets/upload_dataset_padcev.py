import json
from opik import Opik
from dotenv import load_dotenv
import os

load_dotenv()
OPIK_API = os.getenv("OPIK_API_KEY")
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE")

# Load JSON file (it's a JSON array, not JSONL)
json_file = "data/padcev.jsonl"  # Ensure it's named .json
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)  # ✅ Use json.load() for reading a JSON array

# Transform data
transformed_data = []
for item in data:
    transformed_data.append({
        "input": item["input"],
        "expected_output": item["expected_output"],
        "context": item.get("context", ""),  # Ensure context is included, default to empty string if missing
        "reference": item["expected_output"],  # Keeping reference same as expected_output
    })

# Get or create the dataset in Opik
client = Opik(api_key=OPIK_API, workspace=OPIK_WORKSPACE)
dataset = client.get_or_create_dataset(name="PadcevQA")

# Insert transformed data into the dataset
dataset.insert(transformed_data)

print("PadcevQA dataset uploaded successfully!")
