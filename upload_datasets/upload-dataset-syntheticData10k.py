import json
import csv
from opik import Opik
from dotenv import load_dotenv
import os

load_dotenv()
OPIK_API = os.getenv("OPIK_API_KEY")
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE")

# Load CSV file ~ Synthetic_Data_10K.csv
with open("data/Synthetic_Data_10K.csv", "r") as csv_file:
    csv_reader = csv.DictReader(csv_file)  # Read CSV as a dictionary
    data = list(csv_reader)  # Convert to a list of dictionaries

# Translate keys to the required format
transformed_data = []
for item in data:
    transformed_data.append({
        "input": item["input"],
        "expected_output": item["output"],
        "context": item["instruction"], # TODO: This might be the wrong spot for the 'instruction' field
        "reference": item["output"],  # Keeping reference same as expected_output
    })

# Get or create a dataset in Opik
client = Opik(api_key=OPIK_API, workspace=OPIK_WORKSPACE)
dataset = client.get_or_create_dataset(name="synthetic10k")

# Insert transformed data into the dataset
dataset.insert(transformed_data)

print("Dataset uploaded successfully!")


