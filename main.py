from dotenv import load_dotenv
import os
from upload_datasets.upload_dataset_padcev_mc import padcev_mc
from upload_datasets.upload_dataset_padcev import padcev

load_dotenv()
OPIK_API = os.getenv("OPIK_API_KEY")
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE")

if __name__ == "__main__":
    padcev_mc(opik_api=OPIK_API, opik_workspace=OPIK_WORKSPACE)
    padcev(opik_api=OPIK_API, opik_workspace=OPIK_WORKSPACE)
    