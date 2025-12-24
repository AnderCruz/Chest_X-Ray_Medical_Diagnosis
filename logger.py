import json
from datetime import datetime

LOG_FILE = "clinical_inference_log.jsonl"

def log_inference(result):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")
