import os
import time
import json
import sys
from pathlib import Path
import requests

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

def upload_sample(sample_path: Path):
    if not sample_path.exists():
        print(f"[WARN] sample not found at {sample_path}")
        return None
    files = {'file': (sample_path.name, open(sample_path, 'rb'), 'application/pdf')}
    r = requests.post(f"{API_BASE}/api/upload", files=files)
    if r.status_code != 200:
        print("[ERROR] upload failed:", r.text)
        return None
    data = r.json()
    print("[INFO] upload ok:", data)
    return data["doc_id"]

def wait_progress(doc_id: str, timeout_s: int = 120):
    start = time.time()
    while time.time() - start < timeout_s:
        r = requests.get(f"{API_BASE}/api/progress/{doc_id}")
        if r.status_code == 200:
            p = r.json()
            print("[INFO] progress:", p["status"], p["progress"], p.get("message"))
            if p["status"] == "completed" and p["progress"] == 100:
                return True
        else:
            print("[WARN] progress get failed:", r.text)
        time.sleep(2)
    return False

def get_blocks(doc_id: str):
    r = requests.get(f"{API_BASE}/api/documents/{doc_id}/blocks")
    if r.status_code != 200:
        print("[ERROR] blocks failed:", r.text)
        return []
    data = r.json()
    print("[INFO] blocks total:", data.get("total"))
    return data.get("blocks", [])

def run():
    root = Path(__file__).resolve().parents[1]
    sample = root / "tests" / "data" / "sample.pdf"
    doc_id = upload_sample(sample)
    if not doc_id:
        print("[FATAL] no doc_id")
        return 1
    if not wait_progress(doc_id):
        print("[FATAL] timeout waiting progress")
        return 2
    blocks = get_blocks(doc_id)
    if not blocks:
        print("[FATAL] no blocks")
        return 3
    # simple check
    text_count = sum(1 for b in blocks if b.get("block_label","").lower()=="text")
    print("[INFO] text blocks:", text_count)
    print("[INFO] E2E done")
    return 0

if __name__ == "__main__":
    sys.exit(run())
