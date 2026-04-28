"""Test script to verify POST /run-model endpoint works."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
import backend.main as backend_main

client = TestClient(backend_main.app)

print("Testing POST /run-model with ahp model...")
try:
    response = client.post(
        "/run-model",
        json={"model": "ahp", "force_rerun": False}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success! Response keys: {list(data.keys())}")
        print(f"  lod0: {len(data.get('lod0', []))} items")
        print(f"  lod1: {len(data.get('lod1', []))} items")
        print(f"  lod3: {len(data.get('lod3', []))} items")
        print(f"  from_cache: {data.get('from_cache')}")
    else:
        print(f"✗ Error: {response.text}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"✗ Exception: {e}")
