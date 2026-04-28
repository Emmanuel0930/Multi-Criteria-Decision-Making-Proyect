"""Test script to verify lookup tables are returned in params."""
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
import backend.main as backend_main

client = TestClient(backend_main.app)

print("Testing POST /run-model with ahp model - checking for lookup tables...")
try:
    response = client.post(
        "/run-model",
        json={"model": "ahp", "force_rerun": False}
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Success! Response keys: {list(data.keys())}")
        print(f"  lod0: {len(data.get('lod0', []))} items")
        print(f"  lod1: {len(data.get('lod1', []))} items")
        print(f"  lod3: {len(data.get('lod3', []))} items")
        print(f"  from_cache: {data.get('from_cache')}")
        
        params = data.get('params', {})
        print(f"\nParams keys: {list(params.keys())}")
        
        # Check for lookup tables (they are arrays, not dicts)
        muni_table = params.get('muni_table', [])
        dept_table = params.get('dept_table', [])
        divi_table = params.get('divi_table', [])
        
        print(f"\n✓ Lookup tables present:")
        print(f"  muni_table: {len(muni_table)} entries (array)")
        print(f"  dept_table: {len(dept_table)} entries (array)")
        print(f"  divi_table: {len(divi_table)} entries (array)")
        
        # Sample entries (access by index)
        if isinstance(muni_table, list) and len(muni_table) > 0:
            print(f"\n  Sample muni_table entries:")
            for i in range(min(3, len(muni_table))):
                print(f"    [{i}] = {muni_table[i]}")
        if isinstance(dept_table, list) and len(dept_table) > 0:
            print(f"\n  Sample dept_table entries:")
            for i in range(min(3, len(dept_table))):
                print(f"    [{i}] = {dept_table[i]}")
        if isinstance(divi_table, list) and len(divi_table) > 0:
            print(f"\n  Sample divi_table entries:")
            for i in range(min(3, len(divi_table))):
                print(f"    [{i}] = {divi_table[i]}")
        
        # Check if lod1 has the metadata indices
        if data.get('lod1'):
            sample_lod1 = data['lod1'][0]
            print(f"\n✓ Sample LOD1 record (first item):")
            print(f"  Format: [lat, lon, score, rank, mi, di, dpi, ws, sl, dg, dr, lu, pa, cr]")
            print(f"  Data: {sample_lod1}")
            if len(sample_lod1) >= 7:
                mi, di, dpi = sample_lod1[4], sample_lod1[5], sample_lod1[6]
                print(f"\n  Indices: mi={mi}, di={di}, dpi={dpi}")
                if isinstance(muni_table, list) and mi < len(muni_table):
                    print(f"  ✓ muni_table[{mi}] = {muni_table[mi]}")
                if isinstance(dept_table, list) and di < len(dept_table):
                    print(f"  ✓ dept_table[{di}] = {dept_table[di]}")
                if isinstance(divi_table, list) and dpi < len(divi_table):
                    print(f"  ✓ divi_table[{dpi}] = {divi_table[dpi]}")
    else:
        print(f"✗ Error: {response.text}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"✗ Exception: {e}")

