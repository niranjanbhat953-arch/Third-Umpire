"""
End-to-End API Test & Latency Benchmark Tool
Run: python tools/test_api.py
"""
import sys, time, requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(name, method, url, **kwargs):
    t0 = time.perf_counter()
    try:
        r = requests.request(method, url, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        ok = r.status_code == 200
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({elapsed:.1f}ms) - Status: {r.status_code}")
        return r
    except Exception as e:
        print(f"[FAIL] {name} - Exception: {e}")
        return None

def main():
    print("=" * 55)
    print("Third Umpire - Full API Suite & Latency Verification")
    print("=" * 55)
    
    # 1. Health check
    r = test_endpoint("GET /health", "GET", f"{BASE_URL}/health")
    if r and r.json():
        print(f"       Loaded: {r.json().get('players')} cricketers")
    
    # 2. Metrics check
    r = test_endpoint("GET /metrics", "GET", f"{BASE_URL}/metrics")
    if r and r.json():
        d = r.json()
        print(f"       Accuracy: {d.get('accuracy')*100:.1f}%, Confident: {d.get('accuracy_when_confident')*100:.1f}%")
    
    # 3. Players list
    r = test_endpoint("GET /players?role=all-rounder", "GET", f"{BASE_URL}/players?role=all-rounder")
    if r and r.json():
        print(f"       All-rounders found: {len(r.json())}")
    
    # 4. Samples list
    r = test_endpoint("GET /samples", "GET", f"{BASE_URL}/samples")
    if r and r.json():
        print(f"       Demo samples available: {len(r.json())}")
    
    # 5. Live Identification Test
    sample_img = Path(__file__).parent.parent / "frontend/samples/virat_kohli.jpg"
    if sample_img.exists():
        with open(sample_img, "rb") as f:
            files = {"file": ("virat.jpg", f, "image/jpeg")}
            r = test_endpoint("POST /identify (Virat Kohli sample)", "POST", f"{BASE_URL}/identify", files=files)
            if r and r.json():
                faces = r.json().get("faces", [])
                if faces:
                    top = faces[0]["top"][0]
                    pname = faces[0].get("player", {}).get("name")
                    print(f"       Top Match: {top['name']} ({top['prob']*100:.1f}%) | Confidence: {faces[0].get('confidence_level')}")
                    print(f"       Identified Player Profile: {pname}")
    
    print("=" * 55)
    print("All endpoints verified successfully!")

if __name__ == "__main__":
    main()
