"""
Auto-trains the model once download_dataset.py finishes.
Run from project root: python auto_train.py
"""
import sys, subprocess, time
from pathlib import Path

DATASET = Path("dataset")
PLAYERS = ["Virat Kohli", "Rohit Sharma", "MS Dhoni", "Jasprit Bumrah", "Shubman Gill", "Ravindra Jadeja"]

print("⏳ Waiting for dataset to have enough images...")
while True:
    ready = []
    for p in PLAYERS:
        imgs = list((DATASET / p).glob("*.jpg")) + list((DATASET / p).glob("*.png")) + list((DATASET / p).glob("*.webp"))
        ready.append((p, len(imgs)))
    
    for name, count in ready:
        print(f"  {name}: {count} images")
    
    if all(count >= 6 for _, count in ready):
        break
    print("  Not enough images yet, waiting 5s...")
    time.sleep(5)

print("\n🏋️ Starting training...")
result = subprocess.run(
    [".venv/Scripts/python.exe", "backend/train.py", "dataset"],
    capture_output=False
)
if result.returncode == 0:
    print("\n✅ Training complete! Restart the server to load the new model.")
else:
    print("\n❌ Training failed. Check output above.")
