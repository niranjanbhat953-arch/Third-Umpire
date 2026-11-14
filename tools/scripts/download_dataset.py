"""
Multi-source image downloader for cricketers.
Tries Google first, then Bing as fallback.
"""
import os, time, logging
from pathlib import Path

logging.disable(logging.CRITICAL)  # suppress crawler noise

PLAYERS = {
    "Virat Kohli":     ["Virat Kohli India cricket", "Virat Kohli batsman", "Virat Kohli portrait"],
    "Rohit Sharma":    ["Rohit Sharma cricket", "Rohit Sharma India captain"],
    "MS Dhoni":        ["MS Dhoni cricket", "Mahendra Singh Dhoni CSK", "Dhoni wicketkeeper"],
    "Jasprit Bumrah":  ["Jasprit Bumrah cricket", "Jasprit Bumrah India fast bowler"],
    "Shubman Gill":    ["Shubman Gill cricket", "Shubman Gill India batsman"],
    "Ravindra Jadeja": ["Ravindra Jadeja cricket", "Jadeja India all-rounder"],
}

DATASET = Path("dataset")
DATASET.mkdir(exist_ok=True)

TARGET = 25

try:
    from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
except ImportError:
    raise SystemExit("icrawler not installed")

def count_imgs(folder):
    folder = Path(folder)
    if not folder.exists(): return 0
    return len([f for f in folder.rglob("*") if f.suffix.lower() in {".jpg",".jpeg",".png",".webp",".gif"}])

for name, queries in PLAYERS.items():
    dest = DATASET / name
    dest.mkdir(exist_ok=True)
    existing = count_imgs(dest)
    if existing >= TARGET:
        print(f"[SKIP] {name}: {existing} images already")
        continue
    print(f"\n--- {name} (have {existing}, need {TARGET}) ---")
    for q in queries:
        current = count_imgs(dest)
        if current >= TARGET:
            break
        need = TARGET - current
        print(f"  Google: '{q}' → need {need} more")
        try:
            c = GoogleImageCrawler(storage={"root_dir": str(dest)}, feeder_threads=1, parser_threads=1, downloader_threads=4)
            c.crawl(keyword=q, max_num=need+5, min_size=(80,80), file_idx_offset=current)
        except Exception as e:
            print(f"    Google failed: {e}")
        time.sleep(1)
        current = count_imgs(dest)
        if current >= TARGET:
            break
        need = TARGET - current
        print(f"  Bing: '{q}' → need {need} more")
        try:
            c = BingImageCrawler(storage={"root_dir": str(dest)}, feeder_threads=1, parser_threads=1, downloader_threads=4)
            c.crawl(keyword=q, max_num=need+5, min_size=(80,80), file_idx_offset=current)
        except Exception as e:
            print(f"    Bing failed: {e}")
        time.sleep(1)
    final = count_imgs(dest)
    print(f"  → {final} images for {name}")

print("\n")
for name in PLAYERS:
    n = count_imgs(DATASET / name)
    status = "✅" if n >= 6 else "❌"
    print(f"  {status} {name}: {n} images")

print("\n✅ Download complete!")
