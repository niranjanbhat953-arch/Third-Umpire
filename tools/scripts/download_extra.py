"""Download extra images for players that need more faces."""
import requests, time
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
DATASET = Path("dataset")

EXTRA = {
    "Jasprit Bumrah": [
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388180.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388000/388040.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374975.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364290.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/357900/357942.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/333300/333320.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/333300/333321.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/333300/333322.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374980.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374981.jpg",
    ],
    "Shubman Gill": [
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388182.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388000/388042.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374977.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364292.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/357900/357944.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/333300/333323.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/333300/333325.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374982.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374983.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364293.jpg",
    ],
    "Virat Kohli": [
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388183.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388000/388043.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374978.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364294.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/357900/357945.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/333300/333326.jpg",
    ],
}

sess = requests.Session()
sess.headers.update(HEADERS)

for name, urls in EXTRA.items():
    dest = DATASET / name
    dest.mkdir(exist_ok=True)
    existing = [f for f in dest.rglob("*") if f.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
    idx = len(existing)
    got = 0
    print(f"\n[{name}] fetching extras (currently {idx} files)...")
    for url in urls:
        fname = dest / f"extra_{idx+got:03d}.jpg"
        try:
            r = sess.get(url, timeout=10)
            if r.status_code == 200 and len(r.content) > 5000:
                fname.write_bytes(r.content)
                got += 1
                print(f"  ✓ {fname.name} ({len(r.content)//1024}KB)")
            else:
                print(f"  ✗ {url[-40:]} [{r.status_code}]")
        except Exception as e:
            print(f"  ✗ {url[-40:]}: {e}")
        time.sleep(0.2)
    total = len([f for f in dest.rglob("*") if f.suffix.lower() in {".jpg",".jpeg",".png",".webp"}])
    print(f"  Total: {total} files for {name}")

print("\nDone!")
