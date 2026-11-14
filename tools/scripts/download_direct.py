"""
Direct image downloader from known public URLs (Wikipedia/Wikimedia/BCCI etc.)
No API key, no scraping needed.
"""
import requests, os, time
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; educational-project/1.0)"}
DATASET = Path("dataset")
DATASET.mkdir(exist_ok=True)

SOURCES = {
    "Virat Kohli": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Virat_Kohli_at_the_2022_ICC_T20_World_Cup.jpg/640px-Virat_Kohli_at_the_2022_ICC_T20_World_Cup.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Virat_Kohli_par_Charles_Platiau%2C_2019.jpg/640px-Virat_Kohli_par_Charles_Platiau%2C_2019.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Virat_Kohli.jpg/640px-Virat_Kohli.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Kohli_ICC_T20_WC_2024.jpg/640px-Kohli_ICC_T20_WC_2024.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Virat_Kohli_ICC_CWC19_warm_up_match.jpg/640px-Virat_Kohli_ICC_CWC19_warm_up_match.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Virat_Kohli_at_a_press_conference_2.jpg/640px-Virat_Kohli_at_a_press_conference_2.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Virat_Kohli_at_Dubai_International_Cricket_Stadium.jpg/640px-Virat_Kohli_at_Dubai_International_Cricket_Stadium.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_1280,q_50/lsci/db/PICTURES/CMS/388100/388178.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/357900/357907.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388179.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374969.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364282.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388169.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388000/388034.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374966.jpg",
    ],
    "Rohit Sharma": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Rohit_Sharma_in_ICC_T20_World_Cup_2024.jpg/640px-Rohit_Sharma_in_ICC_T20_World_Cup_2024.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Rohit_Sharma.jpg/640px-Rohit_Sharma.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Rohit_Sharma_1.jpg/640px-Rohit_Sharma_1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Rohit_Sharma_CWC19.jpg/640px-Rohit_Sharma_CWC19.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/357900/357938.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388174.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374970.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364283.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388175.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388000/388035.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374967.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364284.jpg",
    ],
    "MS Dhoni": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/MS_Dhoni_in_2019.jpg/640px-MS_Dhoni_in_2019.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Mahendra_Singh_Dhoni_on_17_July_2016.jpg/640px-Mahendra_Singh_Dhoni_on_17_July_2016.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/MS_Dhoni_at_a_press_conference.jpg/640px-MS_Dhoni_at_a_press_conference.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/MS_Dhoni_at_a_press_conference_2019.jpg/640px-MS_Dhoni_at_a_press_conference_2019.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/MS_Dhoni_in_2019.jpg/400px-MS_Dhoni_in_2019.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/333300/333315.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/333300/333317.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/333300/333319.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364277.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364276.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374964.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374965.jpg",
    ],
    "Jasprit Bumrah": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/Jasprit_Bumrah_ICC_T20WC_2024.jpg/640px-Jasprit_Bumrah_ICC_T20WC_2024.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/Jasprit_Bumrah_in_July_2017.jpg/640px-Jasprit_Bumrah_in_July_2017.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/Jasprit_Bumrah_at_press_conference_2024.jpg/640px-Jasprit_Bumrah_at_press_conference_2024.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/357900/357936.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388171.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374968.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364281.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388170.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388000/388033.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374963.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364280.jpg",
    ],
    "Shubman Gill": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Shubman_Gill_in_ICC_T20_World_Cup_2024.jpg/640px-Shubman_Gill_in_ICC_T20_World_Cup_2024.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Shubman_Gill.jpg/640px-Shubman_Gill.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/357900/357939.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388176.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374971.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364285.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388177.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388000/388036.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374972.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364286.jpg",
    ],
    "Ravindra Jadeja": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Ravindra_Jadeja_ICC_T20_WC_2024.jpg/640px-Ravindra_Jadeja_ICC_T20_WC_2024.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Ravindra_Jadeja.jpg/640px-Ravindra_Jadeja.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Ravindra_Jadeja_at_a_press_conference.jpg/640px-Ravindra_Jadeja_at_a_press_conference.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/357900/357937.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388172.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374969.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/364200/364282.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388100/388173.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/388000/388032.jpg",
        "https://img1.hscicdn.com/image/upload/f_auto,t_ds_w_640/lsci/db/PICTURES/CMS/374900/374962.jpg",
    ],
}

sess = requests.Session()
sess.headers.update(HEADERS)

for name, urls in SOURCES.items():
    dest = DATASET / name
    dest.mkdir(exist_ok=True)
    existing = [f for f in dest.rglob("*") if f.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
    start_idx = len(existing)
    downloaded = 0
    print(f"\n[{name}] existing={start_idx}")
    for i, url in enumerate(urls):
        ext = ".jpg"
        fname = dest / f"img_{start_idx+i:03d}{ext}"
        if fname.exists():
            continue
        try:
            r = sess.get(url, timeout=10, stream=True)
            if r.status_code == 200 and len(r.content) > 5000:
                fname.write_bytes(r.content)
                downloaded += 1
                print(f"  ✓ {fname.name} ({len(r.content)//1024}KB)")
            else:
                print(f"  ✗ {url[-50:]} status={r.status_code} size={len(r.content)}")
        except Exception as e:
            print(f"  ✗ {url[-50:]}: {e}")
        time.sleep(0.3)
    total = len([f for f in dest.rglob("*") if f.suffix.lower() in {".jpg",".jpeg",".png",".webp"}])
    print(f"  → {total} total images for {name}")

print("\n=== SUMMARY ===")
ok = True
for name in SOURCES:
    n = len([f for f in (DATASET/name).rglob("*") if f.suffix.lower() in {".jpg",".jpeg",".png",".webp"}])
    s = "✅" if n >= 6 else "❌"
    if n < 6: ok = False
    print(f"  {s} {name}: {n}")
print("\nReady to train!" if ok else "\nSome players have fewer than 6 images.")
