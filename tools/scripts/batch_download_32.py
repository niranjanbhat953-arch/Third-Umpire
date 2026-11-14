"""
Fast multi-threaded downloader for 34 Indian cricketers from Wikimedia Commons.
"""
import requests, io, time, sys, os
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent / "backend"))
import faceid

HEADERS = {"User-Agent": "EdufyFaceID32/1.0 (cricket-research-project; contact: admin@edufy.local)"}
DATASET = Path("dataset")
DATASET.mkdir(exist_ok=True)

PLAYERS_SEARCH = {
    "Sachin Tendulkar": "Sachin Tendulkar",
    "Mohammed Shami": "Mohammed Shami",
    "Mohammed Siraj": "Mohammed Siraj",
    "Shreyas Iyer": "Shreyas Iyer",
    "Kuldeep Yadav": "Kuldeep Yadav",
    "Axar Patel": "Axar Patel",
    "Yuzvendra Chahal": "Yuzvendra Chahal",
    "Shikhar Dhawan": "Shikhar Dhawan",
    "Suresh Raina": "Suresh Raina",
    "Yuvraj Singh": "Yuvraj Singh",
    "Gautam Gambhir": "Gautam Gambhir",
    "Virender Sehwag": "Virender Sehwag",
    "Sourav Ganguly": "Sourav Ganguly",
    "Rahul Dravid": "Rahul Dravid",
    "Anil Kumble": "Anil Kumble",
    "Zaheer Khan": "Zaheer Khan",
    "Harbhajan Singh": "Harbhajan Singh",
    "Kapil Dev": "Kapil Dev",
    "Ishant Sharma": "Ishant Sharma",
    "Bhuvneshwar Kumar": "Bhuvneshwar Kumar",
    "Ravichandran Ashwin": "Ravichandran Ashwin",
    "Ajinkya Rahane": "Ajinkya Rahane",
    "Yashasvi Jaiswal": "Yashasvi Jaiswal",
    "Sanju Samson": "Sanju Samson",
    "Rishabh Pant": "Rishabh Pant"
}

def fetch_urls(name, query):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query", "generator": "search", "gsrsearch": query, "gsrnamespace": "6",
        "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 800, "format": "json", "gsrlimit": 25
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=12).json()
        pages = r.get("query", {}).get("pages", {})
        urls = []
        for k, v in pages.items():
            ii = v.get("imageinfo", [{}])[0]
            u = ii.get("thumburl") or ii.get("url")
            if u and not any(u.endswith(e) for e in [".svg", ".webm", ".ogg", ".pdf"]):
                urls.append(u)
        return name, urls
    except Exception as e:
        return name, []

def download_one(player, url, idx):
    dest = DATASET / player
    dest.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200 and len(r.content) > 6000:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.thumbnail((1200, 1200))
            boxes, probs, embs = faceid.embed(img, largest_only=True)
            if len(embs) > 0:
                out_path = dest / f"face_{idx:03d}.jpg"
                img.save(out_path, "JPEG", quality=90)
                return player, True
    except Exception:
        pass
    return player, False

def main():
    print("Step 1: Finding image URLs from Wikimedia Commons...")
    with ThreadPoolExecutor(max_workers=6) as executor:
        url_results = list(executor.map(lambda item: fetch_urls(item[0], item[1]), PLAYERS_SEARCH.items()))
    
    print("Step 2: Downloading & verifying faces with MTCNN...")
    tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for name, urls in url_results:
            folder = DATASET / name
            existing = len(list(folder.glob("*.jpg")))
            if existing >= 12:
                print(f"[SKIP] {name} already has {existing} images")
                continue
            for i, u in enumerate(urls[:20]):
                tasks.append(executor.submit(download_one, name, u, existing + i))
        
        counts = {}
        for future in as_completed(tasks):
            pl, success = future.result()
            if success:
                counts[pl] = counts.get(pl, 0) + 1
    
    print("\nDownload Summary:")
    for pl in PLAYERS_SEARCH:
        fcount = len(list((DATASET / pl).glob("*.jpg")))
        print(f"  {pl}: {fcount} images")

if __name__ == "__main__":
    main()
