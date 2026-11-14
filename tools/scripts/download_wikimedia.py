"""
Robust downloader using Wikimedia Commons API with live MTCNN face verification.
Ensures every downloaded file has a real, detectable face.
"""
import requests, io, time, sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "backend"))
import faceid

HEADERS = {"User-Agent": "EdufyCricketerFaceID/2.0 (student-research-project; contact: niranjan@edufy.local)"}
DATASET = Path("dataset")

SEARCH_TERMS = {
    "Virat Kohli": ["Virat Kohli", "Kohli cricket", "Virat Kohli 2019", "Virat Kohli 2022", "Virat Kohli RCB"],
    "Rohit Sharma": ["Rohit Sharma", "Rohit Sharma captain", "Rohit Sharma 2019", "Rohit Sharma 2023"],
    "MS Dhoni": ["MS Dhoni", "Mahendra Singh Dhoni", "Dhoni cricket", "Dhoni 2019"],
    "Jasprit Bumrah": ["Jasprit Bumrah", "Bumrah cricket", "Bumrah 2019", "Bumrah bowler"],
    "Shubman Gill": ["Shubman Gill", "Gill cricketer", "Shubman Gill batting"],
    "Ravindra Jadeja": ["Ravindra Jadeja", "Jadeja cricket", "Ravindra Jadeja all-rounder"]
}

def search_wikimedia(query, limit=30):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": 960,
        "format": "json",
        "gsrlimit": limit
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15).json()
        pages = r.get("query", {}).get("pages", {})
        urls = []
        for k, v in pages.items():
            ii = v.get("imageinfo", [{}])[0]
            u = ii.get("thumburl") or ii.get("url")
            if u and not any(u.endswith(ext) for ext in [".svg", ".webm", ".ogg", ".gif", ".tif"]):
                urls.append(u)
        return urls
    except Exception as e:
        print(f"Search error for {query}: {e}")
        return []

def main():
    sess = requests.Session()
    sess.headers.update(HEADERS)
    
    for player, queries in SEARCH_TERMS.items():
        folder = DATASET / player
        folder.mkdir(parents=True, exist_ok=True)
        existing = list(folder.glob("*.*"))
        print(f"\n[{player}] currently has {len(existing)} images")
        
        saved = 0
        seen_urls = set()
        
        for q in queries:
            urls = search_wikimedia(q, limit=25)
            for u in urls:
                if u in seen_urls:
                    continue
                seen_urls.add(u)
                
                try:
                    res = sess.get(u, timeout=12)
                    if res.status_code != 200 or len(res.content) < 8000:
                        continue
                    
                    # Verify face with MTCNN
                    pil_img = Image.open(io.BytesIO(res.content)).convert("RGB")
                    pil_img.thumbnail((1600, 1600))
                    boxes, probs, embs = faceid.embed(pil_img, largest_only=True)
                    
                    if len(embs) > 0:
                        fname = folder / f"wiki_{int(time.time()*1000)%10000000}_{saved:03d}.jpg"
                        pil_img.save(fname, "JPEG", quality=92)
                        saved += 1
                        print(f"  + Face verified & saved ({len(embs)} face): {fname.name}")
                    else:
                        print(f"  - No clear face detected in image from {u[:45]}...")
                except Exception as e:
                    pass
                time.sleep(0.15)
        
        final_count = len(list(folder.glob("*.*")))
        print(f"[{player}] Total after download: {final_count} images")

if __name__ == "__main__":
    main()
