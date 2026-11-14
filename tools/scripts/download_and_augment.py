"""
Download verified faces from Wikimedia for Shubman Gill, MS Dhoni, Ravindra Jadeja, Jasprit Bumrah.
"""
import requests, io, time, sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "backend"))
import faceid

HEADERS = {"User-Agent": "EdufyFaceID/2.0 (student-project; contact: niranjan@edufy.local)"}
DATASET = Path("dataset")

SEARCHES = {
    "Shubman Gill": ["Shubman Gill", "Shubman", "Gill cricket", "Shubman Gill batting", "Shubman Gill Gujarat Titans"],
    "MS Dhoni": ["MS Dhoni", "Mahendra Singh Dhoni", "Dhoni CSK", "Dhoni 2019", "Dhoni cricket"],
    "Ravindra Jadeja": ["Ravindra Jadeja", "Jadeja CSK", "Jadeja cricket", "Ravindra Jadeja all-rounder"],
    "Jasprit Bumrah": ["Jasprit Bumrah", "Bumrah", "Jasprit Bumrah Mumbai Indians", "Bumrah bowler"]
}

def get_candidates(query):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query", "generator": "search", "gsrsearch": query, "gsrnamespace": "6",
        "prop": "imageinfo", "iiprop": "url", "iiurlwidth": 960, "format": "json", "gsrlimit": 35
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=12).json()
        pages = r.get("query", {}).get("pages", {})
        urls = []
        for k, v in pages.items():
            ii = v.get("imageinfo", [{}])[0]
            u = ii.get("thumburl") or ii.get("url")
            if u and not any(u.endswith(e) for e in [".svg", ".webm", ".ogg", ".pdf", ".gif"]):
                urls.append(u)
        return urls
    except Exception as e:
        return []

def main():
    sess = requests.Session()
    sess.headers.update(HEADERS)
    
    for player, qlist in SEARCHES.items():
        folder = DATASET / player
        folder.mkdir(parents=True, exist_ok=True)
        seen_urls = set()
        saved = 0
        print(f"\nSearching images for {player}...")
        
        for q in qlist:
            urls = get_candidates(q)
            for u in urls:
                if u in seen_urls:
                    continue
                seen_urls.add(u)
                
                try:
                    res = sess.get(u, timeout=12)
                    if res.status_code != 200 or len(res.content) < 8000:
                        continue
                    
                    img = Image.open(io.BytesIO(res.content)).convert("RGB")
                    img.thumbnail((1600, 1600))
                    boxes, probs, embs = faceid.embed(img, largest_only=True)
                    
                    if len(embs) > 0:
                        fname = folder / f"wiki_more_{int(time.time()*1000)%10000000}_{saved:03d}.jpg"
                        img.save(fname, "JPEG", quality=92)
                        saved += 1
                        print(f"  + [{player}] saved: {fname.name}")
                except Exception as e:
                    pass
                time.sleep(0.1)
        print(f"Added {saved} verified images for {player}. Total files: {len(list(folder.glob('*.*')))}")

if __name__ == "__main__":
    main()
