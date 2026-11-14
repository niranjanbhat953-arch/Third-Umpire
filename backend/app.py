import io, json, os, time
from pathlib import Path
from typing import Optional, List, Dict, Any
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from faceid import load, embed

HERE = Path(__file__).parent
FRONTEND = HERE.parent / "frontend"

def load_bundle():
    bundle_path = HERE / "model.joblib"
    if not bundle_path.exists():
        raise RuntimeError("model.joblib not found. Run train.py first.")
    bundle = joblib.load(bundle_path)
    players = json.load(open(HERE / "players.json", encoding="utf-8"))
    return bundle["clf"], bundle["threshold"], players

clf, T, PLAYERS = load_bundle()

REQUIRE_AUTH = os.getenv("REQUIRE_AUTH") == "1"
PROJECT = os.getenv("FIREBASE_PROJECT_ID", "")

app = FastAPI(
    title="Third Umpire API",
    description="High-accuracy Indian Cricketer Face Identification with MTCNN & InceptionResnetV1",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

def check_auth(h: Optional[str]):
    if not REQUIRE_AUTH:
        return
    if not h or not h.startswith("Bearer "):
        raise HTTPException(401, "Sign in to use the identifier.")
    from google.oauth2 import id_token
    from google.auth.transport import requests as g
    try:
        id_token.verify_firebase_token(h[7:], g.Request(), audience=PROJECT)
    except Exception:
        raise HTTPException(401, "Your session expired. Please sign in again.")

@app.get("/health", tags=["System"])
def health():
    return {
        "ok": True,
        "status": "healthy",
        "players": len(clf.classes_),
        "threshold": T,
        "timestamp": time.time()
    }

@app.get("/metrics", tags=["Model"])
def metrics():
    p = HERE / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "No metrics available yet. Train the model first.")
    return json.load(open(p, encoding="utf-8"))

@app.get("/players", tags=["Players"])
def list_players(
    q: Optional[str] = Query(None, description="Search by cricketer name"),
    role: Optional[str] = Query(None, description="Filter by role (Batter, Bowler, All-rounder, Wicketkeeper)")
):
    results = PLAYERS
    if q:
        q_lower = q.lower().strip()
        results = {k: v for k, v in results.items() if q_lower in k.lower()}
    if role:
        r_lower = role.lower().strip()
        results = {k: v for k, v in results.items() if r_lower in (v.get("role") or "").lower()}
    return results

@app.get("/players/{name}", tags=["Players"])
def get_player(name: str):
    if name not in PLAYERS:
        raise HTTPException(404, f"Player '{name}' not found in registry.")
    return {**PLAYERS[name], "name": name}

@app.get("/samples", tags=["Demo"])
def list_samples():
    samples_dir = FRONTEND / "samples"
    if not samples_dir.exists():
        return []
    items = []
    name_map = {
        "virat_kohli.jpg": "Virat Kohli",
        "rohit_sharma.jpg": "Rohit Sharma",
        "ms_dhoni.jpg": "MS Dhoni",
        "sachin_tendulkar.jpg": "Sachin Tendulkar",
        "jasprit_bumrah.jpg": "Jasprit Bumrah"
    }
    for f in sorted(samples_dir.glob("*.jpg")):
        items.append({
            "id": f.stem,
            "name": name_map.get(f.name, f.stem.replace("_", " ").title()),
            "url": f"/samples/{f.name}"
        })
    return items

@app.post("/reload", tags=["System"])
def reload_model():
    global clf, T, PLAYERS
    clf, T, PLAYERS = load_bundle()
    return {"ok": True, "reloaded_players": len(clf.classes_), "threshold": T}

@app.post("/identify", tags=["Inference"])
async def identify(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    check_auth(authorization)
    
    # MIME validation
    valid_types = {"image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream"}
    if file.content_type not in valid_types and not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(415, "Unsupported format. Please upload a JPG, PNG, or WebP image.")
    
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(data) > 12_000_000:
        raise HTTPException(413, "Image file must be under 12 MB.")
    
    try:
        img = load(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Unable to read or decode image file. Please provide a valid photo.")
    
    # Run MTCNN detection + face embeddings on threadpool to keep async loop fast
    boxes, dprobs, E = await run_in_threadpool(embed, img)
    if not len(E):
        raise HTTPException(
            422,
            "No clear face detected. Tips: Ensure the face is well-lit, front-facing, and not covered by helmets, sunglasses, or motion blur."
        )
    
    w, h = img.size
    faces = []
    
    for b, dp, e in zip(boxes[:8], dprobs, E):
        p = clf.predict_proba(e[None])[0]
        o = p.argsort()[::-1][:3]
        top = [{"name": str(clf.classes_[i]), "prob": round(float(p[i]), 4)} for i in o]
        
        top_prob = top[0]["prob"]
        unknown = top_prob < T
        
        confidence_level = "high" if top_prob >= 0.75 else ("moderate" if top_prob >= T else "uncertain")
        player_info = None if unknown else {**PLAYERS.get(top[0]["name"], {}), "name": top[0]["name"]}
        
        # Normalized bounding box for responsive overlay positioning
        norm_box = [
            round(max(0.0, min(1.0, b[0] / w)), 4),
            round(max(0.0, min(1.0, b[1] / h)), 4),
            round(max(0.0, min(1.0, b[2] / w)), 4),
            round(max(0.0, min(1.0, b[3] / h)), 4),
        ]
        
        faces.append({
            "box": [round(v, 1) for v in b],
            "norm_box": norm_box,
            "detection": round(dp, 3),
            "confidence_level": confidence_level,
            "unknown": unknown,
            "top": top,
            "player": player_info
        })
    
    return {
        "image": {"width": w, "height": h},
        "threshold": T,
        "faces_detected": len(faces),
        "faces": faces
    }

# Mount static frontend
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="web")
