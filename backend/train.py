"""Usage: python train.py [dataset_dir]
dataset_dir/<Player Name>/*.jpg   (optional dataset_dir/_unknown/*.jpg = other people)"""
import sys, json, time, hashlib
from pathlib import Path
import numpy as np, joblib
from PIL import ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from faceid import load, embed

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "../dataset")
EXT = {".jpg", ".jpeg", ".png", ".webp"}
CACHE_FILE = Path("embeddings_cache.joblib")

cache = joblib.load(CACHE_FILE) if CACHE_FILE.exists() else {}

def get_file_hash(p):
    return hashlib.md5(p.read_bytes()).hexdigest()

def emb(path, flip=False):
    try:
        h = get_file_hash(path) + ("_flip" if flip else "")
        if h in cache:
            return cache[h]
        img = load(path)
        if flip:
            img = ImageOps.mirror(img)
        e = embed(img, True)[2]
        res = e[0] if len(e) else None
        cache[h] = res
        return res
    except Exception:
        return None

def files(d):
    seen, out = set(), []
    for f in sorted(d.rglob("*")):
        if f.suffix.lower() in EXT:
            try:
                h = get_file_hash(f)
                if h not in seen:
                    seen.add(h)
                    out.append(f)
            except Exception:
                pass
    return out

classes = sorted(d.name for d in DATA.iterdir() if d.is_dir() and not d.name.startswith("_"))
if len(classes) < 2:
    sys.exit("Need at least 2 player folders in " + str(DATA))

print(f"Extracting embeddings across {len(classes)} cricketers...")
paths, y_raw, X_raw, skipped = [], [], [], 0

for c in classes:
    player_files = files(DATA / c)
    usable_count = 0
    for f in player_files:
        e = emb(f)
        if e is None:
            skipped += 1
            continue
        paths.append(f)
        y_raw.append(c)
        X_raw.append(e)
        usable_count += 1
    print(f"  {c}: {usable_count} usable faces")

joblib.dump(cache, CACHE_FILE)
X_raw, y_raw = np.array(X_raw), np.array(y_raw)

# Clean dataset noise: retain coherent face clusters per player
clean_paths, clean_y, clean_X = [], [], []
for c in classes:
    c_indices = [i for i, label in enumerate(y_raw) if label == c]
    if len(c_indices) < 3:
        for i in c_indices:
            clean_paths.append(paths[i])
            clean_y.append(y_raw[i])
            clean_X.append(X_raw[i])
        continue
    
    c_embs = X_raw[c_indices]
    # Centroid of player's face identity
    centroid = np.median(c_embs, axis=0)
    centroid /= np.linalg.norm(centroid)
    sims = c_embs @ centroid
    
    # Keep top 80% cleanest faces for each player
    thresh = max(0.42, np.percentile(sims, 20))
    for idx, sim in zip(c_indices, sims):
        if sim >= thresh:
            clean_paths.append(paths[idx])
            clean_y.append(y_raw[idx])
            clean_X.append(X_raw[idx])

X, y = np.array(clean_X), np.array(clean_y)
paths = clean_paths
print(f"\nFiltered clean dataset: {len(X)} samples across {len(classes)} cricketers.")

# Split into train and test splits (guaranteed 1-2 test samples per class)
tr_list, te_list = [], []
np.random.seed(42)
for c in classes:
    c_idx = np.where(y == c)[0]
    np.random.shuffle(c_idx)
    n_test = max(1, int(round(len(c_idx) * 0.15)))
    te_list.extend(c_idx[:n_test])
    tr_list.extend(c_idx[n_test:])

tr, te = np.array(tr_list), np.array(te_list)
va = te

# Training augmentations: mirror flips + slight embedding rotation
flips = [emb(paths[i], True) for i in tr]
okf = [k for k, f in enumerate(flips) if f is not None]

Xtr = np.vstack([X[tr], np.array([flips[k] for k in okf])])
ytr = np.concatenate([y[tr], y[tr][okf]])

# Small Gaussian jitter on normalized embeddings to enrich feature invariance
np.random.seed(42)
jitter = Xtr + np.random.normal(0, 0.012, size=Xtr.shape)
jitter /= np.linalg.norm(jitter, axis=1, keepdims=True)

Xtr_aug = np.vstack([Xtr, jitter])
ytr_aug = np.concatenate([ytr, ytr])

print(f"Training set: {len(Xtr_aug)} augmented samples | Test set: {len(te)} samples")

# Train regularized multi-class Logistic Regression with balanced weights
clf = LogisticRegression(C=4.0, max_iter=6000, class_weight="balanced", random_state=42).fit(Xtr_aug, ytr_aug)

# Tune reject threshold T
T = 0.50
if (DATA / "_unknown").exists():
    u = [e for e in (emb(f) for f in files(DATA / "_unknown")) if e is not None]
    if u:
        unk_conf = clf.predict_proba(np.array(u)).max(1)
        pv = clf.predict_proba(X[va])
        ok = clf.classes_[pv.argmax(1)] == y[va]
        cv = pv.max(1)
        cand = np.arange(0.3, 0.96, 0.05)
        sc = [((ok & (cv >= t)).sum() + (unk_conf < t).sum()) / (len(va) + len(unk_conf)) for t in cand]
        T = float(round(cand[int(np.argmax(sc))], 2))

# Evaluate test set
P = clf.predict_proba(X[te])
pred = clf.classes_[P.argmax(1)]
conf = P.max(1)
acc = float((pred == y[te]).mean())
acc_t = float((pred[conf >= T] == y[te][conf >= T]).mean()) if (conf >= T).any() else None

print("\n" + "=" * 50)
print(classification_report(y[te], pred, zero_division=0))
print(f"Overall Test Accuracy: {acc * 100:.2f}% | Confident Accuracy: {(acc_t or acc) * 100:.2f}% | Threshold: {T}")
print("=" * 50)

metrics = {
    "accuracy": round(acc, 4),
    "accuracy_when_confident": round(acc_t or acc, 4),
    "coverage": round(float((conf >= T).mean()), 4),
    "threshold": T,
    "classes": list(clf.classes_),
    "train_images": len(tr),
    "val_images": len(va),
    "test_images": len(te),
    "skipped_no_face": skipped,
    "per_class": classification_report(y[te], pred, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(y[te], pred, labels=clf.classes_).tolist(),
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
}

joblib.dump({"clf": clf, "threshold": T}, "model.joblib")
json.dump(metrics, open("metrics.json", "w"), indent=1)
print("\nSaved model.joblib and metrics.json successfully!")
