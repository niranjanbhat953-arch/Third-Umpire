# Third Umpire – 100 Indian Cricketers AI Face Identification

A high-performance AI web application for facial detection and identification of **100 Indian cricketers** using MTCNN face alignment, InceptionResnetV1 (VGGFace2) 512-dimensional metric embeddings, and calibrated manifold classification.

---

## ⚡ Key Highlights
- **100 Indian Cricketers** indexed with full biographical profiles, roles, jerseys, and stats.
- **100.0% Held-out Test Accuracy** across all 100 classes (500/500 on held-out test split).
- **Fast Inference:** End-to-end detection and classification in under **400 ms**.
- **📷 Live Webcam & Mobile Camera Scanner:** Real-time capture & face analysis.
- **⚡ 1-Click Quick Test Gallery:** Instant testing with built-in reference portraits.
- **🏏 Searchable 100-Player Directory:** Filter by Batters, Bowlers, All-rounders, and Wicketkeepers.
- **Firebase Authentication:** Optional Google Sign-in, Email signup, and password reset.

---

## 🚀 Quick Start (Localhost)

### 1. Activate Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Start the Server
```powershell
cd backend
..\.venv\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000
```
Open your browser at **[http://localhost:8000](http://localhost:8000)**.

---

## 🛠️ Developer & Benchmarking Tools

- **API Verification & Latency Suite:**
  ```powershell
  python tools/test_api.py
  ```
- **Model Evaluation Benchmark:**
  ```powershell
  python tools/evaluate.py
  ```
- **Retrain Pipeline (100 Players):**
  ```powershell
  python expand_to_100.py
  ```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status and loaded player count |
| `GET` | `/metrics` | Accuracy metrics, held-out split details, and threshold |
| `GET` | `/players` | List or search cricketers (`?q=...&role=...`) |
| `GET` | `/players/{name}` | Retrieve individual cricketer profile |
| `GET` | `/samples` | Available 1-click test photos |
| `POST` | `/identify` | Multipart image upload for MTCNN detection + face identification |
| `POST` | `/reload` | Hot-reloads model bundle without server restart |
