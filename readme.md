# 🚦 Pantau — AI ETLE Enterprise

Bot Telegram untuk deteksi pelanggaran lalu lintas menggunakan YOLOv8 + EasyOCR.

## Fitur
- Deteksi kendaraan & pengendara tanpa helm
- Estimasi kecepatan real-time
- Baca plat nomor Indonesia
- Simpan bukti pelanggaran ke PostgreSQL

---

## ⚠️ Setup Model AI

Ada **2 model custom** yang harus kamu sediakan sendiri dan letakkan di folder `models/`:

| File | Keterangan |
|------|-----------|
| `models/helmet.pt` | Model YOLO deteksi helm (custom, train sendiri) |
| `models/license_plate_detector.pt` | Model YOLO deteksi plat nomor (custom) |

Model `yolov8n.pt` (deteksi kendaraan) akan **otomatis didownload** saat Docker build.

---

## 🚀 Deploy ke Railway

### 1. Push ke GitHub
```bash
git add .
git commit -m "Initial commit"
git push
```

### 2. Buat project di Railway
- Login ke [railway.app](https://railway.app)
- New Project → Deploy from GitHub repo
- Tambahkan plugin **PostgreSQL**

### 3. Set Environment Variables di Railway
Masuk ke Settings > Variables, tambahkan:

```
BOT_TOKEN     = token bot telegram kamu
CHAT_ID       = chat id kamu
DATABASE_URL  = otomatis terisi dari plugin PostgreSQL
PORT          = 8080
TEMP_FOLDER   = temp
```

### 4. Deploy!
Railway akan otomatis build Dockerfile dan deploy.

---

## 💻 Development Lokal

```bash
# Copy env
cp .env.example .env
# Edit .env dengan nilai asli

# Install dependencies
pip install -r requirements.txt

# Jalankan
python app.py
```
