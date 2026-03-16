import os
from dotenv import load_dotenv

load_dotenv()

# ===============================
# TELEGRAM CONFIG
# ===============================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# ===============================
# DATABASE
# ===============================
DATABASE_URL = os.getenv("DATABASE_URL")

# ===============================
# SERVER & TEMP
# ===============================
PORT = int(os.getenv("PORT", 8080))
TEMP_FOLDER = os.getenv("TEMP_FOLDER", "temp")

# ===============================
# AI MODEL PATH
# ===============================
VEHICLE_MODEL = "models/yolov8n.pt"
HELMET_MODEL = "models/helmet.pt"
PLATE_MODEL = "models/license_plate_detector.pt"
