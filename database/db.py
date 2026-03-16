import psycopg2
import base64
import os
from config import DATABASE_URL

def get_connection():
    if not DATABASE_URL:
        return None
    try:
        return psycopg2.connect(DATABASE_URL)
    except:
        return None

def init_db():
    print("[SYSTEM] Inisialisasi Database...")
    conn = get_connection()
    if not conn:
        print("[WARNING] DATABASE_URL tidak valid. DB dilewati.")
        return

    try:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS violations(
            id SERIAL PRIMARY KEY,
            plate TEXT,
            violation TEXT,
            speed INTEGER,
            image TEXT,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("[SYSTEM] Database PostgreSQL Ready!")
    except Exception as e:
        print(f"[ERROR] Database init error: {e}")

def save_violation(plate, violation, speed, image_path):
    conn = get_connection()
    if not conn: return
    try:
        image_data = None
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

        cur = conn.cursor()
        cur.execute(
            "INSERT INTO violations (plate, violation, speed, image) VALUES (%s, %s, %s, %s)",
            (plate, violation, speed, image_data)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Gagal simpan ke DB: {e}")
