import os
import threading
from database.db import init_db
from telegram_bot import start_bot
from flask import Flask, jsonify
from config import PORT

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"status": "ETLE Enterprise System RUNNING"}), 200

def run_flask():
    app.run(host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    print("[SYSTEM] Starting ETLE Modular Application...")
    
    if not os.path.exists("temp"):
        os.makedirs("temp")
        
    init_db()
    
    threading.Thread(target=run_flask, daemon=True).start()
    
    start_bot()
