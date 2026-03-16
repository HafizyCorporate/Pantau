import telebot
import os
import time
from config import BOT_TOKEN, TEMP_FOLDER
from database.db import save_violation
from engine.ai_engine import AIEngine
from engine.video_processor import process_video

bot = telebot.TeleBot(BOT_TOKEN)
ai_engine = None 

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "🚦 AI ETLE Enterprise Aktif!\nKirimkan video untuk di-scan.")

@bot.message_handler(content_types=['video', 'document'])
def handle_video(message):
    bot.reply_to(message, "⚙️ [SYSTEM] Memproses Video... (Arsitektur MVC Docker Aktif!)")
    
    ts = int(time.time())
    input_path = f"{TEMP_FOLDER}/input_{ts}.mp4"
    output_path = f"{TEMP_FOLDER}/output_{ts}.mp4"
    
    try:
        if message.content_type == 'video':
            file_id = message.video.file_id
        else:
            file_id = message.document.file_id

        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        with open(input_path, 'wb') as f:
            f.write(downloaded_file)
            
        hasil = process_video(input_path, output_path, ai_engine)
        
        with open(output_path, 'rb') as video_file:
            bot.send_video(message.chat.id, video_file, caption="🎥 *REKAMAN SELESAI*", parse_mode='Markdown')
            
        if hasil["has_violation"]:
            plat = hasil["plate"]
            pel = hasil["violation"]
            speed = hasil["speed"]
            waktu = hasil["time"]
            bukti_path = hasil["evidence_path"]
            
            save_violation(plat, pel, speed, bukti_path)
            
            surat = (
                f"🚨 *BUKTI PELANGGARAN*\n\n"
                f"No plat : `{plat}`\n"
                f"Pelanggarannya : {pel}\n"
                f"Kecepatan : ~{speed} KM/JAM\n"
                f"waktu : {waktu}"
            )
            
            with open(bukti_path, 'rb') as foto_bukti:
                bot.send_photo(message.chat.id, foto_bukti, caption=surat, parse_mode='Markdown')
                
            if os.path.exists(bukti_path): os.remove(bukti_path)
        else:
            bot.send_message(message.chat.id, "✅ *AMAN:* Tidak ada pelanggar yang melintas.", parse_mode='Markdown')
            
    except Exception as e:
        bot.reply_to(message, f"❌ [ERROR] {str(e)}")
    finally:
        # Garbage Collector Mutlak: Hapus sisa file biar DB Docker gak penuh!
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)

def start_bot():
    global ai_engine
    if ai_engine is None:
        ai_engine = AIEngine()
    print("[SYSTEM] Telegram Bot Running...")
    bot.infinity_polling()
