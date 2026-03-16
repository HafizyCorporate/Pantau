import cv2
import time
import os

def process_video(input_path, output_path, ai_engine):
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_asli = int(cap.get(cv2.CAP_PROP_FPS))
    if fps_asli == 0: fps_asli = 30
    
    # 10 FPS + Frame Skip = Super aman buat server Docker kecil
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10, (w, h)) 
    
    largest_violator = 0
    best_frame = None 
    best_clean = None
    target_box = None
    all_plates = []
    final_speed = 0
    waktu_ms = 0
    prev_centers = [] 
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        frame_count += 1
        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # PROSES 1 FRAME, SKIP 2 FRAME (Biar gak meledak RAM-nya)
        if frame_count % 3 != 0: continue 
            
        hasil = ai_engine.process_frame(frame, prev_centers, fps_asli)
        prev_centers = hasil["centers"]
        
        if hasil["max_area"] > largest_violator:
            largest_violator = hasil["max_area"]
            best_frame = hasil["frame"].copy()
            best_clean = frame.copy()
            target_box = hasil["target_motor"]
            all_plates = hasil["valid_plates"]
            final_speed = hasil["speed"]
            waktu_ms = current_ms
            
        out.write(hasil["frame"])
        
    cap.release()
    out.release()
    
    # BACA PLAT BUKTI
    if best_frame is not None and largest_violator > 0:
        plat_pelanggar = "Tidak Terbaca"
        if target_box is not None and len(all_plates) > 0:
            mx1, my1, mx2, my2 = target_box
            mw = mx2 - mx1
            mh = my2 - my1
            
            for (px1, py1, px2, py2) in all_plates:
                pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                if (mx1 - mw*0.6) <= pcx <= (mx2 + mw*0.6) and (my1 - mh*0.6) <= pcy <= (my2 + mh*0.6):
                    potongan = best_clean[py1:py2, px1:px2]
                    zoom = cv2.resize(potongan, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    teks_hasil = ai_engine.reader.readtext(zoom, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    if teks_hasil:
                        res = ai_engine.cek_plat_indonesia(teks_hasil)
                        if res:
                            plat_pelanggar = res
                            cv2.rectangle(best_frame, (px1, py1), (px2, py2), (0, 0, 255), 4)
                            break
                            
        ts = int(time.time())
        bukti_path = f"temp/bukti_{ts}.jpg"
        cv2.imwrite(bukti_path, best_frame)
        
        detik_total = int(waktu_ms / 1000)
        waktu_str = f"{detik_total // 60:02d}:{detik_total % 60:02d}"
        
        return {
            "has_violation": True,
            "evidence_path": bukti_path,
            "plate": plat_pelanggar,
            "speed": int(final_speed),
            "violation": "Tidak Menggunakan Helm",
            "time": waktu_str
        }
        
    return {"has_violation": False}
