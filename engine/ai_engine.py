import cv2
import re
import numpy as np
import easyocr
from ultralytics import YOLO
from config import VEHICLE_MODEL, HELMET_MODEL, PLATE_MODEL

class AIEngine:
    def __init__(self):
        print("[SYSTEM] Memuat AIEngine (YOLO & EasyOCR)...")
        self.model_kendaraan = YOLO(VEHICLE_MODEL)
        self.model_helm = YOLO(HELMET_MODEL)
        self.model_plat = YOLO(PLATE_MODEL)
        self.reader = easyocr.Reader(['en'], gpu=False)

        # Kalibrasi: berapa pixel = 1 meter (sesuaikan dengan kamera kamu)
        self.PIXEL_PER_METER = 8.0

        # Noise threshold: gerakan < N pixel diabaikan (bukan kecepatan)
        self.MIN_PIXEL_MOVE = 8.0

        # Buffer histori kecepatan per kendaraan untuk smoothing
        self.speed_history = {}
        self.HISTORY_SIZE = 5

    def _smooth_speed(self, vehicle_id, new_speed):
        if vehicle_id not in self.speed_history:
            self.speed_history[vehicle_id] = []
        history = self.speed_history[vehicle_id]
        history.append(new_speed)
        if len(history) > self.HISTORY_SIZE:
            history.pop(0)
        # Buang outlier: kalau tiba-tiba 3x lebih besar dari rata-rata, abaikan
        if len(history) >= 3:
            avg = sum(history[:-1]) / len(history[:-1])
            if avg > 0 and history[-1] > avg * 3:
                history[-1] = avg
        return sum(history) / len(history)

    def cek_plat_indonesia(self, teks_list):
        teks = "".join(teks_list).upper()
        teks_bersih = re.sub(r'[^A-Z0-9]', '', teks)
        pola = r"^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$"
        if re.match(pola, teks_bersih):
            return teks_bersih
        return None

    def process_frame(self, frame, prev_centers, fps_video):
        frame_gambar = frame.copy()
        jumlah_pelanggar = 0
        max_area_pelanggar = 0
        kotak_motor_pelanggar = None
        kecepatan_pelanggar = 0

        hasil_kendaraan = self.model_kendaraan(frame, conf=0.45, classes=[2, 3, 5, 7], imgsz=640, verbose=False)
        hasil_helm = self.model_helm(frame, conf=0.45, imgsz=640, verbose=False)
        hasil_plat = self.model_plat(frame, conf=0.35, imgsz=960, verbose=False)

        kendaraan_mentah = []
        for box in hasil_kendaraan[0].boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            kendaraan_mentah.append({'box': (x1, y1, x2, y2), 'cls': cls_id})

        current_centers = []
        tracked_vehicles = []

        for idx, v in enumerate(kendaraan_mentah):
            x1, y1, x2, y2 = v['box']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            current_centers.append((cx, cy))

            is_moving = False
            speed_kmh = 0.0

            if prev_centers:
                # Cari kendaraan terdekat dari frame sebelumnya
                dists = [np.hypot(cx - px, cy - py) for px, py in prev_centers]
                jarak_pixel = min(dists)

                if jarak_pixel >= self.MIN_PIXEL_MOVE:
                    is_moving = True
                    jarak_meter = jarak_pixel / self.PIXEL_PER_METER
                    raw_speed = (jarak_meter * fps_video) * 3.6
                    # Smoothing per kendaraan
                    speed_kmh = self._smooth_speed(idx, raw_speed)
                    # Hard cap maksimum realistis
                    speed_kmh = min(speed_kmh, 100.0)

            v['speed'] = speed_kmh
            v['status'] = "MOVING" if is_moving else "STOPPED"
            tracked_vehicles.append(v)

            nama = "MOTOR" if v['cls'] == 3 else ("MOBIL" if v['cls'] == 2 else ("BUS" if v['cls'] == 5 else "TRUK"))
            teks_speed = f"{nama} | {int(speed_kmh)} KM/H" if is_moving else f"{nama} | BERHENTI"
            warna_kendaraan = (0, 165, 255) if v['cls'] != 3 else (200, 200, 200)
            cv2.rectangle(frame_gambar, (x1, y1), (x2, y2), warna_kendaraan, 1)
            cv2.putText(frame_gambar, teks_speed, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, warna_kendaraan, 1)

        # VALIDASI PLAT INDONESIA
        valid_plates = []
        for box in hasil_plat[0].boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            lebar_plat = px2 - px1

            is_valid_position = False
            for v in tracked_vehicles:
                vx1, vy1, vx2, vy2 = v['box']
                v_cls = v['cls']
                vw = vx2 - vx1
                vh = vy2 - vy1

                if (vx1 - 30) <= pcx <= (vx2 + 30) and (vy1 - 30) <= pcy <= (vy2 + 30):
                    if v_cls == 3:
                        if pcy < (vy1 + vh * 0.4) or pcy > (vy1 + vh * 0.6) or pcx < (vx1 + vw * 0.2) or pcx > (vx1 + vw * 0.8):
                            is_valid_position = True
                            break
                    else:
                        if pcy > (vy1 + vh * 0.4):
                            is_valid_position = True
                            break

            if is_valid_position and lebar_plat > 20:
                potongan_plat = frame[py1:py2, px1:px2]
                if potongan_plat.size > 0:
                    try:
                        plat_zoom = cv2.resize(potongan_plat, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                        teks_hasil = self.reader.readtext(plat_zoom, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                        if teks_hasil:
                            plat_resmi = self.cek_plat_indonesia(teks_hasil)
                            if plat_resmi:
                                valid_plates.append((px1, py1, px2, py2))
                                cv2.rectangle(frame_gambar, (px1, py1), (px2, py2), (255, 0, 0), 2)
                                cv2.putText(frame_gambar, plat_resmi, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    except:
                        pass

        # ANALISIS HELM
        current_head_centers = []
        for box in hasil_helm[0].boxes:
            hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
            kelas_id = int(box.cls[0])
            nama_objek = self.model_helm.names[kelas_id].lower()

            hcx, hcy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
            h_area = (hx2 - hx1) * (hy2 - hy1)
            current_head_centers.append((hcx, hcy))

            status_kepala = "PEJALAN_KAKI"
            motor_terkait = None
            kecepatan_terkait = 0

            for v in tracked_vehicles:
                if v['cls'] == 3:
                    mx1, my1, mx2, my2 = v['box']
                    mw = mx2 - mx1
                    mh = my2 - my1
                    if (mx1 - mw*0.4) <= hcx <= (mx2 + mw*0.4) and (my1 - mh*1.5) <= hcy <= my2:
                        status_kepala = v['status']
                        motor_terkait = v['box']
                        kecepatan_terkait = v['speed']
                        break

            if status_kepala == "PEJALAN_KAKI":
                warna = (128, 128, 128)
                label = "PEJALAN KAKI"
            elif status_kepala == "STOPPED":
                warna = (0, 255, 255)
                label = "BERHENTI (AMAN)"
            else:
                if "no" in nama_objek or "without" in nama_objek or "bare" in nama_objek:
                    warna = (0, 0, 255)
                    label = f"NO HELM | {int(kecepatan_terkait)} KMH"
                    jumlah_pelanggar += 1
                    if h_area > max_area_pelanggar:
                        max_area_pelanggar = h_area
                        kotak_motor_pelanggar = motor_terkait
                        kecepatan_pelanggar = kecepatan_terkait
                else:
                    warna = (0, 255, 0)
                    label = f"HELM | {int(kecepatan_terkait)} KMH"

            cv2.rectangle(frame_gambar, (hx1, hy1), (hx2, hy2), warna, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_gambar, (hx1, hy1 - 20), (hx1 + tw, hy1), warna, -1)
            cv2.putText(frame_gambar, label, (hx1, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        all_centers_memory = current_centers + current_head_centers

        return {
            "frame": frame_gambar,
            "max_area": max_area_pelanggar,
            "target_motor": kotak_motor_pelanggar,
            "speed": kecepatan_pelanggar,
            "valid_plates": valid_plates,
            "centers": all_centers_memory
        }
