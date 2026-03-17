import cv2
import re
import numpy as np
import easyocr
from ultralytics import YOLO
from config import VEHICLE_MODEL, HELMET_MODEL, PLATE_MODEL, SEATBELT_MODEL

class AIEngine:
    def __init__(self):
        print("[SYSTEM] Memuat AIEngine (YOLO & EasyOCR)...")
        self.model_kendaraan = YOLO(VEHICLE_MODEL)
        self.model_helm = YOLO(HELMET_MODEL)
        self.model_plat = YOLO(PLATE_MODEL)
        self.reader = easyocr.Reader(['en'], gpu=False)

        try:
            self.model_seatbelt = YOLO(SEATBELT_MODEL)
            self.seatbelt_enabled = True
            print("[SYSTEM] Seatbelt model loaded!")
        except Exception as e:
            self.model_seatbelt = None
            self.seatbelt_enabled = False
            print(f"[WARNING] Seatbelt model tidak ditemukan: {e}")

        self.PIXEL_PER_METER = 8.0
        self.MIN_PIXEL_MOVE = 8.0
        self.HISTORY_SIZE = 5
        self.speed_history = {}
        self.direction_history = {}
        self.dominant_left = None
        self.dominant_right = None
        self.frame_width = None

        # Posisi garis tengah jalur (0.0-1.0), sesuaikan dengan posisi jalan di video
        self.LANE_RATIO = 0.62

    def _smooth_speed(self, vehicle_id, new_speed):
        if vehicle_id not in self.speed_history:
            self.speed_history[vehicle_id] = []
        history = self.speed_history[vehicle_id]
        history.append(new_speed)
        if len(history) > self.HISTORY_SIZE:
            history.pop(0)
        if len(history) >= 3:
            avg = sum(history[:-1]) / len(history[:-1])
            if avg > 0 and history[-1] > avg * 3:
                history[-1] = avg
        return sum(history) / len(history)

    def _update_direction(self, vehicle_id, dx, dy):
        if vehicle_id not in self.direction_history:
            self.direction_history[vehicle_id] = []
        self.direction_history[vehicle_id].append((dx, dy))
        if len(self.direction_history[vehicle_id]) > self.HISTORY_SIZE:
            self.direction_history[vehicle_id].pop(0)

    def _get_avg_direction(self, vehicle_id):
        if vehicle_id not in self.direction_history:
            return (0, 0)
        history = self.direction_history[vehicle_id]
        avg_dx = sum(d[0] for d in history) / len(history)
        avg_dy = sum(d[1] for d in history) / len(history)
        return (avg_dx, avg_dy)

    def _get_lane(self, cx):
        if self.frame_width is None:
            return "left"
        mid = int(self.frame_width * self.LANE_RATIO)
        return "left" if cx < mid else "right"

    def _update_dominant_directions(self, vehicles_with_direction):
        left_dy, right_dy = [], []
        for (cx, dy) in vehicles_with_direction:
            if abs(dy) < 3:
                continue
            if self._get_lane(cx) == "left":
                left_dy.append(dy)
            else:
                right_dy.append(dy)

        def majority(dy_list):
            if len(dy_list) < 2: return None
            pos = sum(1 for d in dy_list if d > 0)
            neg = sum(1 for d in dy_list if d < 0)
            if pos > neg: return "DOWN"
            elif neg > pos: return "UP"
            return None

        r = majority(left_dy)
        if r: self.dominant_left = r
        r = majority(right_dy)
        if r: self.dominant_right = r

    def _is_wrong_way(self, vehicle_id, cx):
        dominant = self.dominant_left if self._get_lane(cx) == "left" else self.dominant_right
        if dominant is None: return False
        avg_dx, avg_dy = self._get_avg_direction(vehicle_id)
        if abs(avg_dy) < 3: return False
        if dominant == "DOWN" and avg_dy < -5: return True
        if dominant == "UP" and avg_dy > 5: return True
        return False

    def cek_plat_indonesia(self, teks_list):
        teks = "".join(teks_list).upper()
        teks_bersih = re.sub(r'[^A-Z0-9]', '', teks)
        # Pola lebih longgar: minimal 1 huruf + 1 angka
        pola = r"^[A-Z]{1,2}\d{1,4}[A-Z]{0,3}$"
        if re.match(pola, teks_bersih):
            return teks_bersih
        # Fallback: kalau panjang minimal 4 karakter, tetap kembalikan
        if len(teks_bersih) >= 4:
            return teks_bersih
        return None

    def baca_plat(self, frame, px1, py1, px2, py2):
        """Baca plat dengan multi-scale untuk akurasi lebih tinggi"""
        potongan = frame[py1:py2, px1:px2]
        if potongan.size == 0:
            return None
        try:
            hasil_terbaik = None
            for scale in [2, 3, 4]:
                zoom = cv2.resize(potongan, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                # Sharpen
                kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                zoom = cv2.filter2D(zoom, -1, kernel)
                teks = self.reader.readtext(zoom, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                if teks:
                    hasil = self.cek_plat_indonesia(teks)
                    if hasil and len(hasil) > len(hasil_terbaik or ""):
                        hasil_terbaik = hasil
            return hasil_terbaik
        except:
            return None

    def process_frame(self, frame, prev_centers, fps_video):
        frame_gambar = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        self.frame_width = w_frame

        jumlah_pelanggar = 0
        max_area_pelanggar = 0
        kotak_motor_pelanggar = None
        kecepatan_pelanggar = 0
        pelanggaran_pelanggar = "Pelanggaran Lalu Lintas"

        hasil_kendaraan = self.model_kendaraan(frame, conf=0.4, classes=[2, 3, 5, 7], imgsz=640, verbose=False)
        hasil_helm = self.model_helm(frame, conf=0.4, imgsz=640, verbose=False)
        # Plat: conf lebih rendah + imgsz lebih besar untuk deteksi lebih sensitif
        hasil_plat = self.model_plat(frame, conf=0.25, imgsz=1280, verbose=False)

        kendaraan_mentah = []
        for box in hasil_kendaraan[0].boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            kendaraan_mentah.append({'box': (x1, y1, x2, y2), 'cls': cls_id})

        current_centers = []
        tracked_vehicles = []
        vehicles_with_direction = []

        for idx, v in enumerate(kendaraan_mentah):
            x1, y1, x2, y2 = v['box']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            current_centers.append((cx, cy))

            is_moving = False
            speed_kmh = 0.0
            is_wrong = False

            if prev_centers:
                dists = [np.hypot(cx - px, cy - py) for px, py in prev_centers]
                min_idx = int(np.argmin(dists))
                jarak_pixel = dists[min_idx]
                if jarak_pixel >= self.MIN_PIXEL_MOVE:
                    is_moving = True
                    px_prev, py_prev = prev_centers[min_idx]
                    dx = cx - px_prev
                    dy = cy - py_prev
                    self._update_direction(idx, dx, dy)
                    vehicles_with_direction.append((cx, dy))
                    jarak_meter = jarak_pixel / self.PIXEL_PER_METER
                    raw_speed = (jarak_meter * fps_video) * 3.6
                    speed_kmh = self._smooth_speed(idx, raw_speed)
                    speed_kmh = min(speed_kmh, 100.0)
                    if len(self.direction_history.get(idx, [])) >= 3:
                        is_wrong = self._is_wrong_way(idx, cx)

            v['speed'] = speed_kmh
            v['status'] = "MOVING" if is_moving else "STOPPED"
            v['wrong_way'] = is_wrong
            v['cx'] = cx
            tracked_vehicles.append(v)

            nama = "MOTOR" if v['cls'] == 3 else ("MOBIL" if v['cls'] == 2 else ("BUS" if v['cls'] == 5 else "TRUK"))

            if is_wrong:
                warna_kendaraan = (0, 0, 255)
                teks_speed = f"! {nama} LAWAN ARAH | {int(speed_kmh)} KM/H"
                area = (x2 - x1) * (y2 - y1)
                if area > max_area_pelanggar:
                    max_area_pelanggar = area
                    kotak_motor_pelanggar = v['box']
                    kecepatan_pelanggar = speed_kmh
                    pelanggaran_pelanggar = "Melawan Arah"
                    jumlah_pelanggar += 1
            elif is_moving:
                warna_kendaraan = (0, 165, 255) if v['cls'] != 3 else (200, 200, 200)
                teks_speed = f"{nama} | {int(speed_kmh)} KM/H"
            else:
                warna_kendaraan = (0, 165, 255) if v['cls'] != 3 else (200, 200, 200)
                teks_speed = f"{nama} | BERHENTI"

            cv2.rectangle(frame_gambar, (x1, y1), (x2, y2), warna_kendaraan, 2 if is_wrong else 1)
            cv2.putText(frame_gambar, teks_speed, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, warna_kendaraan, 1)

        # CEK SABUK
        if self.seatbelt_enabled:
            for v in tracked_vehicles:
                if v['cls'] not in [2, 5, 7]:
                    continue
                x1, y1, x2, y2 = v['box']
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                try:
                    hasil_belt = self.model_seatbelt(crop, conf=0.5, imgsz=320, verbose=False)
                    for box in hasil_belt[0].boxes:
                        kelas_id = int(box.cls[0])
                        nama_objek = self.model_seatbelt.names[kelas_id].lower()
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        abs_x1 = x1 + bx1
                        abs_y1 = y1 + by1
                        abs_x2 = x1 + bx2
                        abs_y2 = y1 + by2
                        b_area = (abs_x2 - abs_x1) * (abs_y2 - abs_y1)
                        no_sabuk = any(k in nama_objek for k in ["no", "without", "not", "0"])
                        if no_sabuk:
                            warna = (0, 0, 255)
                            label = f"NO SABUK | {int(v['speed'])} KMH"
                            cv2.rectangle(frame_gambar, (abs_x1, abs_y1), (abs_x2, abs_y2), warna, 2)
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(frame_gambar, (abs_x1, abs_y1 - 20), (abs_x1 + tw, abs_y1), warna, -1)
                            cv2.putText(frame_gambar, label, (abs_x1, abs_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            jumlah_pelanggar += 1
                            if b_area > max_area_pelanggar:
                                max_area_pelanggar = b_area
                                kotak_motor_pelanggar = v['box']
                                kecepatan_pelanggar = v['speed']
                                pelanggaran_pelanggar = "Tidak Menggunakan Sabuk Pengaman"
                        else:
                            cv2.putText(frame_gambar, "SABUK OK", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                except:
                    pass

        self._update_dominant_directions(vehicles_with_direction)

        # Garis tengah jalur
        mid_x = int(w_frame * self.LANE_RATIO)
        cv2.line(frame_gambar, (mid_x, 0), (mid_x, h_frame), (255, 255, 0), 1)

        # VALIDASI PLAT - lebih longgar & multi-scale
        valid_plates = []
        for box in hasil_plat[0].boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            lebar_plat = px2 - px1

            # Validasi posisi plat terhadap kendaraan - lebih longgar
            is_valid_position = False
            for v in tracked_vehicles:
                vx1, vy1, vx2, vy2 = v['box']
                vw = vx2 - vx1
                vh = vy2 - vy1
                # Area pencarian diperlebar
                if (vx1 - vw*0.5) <= pcx <= (vx2 + vw*0.5) and (vy1 - vh*0.3) <= pcy <= (vy2 + vh*0.3):
                    is_valid_position = True
                    break

            # Kalau tidak ada kendaraan terdeteksi di sekitarnya, tetap proses kalau ukuran wajar
            if not is_valid_position and lebar_plat > 30:
                is_valid_position = True

            if is_valid_position and lebar_plat > 15:
                plat_resmi = self.baca_plat(frame, px1, py1, px2, py2)
                if plat_resmi:
                    valid_plates.append((px1, py1, px2, py2))
                    cv2.rectangle(frame_gambar, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    # Label tidak keluar frame
                    label_x = max(0, min(px1, w_frame - 150))
                    cv2.putText(frame_gambar, plat_resmi, (label_x, max(15, py1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
            motor_wrong_way = False

            for v in tracked_vehicles:
                if v['cls'] == 3:
                    mx1, my1, mx2, my2 = v['box']
                    mw = mx2 - mx1
                    mh = my2 - my1
                    # Area pencarian helm diperlebar
                    if (mx1 - mw*0.6) <= hcx <= (mx2 + mw*0.6) and (my1 - mh*2.0) <= hcy <= (my2 + mh*0.3):
                        status_kepala = v['status']
                        motor_terkait = v['box']
                        kecepatan_terkait = v['speed']
                        motor_wrong_way = v['wrong_way']
                        break

            if status_kepala == "PEJALAN_KAKI":
                warna = (128, 128, 128)
                label = "PEJALAN KAKI"
                
           elif status_kepala == "STOPPED":
    # Kalau berhenti, tidak dihitung pelanggaran meski tidak pakai helm
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
                        pelanggaran_pelanggar = "Tidak Menggunakan Helm"
                        if motor_wrong_way:
                            pelanggaran_pelanggar = "Lawan Arah & Tidak Menggunakan Helm"
                else:
                    warna = (0, 255, 0)
                    label = f"HELM | {int(kecepatan_terkait)} KMH"

            cv2.rectangle(frame_gambar, (hx1, hy1), (hx2, hy2), warna, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Label tidak keluar frame
            label_x = max(0, min(hx1, w_frame - tw - 5))
            cv2.rectangle(frame_gambar, (label_x, hy1 - 20), (label_x + tw, hy1), warna, -1)
            cv2.putText(frame_gambar, label, (label_x, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        all_centers_memory = current_centers + current_head_centers

        return {
            "frame": frame_gambar,
            "max_area": max_area_pelanggar,
            "target_motor": kotak_motor_pelanggar,
            "speed": kecepatan_pelanggar,
            "violation": pelanggaran_pelanggar,
            "valid_plates": valid_plates,
            "centers": all_centers_memory
        }
Dan jangan lupa update engine/video_processor.py juga — fix kecepatan 3x lipat:
Cari baris:
hasil = ai_engine.process_frame(frame, prev_centers, fps_asli)
Ganti jadi:
hasil = ai_engine.process_frame(frame, prev_centers, fps_asli / 3)
