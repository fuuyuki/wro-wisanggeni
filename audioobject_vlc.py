import numpy as np
import cv2
import os
cv2.setNumThreads(0)   # optional
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # optional, not needed
import time
import serial
import RPi.GPIO as GPIO
from tensorflow.lite.python.interpreter import Interpreter
import vlc

# ================= KONFIGURASI GLOBAL =================
MODEL_PATH = 'detect.tflite'
LABEL_PATH = 'labelmap.txt'
MIN_CONF = 0.7
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

#Konfigurasi Pin gpio
#BUTTON_PIN = 17
#BUTTON_EXIT = 27
#BUTTONS_AUDIO = [4, 22, 23]

BUTTON_PIN = 24 #bisa diganti 17, asli 25-----------------------------------------------------------------
BUTTON_EXIT = 23
BUTTONS_AUDIO = [17, 27, 22]

AUDIO_FOLDER = 'audio'

capture_count = {} #variabel global untuk menyimpan data hitungan total (dataset)

# Variabel Global untuk VLC Player
vlc_instance = None
vlc_player = None

# ================= 1. FUNGSI INISIALISASI =================

def inisialisasi_gpio():
    """Mengatur konfigurasi pin tombol Raspberry Pi"""
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) #pin 17 scan

    GPIO.setup(BUTTON_EXIT, GPIO.IN, pull_up_down=GPIO.PUD_UP) #pin 27 exit

    GPIO.setup(BUTTONS_AUDIO, GPIO.IN, pull_up_down=GPIO.PUD_UP) #pin 4,22,23,24
    print("GPIO semua Ok")



def inisialisasi_audio():
    global vlc_instance, vlc_player

    try:
        vlc_instance = vlc.Instance()
        vlc_player = vlc_instance.media_player_new()

        #vlc_player.audio_set_volume(200)

        # AKTIFKAN EQUALIZER AKTIF AGAR SUARA SPEAKER BOSE KELUAR MAKSIMAL
        eq = vlc.AudioEqualizer('Rock')  # Preset Rock membuka suara treble & bass yang mendem
        eq.set_preamp(1000.0)              # Menaikkan power dasar sinyal Raspi sebesar +12dB
        vlc_player.set_equalizer(eq)

        print("Audio VLC + Active Equalizer Ok")
    except Exception as e:
        print(f"Gagal menginisialisasi VLC Audio: {e}")

def inisialisasi_serial():
    """Mengatur koneksi komunikasi data ke Arduino"""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(3)
        ser.reset_input_buffer()
        print("Serial Ok")
        return ser
    except Exception as e:
        print(f"Gagal membuka Serial Port: {e}")
        return None

def inisialisasi_model():
    """Memuat model TFLite dan membaca file label objek"""
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Membaca daftar label objek
    with open(LABEL_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    print("Model TFLite Ok")
    return interpreter, input_details, output_details, labels

# ================= 2. FUNGSI FITUR UTAMA =================

def putar_suara(class_id, nomor_tombol=None):
    """Memutar file suara berdasarkan indeks kelas objek tanpa membuat layar macet"""
    global vlc_instance, vlc_player

    if nomor_tombol is None:
        nama_file_suara = f"suara_{class_id}.mp3"
    else:
        nama_file_suara = f"suara_{class_id}_tombol_{nomor_tombol}.mp3"

    path_lengkap = os.path.join(AUDIO_FOLDER, nama_file_suara)

    if os.path.exists(path_lengkap):
        print(f"Memuat audio baru ke sistem: {path_lengkap}")
        vlc_player.stop()

        media = vlc_instance.media_new(path_lengkap)
        vlc_player.set_media(media)
        vlc_player.play()
        vlc_player.audio_set_volume(200)

    else:
        print(f"Peringatan: File berkas suara '{path_lengkap}' tidak ditemukan!")


def kirim_ke_esp32(ser):
    """Mengirim string 'e' ke esp32 via Serial"""
    if ser is not None:
        try:
            message = "e"
            ser.write(message.encode('utf-8'))
            #ser.write(b"e")
            print("data terkirim ke esp32: e")
        except Exception as e:
            print(f"Gagal mengirim data Serial: {e}")

# ================ 3. fungsi scanning objek =====================
def jalankan_kamera_dan_deteksi(interpreter, input_details, output_details, labels,ser): #di sini nanti tambahkan variabel ser kalau udah dipasang ke esp32
    """Membuka kamera, capture objek, MATIKAN KAMERA SEGERA, lalu tampilkan gambar diam + audio"""
    global capture_count, vlc_player

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)

    print("Membuka Kamera & Mulai Deteksi...")
    cap = cv2.VideoCapture(0)

    frame_terakhir = None
    class_id = None
    capture_message = ""
    prev_time = 0

    try:
        # --- TAHAP 1: NYALAKAN KAMERA & CARI OBJEK ---
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame kamera.")
                break

            current_time = time.time()

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imH, imW, _ = frame.shape
            image_resized = cv2.resize(image_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)

            if float_input:
                input_data = (np.float32(input_data) - 127.5) / 127.5

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            boxes = interpreter.get_tensor(output_details[1]['index'])[0]
            classes = interpreter.get_tensor(output_details[3]['index'])[0]
            scores = interpreter.get_tensor(output_details[0]['index'])[0]

            objek_ditemukan = False

            for i in range(len(scores)):
                if (scores[i] > MIN_CONF) and (scores[i] <= 1.0):
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    class_id = int(classes[i])
                    object_name = labels[class_id]

                    # Gambar kotak dan label deteksi di frame
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                    label = f"{object_name}: {int(scores[i]*100)}%"
                    cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # Logika Hitungan / Capture
                    if class_id not in capture_count:
                        capture_count[class_id] = 0
                    capture_count[class_id] += 1

                    capture_message = f"Captured class {class_id}"
                    print(f"Berhasil Capture! {capture_message} | Total: {capture_count[class_id]}")

                    # Tampilkan notifikasi teks di atas foto hasil capture
                    cv2.putText(frame, capture_message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)

                    # Simpan gambar terakhir yang sudah ada bounding box-nya ke memori
                    frame_terakhir = frame.copy()

                    # Pemicu Audio dinyalakan
                    putar_suara(class_id)
                    objek_ditemukan = True
                    break

            if objek_ditemukan:
                break # Keluar dari loop kamera jika objek sudah didapatkan

            time_diff = current_time - prev_time
            fps = 1/time_diff if time_diff > 0 else 0
            prev_time = current_time

            fps_text = f"fps: {fps:.1f}"
            cv2.putText(frame, fps_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

            # Jika belum ada objek, tetap tampilkan video live kamera seperti biasa
            # cv2.imshow('output', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("Batal scan, kembali ke standby.")
            #     return True

    finally:
        # --- TAHAP 2: KAMERA LANGSUNG DIMATIKAN DI SINI ---
        cap.release()
        print("Hardware kamera berhasil dimatikan (Hemat Baterai).")

    # --- TAHAP 3: SUBMENU INTERAKTIF AUDIO DENGAN SISTEM KUNCI MUTLAK ---
    if frame_terakhir is not None and class_id is not None:
        print (f"--- Masuk ke menu audio objek kelas {class_id}. ---")
        print("CATATAN: AUDIO YANG SEDANG BERJALAN HARUS SELESAI TOTAL SEBELUM TOMBOL APAPUN BISA MERESPON")

        while True:
            #Tampilkan foto diam hasil capture secara berulang agar os tidak hang
            # cv2.imshow('output', frame_terakhir)
            # cv2.waitKey(30)

            #cek status secara real-tme: apakah ada audio yang sedang berputar?
            audio_sedang_berputar = vlc_player.is_playing()

            #A. Cek tombol gpio 27 (exit utama)
            if GPIO.input(BUTTON_EXIT) == GPIO.LOW:
                if audio_sedang_berputar:
                    print("Aksi ditolak: tunggu sampai audio yang sedang berjalan selesai!")
                    time.sleep(0.3) #mencegah double trigger

                else:
                    print("Tombol GPIO 27 ditekan! Mengirim pesan exit ke ESP32...")

                    nama_file_exit = os.path.join(AUDIO_FOLDER, f"suara_{class_id}_exit.mp3")
                    if os.path.exists(nama_file_exit):
                        vlc_player.stop()

                        media_exit = vlc_instance.media_new(nama_file_exit)
                        vlc_player.set_media(media_exit)
                        vlc_player.play()

                        time.sleep(0.3)

                        print("Memutar audio exit, menahan pengiriman pesan ke ESP32...")

                        while vlc_player.get_state() not in [vlc.State.Ended, vlc.State.Stopped, vlc.State.Error]:
                            # cv2.imshow('output', frame_terakhir)
                            # cv2.waitKey(30)

                            print("Audio exit SELESAI TOTAL!")
                    else:
                        print(f"Peringatan: file audio exit '{nama_file_exit}' tidak ditemukan, lanjut tanpa suara")

                    print("Audio exit selesai. Mengirim pesan e ke ESP32...")
                    kirim_ke_esp32(ser)
                    break #keluar dari loop audio, menutup layar

            #B. Cek tombol pilihan audio tambahan GPIO 4,22,23,24
            for pin_tombol in BUTTONS_AUDIO:
                if GPIO.input(pin_tombol) == GPIO.LOW:
                    if audio_sedang_berputar:
                        print("Aksi ditolak: tunggu sampai audio yang sedang berjalan selesai!")
                        time.sleep(0.3) #mencegah double trigger
                        continue #lewati proses di bawah, cek tombol berikutnya

                    #jika kondisi hening/audio selesai, barulah lagu baru diizinkan berputar
                    print(f"tombol audio pin GPIO {pin_tombol} diizinkan ditekan.")
                    putar_suara(class_id, nomor_tombol=pin_tombol)
                    time.sleep(0.4)


            # Tombol q untuk skip audio dan langsung menutup layar
            if cv2.waitKey(100) & 0xFF == ord('q'):
                print("Audio dilewati oleh user.")
                vlc_player.stop()
                break

    # --- TAHAP 4: BERSIHKAN MONITOR & KEMBALI STANDBY ---
    cv2.destroyAllWindows()
    print("Jendela monitor ditutup. Kembali ke mode Standby.")


# ================= 4. FUNGSI UTAMA (MAIN LOOP) =================

def main():
    # Panggil semua fungsi inisialisasi diawal
    inisialisasi_gpio()
    inisialisasi_audio()
    ser = inisialisasi_serial()
    interpreter, input_details, output_details, labels = inisialisasi_model()

    print("\n system standby: menunggu perintah dari ESP32")

    try:
        while True:
            #cek penekanan tombol fisik saat mode standby
            if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
                print("\n tombol ditekan")
                time.sleep(0.4)

                jalankan_kamera_dan_deteksi(interpreter, input_details, output_details, labels,ser) #tambahkan ser kalau sudah disambungkan ke esp32

                print("\n=== SYSTEM STANDBY: Siap Menerima Trigger Berikutnya ===")

            time.sleep(0.05)

    finally:
        # Menjamin pembersihan resource perangkat keras saat program ditutup
        print("\nMembersihkan Resource...")
        if ser is not None:
            ser.close()
        GPIO.cleanup()
        print("Program Selesai.")

# Jalankan fungsi utama program
if __name__ == '__main__':
    main()
