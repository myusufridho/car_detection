# Deep Learning & Visual Grounding Project

## Objectives
Proyek ini dibuat untuk menjawab soal dengan dua tujuan utama:  
1. Objective #1 (Mandatory): Build a deep learning model from scratch  
2. Objective #3 (Bonus Task): Visual Grounding Model  

---

## 1. Deep Learning Model - Real-time 3D Object Detection (`app.py`)
Model CNN (Convolutional Neural Network) dibangun dari awal untuk melakukan deteksi objek mobil secara real-time 3D.  

### Dataset dan Data Acquisition
- Data diperoleh dari video screen recording.  
- Video diekstraksi menjadi frame gambar untuk dijadikan dataset.  
- Total 5 objek digunakan dalam training, yaitu:  
  - Hood  
  - Rear Left Door  
  - Rear Right Door  
  - Front Left Door  
  - Front Right Door  
- Masing-masing objek memiliki 2 kelas (misalnya: terbuka dan tertutup).  

### Training
- Model CNN dilatih untuk mengenali 5 objek tersebut.  
- Hasil training menunjukkan performa baik pada realtime detection.  

### Realtime Detection
- Model hasil training dipanggil di `app.py` untuk mendeteksi kondisi objek mobil secara langsung.  

---

## 2. Visual Grounding Model - GroundingDINO (`grounding.py`)
Untuk tugas tambahan (Objective #3), digunakan pretrained model **GroundingDINO**.  

### Visual Grounding
Visual Grounding adalah metode untuk menghubungkan deskripsi teks (prompt) dengan objek pada gambar.  
Contoh: jika pengguna memasukkan prompt `"rear left door"`, sistem akan mendeteksi objek pintu kiri belakang pada gambar.  

### GroundingDINO
- Menggunakan pretrained model `GroundingDINO_SwinT_OGC`.  
- Konfigurasi yang digunakan:  
  - `config_path = "GroundingDINO_SwinT_OGC.cfg.py"`  
  - `weights_path = "groundingdino_swint_ogc.pth"`  
- Diimplementasikan dengan Flask di `grounding.py`.  

### Alur penggunaan
1. Upload gambar melalui web app.  
2. Masukkan prompt deskripsi objek (misalnya: `"rear left door, person, car"`).  
3. Sistem akan memproses gambar dengan GroundingDINO.  
4. Gambar hasil deteksi ditampilkan dengan bounding box dan dapat diunduh.  

---

## Model
Model yang digunakan tersedia di Google Drive:  
- CNN Model: [Google Drive Link CNN]  
- GroundingDINO Model: [Google Drive Link GroundingDINO]  

*(silakan ganti dengan link asli setelah diunggah)*  

---

Aplikasi dapat diakses melalui http://127.0.0.1:5000/.


Untuk menjalankan realtime 3D Detection car view, jalankan perintah:
python app.py



Untuk menjalankan visual grounding berbasis GroundingDINO, jalankan perintah:
python grounding.py



