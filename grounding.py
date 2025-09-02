from flask import Flask, render_template_string, request, send_from_directory
import cv2
import os
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate

config_path = "GroundingDINO_SwinT_OGC.cfg.py"
weights_path = "groundingdino_swint_ogc.pth"
model = load_model(config_path, weights_path)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GroundingDINO - Object Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        .upload-area {
            border: 3px dashed #dee2e6;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            background: #f8f9fa;
        }
        .upload-area:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }
        .upload-area.dragover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }
        .btn-primary {
            background: linear-gradient(45deg, #667eea, #764ba2);
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        .result-card {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        .loading {
            display: none;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
        .form-control, .form-select {
            border-radius: 10px;
            border: 2px solid #e9ecef;
            padding: 12px 15px;
            transition: all 0.3s ease;
        }
        .form-control:focus, .form-select:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
        }
        .header-icon {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin: 0 auto 20px;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="main-card p-5">
                    <div class="text-center mb-5">
                        <div class="header-icon">
                            <i class="fas fa-eye"></i>
                        </div>
                        <h1 class="display-6 fw-bold text-dark mb-3">GroundingDINO</h1>
                        <p class="lead text-muted">Upload gambar dan masukkan prompt untuk deteksi objek yang akurat</p>
                    </div>
                    <form method="POST" enctype="multipart/form-data" id="uploadForm">
                        <div class="row g-4">
                            <div class="col-12">
                                <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                                    <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                                    <h5 class="text-muted">Klik untuk pilih gambar atau drag & drop</h5>
                                    <p class="text-muted small mb-0">Format: JPG, PNG, GIF (Max: 10MB)</p>
                                    <input type="file" name="image" id="imageInput" class="d-none" accept="image/*" required>
                                </div>
                                <div id="imagePreview" class="mt-3 text-center d-none">
                                    <img id="previewImg" src="" class="img-thumbnail" style="max-height: 200px;">
                                    <p class="mt-2 small text-muted">File terpilih: <span id="fileName"></span></p>
                                </div>
                            </div>
                            <div class="col-12">
                                <label class="form-label fw-semibold">
                                    <i class="fas fa-comments me-2"></i>Prompt Deteksi
                                </label>
                                <input type="text" name="prompt" class="form-control form-control-lg" 
                                       placeholder="Contoh: rear left door, person, car, building..." required>
                                <div class="form-text">
                                    <i class="fas fa-info-circle me-1"></i>
                                    Masukkan deskripsi objek yang ingin dideteksi dalam bahasa Inggris
                                </div>
                            </div>
                            <div class="col-12 text-center">
                                <button type="submit" class="btn btn-primary btn-lg px-5">
                                    <i class="fas fa-magic me-2"></i>Proses Gambar
                                </button>
                            </div>
                        </div>
                    </form>
                    <div class="loading text-center my-5" id="loading">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p class="mt-3 text-muted">Sedang memproses gambar, mohon tunggu...</p>
                    </div>
                    {% if result %}
                    <div class="mt-5">
                        <div class="row">
                            <div class="col-12">
                                <div class="result-card bg-white border-0">
                                    <div class="card-header bg-success text-white text-center py-3">
                                        <h5 class="mb-0">
                                            <i class="fas fa-check-circle me-2"></i>Hasil Deteksi
                                        </h5>
                                    </div>
                                    <div class="card-body p-0">
                                        <img src="{{ url_for('output_file', filename=result) }}" 
                                             class="img-fluid w-100" alt="Hasil deteksi">
                                    </div>
                                    <div class="card-footer text-center bg-light">
                                        <a href="{{ url_for('output_file', filename=result) }}" 
                                           download class="btn btn-outline-success">
                                            <i class="fas fa-download me-2"></i>Download Hasil
                                        </a>
                                        <button onclick="window.location.reload()" class="btn btn-outline-primary ms-2">
                                            <i class="fas fa-redo me-2"></i>Proses Lagi
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                    <div class="text-center mt-5 pt-4 border-top">
                    
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('previewImg').src = e.target.result;
                    document.getElementById('fileName').textContent = file.name;
                    document.getElementById('imagePreview').classList.remove('d-none');
                };
                reader.readAsDataURL(file);
            }
        });
        const uploadArea = document.querySelector('.upload-area');
        const imageInput = document.getElementById('imageInput');
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
        });
        function highlight(e) {
            uploadArea.classList.add('dragover');
        }
        function unhighlight(e) {
            uploadArea.classList.remove('dragover');
        }
        uploadArea.addEventListener('drop', handleDrop, false);
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                imageInput.files = files;
                imageInput.dispatchEvent(new Event('change'));
            }
        }
        document.getElementById('uploadForm').addEventListener('submit', function() {
            document.getElementById('loading').style.display = 'block';
        });
        document.addEventListener('DOMContentLoaded', function() {
            const card = document.querySelector('.main-card');
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px)';
            setTimeout(() => {
                card.style.transition = 'all 0.6s ease';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 100);
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result_filename = None
    if request.method == "POST":
        file = request.files["image"]
        prompt = request.form["prompt"]
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)
        cv2_bgr = cv2.imread(img_path)
        cv2_rgb = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2RGB)
        image_source, image_tensor = load_image(img_path)
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=prompt,
            box_threshold=0.35,
            text_threshold=0.35,
            device="cpu"
        )
        annotated_frame = annotate(
            image_source=cv2_rgb,
            boxes=boxes,
            logits=logits,
            phrases=phrases
        )
        result_filename = f"result_{file.filename}"
        out_path = os.path.join(OUTPUT_FOLDER, result_filename)
        cv2.imwrite(out_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
    return render_template_string(HTML, result=result_filename)

@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
