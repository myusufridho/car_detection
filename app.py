from flask import Flask, render_template_string, request, jsonify, send_file
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
import logging
import time
import threading
from datetime import datetime

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
    print("Selenium available - Real 3D screenshots enabled")
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not available - Using fallback screenshot method")
    print("Install: pip install selenium webdriver-manager")

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.getcwd(), "screenshots")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
OUTPUT_FILE = os.path.join(UPLOAD_DIR, "output.jpg")
REAL_SCREENSHOT_FILE = os.path.join(UPLOAD_DIR, "real_3d_capture.jpg")

models_info = {
    "hood": "hood_5000.keras",
    "rear_left": "rear_left.keras", 
    "rear_right": "rear_right.keras",
    "front_left": "front_left.keras",
    "front_right": "front_right.keras"
}

models = {}
class_names = ["closed", "open"]

def load_models():
    """Load all models with error handling"""
    global models
    for name, path in models_info.items():
        try:
            if os.path.exists(path):
                models[name] = tf.keras.models.load_model(path)
                logger.info(f"Model {name} loaded successfully")
            else:
                logger.warning(f"Model file {path} not found, using dummy model")
                models[name] = create_dummy_model()
        except Exception as e:
            logger.error(f"Error loading model {name}: {str(e)}")
            models[name] = create_dummy_model()

def create_dummy_model():
    """Create a simple dummy model for testing"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(256, 256, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def predict_all_models(pil_img):
    """Predict using all loaded models"""
    try:
        img = pil_img.resize((256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        results = {}
        for part, model in models.items():
            try:
                prediction = model.predict(img_array, verbose=0)[0][0]
                predicted_class = class_names[int(prediction > 0.5)]
                confidence = prediction if prediction > 0.5 else 1 - prediction
                results[part] = {"status": predicted_class, "conf": float(confidence)}
            except Exception as e:
                logger.error(f"Error predicting for {part}: {str(e)}")
                results[part] = {"status": "unknown", "conf": 0.0}
        return results
    except Exception as e:
        logger.error(f"Error in predict_all_models: {str(e)}")
        return {part: {"status": "unknown", "conf": 0.0} for part in models_info.keys()}

def capture_real_3d_screenshot():
    """Capture real 3D car viewer using Selenium"""
    if not SELENIUM_AVAILABLE:
        return False, "Selenium not available"
    
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--window-size=1200,800")
        chrome_options.add_argument("--hide-scrollbars")
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        logger.info("Loading 3D car viewer...")
        driver.get("https://euphonious-concha-ab5c5d.netlify.app/")
        
        wait = WebDriverWait(driver, 15)
        time.sleep(8)
        
        driver.save_screenshot(REAL_SCREENSHOT_FILE)
        driver.quit()
        
        if os.path.exists(REAL_SCREENSHOT_FILE):
            file_size = os.path.getsize(REAL_SCREENSHOT_FILE)
            logger.info(f"Real 3D screenshot captured: {file_size} bytes")
            return True, f"Success: {file_size} bytes"
        else:
            return False, "Screenshot file not created"
            
    except Exception as e:
        logger.error(f"Real screenshot error: {str(e)}")
        return False, str(e)

def create_enhanced_placeholder(width=1200, height=800):
    """Create enhanced placeholder with car detection overlay"""
    img = Image.new('RGB', (width, height), color='#f8f9fa')
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    except:
        font_large = None
        font_small = None
    
    draw.rectangle([(0, 0), (width, 100)], fill='#007bff')
    draw.text((50, 30), "AI Car Component Detection System", fill='white', font=font_large)
    draw.text((50, 60), f"Screenshot captured at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fill='white', font=font_small)
    
    car_x = width // 2 - 200
    car_y = height // 2 - 100
    
    draw.rectangle([(car_x, car_y), (car_x + 400, car_y + 200)], fill='#e9ecef', outline='#6c757d', width=3)
    draw.text((car_x + 150, car_y + 90), "3D Car Model", fill='#495057', font=font_large)
    draw.text((car_x + 120, car_y + 120), "(Real-time detection active)", fill='#6c757d', font=font_small)
    
    components = [
        ("Hood", car_x + 180, car_y - 30),
        ("Front Left", car_x - 80, car_y + 50),
        ("Front Right", car_x + 420, car_y + 50),
        ("Rear Left", car_x - 80, car_y + 150),
        ("Rear Right", car_x + 420, car_y + 150)
    ]
    
    for comp_name, x, y in components:
        draw.rectangle([(x, y), (x + 80, y + 25)], fill='#28a745', outline='#1e7e34')
        draw.text((x + 5, y + 5), comp_name, fill='white', font=font_small)
    
    draw.text((50, height - 80), "This is a placeholder - Real 3D model is processed by AI", fill='#6c757d', font=font_small)
    draw.text((50, height - 50), "AI Detection Results are displayed in the table below", fill='#6c757d', font=font_small)
    
    return img

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Car Detection System</title>
    <meta charset="UTF-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            text-align: center; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }
        .header {
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        .iframe-container {
            position: relative;
            background: #343a40;
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            min-height: 500px;
        }
        iframe { 
            width: 100%; 
            height: 500px; 
            border: none; 
            border-radius: 10px;
        }
        .iframe-overlay {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 12px;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        button {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
        }
        button:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .btn-primary { background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); }
        .btn-danger { background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); }
        .btn-warning { background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%); }
        .btn-secondary { background: linear-gradient(135deg, #6c757d 0%, #5a6268 100%); }
        
        .status-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 30px;
            margin: 30px 0;
        }
        .stats-panel {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #007bff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
        }
        
        .detection-panel {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
        }
        table { 
            width: 100%;
            border-collapse: collapse; 
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
        th, td { 
            padding: 15px; 
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        th {
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            font-weight: 600;
        }
        .status-open { 
            color: #dc3545; 
            font-weight: bold;
            background: rgba(220, 53, 69, 0.1);
            padding: 4px 8px;
            border-radius: 15px;
        }
        .status-closed { 
            color: #28a745; 
            font-weight: bold;
            background: rgba(40, 167, 69, 0.1);
            padding: 4px 8px;
            border-radius: 15px;
        }
        .status-unknown { 
            color: #6c757d; 
            font-style: italic;
            background: rgba(108, 117, 125, 0.1);
            padding: 4px 8px;
            border-radius: 15px;
        }
        
        .log-panel {
            background: #1e1e1e;
            color: #00ff00;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 300px;
            overflow-y: auto;
            text-align: left;
        }
        .log-panel::-webkit-scrollbar {
            width: 8px;
        }
        .log-panel::-webkit-scrollbar-track {
            background: #2d2d2d;
        }
        .log-panel::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 4px;
        }
        
        @media (max-width: 768px) {
            .status-grid { grid-template-columns: 1fr; }
            .controls { justify-content: center; }
            button { margin: 5px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Car Component Detection System</h1>
            <p>Real-time analysis of car door and hood states using advanced computer vision</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="totalCaptures">0</div>
                <div class="stat-label">Total Captures</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="successRate">0%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="openComponents">0</div>
                <div class="stat-label">Open Components</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="lastUpdate">Never</div>
                <div class="stat-label">Last Update</div>
            </div>
        </div>
        
        <div class="iframe-container">
            <iframe id="carView" src="https://euphonious-concha-ab5c5d.netlify.app/" allowfullscreen></iframe>
            <div class="iframe-overlay">
                <div>3D Car Model Viewer</div>
                <div style="font-size: 10px; opacity: 0.8;">Interactive 3D model for visualization</div>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="manualCapture()" id="captureBtn" class="btn-primary">Enhanced Screenshot</button>
            <button onclick="captureReal3D()" id="realCaptureBtn" class="btn-warning">Real 3D Capture</button>
            <button onclick="toggleAuto()" id="autoBtn" class="btn-secondary">Start Auto Mode</button>
            <button onclick="clearLogs()" class="btn-secondary">Clear Logs</button>
            <a href="/latest.jpg" target="_blank" style="text-decoration: none;">
                <button class="btn-primary">Download Screenshot</button>
            </a>
        </div>
        
        <div class="status-grid">
            <div class="detection-panel">
                <h3>AI Detection Results</h3>
                <table id="statusTable">
                    <thead>
                        <tr><th>Component</th><th>Status</th><th>Confidence</th><th>Last Updated</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><strong>Hood</strong></td><td id="hood" class="status-unknown">Detecting...</td><td id="hood_conf">-</td><td id="hood_time">-</td></tr>
                        <tr><td><strong>Front Left Door</strong></td><td id="front_left" class="status-unknown">Detecting...</td><td id="front_left_conf">-</td><td id="front_left_time">-</td></tr>
                        <tr><td><strong>Front Right Door</strong></td><td id="front_right" class="status-unknown">Detecting...</td><td id="front_right_conf">-</td><td id="front_right_time">-</td></tr>
                        <tr><td><strong>Rear Left Door</strong></td><td id="rear_left" class="status-unknown">Detecting...</td><td id="rear_left_conf">-</td><td id="rear_left_time">-</td></tr>
                        <tr><td><strong>Rear Right Door</strong></td><td id="rear_right" class="status-unknown">Detecting...</td><td id="rear_right_conf">-</td><td id="rear_right_time">-</td></tr>
                    </tbody>
                </table>
            </div>
            
            <div class="stats-panel">
                <h3>System Statistics</h3>
                <div id="detectionSummary">
                    <p><strong>Current Status:</strong> <span id="overallStatus">Initializing...</span></p>
                    <p><strong>Detection Mode:</strong> <span id="detectionMode">Manual</span></p>
                    <p><strong>Last Screenshot:</strong> <span id="lastScreenshot">None</span></p>
                    <p><strong>AI Model Status:</strong> <span style="color: #28a745;">Active</span></p>
                </div>
            </div>
        </div>
        
        <div class="log-panel" id="logArea">
<strong>AI CAR DETECTION SYSTEM v2.0</strong>
<span style="color: #00ff00;">[SYSTEM]</span> Initializing AI detection models...
<span style="color: #00bfff;">[INFO]</span> Enhanced screenshot system ready
<span style="color: #ffa500;">[NOTICE]</span> Real 3D capture available via Selenium
<span style="color: #00ff00;">[READY]</span> System online and ready for detection
        </div>
    </div>
    
    <script>
        let autoCapture = false;
        let captureInterval = null;
        let totalCaptures = 0;
        let successfulCaptures = 0;
        let openComponentsCount = 0;
        
        function log(message, type = 'INFO') {
            const logArea = document.getElementById('logArea');
            const timestamp = new Date().toLocaleTimeString();
            const colors = {
                'INFO': '#00bfff',
                'SUCCESS': '#00ff00', 
                'ERROR': '#ff4444',
                'WARNING': '#ffa500',
                'SYSTEM': '#ff00ff'
            };
            const color = colors[type] || '#ffffff';
            logArea.innerHTML += `<span style="color: ${color};">[${type}]</span> [${timestamp}] ${message}<br>`;
            logArea.scrollTop = logArea.scrollHeight;
        }
        
        function updateStats() {
            document.getElementById('totalCaptures').textContent = totalCaptures;
            document.getElementById('successRate').textContent = 
                totalCaptures > 0 ? Math.round((successfulCaptures / totalCaptures) * 100) + '%' : '0%';
            document.getElementById('openComponents').textContent = openComponentsCount;
            document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            document.getElementById('lastScreenshot').textContent = new Date().toLocaleString();
        }
        
        function updateStatus(results) {
            const currentTime = new Date().toLocaleTimeString();
            openComponentsCount = 0;
            
            Object.keys(results).forEach(part => {
                const statusElement = document.getElementById(part);
                const confElement = document.getElementById(part + '_conf');
                const timeElement = document.getElementById(part + '_time');
                
                if (statusElement && confElement && timeElement) {
                    const status = results[part].status;
                    const confidence = (results[part].conf * 100).toFixed(1);
                    
                    statusElement.textContent = status.toUpperCase();
                    statusElement.className = `status-${status}`;
                    confElement.textContent = confidence + '%';
                    timeElement.textContent = currentTime;
                    
                    if (status === 'open') openComponentsCount++;
                }
            });
            
            const overallStatus = document.getElementById('overallStatus');
            if (openComponentsCount > 0) {
                overallStatus.innerHTML = `<span style="color: #dc3545;">${openComponentsCount} component(s) OPEN</span>`;
            } else {
                overallStatus.innerHTML = `<span style="color: #28a745;">All components CLOSED</span>`;
            }
        }
        
        async function captureAndPredict() {
            totalCaptures++;
            
            try {
                log('Starting enhanced webpage screenshot...', 'INFO');
                
                const canvas = await html2canvas(document.body, {
                    allowTaint: true,
                    useCORS: true,
                    scale: 0.9,
                    width: window.innerWidth,
                    height: Math.max(document.body.scrollHeight, window.innerHeight),
                    scrollX: 0,
                    scrollY: 0,
                    backgroundColor: '#ffffff',
                    removeContainer: false,
                    logging: false
                });
                
                const dataURL = canvas.toDataURL('image/jpeg', 0.9);
                log(`Enhanced canvas created: ${canvas.width}x${canvas.height}px`, 'SUCCESS');
                
                const response = await fetch('/save_screenshot', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image: dataURL })
                });
                
                if (response.ok) {
                    const result = await response.json();
                    log(`Screenshot saved: ${result.size} bytes (${result.dimensions})`, 'SUCCESS');
                    
                    const predictResponse = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ image: dataURL })
                    });
                    
                    if (predictResponse.ok) {
                        const predictions = await predictResponse.json();
                        log('AI analysis completed successfully', 'SUCCESS');
                        updateStatus(predictions);
                        successfulCaptures++;
                        
                        const openParts = Object.entries(predictions)
                            .filter(([_, data]) => data.status === 'open')
                            .map(([part, _]) => part.replace('_', ' '));
                        
                        if (openParts.length > 0) {
                            log(`ALERT: Open detected - ${openParts.join(', ')}`, 'WARNING');
                        } else {
                            log('All components secure (closed)', 'SUCCESS');
                        }
                    } else {
                        log('AI prediction failed', 'ERROR');
                    }
                } else {
                    log('Screenshot upload failed', 'ERROR');
                }
            } catch (err) {
                log(`System error: ${err.message}`, 'ERROR');
                console.error('Detailed error:', err);
            }
            
            updateStats();
        }
        
        async function captureReal3D() {
            const btn = document.getElementById('realCaptureBtn');
            btn.disabled = true;
            btn.textContent = 'Capturing Real 3D...';
            
            try {
                log('Initiating real 3D model capture via Selenium...', 'SYSTEM');
                
                const response = await fetch('/capture_real_3d', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                });
                
                if (response.ok) {
                    const result = await response.json();
                    log(`Real 3D capture completed: ${result.message}`, 'SUCCESS');
                    
                    if (result.predictions) {
                        updateStatus(result.predictions);
                        successfulCaptures++;
                        totalCaptures++;
                        updateStats();
                        log('AI analysis on real 3D model completed', 'SUCCESS');
                    }
                } else {
                    const error = await response.json();
                    log(`Real 3D capture failed: ${error.error}`, 'ERROR');
                }
            } catch (err) {
                log(`Real capture error: ${err.message}`, 'ERROR');
            }
            
            btn.disabled = false;
            btn.textContent = 'Real 3D Capture';
        }
        
        function manualCapture() {
            const btn = document.getElementById('captureBtn');
            btn.disabled = true;
            btn.textContent = 'Processing...';
            
            captureAndPredict().finally(() => {
                btn.disabled = false;
                btn.textContent = 'Enhanced Screenshot';
            });
        }
        
        function toggleAuto() {
            const btn = document.getElementById('autoBtn');
            
            if (!autoCapture) {
                autoCapture = true;
                btn.textContent = 'Stop Auto Mode';
                btn.className = 'btn-danger';
                document.getElementById('detectionMode').textContent = 'Automatic (20s interval)';
                log('Auto detection mode activated (20 second intervals)', 'SYSTEM');
                
                captureInterval = setInterval(captureAndPredict, 20000);
                captureAndPredict();
            } else {
                autoCapture = false;
                btn.textContent = 'Start Auto Mode';
                btn.className = 'btn-secondary';
                document.getElementById('detectionMode').textContent = 'Manual';
                log('Auto detection mode deactivated', 'SYSTEM');
                
                if (captureInterval) {
                    clearInterval(captureInterval);
                    captureInterval = null;
                }
            }
        }
        
        function clearLogs() {
            const logArea = document.getElementById('logArea');
            logArea.innerHTML = '<strong>AI CAR DETECTION SYSTEM v2.0</strong><br>' +
                               '<span style="color: #00ff00;">[READY]</span> System logs cleared<br>';
        }
        
        setTimeout(() => {
            log('System initialization complete', 'SYSTEM');
            log('Performing initial detection scan...', 'INFO');
            captureAndPredict();
        }, 2000);
        
        setInterval(() => {
            const iframe = document.getElementById('carView');
            if (iframe) {
                log('Maintaining 3D viewer connection...', 'INFO');
            }
        }, 60000);
        
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    """AI Prediction with enhanced error handling"""
    try:
        data = request.get_json()
        if not data or "image" not in data:
            logger.error("No image data received")
            return jsonify({"error": "No image data provided"}), 400

        img_data = data["image"]
        if not img_data.startswith("data:image"):
            logger.error("Invalid image format")
            return jsonify({"error": "Invalid image format"}), 400

        img_base64 = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_base64)
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        results = predict_all_models(pil_img)
        logger.info(f"AI Prediction: {results}")
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/save_screenshot", methods=["POST"])
def save_screenshot():
    """Enhanced screenshot saving with placeholder generation"""
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data"}), 400
            
        img_data = data["image"]
        if not img_data.startswith("data:image"):
            return jsonify({"error": "Invalid format"}), 400
            
        img_base64 = img_data.split(",")[1]
        img_bytes = base64.b64decode(img_base64)
        
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        img_array = np.array(pil_img)
        avg_brightness = np.mean(img_array)
        
        if avg_brightness > 240:
            logger.warning("Detected empty screenshot, creating enhanced placeholder")
            pil_img = create_enhanced_placeholder(pil_img.width, pil_img.height)
        
        pil_img.save(OUTPUT_FILE, "JPEG", quality=90)
        
        timestamped_file = os.path.join(UPLOAD_DIR, f"enhanced_{int(time.time())}.jpg")
        pil_img.save(timestamped_file, "JPEG", quality=90)
        
        file_size = os.path.getsize(OUTPUT_FILE)
        logger.info(f"Enhanced screenshot saved: {file_size} bytes")
        
        return jsonify({
            "status": "success", 
            "size": file_size,
            "dimensions": f"{pil_img.width}x{pil_img.height}",
            "enhanced": avg_brightness > 240
        })
        
    except Exception as e:
        logger.error(f"Screenshot error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/capture_real_3d", methods=["POST"])
def capture_real_3d_endpoint():
    """Real 3D capture endpoint using Selenium"""
    try:
        success, message = capture_real_3d_screenshot()
        
        if success:
            with open(REAL_SCREENSHOT_FILE, 'rb') as f:
                img_bytes = f.read()
                pil_img = Image.open(BytesIO(img_bytes))
                predictions = predict_all_models(pil_img)
            
            return jsonify({
                "status": "success",
                "message": message,
                "predictions": predictions,
                "file_size": os.path.getsize(REAL_SCREENSHOT_FILE)
            })
        else:
            return jsonify({
                "status": "error",
                "error": message
            }), 500
            
    except Exception as e:
        logger.error(f"Real 3D capture error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/latest.jpg")
def latest_screenshot():
    """Serve latest screenshot with fallback"""
    try:
        if os.path.exists(REAL_SCREENSHOT_FILE) and os.path.getsize(REAL_SCREENSHOT_FILE) > 1000:
            return send_file(REAL_SCREENSHOT_FILE, mimetype="image/jpeg")
        elif os.path.exists(OUTPUT_FILE):
            return send_file(OUTPUT_FILE, mimetype="image/jpeg")
        else:
            placeholder = create_enhanced_placeholder()
            placeholder_io = BytesIO()
            placeholder.save(placeholder_io, 'JPEG', quality=90)
            placeholder_io.seek(0)
            return send_file(placeholder_io, mimetype="image/jpeg")
    except Exception as e:
        logger.error(f"Error serving screenshot: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route("/real_3d.jpg")
def real_3d_screenshot():
    """Serve real 3D screenshot specifically"""
    try:
        if os.path.exists(REAL_SCREENSHOT_FILE):
            return send_file(REAL_SCREENSHOT_FILE, mimetype="image/jpeg")
        else:
            return "Real 3D screenshot not available", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route("/status")
def status():
    """Enhanced system status"""
    try:
        return jsonify({
            "status": "running",
            "version": "2.0",
            "models_loaded": len(models),
            "models": list(models.keys()),
            "selenium_available": SELENIUM_AVAILABLE,
            "screenshots": {
                "regular": os.path.exists(OUTPUT_FILE),
                "real_3d": os.path.exists(REAL_SCREENSHOT_FILE)
            },
            "tensorflow_version": tf.__version__,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

load_models()

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("AI CAR DETECTION SYSTEM v2.0")
    logger.info("=" * 60)
    logger.info(f"Screenshots: {UPLOAD_DIR}")
    logger.info(f"Models: {list(models.keys())}")
    logger.info(f"TensorFlow: {tf.__version__}")
    logger.info(f"Selenium: {'Available' if SELENIUM_AVAILABLE else 'Not Available'}")
    logger.info(f"Server: http://127.0.0.1:5000")
    logger.info("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
