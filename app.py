import os
import io
import base64
import time
from typing import List, Dict, Tuple

from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import cv2
import requests


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Model sources with multiple mirrors/fallbacks
AGE_PROTOTXT_URLS = [
    "https://raw.githubusercontent.com/akshtsng/Gender_Detection_and_Age_Prediction/master/age_deploy.prototxt",
    "https://raw.githubusercontent.com/OshaPandey/Age_Gender_Detection/main/age_deploy.prototxt",
    "https://raw.githubusercontent.com/HardShell1307/DeepLearning_Gender-and-Age-Detection-OpenCV-Python/main/age_deploy.prototxt"
]
AGE_MODEL_URLS = [
    "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
    "https://raw.githubusercontent.com/OshaPandey/Age_Gender_Detection/main/age_net.caffemodel",
    "https://raw.githubusercontent.com/HardShell1307/DeepLearning_Gender-and-Age-Detection-OpenCV-Python/main/age_net.caffemodel"
]
GENDER_PROTOTXT_URLS = [
    "https://raw.githubusercontent.com/akshtsng/Gender_Detection_and_Age_Prediction/master/gender_deploy.prototxt",
    "https://raw.githubusercontent.com/OshaPandey/Age_Gender_Detection/main/gender_deploy.prototxt",
    "https://raw.githubusercontent.com/HardShell1307/DeepLearning_Gender-and-Age-Detection-OpenCV-Python/main/gender_deploy.prototxt"
]
GENDER_MODEL_URLS = [
    "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel",
    "https://raw.githubusercontent.com/OshaPandey/Age_Gender_Detection/main/gender_net.caffemodel",
    "https://raw.githubusercontent.com/HardShell1307/DeepLearning_Gender-and-Age-Detection-OpenCV-Python/main/gender_net.caffemodel"
]

AGE_PROTOTXT_PATH = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL_PATH = os.path.join(MODELS_DIR, "age_net.caffemodel")
GENDER_PROTOTXT_PATH = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL_PATH = os.path.join(MODELS_DIR, "gender_net.caffemodel")

AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)"
]
GENDERS = ["Male", "Female"]

MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Global camera object
camera = None


def _download_file(url: str, target_path: str) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(target_path, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out_file.write(chunk)


def _download_with_fallback(urls, target_path: str) -> None:
    last_err = None
    for url in urls:
        try:
            _download_file(url, target_path)
            # basic sanity check: ensure file has some bytes
            if os.path.getsize(target_path) > 1024:
                return
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError(f"Failed to download {target_path}")


def ensure_models_downloaded() -> None:
    to_fetch = []
    if not os.path.exists(AGE_PROTOTXT_PATH) or os.path.getsize(AGE_PROTOTXT_PATH) == 0:
        to_fetch.append(("age prototxt", AGE_PROTOTXT_URLS, AGE_PROTOTXT_PATH))
    if not os.path.exists(AGE_MODEL_PATH) or os.path.getsize(AGE_MODEL_PATH) == 0:
        to_fetch.append(("age model", AGE_MODEL_URLS, AGE_MODEL_PATH))
    if not os.path.exists(GENDER_PROTOTXT_PATH) or os.path.getsize(GENDER_PROTOTXT_PATH) == 0:
        to_fetch.append(("gender prototxt", GENDER_PROTOTXT_URLS, GENDER_PROTOTXT_PATH))
    if not os.path.exists(GENDER_MODEL_PATH) or os.path.getsize(GENDER_MODEL_PATH) == 0:
        to_fetch.append(("gender model", GENDER_MODEL_URLS, GENDER_MODEL_PATH))

    for label, urls, path in to_fetch:
        _download_with_fallback(urls, path)


def load_networks() -> Tuple[cv2.dnn_Net, cv2.dnn_Net]:
    ensure_models_downloaded()
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT_PATH, AGE_MODEL_PATH)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTOTXT_PATH, GENDER_MODEL_PATH)
    return age_net, gender_net


def read_image_from_request(file_storage) -> np.ndarray:
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def detect_faces(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def analyze_faces(image_bgr: np.ndarray, age_net, gender_net) -> Tuple[List[Dict], np.ndarray]:
    results = []
    annotated = image_bgr.copy()

    faces = detect_faces(image_bgr)
    if not faces:
        return results, annotated

    for (x, y, w, h) in faces:
        face_img = image_bgr[max(0, y): y + h, max(0, x): x + w]
        if face_img.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face_img, scalefactor=1.0, size=(227, 227), mean=MEAN_VALUES, swapRB=False, crop=False)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()[0]
        gender_idx = int(np.argmax(gender_preds))
        gender_label = GENDERS[gender_idx]
        gender_conf = float(gender_preds[gender_idx])

        age_net.setInput(blob)
        age_preds = age_net.forward()[0]
        age_idx = int(np.argmax(age_preds))
        age_label = AGE_BUCKETS[age_idx]
        age_conf = float(age_preds[age_idx])

        results.append({
            "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "gender": gender_label,
            "gender_confidence": round(gender_conf, 4),
            "age_range": age_label,
            "age_confidence": round(age_conf, 4)
        })

        label = f"{gender_label}, {age_label} ({int(max(gender_conf, age_conf) * 100)}%)"
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(annotated, (x, y - 22), (x + w, y), (0, 255, 0), cv2.FILLED)
        cv2.putText(annotated, label, (x + 5, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return results, annotated


def image_to_base64_png(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode('.png', image_bgr)
    if not success:
        return ""
    b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = None
    return camera


def generate_frames():
    global _age_net, _gender_net
    
    if _age_net is None or _gender_net is None:
        _age_net, _gender_net = load_networks()
    
    cam = get_camera()
    if cam is None:
        return
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        # Analyze the frame
        results, annotated_frame = analyze_faces(frame, _age_net, _gender_net)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


app = Flask(__name__)

# Lazy-load networks on first request to reduce startup time
_age_net = None
_gender_net = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return {"status": "ok"}


@app.route("/analyze", methods=["POST"]) 
def analyze():
    global _age_net, _gender_net

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided with key 'image'"}), 400

    file = request.files['image']
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image = read_image_from_request(file)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    if _age_net is None or _gender_net is None:
        try:
            _age_net, _gender_net = load_networks()
        except Exception as e:
            return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

    try:
        results, annotated = analyze_faces(image, _age_net, _gender_net)
    except Exception as e:
        return jsonify({"error": f"Failed to analyze image: {str(e)}"}), 500

    annotated_b64 = image_to_base64_png(annotated)
    return jsonify({
        "faces": results,
        "annotated_image": annotated_b64
    })


@app.route("/camera")
def camera_page():
    return render_template("camera.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/camera_status")
def camera_status():
    cam = get_camera()
    available = cam is not None and cam.isOpened()
    return jsonify({"camera_available": available})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)