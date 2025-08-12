import os
import io
import base64
import time
import glob
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import cv2
import requests
from PIL import Image


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_ROOT, "models")
SAMPLES_DIR = os.path.join(APP_ROOT, "samples")
MISSING_PERSONS_DIR = os.path.join(APP_ROOT, "missing_persons")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(MISSING_PERSONS_DIR, exist_ok=True)

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

# Sample images URLs for demo mode
SAMPLE_IMAGE_URLS = [
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=400&fit=crop&crop=face",
    "https://images.unsplash.com/photo-1494790108755-2616b612b589?w=400&h=400&fit=crop&crop=face",
    "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400&h=400&fit=crop&crop=face",
    "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400&h=400&fit=crop&crop=face",
    "https://images.unsplash.com/photo-1547425260-76bcadfb4f2c?w=400&h=400&fit=crop&crop=face",
    "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&h=400&fit=crop&crop=face"
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

# Global camera object and demo state
camera = None
demo_images = []
demo_frame_index = 0

# Initialize Flask app with database
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///missing_persons.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = MISSING_PERSONS_DIR

db = SQLAlchemy(app)

# Database Models
class MissingPerson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=False)
    description = db.Column(db.Text, nullable=True)
    photo_path = db.Column(db.String(200), nullable=True)
    last_seen_location = db.Column(db.String(200), nullable=True)
    date_reported = db.Column(db.DateTime, default=datetime.utcnow)
    contact_phone = db.Column(db.String(20), nullable=True)
    contact_email = db.Column(db.String(100), nullable=True)
    status = db.Column(db.String(20), default='active')  # active, found, closed
    
    # For ML matching
    detected_age_range = db.Column(db.String(20), nullable=True)
    detected_gender = db.Column(db.String(10), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'description': self.description,
            'photo_path': self.photo_path,
            'last_seen_location': self.last_seen_location,
            'date_reported': self.date_reported.isoformat() if self.date_reported else None,
            'contact_phone': self.contact_phone,
            'contact_email': self.contact_email,
            'status': self.status,
            'detected_age_range': self.detected_age_range,
            'detected_gender': self.detected_gender
        }

class Sighting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    missing_person_id = db.Column(db.Integer, db.ForeignKey('missing_person.id'), nullable=False)
    location = db.Column(db.String(200), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    confidence_score = db.Column(db.Float, nullable=True)
    photo_path = db.Column(db.String(200), nullable=True)
    verified = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text, nullable=True)
    
    missing_person = db.relationship('MissingPerson', backref=db.backref('sightings', lazy=True))


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


def download_sample_images() -> None:
    """Download sample images for demo mode"""
    global demo_images
    demo_images = []
    
    for i, url in enumerate(SAMPLE_IMAGE_URLS):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Convert to OpenCV format
                nparr = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    # Resize to a reasonable size
                    img = cv2.resize(img, (640, 480))
                    demo_images.append(img)
        except Exception as e:
            print(f"Failed to download sample image {i}: {e}")
    
    # If no images downloaded, create a simple placeholder
    if not demo_images:
        placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.putText(placeholder, "No Camera Available", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(placeholder, "Demo Mode", (250, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        demo_images.append(placeholder)


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


def calculate_age_match_score(detected_age_range: str, person_age: int) -> float:
    """Calculate how well detected age range matches person's age"""
    if not detected_age_range or not person_age:
        return 0.0
    
    # Extract age range numbers
    age_range = detected_age_range.strip('()')
    if '-' not in age_range:
        return 0.0
    
    try:
        min_age, max_age = map(int, age_range.split('-'))
        if min_age <= person_age <= max_age:
            return 1.0
        elif abs(person_age - min_age) <= 5 or abs(person_age - max_age) <= 5:
            return 0.7  # Close match
        elif abs(person_age - min_age) <= 10 or abs(person_age - max_age) <= 10:
            return 0.4  # Somewhat close
        else:
            return 0.0
    except ValueError:
        return 0.0

def find_potential_matches(detected_faces: List[Dict]) -> List[Dict]:
    """Find potential missing person matches based on detected faces"""
    matches = []
    
    # Get all active missing persons
    missing_persons = MissingPerson.query.filter_by(status='active').all()
    
    for face in detected_faces:
        detected_gender = face.get('gender', '')
        detected_age_range = face.get('age_range', '')
        
        for person in missing_persons:
            # Calculate match score
            gender_match = 1.0 if detected_gender.lower() == person.gender.lower() else 0.0
            age_match = calculate_age_match_score(detected_age_range, person.age)
            
            # Combined confidence score
            overall_confidence = (gender_match * 0.6 + age_match * 0.4)
            
            if overall_confidence >= 0.5:  # Threshold for potential match
                matches.append({
                    'missing_person': person.to_dict(),
                    'detected_face': face,
                    'confidence': round(overall_confidence * 100, 1),
                    'match_reasons': {
                        'gender_match': gender_match > 0,
                        'age_match': age_match > 0,
                        'age_score': round(age_match * 100, 1)
                    }
                })
    
    # Sort by confidence score
    matches.sort(key=lambda x: x['confidence'], reverse=True)
    return matches

def analyze_faces_with_matching(image_bgr: np.ndarray, age_net, gender_net) -> Tuple[List[Dict], np.ndarray, List[Dict]]:
    """Enhanced face analysis that includes missing person matching"""
    results, annotated = analyze_faces(image_bgr, age_net, gender_net)
    
    # Find potential matches
    potential_matches = find_potential_matches(results)
    
    # Enhance annotation with match info
    if potential_matches:
        for match in potential_matches:
            face_box = match['detected_face']['box']
            x, y, w, h = face_box['x'], face_box['y'], face_box['w'], face_box['h']
            
            # Add red border for potential matches
            cv2.rectangle(annotated, (x-2, y-2), (x + w + 2, y + h + 2), (0, 0, 255), 3)
            
            # Add match indicator
            match_text = f"POTENTIAL MATCH: {match['missing_person']['name']} ({match['confidence']}%)"
            cv2.putText(annotated, match_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    return results, annotated, potential_matches


def image_to_base64_png(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode('.png', image_bgr)
    if not success:
        return ""
    b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def get_camera() -> Optional[cv2.VideoCapture]:
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = None
    return camera


def get_demo_frame() -> np.ndarray:
    """Get next frame from demo images"""
    global demo_frame_index, demo_images
    
    if not demo_images:
        download_sample_images()
    
    if not demo_images:
        # Fallback placeholder
        placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 64
        cv2.putText(placeholder, "DEMO MODE", (220, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return placeholder
    
    frame = demo_images[demo_frame_index % len(demo_images)].copy()
    
    # Add demo watermark
    cv2.putText(frame, "DEMO MODE", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Cycle through images slowly (change every ~3 seconds at 10 FPS)
    if demo_frame_index % 30 == 0:
        demo_frame_index = (demo_frame_index // 30 + 1) % len(demo_images) * 30
    else:
        demo_frame_index += 1
    
    return frame


def generate_frames():
    global _age_net, _gender_net
    
    if _age_net is None or _gender_net is None:
        _age_net, _gender_net = load_networks()
    
    cam = get_camera()
    use_demo = cam is None
    
    if use_demo:
        print("Using demo mode - no camera available")
    
    while True:
        if use_demo:
            frame = get_demo_frame()
            time.sleep(0.1)  # ~10 FPS for demo
        else:
            success, frame = cam.read()
            if not success:
                break
        
        # Analyze the frame with missing person matching
        results, annotated_frame, potential_matches = analyze_faces_with_matching(frame, _age_net, _gender_net)
        
        # Log potential matches (in production, this would trigger alerts)
        if potential_matches:
            print(f"ALERT: {len(potential_matches)} potential missing person matches detected!")
            for match in potential_matches:
                print(f"  - {match['missing_person']['name']} ({match['confidence']}% confidence)")
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# Lazy-load networks on first request to reduce startup time
_age_net = None
_gender_net = None

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/missing-persons")
def missing_persons_dashboard():
    persons = MissingPerson.query.filter_by(status='active').all()
    return render_template("missing_persons.html", persons=persons)

@app.route("/report-missing", methods=["GET", "POST"])
def report_missing():
    if request.method == "POST":
        # Handle form submission
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        description = request.form.get('description')
        last_seen_location = request.form.get('last_seen_location')
        contact_phone = request.form.get('contact_phone')
        contact_email = request.form.get('contact_email')
        
        # Create new missing person record
        missing_person = MissingPerson(
            name=name,
            age=int(age) if age else None,
            gender=gender,
            description=description,
            last_seen_location=last_seen_location,
            contact_phone=contact_phone,
            contact_email=contact_email
        )
        
        # Handle photo upload
        if 'photo' in request.files:
            photo = request.files['photo']
            if photo.filename:
                # Save photo
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"missing_{timestamp}_{photo.filename}"
                photo_path = os.path.join(MISSING_PERSONS_DIR, filename)
                photo.save(photo_path)
                missing_person.photo_path = filename
                
                # Analyze uploaded photo to extract age/gender for better matching
                try:
                    image = cv2.imread(photo_path)
                    if image is not None:
                        if _age_net is None or _gender_net is None:
                            load_networks()
                        results, _ = analyze_faces(image, _age_net, _gender_net)
                        if results:
                            # Use the first detected face for the person's profile
                            face_data = results[0]
                            missing_person.detected_age_range = face_data.get('age_range')
                            missing_person.detected_gender = face_data.get('gender')
                except Exception as e:
                    print(f"Error analyzing uploaded photo: {e}")
        
        db.session.add(missing_person)
        db.session.commit()
        
        flash('Missing person report submitted successfully!', 'success')
        return redirect(url_for('missing_persons_dashboard'))
    
    return render_template("report_missing.html")

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
        results, annotated, potential_matches = analyze_faces_with_matching(image, _age_net, _gender_net)
    except Exception as e:
        return jsonify({"error": f"Failed to analyze image: {str(e)}"}), 500

    annotated_b64 = image_to_base64_png(annotated)
    return jsonify({
        "faces": results,
        "annotated_image": annotated_b64,
        "potential_matches": potential_matches
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
    camera_available = cam is not None and cam.isOpened()
    demo_mode = not camera_available
    
    return jsonify({
        "camera_available": camera_available,
        "demo_mode": demo_mode,
        "message": "Demo mode with sample images" if demo_mode else "Physical camera detected"
    })

@app.route("/api/missing-persons")
def api_missing_persons():
    """API endpoint to get missing persons data"""
    persons = MissingPerson.query.filter_by(status='active').all()
    return jsonify([person.to_dict() for person in persons])

@app.route("/missing-person/<int:person_id>")
def missing_person_detail(person_id):
    person = MissingPerson.query.get_or_404(person_id)
    sightings = Sighting.query.filter_by(missing_person_id=person_id).order_by(Sighting.timestamp.desc()).all()
    return render_template("person_detail.html", person=person, sightings=sightings)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)