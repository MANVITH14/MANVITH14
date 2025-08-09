import os
import base64
import time
from datetime import datetime
from typing import List, Dict

from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import cv2

# Initialize Flask app with database
app = Flask(__name__)
app.config['SECRET_KEY'] = 'demo-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///missing_persons_demo.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create directories
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MISSING_PERSONS_DIR = os.path.join(APP_ROOT, "missing_persons")
os.makedirs(MISSING_PERSONS_DIR, exist_ok=True)

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
    status = db.Column(db.String(20), default='active')
    
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
            'status': self.status
        }

class Sighting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    missing_person_id = db.Column(db.Integer, db.ForeignKey('missing_person.id'), nullable=False)
    location = db.Column(db.String(200), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    confidence_score = db.Column(db.Float, nullable=True)
    verified = db.Column(db.Boolean, default=False)
    
    missing_person = db.relationship('MissingPerson', backref=db.backref('sightings', lazy=True))

# Mock AI functions for demo
def mock_detect_faces(image):
    """Mock face detection - returns demo data"""
    height, width = image.shape[:2]
    return [(100, 100, 150, 150)]  # Mock face coordinates

def mock_analyze_faces(image):
    """Mock face analysis - returns demo data"""
    faces = mock_detect_faces(image)
    results = []
    annotated = image.copy()
    
    for i, (x, y, w, h) in enumerate(faces):
        # Draw mock detection box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, "Demo Detection", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Mock results
        results.append({
            "box": {"x": x, "y": y, "w": w, "h": h},
            "gender": "Male" if i % 2 == 0 else "Female",
            "gender_confidence": 0.85,
            "age_range": "(25-32)",
            "age_confidence": 0.78
        })
    
    return results, annotated

def image_to_base64_png(image):
    success, buffer = cv2.imencode('.png', image)
    if not success:
        return ""
    b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def read_image_from_request(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

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
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        description = request.form.get('description')
        last_seen_location = request.form.get('last_seen_location')
        contact_phone = request.form.get('contact_phone')
        contact_email = request.form.get('contact_email')
        
        missing_person = MissingPerson(
            name=name,
            age=int(age) if age else None,
            gender=gender,
            description=description,
            last_seen_location=last_seen_location,
            contact_phone=contact_phone,
            contact_email=contact_email
        )
        
        if 'photo' in request.files:
            photo = request.files['photo']
            if photo.filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"missing_{timestamp}_{photo.filename}"
                photo_path = os.path.join(MISSING_PERSONS_DIR, filename)
                photo.save(photo_path)
                missing_person.photo_path = filename
        
        db.session.add(missing_person)
        db.session.commit()
        
        flash('Missing person report submitted successfully!', 'success')
        return redirect(url_for('missing_persons_dashboard'))
    
    return render_template("report_missing.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        image = read_image_from_request(file)
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        # Use mock analysis for demo
        results, annotated = mock_analyze_faces(image)
        
        # Mock potential matches
        persons = MissingPerson.query.filter_by(status='active').all()
        potential_matches = []
        
        for person in persons[:2]:  # Show max 2 demo matches
            potential_matches.append({
                'missing_person': person.to_dict(),
                'detected_face': results[0] if results else {},
                'confidence': 75.0,
                'match_reasons': {
                    'gender_match': True,
                    'age_match': True,
                    'age_score': 80.0
                }
            })
        
        annotated_b64 = image_to_base64_png(annotated)
        return jsonify({
            "faces": results,
            "annotated_image": annotated_b64,
            "potential_matches": potential_matches
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/camera")
def camera_page():
    return render_template("camera.html")

@app.route("/video_feed")
def video_feed():
    def generate():
        # Mock video feed for demo
        while True:
            # Create a simple demo frame
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
            cv2.putText(frame, "DEMO MODE - AI Missing Person Finder", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Live detection would appear here", (100, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (50, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.1)  # 10 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/camera_status")
def camera_status():
    return jsonify({
        "camera_available": False,
        "demo_mode": True,
        "message": "Demo mode - simulated video feed"
    })

@app.route("/api/missing-persons")
def api_missing_persons():
    persons = MissingPerson.query.filter_by(status='active').all()
    return jsonify([person.to_dict() for person in persons])

@app.route("/missing-person/<int:person_id>")
def missing_person_detail(person_id):
    person = MissingPerson.query.get_or_404(person_id)
    sightings = Sighting.query.filter_by(missing_person_id=person_id).order_by(Sighting.timestamp.desc()).all()
    return render_template("person_detail.html", person=person, sightings=sightings)

@app.route("/api/missing-persons/<int:person_id>/mark-found", methods=["POST"])
def mark_person_found(person_id):
    try:
        person = MissingPerson.query.get_or_404(person_id)
        person.status = 'found'
        db.session.commit()
        return jsonify({"success": True, "message": "Person marked as found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/healthz")
def healthz():
    return {"status": "ok", "mode": "demo"}

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        
        # Add some demo data if database is empty
        if MissingPerson.query.count() == 0:
            demo_persons = [
                MissingPerson(
                    name="Demo Person A", 
                    age=25, 
                    gender="Female", 
                    description="Demo missing person for testing",
                    last_seen_location="Demo City Center"
                ),
                MissingPerson(
                    name="Demo Person B", 
                    age=35, 
                    gender="Male", 
                    description="Another demo person",
                    last_seen_location="Demo Park"
                )
            ]
            for person in demo_persons:
                db.session.add(person)
            db.session.commit()
            print("Added demo missing persons data")
    
    print("🔍 AI Missing Person Finder (Demo Mode)")
    print("📱 Access at: http://localhost:5000")
    print("✨ Features: Face detection, Missing person database, Real-time alerts")
    print("⚡ Demo mode: Fast startup, simulated AI detection")
    
    app.run(host="0.0.0.0", port=5000, debug=True)