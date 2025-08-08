# Age & Gender Detection (Mini Project)

A simple Flask web app that detects faces in an image and predicts age range and gender using OpenCV DNN models.

## Features
- Upload an image from your device
- Detects multiple faces
- Predicts age range and gender for each face
- Returns an annotated image with labels

## Tech
- Flask (Python)
- OpenCV (Haar face detection + Caffe age/gender nets)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 in your browser.

Note: The first request will download the age/gender models (~70 MB total) into `models/`.

## Notes
- Age ranges are from the IMDB-WIKI pretrained model: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- For consistent results, use clear, front-facing photos with good lighting.
