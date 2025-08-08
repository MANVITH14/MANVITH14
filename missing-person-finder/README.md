# Missing Person Finder (Mini Project)

A minimal face search app to help find a missing person in a gallery of photos using AI. It detects faces, builds an embedding index with InsightFace, and returns the most similar faces to the query photo.

## Features
- Face detection and embeddings via InsightFace
- Fast similarity search with cosine similarity (NumPy)
- Streamlit UI: upload gallery photos, build index, and search by query image(s)

## Setup
1. Create and activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run
```bash
streamlit run app.py
```
Then open the URL shown in the terminal.

## Usage
- In the "1) Build Gallery Index" section, upload multiple images that may contain the missing person. The app detects faces and builds an index.
- In the "2) Search Missing Person" section, upload one or more photos of the missing person. The app computes an averaged face embedding and returns the top matches from the gallery.

## Notes
- All computation is local. No data leaves your machine.
- For best results, use clear, frontal face photos.
- If no face is detected in a photo, it will be skipped.