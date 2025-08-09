import threading
from typing import List, Tuple, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis


_face_app: Optional[FaceAnalysis] = None
_face_app_lock = threading.Lock()


def _init_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is None:
        with _face_app_lock:
            if _face_app is None:
                app = FaceAnalysis(name="buffalo_l")
                # Use CPU by default for compatibility
                app.prepare(ctx_id=-1, det_size=(640, 640))
                _face_app = app
    return _face_app  # type: ignore


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # OpenCV loads as BGR by default
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_face_embeddings(image_bgr: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
    """
    Returns list of tuples: (embedding[512], bbox(x1,y1,x2,y2), detection_score)
    Embeddings are L2-normalized float32 vectors.
    """
    app = _init_face_app()
    rgb = ensure_rgb(image_bgr)
    faces = app.get(rgb)

    results: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]] = []
    for f in faces:
        emb = f.normed_embedding.astype(np.float32)
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        results.append((emb, (x1, y1, x2, y2), float(f.det_score)))
    return results


def select_best_face_embedding(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Returns the highest-confidence face embedding in the image, or None if none found."""
    faces = get_face_embeddings(image_bgr)
    if not faces:
        return None
    faces.sort(key=lambda t: t[2], reverse=True)
    return faces[0][0]