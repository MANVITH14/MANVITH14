import io
from typing import List, Dict, Any

import numpy as np
import streamlit as st
from PIL import Image

from app.embedder import get_face_embeddings, select_best_face_embedding
from app.indexer import FaceIndex
from app.utils import pil_to_bgr, crop_bgr, bgr_to_pil


st.set_page_config(page_title="Missing Person Finder", layout="wide")

if "face_index" not in st.session_state:
    st.session_state.face_index = None  # type: FaceIndex | None
    st.session_state.gallery_faces = []  # list of dicts with image and meta


st.title("Missing Person Finder (Mini Project)")

st.markdown(
    "Upload gallery photos to build an index, then upload one or more photos of the missing person to search for matches."
)

# --- 1) Build Gallery Index ---
with st.expander("1) Build Gallery Index", expanded=True):
    gallery_files = st.file_uploader(
        "Upload gallery images (png/jpg)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )
    build_col1, build_col2 = st.columns([1, 2])

    with build_col1:
        min_det_score = st.slider("Min face detection score", 0.0, 1.0, 0.45, 0.01)
        top_k_per_image = st.number_input("Max faces per image", 1, 10, 3)
        build_btn = st.button("Build/Update Index")

    with build_col2:
        st.write("Uploaded:", len(gallery_files) if gallery_files else 0, "files")

    if build_btn and gallery_files:
        embeddings: List[np.ndarray] = []
        metadatas: List[Dict[str, Any]] = []
        gallery_faces: List[Dict[str, Any]] = []

        prog = st.progress(0.0)
        for idx, f in enumerate(gallery_files):
            try:
                image = Image.open(io.BytesIO(f.getvalue()))
                image_bgr = pil_to_bgr(image)
            except Exception as e:
                st.warning(f"Failed to read {f.name}: {e}")
                continue
            faces = get_face_embeddings(image_bgr)
            faces = [t for t in faces if t[2] >= min_det_score]
            faces.sort(key=lambda t: t[2], reverse=True)
            faces = faces[: int(top_k_per_image)]
            for face_idx, (emb, bbox, score) in enumerate(faces):
                embeddings.append(emb)
                metadatas.append({
                    "source_name": f.name,
                    "face_index": face_idx,
                    "det_score": float(score),
                })
                face_crop = crop_bgr(image_bgr, bbox)
                gallery_faces.append({
                    "image": bgr_to_pil(face_crop),
                    "meta": metadatas[-1],
                })
            prog.progress((idx + 1) / len(gallery_files))

        if not embeddings:
            st.error("No faces detected in uploaded gallery images.")
        else:
            embeddings_np = np.stack(embeddings, axis=0)
            if st.session_state.face_index is None:
                st.session_state.face_index = FaceIndex(dim=embeddings_np.shape[1])
            st.session_state.face_index.add(embeddings_np, metadatas)
            st.session_state.gallery_faces.extend(gallery_faces)
            st.success(f"Indexed {len(embeddings)} faces from {len(gallery_files)} images.")

# --- 2) Search Missing Person ---
with st.expander("2) Search Missing Person", expanded=True):
    query_files = st.file_uploader(
        "Upload 1-3 photos of the missing person (png/jpg)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="query_uploader"
    )
    col_q1, col_q2 = st.columns([1, 2])
    with col_q1:
        top_k = st.number_input("Top K matches", 1, 50, 10)
        search_btn = st.button("Search")
    with col_q2:
        if st.session_state.face_index is None:
            st.info("Please build the gallery index first.")

    if search_btn:
        if st.session_state.face_index is None:
            st.error("No index. Build the gallery index first.")
        elif not query_files:
            st.error("Please upload at least one query image.")
        else:
            query_embeds: List[np.ndarray] = []
            for f in query_files[:3]:
                try:
                    image = Image.open(io.BytesIO(f.getvalue()))
                    image_bgr = pil_to_bgr(image)
                except Exception as e:
                    st.warning(f"Failed to read {f.name}: {e}")
                    continue
                emb = select_best_face_embedding(image_bgr)
                if emb is not None:
                    query_embeds.append(emb)
                else:
                    st.warning(f"No face detected in {f.name}")

            if not query_embeds:
                st.error("No face detected in query images.")
            else:
                query_stack = np.stack(query_embeds, axis=0)
                query_mean = query_stack.mean(axis=0)
                scores, indices, metas = st.session_state.face_index.search(query_mean, k=int(top_k))
                st.subheader("Matches")
                grid_cols = st.columns(5)
                for rank, (score, idx_meta) in enumerate(zip(scores[0], metas[0])):
                    meta = idx_meta
                    face_item = None
                    # Find matching face in stored gallery_faces by metadata identity
                    for item in st.session_state.gallery_faces:
                        m = item["meta"]
                        if (
                            m.get("source_name") == meta.get("source_name")
                            and m.get("face_index") == meta.get("face_index")
                            and abs(float(m.get("det_score", 0.0)) - float(meta.get("det_score", 0.0))) < 1e-6
                        ):
                            face_item = item
                            break
                    col = grid_cols[rank % len(grid_cols)]
                    with col:
                        if face_item is not None:
                            st.image(face_item["image"], caption=f"{meta.get('source_name')}\nscore={score:.3f}")
                        else:
                            st.write(f"score={score:.3f}")

# --- Sidebar: session info ---
with st.sidebar:
    st.markdown("### Session")
    num_faces = len(st.session_state.gallery_faces)
    st.write(f"Indexed faces: {num_faces}")
    if st.button("Clear session"):
        st.session_state.face_index = None
        st.session_state.gallery_faces = []
        st.experimental_rerun()