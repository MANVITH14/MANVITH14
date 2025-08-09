from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np


@dataclass
class IndexedFace:
    embedding: np.ndarray
    metadata: Dict[str, Any]


class FaceIndex:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._embeddings: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Embeddings must have shape [N,{self.dim}]")
        if len(metadatas) != embeddings.shape[0]:
            raise ValueError("metadatas length must match number of embeddings")
        embeddings = embeddings.astype(np.float32)
        embeddings = self._normalize(embeddings)
        self._embeddings.extend(list(embeddings))
        self._metadata.extend(metadatas)

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[List[Dict[str, Any]]]]:
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[None, :]
        query_embeddings = query_embeddings.astype(np.float32)
        query_embeddings = self._normalize(query_embeddings)
        if not self._embeddings:
            return (
                np.empty((query_embeddings.shape[0], 0), dtype=np.float32),
                np.empty((query_embeddings.shape[0], 0), dtype=int),
                [[] for _ in range(query_embeddings.shape[0])],
            )
        gallery = np.stack(self._embeddings, axis=0)  # [G, D]
        # Cosine similarity since both sets are normalized
        sims = query_embeddings @ gallery.T  # [Q, G]
        k = min(k, gallery.shape[0])
        # Argpartition for top-k per row
        idx_part = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
        # Sort those top-k indices by similarity descending
        row_indices = np.arange(sims.shape[0])[:, None]
        row_sims = np.take_along_axis(sims, idx_part, axis=1)
        order = np.argsort(-row_sims, axis=1)
        top_idx = np.take_along_axis(idx_part, order, axis=1)
        top_scores = np.take_along_axis(sims, top_idx, axis=1)
        batch_metadata: List[List[Dict[str, Any]]] = []
        for row in top_idx:
            row_meta: List[Dict[str, Any]] = []
            for idx in row:
                if idx < 0 or idx >= len(self._metadata):
                    row_meta.append({})
                else:
                    row_meta.append(self._metadata[int(idx)])
            batch_metadata.append(row_meta)
        return top_scores.astype(np.float32), top_idx.astype(int), batch_metadata

    def save(self, path: str) -> None:
        np.savez_compressed(
            path + ".npz",
            embeddings=np.stack(self._embeddings, axis=0) if self._embeddings else np.zeros((0, self.dim), dtype=np.float32),
            metadata=np.array(self._metadata, dtype=object),
        )

    @classmethod
    def load(cls, path: str) -> "FaceIndex":
        npz = np.load(path + ".npz", allow_pickle=True)
        obj = cls(dim=npz["embeddings"].shape[1] if npz["embeddings"].size else 512)
        obj._embeddings = list(npz["embeddings"]) if npz["embeddings"].size else []
        obj._metadata = list(npz["metadata"]) if npz["metadata"].size else []
        return obj