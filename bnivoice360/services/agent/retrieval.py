"""Read existing vectors lazily; never embed chat or invent missing vectors."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


@lru_cache(maxsize=2)
def _load_vectors(path: str, modified: int, size: int) -> tuple[tuple[str, ...], np.ndarray]:
    del modified, size  # Cache invalidates when the file changes.
    frame = pd.read_parquet(path, columns=["data_id", "embedding"])
    ids = tuple(frame["data_id"].astype(str))
    if not ids or len(set(ids)) != len(ids) or frame["data_id"].isna().any():
        raise ValueError("Embedding IDs must be unique and nonempty.")
    vectors = np.asarray(frame["embedding"].tolist(), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 384 or not np.isfinite(vectors).all():
        raise ValueError("Expected finite 384-dimensional voice embeddings.")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms == 0).any():
        raise ValueError("Embedding file contains zero vectors.")
    vectors /= norms[:, None]
    vectors.setflags(write=False)
    return ids, vectors


def load_voice_vectors(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Reuse validated, normalized vectors; file changes invalidate the small cache."""
    stat = path.stat()
    return _load_vectors(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def similar_voices(
    path: Path, frame: pd.DataFrame, anchor_id: str, threshold: float, limit: int
) -> pd.DataFrame:
    if not 0 <= threshold <= 1 or not 1 <= limit <= 50:
        raise ValueError("Similarity must be 0–1 and candidate limit 1–50.")
    ids, vectors = load_voice_vectors(path)
    index = {record_id: i for i, record_id in enumerate(ids)}
    if anchor_id not in index:
        raise ValueError("Keluhan awal belum memiliki embedding; pencarian tidak tersedia.")
    eligible = frame.loc[frame["escalation_status"].eq("Open")].copy()
    eligible["has_embedding"] = eligible["suara_nasabah_id"].astype(str).isin(index)
    missing = int((~eligible["has_embedding"]).sum())
    eligible = eligible.loc[eligible["has_embedding"]].copy()
    query = vectors[index[anchor_id]]
    eligible["similarity"] = [
        float(np.clip(vectors[index[str(record_id)]] @ query, -1, 1))
        for record_id in eligible["suara_nasabah_id"]
    ]
    eligible = (
        eligible.loc[
            eligible["similarity"].ge(threshold) & eligible["suara_nasabah_id"].ne(anchor_id)
        ]
        .sort_values(["similarity", "suara_nasabah_id"], ascending=[False, True])
        .head(limit)
    )
    eligible.attrs["missing_embeddings"] = missing
    return eligible
