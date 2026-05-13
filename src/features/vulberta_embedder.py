from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/codebert-base"
EMB_DIM = 32          # PCA output dimensions
CACHE_PATH = Path("data/vulberta_emb_cache.pkl")
PCA_PATH = Path("data/vulberta_pca.pkl")
_MAX_LEN = 256        # VulBERTa context window


class VulBertaEmbedder:
    """
    Wraps mrm8488/vulBERTa-mlm to produce fixed-size CVE description embeddings.

    Pipeline:
        CVE description text
          → VulBERTa encoder (mean-pool last hidden state) → 768-dim
          → PCA (fit on first batch, cached) → EMB_DIM-dim
    Embeddings are cached on disk by SHA-256 of the input text.
    """

    def __init__(
        self,
        cache_path: Path = CACHE_PATH,
        pca_path: Path = PCA_PATH,
        device: str | None = None,
    ) -> None:
        self._cache_path = cache_path
        self._pca_path = pca_path
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None
        self._pca: PCA | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._load_cache()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return (N, EMB_DIM) float32 array, one row per text."""
        self._ensure_model()

        keys = [_sha(t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]

        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            raw = self._embed_raw(missing_texts)          # (M, 768)
            projected = self._project(raw)                # (M, EMB_DIM)
            for i, idx in enumerate(missing_idx):
                self._cache[keys[idx]] = projected[i]
            self._save_cache()

        result = np.stack([self._cache[k] for k in keys])
        return result.astype(np.float32)

    @property
    def dim(self) -> int:
        return EMB_DIM

    @property
    def col_names(self) -> list[str]:
        return [f"emb_{i}" for i in range(EMB_DIM)]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading VulBERTa from HuggingFace: %s (device=%s)", MODEL_ID, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = AutoModel.from_pretrained(MODEL_ID)
        self._model.eval()
        self._model.to(self._device)
        logger.info("VulBERTa loaded.")

    @torch.no_grad()
    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        """Encode texts → mean-pooled last hidden state → (N, 768)."""
        all_embs: list[np.ndarray] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=_MAX_LEN,
                return_tensors="pt",
            ).to(self._device)
            out = self._model(**enc)
            # mean pool over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_embs.append(pooled.cpu().numpy())
        return np.vstack(all_embs)

    def _project(self, raw: np.ndarray) -> np.ndarray:
        """PCA-reduce (N, 768) → (N, EMB_DIM). Fits PCA on first call."""
        if self._pca is None:
            if self._pca_path.exists():
                self._pca = joblib.load(self._pca_path)
                logger.info("PCA loaded from %s", self._pca_path)
            else:
                n_components = min(EMB_DIM, raw.shape[0], raw.shape[1])
                self._pca = PCA(n_components=n_components, random_state=42)
                self._pca.fit(raw)
                self._pca_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self._pca, self._pca_path)
                logger.info("PCA fitted and saved → %s", self._pca_path)

        projected = self._pca.transform(raw)
        # pad to EMB_DIM if n_components < EMB_DIM (small batch edge case)
        if projected.shape[1] < EMB_DIM:
            pad = np.zeros((projected.shape[0], EMB_DIM - projected.shape[1]), dtype=np.float32)
            projected = np.hstack([projected, pad])
        return projected

    def _load_cache(self) -> None:
        if self._cache_path.exists():
            self._cache = joblib.load(self._cache_path)
            logger.debug("Embedding cache loaded: %d entries", len(self._cache))

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._cache, self._cache_path)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
