from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

from src.features.feature_builder import TABULAR_FEATURE_COLS, get_model_feature_cols

logger = logging.getLogger(__name__)

MODEL_PATH = Path("data/risk_model.pkl")
SCALER_PATH = Path("data/risk_scaler.pkl")
COLS_PATH = Path("data/risk_feature_cols.pkl")


class RiskScorer:
    def __init__(self) -> None:
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        )
        self.scaler = MinMaxScaler()
        self._trained = False
        self._feature_cols: list[str] = TABULAR_FEATURE_COLS

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def _make_label(self, df: pd.DataFrame) -> np.ndarray:
        score = (
            0.30 * df["cvss_score"] / 10
            + 0.25 * df["exploitability"] / 10
            + 0.20 * df["asset_criticality"] / 10
            + 0.15 * df["asset_exposure"]
            + 0.10 * df["high_risk_cwe"].astype(float)
        )
        return score.clip(0, 1).to_numpy()

    def train(self, df: pd.DataFrame) -> None:
        self._feature_cols = get_model_feature_cols(df)
        X = df[self._feature_cols].fillna(0).to_numpy()
        y = self._make_label(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._trained = True
        logger.info(
            "RiskScorer trained on %d samples with %d features (%d embedding dims)",
            len(df),
            len(self._feature_cols),
            len(self._feature_cols) - len(TABULAR_FEATURE_COLS),
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        X = df[self._feature_cols].fillna(0).to_numpy()
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).clip(0, 1)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        cols_path: Path = COLS_PATH,
    ) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self._feature_cols, cols_path)
        logger.info("Model saved → %s", model_path)

    def load(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        cols_path: Path = COLS_PATH,
    ) -> bool:
        if model_path.exists() and scaler_path.exists() and cols_path.exists():
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self._feature_cols = joblib.load(cols_path)
            self._trained = True
            logger.info("Model loaded from %s (%d features)", model_path, len(self._feature_cols))
            return True
        return False

    # ------------------------------------------------------------------ #
    # Insights
    # ------------------------------------------------------------------ #

    @property
    def feature_importance(self) -> dict[str, float]:
        if not self._trained:
            return {}
        raw = dict(zip(self._feature_cols, self.model.feature_importances_))
        # Aggregate all emb_* dims into one "vulberta_embeddings" entry for display
        emb_total = sum(v for k, v in raw.items() if k.startswith("emb_"))
        tabular = {k: v for k, v in raw.items() if not k.startswith("emb_")}
        if emb_total:
            tabular["vulberta_embeddings"] = emb_total
        return tabular
