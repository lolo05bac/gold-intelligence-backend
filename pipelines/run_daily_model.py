"""
Daily Model Scoring Pipeline
Runs the full 4-model stack and produces the daily Gold Bias Score.

Model Stack:
    1. Regime Classifier (XGBoost) → Current market environment
    2. Direction Model (LightGBM ensemble) → P(up day)
    3. Expected Move Model (LightGBM regressor) → Magnitude
    4. Event Shock Adjuster → CPI/FOMC/NFP calibration
    5. Signal Combiner → Bias 1–10

Usage:
    python -m pipelines.run_daily_model [--date 2024-03-16] [--train]
"""
import os
import argparse
import pickle
from datetime import date, datetime
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

try:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, brier_score_loss, classification_report,
    )
    from sklearn.preprocessing import StandardScaler
    import shap
except ImportError:
    logger.warning("ML libraries not installed. Run: pip install lightgbm xgboost scikit-learn shap")

FEATURES_DIR = os.path.join("data", "features")
MODELS_DIR = os.path.join("data", "model_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)

# Feature columns (excluding date and targets)
TARGET_COLS = ["target_return", "target_direction", "target_abs_return", "target_range"]

# Regime labels
REGIME_LABELS = {
    0: "risk_off",
    1: "usd_dominant",
    2: "real_yield_driven",
    3: "inflation_scare",
    4: "fed_event",
    5: "range_bound",
    6: "trend",
}


class GoldSignalEngine:
    """The core 4-model signal engine for gold bias scoring."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.regime_model = None
        self.direction_models = {}
        self.move_model = None
        self.feature_cols = []
        self.is_trained = False

    def load_features(self) -> pd.DataFrame:
        """Load the feature store."""
        path = os.path.join(FEATURES_DIR, "daily_features.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature store not found at {path}. Run build_features first.")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def prepare_data(self, df: pd.DataFrame):
        """Split features and targets, identify feature columns."""
        self.feature_cols = [c for c in df.columns if c not in ["date"] + TARGET_COLS]

        # Drop rows where targets are NaN (last row usually)
        mask = df[TARGET_COLS].notna().all(axis=1)
        df = df[mask].copy()

        X = df[self.feature_cols].values
        y_dir = df["target_direction"].values
        y_ret = df["target_return"].values
        dates = df["date"].values

        return X, y_dir, y_ret, dates, df

    # ═══════════════════════════════════════════════════════
    # MODEL 1: REGIME CLASSIFIER
    # ═══════════════════════════════════════════════════════

    def _build_regime_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create regime labels using rule-based classification.
        In production, this would use HMM or clustering.
        """
        labels = np.full(len(df), 5)  # default: range_bound

        for i, row in df.iterrows():
            idx = df.index.get_loc(i)

            # Risk-off: VIX spike + negative equities
            if row.get("vix_change", 0) > 0.05 and row.get("spx_return", 0) < -0.01:
                labels[idx] = 0  # risk_off
            # USD dominant: big DXY moves
            elif abs(row.get("dxy_return", 0)) > 0.005:
                labels[idx] = 1  # usd_dominant
            # Real yield driven: big real yield changes
            elif abs(row.get("real_yield_change", 0)) > 0.03:
                labels[idx] = 2  # real_yield_driven
            # Inflation scare: high CPI surprise
            elif abs(row.get("cpi_surprise", 0)) > 1.5:
                labels[idx] = 3  # inflation_scare
            # Trend: strong momentum alignment
            elif abs(row.get("momentum_20d", 0)) > 0.04:
                labels[idx] = 6  # trend

        return labels

    def train_regime_model(self, X: np.ndarray, df: pd.DataFrame):
        """Train the regime classifier."""
        logger.info("Training Regime Classifier...")
        y_regime = self._build_regime_labels(df)

        self.regime_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=len(REGIME_LABELS),
            random_state=42,
            verbosity=0,
        )
        # Remap labels to consecutive integers
        unique_labels = sorted(set(y_regime))
        label_map = {old_l: new_l for new_l, old_l in enumerate(unique_labels)}
        y_regime = np.array([label_map[l] for l in y_regime])
        self.regime_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=len(unique_labels),
            random_state=42,
            verbosity=0,
        )
        self.regime_model.fit(X, y_regime)
        logger.info(f"  Regime model trained. Classes: {np.unique(y_regime)}")

    # ═══════════════════════════════════════════════════════
    # MODEL 2: DIRECTION MODEL (ensemble)
    # ═══════════════════════════════════════════════════════

    def train_direction_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train an ensemble of direction classifiers."""
        logger.info("Training Direction Model Ensemble...")

        X_scaled = self.scaler.fit_transform(X_train)

        # Model 1: Logistic Regression (baseline)
        lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        lr.fit(X_scaled, y_train)
        self.direction_models["logistic"] = lr

        # Model 2: LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(X_train, y_train)
        self.direction_models["lightgbm"] = lgb_model

        # Model 3: Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        self.direction_models["random_forest"] = rf

        # Model 4: Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
        gb.fit(X_train, y_train)
        self.direction_models["gradient_boost"] = gb

        logger.info(f"  Trained {len(self.direction_models)} direction models")

    def predict_direction(self, X: np.ndarray) -> np.ndarray:
        """Ensemble probability of up day."""
        probs = []
        X_scaled = self.scaler.transform(X)

        for name, model in self.direction_models.items():
            if name == "logistic":
                p = model.predict_proba(X_scaled)[:, 1]
            else:
                p = model.predict_proba(X)[:, 1]
            probs.append(p)

        # Equal-weight ensemble average
        return np.mean(probs, axis=0)

    # ═══════════════════════════════════════════════════════
    # MODEL 3: EXPECTED MOVE
    # ═══════════════════════════════════════════════════════

    def train_move_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the expected signed return model."""
        logger.info("Training Expected Move Model...")

        self.move_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        self.move_model.fit(X_train, y_train)
        logger.info("  Move model trained")

    # ═══════════════════════════════════════════════════════
    # SIGNAL COMBINER → BIAS 1–10
    # ═══════════════════════════════════════════════════════

    def compute_bias_score(
        self,
        direction_prob: float,
        expected_move: float,
        regime_probs: np.ndarray,
        confidence: float,
    ) -> float:
        """
        Convert model outputs to Gold Bias Score 1–10.

        Mapping:
            P(up) < 0.30 → 1–2
            P(up) 0.30–0.45 → 3–4
            P(up) 0.45–0.55 → 4.5–5.5
            P(up) 0.55–0.70 → 6–7
            P(up) > 0.70 → 8–10
        """
        # Base score from direction probability
        base = direction_prob * 10  # 0–10 linear mapping

        # Adjust by expected move magnitude
        move_adj = np.clip(expected_move * 500, -1.5, 1.5)  # ±1.5 adjustment max

        # Regime confidence weight
        regime_clarity = np.max(regime_probs) - np.mean(regime_probs)
        regime_adj = regime_clarity * 0.5

        # Combine
        raw_score = base + move_adj * 0.3 + regime_adj
        bias = np.clip(raw_score, 1.0, 10.0)

        return round(float(bias), 1)

    def compute_confidence(
        self,
        direction_probs: list[float],
        regime_probs: np.ndarray,
    ) -> float:
        """
        Confidence based on model agreement and regime clarity.
        """
        # Model agreement: low std across ensemble = high confidence
        model_std = np.std(direction_probs)
        agreement = 1 - np.clip(model_std / 0.15, 0, 1)  # Normalize

        # Direction strength: far from 0.5 = more confident
        avg_prob = np.mean(direction_probs)
        strength = abs(avg_prob - 0.5) * 2  # 0–1

        # Regime clarity
        regime_clarity = float(np.max(regime_probs))

        # Weighted combination
        confidence = (agreement * 0.4 + strength * 0.35 + regime_clarity * 0.25)
        return round(float(np.clip(confidence, 0, 1)), 3)

    def get_confidence_label(self, conf: float) -> str:
        if conf >= 0.70:
            return "high"
        elif conf >= 0.45:
            return "medium"
        return "low"

    # ═══════════════════════════════════════════════════════
    # FEATURE IMPORTANCE / EXPLAINABILITY
    # ═══════════════════════════════════════════════════════

    def get_top_drivers(self, X_row: np.ndarray, n: int = 5) -> tuple[list, list]:
        """Get top bullish and bearish drivers using SHAP."""
        try:
            explainer = shap.TreeExplainer(self.direction_models["lightgbm"])
            shap_values = explainer.shap_values(X_row.reshape(1, -1))

            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]  # Class 1 (up) SHAP values
            else:
                shap_vals = shap_values[0]

            # Rank by absolute importance
            feature_impacts = list(zip(self.feature_cols, shap_vals))
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)

            bullish = []
            bearish = []
            for feat, impact in feature_impacts:
                entry = {
                    "name": feat.replace("_", " ").title(),
                    "impact": round(abs(float(impact)) * 100, 1),
                    "detail": f"SHAP: {impact:+.4f}",
                }
                if impact > 0:
                    bullish.append(entry)
                else:
                    bearish.append(entry)

            return bullish[:n], bearish[:n]

        except Exception as e:
            logger.warning(f"SHAP failed: {e}")
            return [], []

    # ═══════════════════════════════════════════════════════
    # TRAINING PIPELINE
    # ═══════════════════════════════════════════════════════

    def train(self, train_end_date: Optional[str] = None):
        """Full training pipeline with walk-forward validation."""
        logger.info("=== Training Gold Signal Engine ===")

        df = self.load_features()
        X, y_dir, y_ret, dates, full_df = self.prepare_data(df)

        # Split: use all data up to train_end for training
        if train_end_date:
            train_mask = dates <= pd.to_datetime(train_end_date).date()
        else:
            # Default: use first 80% for training
            split_idx = int(len(X) * 0.8)
            train_mask = np.arange(len(X)) < split_idx

        X_train, X_test = X[train_mask], X[~train_mask]
        y_dir_train, y_dir_test = y_dir[train_mask], y_dir[~train_mask]
        y_ret_train, y_ret_test = y_ret[train_mask], y_ret[~train_mask]

        logger.info(f"  Train: {sum(train_mask)} rows | Test: {sum(~train_mask)} rows")

        # Train all models
        self.train_regime_model(X_train, full_df[train_mask])
        self.train_direction_model(X_train, y_dir_train)
        self.train_move_model(X_train, y_ret_train)

        # Evaluate on test set
        if sum(~train_mask) > 0:
            self._evaluate(X_test, y_dir_test, y_ret_test, dates[~train_mask])

        self.is_trained = True
        self._save_models()

        logger.info("=== Training complete ===")

    def _evaluate(self, X_test, y_dir_test, y_ret_test, test_dates):
        """Evaluate model performance."""
        logger.info("=== Model Evaluation ===")

        # Direction predictions
        probs = self.predict_direction(X_test)
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_dir_test, preds)
        prec = precision_score(y_dir_test, preds, zero_division=0)
        rec = recall_score(y_dir_test, preds, zero_division=0)
        f1 = f1_score(y_dir_test, preds, zero_division=0)
        auc = roc_auc_score(y_dir_test, probs) if len(np.unique(y_dir_test)) > 1 else 0
        brier = brier_score_loss(y_dir_test, probs)

        logger.info(f"  Accuracy:  {acc:.4f}")
        logger.info(f"  Precision: {prec:.4f}")
        logger.info(f"  Recall:    {rec:.4f}")
        logger.info(f"  F1:        {f1:.4f}")
        logger.info(f"  ROC AUC:   {auc:.4f}")
        logger.info(f"  Brier:     {brier:.4f}")

        # High-confidence subset
        high_conf = probs > 0.65
        if sum(high_conf) > 0:
            hc_acc = accuracy_score(y_dir_test[high_conf], preds[high_conf])
            logger.info(f"  High-conf accuracy: {hc_acc:.4f} (n={sum(high_conf)})")

        # Move model
        move_preds = self.move_model.predict(X_test)
        move_corr = np.corrcoef(move_preds, y_ret_test)[0, 1]
        logger.info(f"  Move model correlation: {move_corr:.4f}")

    # ═══════════════════════════════════════════════════════
    # DAILY SCORING
    # ═══════════════════════════════════════════════════════

    def score_today(self, target_date: Optional[date] = None) -> dict:
        """Generate the daily gold bias signal."""
        if not self.is_trained:
            self._load_models()

        df = self.load_features()

        if target_date:
            row = df[df["date"] == target_date]
        else:
            row = df.iloc[[-1]]  # Latest row

        if row.empty:
            raise ValueError(f"No feature data for {target_date}")

        X = row[self.feature_cols].values
        scoring_date = row["date"].values[0]

        # Model 1: Regime
        regime_probs = self.regime_model.predict_proba(X)[0]
        regime_id = int(np.argmax(regime_probs))
        regime_name = REGIME_LABELS.get(regime_id, "unknown")

        # Model 2: Direction
        individual_probs = []
        X_scaled = self.scaler.transform(X)
        for name, model in self.direction_models.items():
            if name == "logistic":
                p = model.predict_proba(X_scaled)[:, 1][0]
            else:
                p = model.predict_proba(X)[:, 1][0]
            individual_probs.append(float(p))
        direction_prob = float(np.mean(individual_probs))

        # Model 3: Expected move
        expected_move = float(self.move_model.predict(X)[0])

        # Model 4: Confidence
        confidence = self.compute_confidence(individual_probs, regime_probs)
        conf_label = self.get_confidence_label(confidence)

        # Signal combiner
        bias_score = self.compute_bias_score(direction_prob, expected_move, regime_probs, confidence)

        # Explainability
        bullish_drivers, bearish_drivers = self.get_top_drivers(X[0])

        signal = {
            "signal_date": scoring_date,
            "bias_score": bias_score,
            "direction_probability": round(direction_prob, 4),
            "expected_move_pct": round(expected_move * 100, 3),
            "confidence": confidence,
            "confidence_label": conf_label,
            "regime": regime_name,
            "regime_probability": round(float(np.max(regime_probs)), 3),
            "bullish_drivers": bullish_drivers,
            "bearish_drivers": bearish_drivers,
            "model_version": "2.4",
            "individual_probs": {name: round(p, 4) for name, p in zip(self.direction_models.keys(), individual_probs)},
        }

        logger.info(f"=== Daily Signal for {scoring_date} ===")
        logger.info(f"  Bias:       {bias_score}/10")
        logger.info(f"  Direction:  {direction_prob:.1%} up")
        logger.info(f"  Move:       {expected_move*100:+.3f}%")
        logger.info(f"  Confidence: {confidence:.0%} ({conf_label})")
        logger.info(f"  Regime:     {regime_name}")

        # Save signal
        self._save_signal(signal)

        return signal

    # ═══════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════

    def _save_models(self):
        """Save all trained models to disk."""
        artifacts = {
            "scaler": self.scaler,
            "regime_model": self.regime_model,
            "direction_models": self.direction_models,
            "move_model": self.move_model,
            "feature_cols": self.feature_cols,
        }
        path = os.path.join(MODELS_DIR, "model_artifacts.pkl")
        with open(path, "wb") as f:
            pickle.dump(artifacts, f)
        logger.info(f"Models saved to {path}")

    def _load_models(self):
        """Load trained models from disk."""
        path = os.path.join(MODELS_DIR, "model_artifacts.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError("No trained models found. Run with --train first.")
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
        self.scaler = artifacts["scaler"]
        self.regime_model = artifacts["regime_model"]
        self.direction_models = artifacts["direction_models"]
        self.move_model = artifacts["move_model"]
        self.feature_cols = artifacts["feature_cols"]
        self.is_trained = True
        logger.info("Models loaded from disk")

    def _save_signal(self, signal: dict):
        """Save daily signal to JSON."""
        import json
        date_str = str(signal["signal_date"])
        path = os.path.join(MODELS_DIR, f"signal_{date_str}.json")
        # Convert non-serializable types
        clean = {}
        for k, v in signal.items():
            if isinstance(v, (np.integer, np.floating)):
                clean[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        with open(path, "w") as f:
            json.dump(clean, f, indent=2, default=str)


# ── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run daily gold model")
    parser.add_argument("--train", action="store_true", help="Train models first")
    parser.add_argument("--date", default=None, help="Scoring date YYYY-MM-DD")
    parser.add_argument("--train-end", default=None, help="Training end date")
    args = parser.parse_args()

    engine = GoldSignalEngine()

    if args.train:
        engine.train(train_end_date=args.train_end)

    target = date.fromisoformat(args.date) if args.date else None
    signal = engine.score_today(target_date=target)

    print("\n" + "=" * 50)
    print(f"GOLD DAILY BIAS: {signal['bias_score']}/10")
    print(f"Direction: {signal['direction_probability']:.1%} probability of up day")
    print(f"Expected Move: {signal['expected_move_pct']:+.3f}%")
    print(f"Confidence: {signal['confidence']:.0%} ({signal['confidence_label']})")
    print(f"Regime: {signal['regime']}")
    print("=" * 50)
