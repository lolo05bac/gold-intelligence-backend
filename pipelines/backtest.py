"""
Walk-Forward Backtesting Pipeline
Proper out-of-sample validation with no future leakage.

Methodology:
    - Walk-forward: train on expanding window, test on next block
    - Minimum train period: 2 years
    - Test block: 3 months
    - Metrics tracked per fold and aggregated

Usage:
    python -m pipelines.backtest [--start 2020-01-01] [--block-months 3]
"""
import os
import argparse
import json
from datetime import date, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, brier_score_loss,
    )
except ImportError:
    pass

from pipelines.run_daily_model import GoldSignalEngine, REGIME_LABELS

FEATURES_DIR = os.path.join("data", "features")
RESULTS_DIR = os.path.join("research", "reports")
os.makedirs(RESULTS_DIR, exist_ok=True)


class WalkForwardBacktester:
    """Walk-forward backtesting with proper time-series methodology."""

    def __init__(
        self,
        min_train_years: int = 2,
        test_block_months: int = 3,
        retrain_frequency_months: int = 3,
    ):
        self.min_train_years = min_train_years
        self.test_block_months = test_block_months
        self.retrain_frequency = retrain_frequency_months
        self.engine = GoldSignalEngine()
        self.results = []

    def run(self, start_date: str = "2020-01-01"):
        """Run the full walk-forward backtest."""
        logger.info("=== Walk-Forward Backtest ===")

        df = self.engine.load_features()
        X, y_dir, y_ret, dates, full_df = self.engine.prepare_data(df)

        test_start = pd.to_datetime(start_date).date()
        data_end = dates[-1]
        block_delta = timedelta(days=self.test_block_months * 30)

        fold = 0
        current_test_start = test_start

        while current_test_start < data_end:
            current_test_end = min(
                current_test_start + block_delta,
                data_end,
            )

            # Train mask: everything before test start
            train_mask = dates < current_test_start
            test_mask = (dates >= current_test_start) & (dates <= current_test_end)

            n_train = sum(train_mask)
            n_test = sum(test_mask)

            if n_train < 250 or n_test < 10:  # Need minimum data
                logger.warning(f"Fold {fold}: insufficient data (train={n_train}, test={n_test})")
                current_test_start = current_test_end + timedelta(days=1)
                fold += 1
                continue

            logger.info(f"\n--- Fold {fold}: Train to {dates[train_mask][-1]}, Test {current_test_start} to {current_test_end} (n={n_test}) ---")

            X_train, X_test = X[train_mask], X[test_mask]
            y_dir_train, y_dir_test = y_dir[train_mask], y_dir[test_mask]
            y_ret_train, y_ret_test = y_ret[train_mask], y_ret[test_mask]

            # Train fresh models for this fold
            self.engine.scaler.fit(X_train)
            self.engine.train_direction_model(X_train, y_dir_train)
            self.engine.train_move_model(X_train, y_ret_train)
            self.engine.train_regime_model(X_train, full_df[train_mask])

            # Predict
            dir_probs = self.engine.predict_direction(X_test)
            dir_preds = (dir_probs > 0.5).astype(int)
            move_preds = self.engine.move_model.predict(X_test)

            # Regime predictions
            regime_preds = self.engine.regime_model.predict(X_test)

            # Compute metrics
            metrics = self._compute_metrics(
                y_dir_test, dir_preds, dir_probs,
                y_ret_test, move_preds,
                regime_preds, dates[test_mask],
            )
            metrics["fold"] = fold
            metrics["train_end"] = str(dates[train_mask][-1])
            metrics["test_start"] = str(current_test_start)
            metrics["test_end"] = str(current_test_end)
            metrics["n_train"] = int(n_train)
            metrics["n_test"] = int(n_test)

            self.results.append(metrics)
            self._log_fold_metrics(metrics)

            # Advance
            current_test_start = current_test_end + timedelta(days=1)
            fold += 1

        # Aggregate results
        self._aggregate_and_save()

    def _compute_metrics(self, y_true, y_pred, y_prob, y_ret_true, y_ret_pred, regimes, dates):
        """Compute comprehensive metrics for one fold."""
        metrics = {}

        # Direction metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision_bull"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["precision_bear"] = float(precision_score(1 - y_true, 1 - y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = 0.5

        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

        # High-confidence accuracy
        high_conf = (y_prob > 0.65) | (y_prob < 0.35)
        if sum(high_conf) > 0:
            hc_preds = (y_prob[high_conf] > 0.5).astype(int)
            metrics["high_conf_accuracy"] = float(accuracy_score(y_true[high_conf], hc_preds))
            metrics["high_conf_n"] = int(sum(high_conf))
        else:
            metrics["high_conf_accuracy"] = None
            metrics["high_conf_n"] = 0

        # Move model correlation
        valid = ~np.isnan(y_ret_true) & ~np.isnan(y_ret_pred)
        if sum(valid) > 2:
            metrics["move_correlation"] = float(np.corrcoef(y_ret_pred[valid], y_ret_true[valid])[0, 1])
        else:
            metrics["move_correlation"] = 0

        # Expected value (simple PnL proxy)
        # If we go long on bullish calls, short on bearish
        positions = np.where(y_prob > 0.5, 1, -1)
        returns = positions * y_ret_true
        metrics["avg_return_per_signal"] = float(np.nanmean(returns))
        metrics["total_return"] = float(np.nansum(returns))
        metrics["sharpe_approx"] = float(np.nanmean(returns) / (np.nanstd(returns) + 1e-8) * np.sqrt(252))

        # Per-regime accuracy
        regime_metrics = {}
        for r_id, r_name in REGIME_LABELS.items():
            mask = regimes == r_id
            if sum(mask) >= 5:
                r_preds = (y_prob[mask] > 0.5).astype(int)
                regime_metrics[r_name] = {
                    "accuracy": float(accuracy_score(y_true[mask], r_preds)),
                    "n": int(sum(mask)),
                }
        metrics["by_regime"] = regime_metrics

        return metrics

    def _log_fold_metrics(self, m: dict):
        """Log key metrics for a fold."""
        logger.info(f"  Accuracy:         {m['accuracy']:.4f}")
        logger.info(f"  ROC AUC:          {m['roc_auc']:.4f}")
        logger.info(f"  Brier Score:      {m['brier_score']:.4f}")
        logger.info(f"  High-Conf Acc:    {m.get('high_conf_accuracy', 'N/A')} (n={m.get('high_conf_n', 0)})")
        logger.info(f"  Avg Return/Trade: {m['avg_return_per_signal']:.5f}")
        logger.info(f"  Sharpe (approx):  {m['sharpe_approx']:.3f}")

    def _aggregate_and_save(self):
        """Aggregate across folds and save report."""
        if not self.results:
            logger.warning("No backtest results to aggregate")
            return

        logger.info("\n" + "=" * 60)
        logger.info("AGGREGATE BACKTEST RESULTS")
        logger.info("=" * 60)

        agg = {}
        for key in ["accuracy", "roc_auc", "brier_score", "f1", "avg_return_per_signal", "sharpe_approx"]:
            vals = [r[key] for r in self.results if r.get(key) is not None]
            if vals:
                agg[key] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                }
                logger.info(f"  {key}: {agg[key]['mean']:.4f} ± {agg[key]['std']:.4f}")

        # High-confidence aggregate
        hc_vals = [r["high_conf_accuracy"] for r in self.results if r.get("high_conf_accuracy") is not None]
        if hc_vals:
            agg["high_conf_accuracy"] = {
                "mean": round(float(np.mean(hc_vals)), 4),
                "std": round(float(np.std(hc_vals)), 4),
            }
            logger.info(f"  high_conf_accuracy: {agg['high_conf_accuracy']['mean']:.4f}")

        # Save full report
        report = {
            "config": {
                "min_train_years": self.min_train_years,
                "test_block_months": self.test_block_months,
                "n_folds": len(self.results),
            },
            "aggregate": agg,
            "folds": self.results,
        }

        path = os.path.join(RESULTS_DIR, "backtest_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\nFull report saved to {path}")

        # Also save as CSV for easy analysis
        fold_df = pd.DataFrame(self.results)
        fold_df.to_csv(os.path.join(RESULTS_DIR, "backtest_folds.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date")
    parser.add_argument("--block-months", type=int, default=3, help="Test block size in months")
    args = parser.parse_args()

    bt = WalkForwardBacktester(test_block_months=args.block_months)
    bt.run(start_date=args.start)
