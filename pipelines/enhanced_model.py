"""
Enhanced Gold Signal Engine v2
"""
import os
import pickle
import numpy as np
import pandas as pd
from loguru import logger

import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

FEATURES_DIR = os.path.join("data", "features")
MODELS_DIR = os.path.join("data", "model_outputs")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_enhance_features():
    path = os.path.join(FEATURES_DIR, "daily_features.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    logger.info(f"Base features: {df.shape}")

    # INTERACTION FEATURES
    if "daily_return" in df.columns and "dxy_return" in df.columns:
        df["gold_x_dxy"] = df["daily_return"] * df["dxy_return"]
    if "daily_return" in df.columns and "real_yield_change" in df.columns:
        df["gold_x_real_yield"] = df["daily_return"] * df["real_yield_change"]
    if "daily_return" in df.columns and "vix_change" in df.columns:
        df["gold_x_vix"] = df["daily_return"] * df["vix_change"]
    if "momentum_20d" in df.columns and "realized_vol_20d" in df.columns:
        df["momentum_vol_ratio"] = df["momentum_20d"] / (df["realized_vol_20d"] + 0.001)
    if "dxy_return" in df.columns and "real_yield_change" in df.columns:
        df["dxy_x_real_yield"] = df["dxy_return"] * df["real_yield_change"]
    if "vix_change" in df.columns and "spx_return" in df.columns:
        df["vix_x_spx"] = df["vix_change"] * df["spx_return"]
    if "oil_return" in df.columns and "inflation_scare" in df.columns:
        df["oil_x_inflation"] = df["oil_return"] * (df["inflation_scare"] / 100)
    if "geopolitical_tension" in df.columns and "safe_haven_demand" in df.columns:
        df["geo_x_safehaven"] = (df["geopolitical_tension"] / 100) * (df["safe_haven_demand"] / 100)

    # ROLLING Z-SCORES
    for col in ["daily_return", "dxy_return", "vix_change", "real_yield_change"]:
        if col in df.columns:
            rm = df[col].rolling(20, min_periods=5).mean()
            rs = df[col].rolling(20, min_periods=5).std()
            df[f"{col}_zscore"] = (df[col] - rm) / (rs + 0.0001)

    # CROSS-ASSET RATIO
    if "momentum_5d" in df.columns and "spx_return" in df.columns:
        df["gold_vs_spx_momentum"] = df["momentum_5d"] - df["spx_return"].rolling(5).sum()

    # WEEKLY TARGET
    df["target_weekly_return"] = df["daily_return"].shift(-1).rolling(5).sum().shift(-4)
    df["target_weekly_direction"] = (df["target_weekly_return"] > 0).astype(int)

    logger.info(f"Enhanced features: {df.shape}")
    return df


def get_feature_cols(df):
    exclude = ["date", "target_return", "target_direction", "target_abs_return",
               "target_range", "target_weekly_return", "target_weekly_direction"]
    return [c for c in df.columns if c not in exclude]


def train_and_evaluate():
    df = load_and_enhance_features()
    feature_cols = get_feature_cols(df)

    # DAILY MODEL
    logger.info("=" * 60)
    logger.info("DAILY DIRECTION MODEL")
    logger.info("=" * 60)

    daily_df = df.dropna(subset=["target_direction"] + feature_cols).copy()
    X = daily_df[feature_cols].values
    y = daily_df["target_direction"].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {}
    models["lgbm"] = lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.02, num_leaves=15, min_child_samples=30, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1)
    models["lgbm"].fit(X_train, y_train)
    models["xgb"] = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.02, min_child_weight=5, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=0)
    models["xgb"].fit(X_train, y_train)
    models["rf"] = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=15, max_features=0.6, random_state=42, n_jobs=-1)
    models["rf"].fit(X_train, y_train)
    models["lr"] = LogisticRegression(max_iter=2000, C=0.05, random_state=42)
    models["lr"].fit(X_train_s, y_train)

    probs = []
    for name, model in models.items():
        if name == "lr":
            probs.append(model.predict_proba(X_test_s)[:, 1])
        else:
            probs.append(model.predict_proba(X_test)[:, 1])
    ensemble_prob = np.mean(probs, axis=0)
    ensemble_pred = (ensemble_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, ensemble_pred)
    auc = roc_auc_score(y_test, ensemble_prob)
    brier = brier_score_loss(y_test, ensemble_prob)
    logger.info(f"  Overall Accuracy:    {acc:.4f}")
    logger.info(f"  ROC AUC:             {auc:.4f}")
    logger.info(f"  Brier Score:         {brier:.4f}")
    logger.info(f"  Test days:           {len(y_test)}")

    logger.info("")
    logger.info("DAILY HIGH-CONVICTION:")
    for t in [0.55, 0.60, 0.65, 0.70]:
        hc = (ensemble_prob > t) | (ensemble_prob < (1 - t))
        if sum(hc) >= 5:
            hc_acc = accuracy_score(y_test[hc], (ensemble_prob[hc] > 0.5).astype(int))
            logger.info(f"  >{t:.0%} confidence: {hc_acc:.1%} accuracy on {sum(hc)} days ({sum(hc)/len(y_test)*100:.0f}%)")

    # WEEKLY MODEL
    logger.info("")
    logger.info("=" * 60)
    logger.info("WEEKLY DIRECTION MODEL (5-day forward)")
    logger.info("=" * 60)

    weekly_df = df.dropna(subset=["target_weekly_direction"] + feature_cols).copy()
    Xw = weekly_df[feature_cols].values
    yw = weekly_df["target_weekly_direction"].values

    sw = int(len(Xw) * 0.8)
    Xw_tr, Xw_te = Xw[:sw], Xw[sw:]
    yw_tr, yw_te = yw[:sw], yw[sw:]
    Xw_tr_s = scaler.fit_transform(Xw_tr)
    Xw_te_s = scaler.transform(Xw_te)

    wmodels = {}
    wmodels["lgbm"] = lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.02, num_leaves=15, min_child_samples=30, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1)
    wmodels["lgbm"].fit(Xw_tr, yw_tr)
    wmodels["xgb"] = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.02, min_child_weight=5, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=0)
    wmodels["xgb"].fit(Xw_tr, yw_tr)
    wmodels["rf"] = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=15, max_features=0.6, random_state=42, n_jobs=-1)
    wmodels["rf"].fit(Xw_tr, yw_tr)
    wmodels["lr"] = LogisticRegression(max_iter=2000, C=0.05, random_state=42)
    wmodels["lr"].fit(Xw_tr_s, yw_tr)

    wprobs = []
    for name, model in wmodels.items():
        if name == "lr":
            wprobs.append(model.predict_proba(Xw_te_s)[:, 1])
        else:
            wprobs.append(model.predict_proba(Xw_te)[:, 1])
    w_ensemble = np.mean(wprobs, axis=0)
    w_pred = (w_ensemble > 0.5).astype(int)

    w_acc = accuracy_score(yw_te, w_pred)
    w_auc = roc_auc_score(yw_te, w_ensemble)
    logger.info(f"  Overall Accuracy:    {w_acc:.4f}")
    logger.info(f"  ROC AUC:             {w_auc:.4f}")
    logger.info(f"  Test days:           {len(yw_te)}")

    logger.info("")
    logger.info("WEEKLY HIGH-CONVICTION:")
    for t in [0.55, 0.60, 0.65, 0.70]:
        hc = (w_ensemble > t) | (w_ensemble < (1 - t))
        if sum(hc) >= 5:
            hc_acc = accuracy_score(yw_te[hc], (w_ensemble[hc] > 0.5).astype(int))
            logger.info(f"  >{t:.0%} confidence: {hc_acc:.1%} accuracy on {sum(hc)} days ({sum(hc)/len(yw_te)*100:.0f}%)")

    # COMBINED SIGNAL
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMBINED SIGNAL (daily + weekly agree)")
    logger.info("=" * 60)

    min_len = min(len(ensemble_prob), len(w_ensemble))
    dp = ensemble_prob[-min_len:]
    wp = w_ensemble[-min_len:]
    yc = y_test[-min_len:]

    both_agree = ((dp > 0.5) & (wp > 0.5)) | ((dp < 0.5) & (wp < 0.5))
    if sum(both_agree) > 10:
        ag_acc = accuracy_score(yc[both_agree], (dp[both_agree] > 0.5).astype(int))
        logger.info(f"  Both agree: {sum(both_agree)} days ({sum(both_agree)/min_len*100:.0f}%)")
        logger.info(f"  Accuracy when both agree: {ag_acc:.1%}")

    strong = ((dp > 0.6) & (wp > 0.6)) | ((dp < 0.4) & (wp < 0.4))
    if sum(strong) > 5:
        st_acc = accuracy_score(yc[strong], (dp[strong] > 0.5).astype(int))
        logger.info(f"  Strong agreement: {sum(strong)} days ({sum(strong)/min_len*100:.0f}%)")
        logger.info(f"  Strong agreement accuracy: {st_acc:.1%}")

    # FEATURE IMPORTANCE
    logger.info("")
    logger.info("TOP 20 FEATURES:")
    imp = models["lgbm"].feature_importances_
    top = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
    for i, (f, v) in enumerate(top[:20]):
        bar = "█" * int(v / max(imp) * 30)
        logger.info(f"  {i+1:2d}. {f:30s} {v:4d}  {bar}")

    # SUMMARY
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Daily accuracy:     {acc:.1%}")
    logger.info(f"  Weekly accuracy:    {w_acc:.1%}")
    if sum(both_agree) > 10:
        logger.info(f"  Combined accuracy:  {ag_acc:.1%}")
    if sum(strong) > 5:
        logger.info(f"  Strong agreement:   {st_acc:.1%}")

    artifacts = {"daily_models": models, "weekly_models": wmodels, "scaler": scaler, "feature_cols": feature_cols}
    with open(os.path.join(MODELS_DIR, "enhanced_model_v2.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    logger.info(f"Models saved!")


if __name__ == "__main__":
    train_and_evaluate()
