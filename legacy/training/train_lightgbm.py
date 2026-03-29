"""
Train LightGBM on sentence embeddings with early stopping.
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import lightgbm as lgb

SEED = 42
ARTIFACTS_DIR = Path("d:/Fake_News_Detection/artifacts")


if __name__ == "__main__":
    print("Loading embeddings...")
    X = joblib.load(ARTIFACTS_DIR / "embeddings.joblib")
    y = joblib.load(ARTIFACTS_DIR / "y_labels.joblib")
    print(f"  X: {X.shape}  |  y: {y.shape}")

    # 70/10/20 split
    print("\nSplitting 70/10/20...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=SEED, stratify=y_temp)
    print(f"  Train: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    joblib.dump(X_test, ARTIFACTS_DIR / "X_test_lgbm.joblib")
    joblib.dump(y_test, ARTIFACTS_DIR / "y_test.joblib")

    # Train
    print("\n--- Training LightGBM on Embeddings ---")
    params = dict(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        device="gpu",
        random_state=SEED,
        verbose=1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iter = model.best_iteration_
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc   = accuracy_score(y_val,   model.predict(X_val))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))

    print(f"\n  Best iteration : {best_iter}")
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Val   Accuracy : {val_acc:.4f}")
    print(f"  Test  Accuracy : {test_acc:.4f}")
    print(f"  Train-Test Gap : {train_acc - test_acc:.4f}")

    # 5-fold CV
    print("\n--- 5-Fold CV ---")
    cv_model  = lgb.LGBMClassifier(**{**params, "n_estimators": best_iter, "verbose": -1})
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(cv_model, X_temp, y_temp, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Folds: {[round(s, 4) for s in cv_scores]}")

    joblib.dump(model, ARTIFACTS_DIR / "lgbm_model.joblib")
    print(f"\nModel saved → lgbm_model.joblib")
    print("LightGBM training done.")
