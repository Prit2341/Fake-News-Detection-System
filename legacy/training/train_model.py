"""
Train LightGBM: 80/20 split with a validation set for early stopping.
Prevents overfitting by stopping when validation loss stops improving.
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
    # Load features
    print("Loading features...")
    X = joblib.load(ARTIFACTS_DIR / "X_features.joblib")
    y = joblib.load(ARTIFACTS_DIR / "y_labels.joblib")
    print(f"  X: {X.shape}  |  y: {y.shape}")

    # Split: 70% train | 10% validation | 20% test
    print("\nSplitting 70/10/20 (stratified)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=SEED, stratify=y_temp
    )
    print(f"  Train: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # Save test set for evaluate.py
    joblib.dump(X_test, ARTIFACTS_DIR / "X_test.joblib")
    joblib.dump(y_test, ARTIFACTS_DIR / "y_test.joblib")

    # Tuned model with early stopping + regularization
    print("\n--- Training with Early Stopping ---")
    params = dict(
        n_estimators=1000,          # high cap — early stopping will find optimal
        max_depth=10,               # reduced from 15 to limit complexity
        learning_rate=0.05,         # slower learning = better generalization
        num_leaves=31,              # reduced from 63
        min_child_samples=50,       # require more samples per leaf
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
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
    print(f"\n  Best iteration: {best_iter}")

    # Train vs Val accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc   = accuracy_score(y_val,   model.predict(X_val))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))

    print(f"\n  Train Accuracy: {train_acc:.4f}")
    print(f"  Val   Accuracy: {val_acc:.4f}")
    print(f"  Test  Accuracy: {test_acc:.4f}")
    print(f"  Train-Test Gap: {train_acc - test_acc:.4f}")

    if train_acc - test_acc < 0.01:
        print("  Overfitting: LOW")
    elif train_acc - test_acc < 0.03:
        print("  Overfitting: MILD")
    else:
        print("  Overfitting: HIGH — consider more regularization")

    # 5-fold CV on full training set
    print("\n--- 5-Fold Cross Validation ---")
    cv_model = lgb.LGBMClassifier(**{**params, "n_estimators": best_iter, "verbose": -1})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(cv_model, X_temp, y_temp, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Fold scores: {[round(s, 4) for s in cv_scores]}")

    # Save
    joblib.dump(model, ARTIFACTS_DIR / "lgbm_model.joblib")
    joblib.dump(params, ARTIFACTS_DIR / "model_params.joblib")

    print(f"\nModel saved → {ARTIFACTS_DIR / 'lgbm_model.joblib'}")
    print("Training done.")
