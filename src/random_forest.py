from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer, precision_recall_curve,
    precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
import pandas as pd
import numpy as np
from numpy.random import default_rng
from src.helper import render_confusion_matrix
import streamlit as st


def perform_primary_rf_training(X_train_encoded, y_train, X_test_encoded, y_test, class_weight, pos_label=1):
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight
    )
    rf_model.fit(X_train_encoded, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test_encoded)
    y_proba = rf_model.predict_proba(X_test_encoded)[:, 1]  # for ROC–AUC

    # Metrics (focus on positive "Churn" class)
    acc = (y_pred == y_test).mean()
    prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # Create result dictionary (rounded)
    row = {
        "Model": "Random Forest",
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1-score": round(f1, 3),
        "ROC-AUC": round(roc_auc, 3),
        "Balanced Acc.": round(bal_acc, 3)
    }

    # Print neatly formatted report
    print("\nRandom Forest Model Evaluation")
    print("-" * 55)
    for key, value in row.items():
        print(f"{key:<15}: {value}")
    print("-" * 55)


def cross_validate_rf_model(X_train_encoded, y_train, cross_validation, class_weight, *,
    pos_label=1, n_splits=5, random_state=42, n_jobs=-1,
    n_estimators=200, max_depth=None,
    min_samples_leaf=1, max_features="sqrt"):

    """
    Cross-validate Random Forest with churn-centric scorers to match
    perform_primary_rf_training.
    Returns a DataFrame with mean ± std for each metric.
    """

    # Churn-focused scorers + global metrics
    scoring = {
        "roc_auc": "roc_auc",  # uses predict_proba
        "accuracy": "accuracy",
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision": make_scorer(precision_score, pos_label=pos_label, zero_division=0),
        "recall":    make_scorer(recall_score,    pos_label=pos_label, zero_division=0),
        "f1":        make_scorer(f1_score,        pos_label=pos_label, zero_division=0),
    }

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features
    )

    # Cross-validate
    cv_results = cross_validate(
        rf, X_train_encoded, y_train,
        cv=cross_validation, scoring=scoring, n_jobs=n_jobs, return_train_score=False
    )

    # Summarize mean ± std
    rows = []
    for k, v in cv_results.items():
        if not k.startswith("test_"):
            continue
        metric = k.replace("test_", "")
        rows.append((metric, float(np.mean(v)), float(np.std(v))))
    summary = pd.DataFrame(rows, columns=["metric", "mean", "std"])\
                .sort_values("metric").reset_index(drop=True)

    # Pretty print to terminal (matches your style)
    print("\nRandom Forest Cross-Validation Results (k={}, class_weight={})"
          .format(n_splits, class_weight))
    print("-" * 65)
    for _, r in summary.iterrows():
        print(f"{r['metric']:<20}: {r['mean']:.3f} ± {r['std']:.3f}")
    print("-" * 65)

def make_max_f1_scorer(pos_label=1, grid=np.linspace(0.05, 0.95, 91)):
    def _score(estimator, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        best = 0.0
        for t in grid:
            pred = (proba >= t).astype(int)
            s = f1_score(y, pred, pos_label=pos_label, zero_division=0)
            if s > best:
                best = s
        return best
    return _score

def best_threshold_for_f1(y_true, y_scores, pos_label=1):
    # PR curve: p, r length = n_thr+1 ; t length = n_thr
    p, r, t = precision_recall_curve(y_true == pos_label, y_scores)

    # Align to thresholds
    p, r = p[:-1], r[:-1]

    if t.size == 0:             # degenerate: all scores equal
        return 0.5, 0.0

    # Safe F1: 2PR / (P+R), but avoid divide-by-zero warnings and NaNs
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    # If any weirdness remains, zero it out
    f1 = np.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0)

    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])

def hyperparameter_tune_rf(
        
    X, y, cv, *,
    pos_label=1,
    n_iter=60,
    random_state=42,
    use_max_f1_scorer=True
):
    """
    Randomized hyperparameter search for Random Forest on encoded features.
    - Reuses your provided StratifiedKFold 'cv'
    - Scoring: standard F1(Churn) at threshold 0.5, or in-fold max-F1 via threshold sweep
    - Returns (best_estimator, results_df)
    """

    # Choose scorer
    scorer = (make_max_f1_scorer(pos_label=pos_label)
              if use_max_f1_scorer
              else make_scorer(f1_score, pos_label=pos_label, zero_division=0))

    # Reasonable param distributions (all discrete -> reproducible)
    rng = default_rng(random_state)
    param_distributions = {
        "n_estimators": rng.integers(200, 1201, size=1000),     # 200..1200
        "max_depth":    [None] + list(range(3, 31)),            # None or 3..30
        "min_samples_split": list(range(2, 21)),                 # 2..20
        "min_samples_leaf":  list(range(1, 21)),                 # 1..20
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 0.9],    # try categorical & frac
        "bootstrap": [True],                                     # keep bootstrap for stability
        "class_weight": [None, "balanced", "balanced_subsample"]
    }

    rf = RandomForestClassifier(
        n_jobs=-1,
        random_state=random_state,
        # leave other params to the sampler
    )

    rs = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,                # refit on full training set with best params
        return_train_score=False,
        verbose=0
    )

    rs.fit(X, y)

    # --- NEW: compute & print best threshold using OOF probs with best params ---
    from sklearn.model_selection import cross_val_predict
    oof_proba = cross_val_predict(rs.best_estimator_, X, y, cv=cv,
                                  method="predict_proba", n_jobs=-1)[:, 1]
    thr, f1_at_thr = best_threshold_for_f1(y, oof_proba, pos_label=pos_label)
    print(f"\nChosen threshold (OOF, best params): {thr:.4f}")
    print(f"OOF F1 at chosen threshold: {f1_at_thr:.3f}")
    # ---------------------------------------------------------------------------

    # Summarize top results
    results = pd.DataFrame(rs.cv_results_).sort_values(
        "mean_test_score", ascending=False
    ).reset_index(drop=True)

    display_cols = [
        "mean_test_score", "std_test_score",
        "param_n_estimators", "param_max_depth",
        "param_min_samples_split", "param_min_samples_leaf",
        "param_max_features", "param_class_weight", "param_bootstrap"
    ]
    print("\n[Random Forest RandomizedSearchCV] Top configs (by CV score):")
    print(results[display_cols].head(10).to_string(index=False))

    print("\nBest params:", rs.best_params_)
    print(f"Best CV score ({'max-F1' if use_max_f1_scorer else 'F1@0.5'}): {rs.best_score_:.3f}")

def retrain_rf_model(X_train_encoded, y_train, X_test_encoded, y_test, threshold,
                      n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth,
                        pos_label=1):
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_depth=max_depth,
        bootstrap=True
    )
    rf_model.fit(X_train_encoded, y_train)

    # Predictions
    y_proba = rf_model.predict_proba(X_test_encoded)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    cm_fig = render_confusion_matrix(
        y_true=y_test, y_proba=y_proba, threshold=threshold
    )

    # Metrics (focus on positive "Churn" class)
    acc = (y_pred == y_test).mean()
    prec = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # Create result dictionary (rounded)
    row = {
        "Confusion matrix": cm_fig,
        "Model": "Random Forest",
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1-score": round(f1, 3),
        "ROC-AUC": round(roc_auc, 3),
        "Balanced Acc.": round(bal_acc, 3)
    }, rf_model

    return row

def display_rf_metrics(metrics_tab, metrics):
    df = pd.DataFrame([metrics])

    metric_cols = ["Accuracy", "Balanced Acc.", "Precision", "Recall", "F1-score", "ROC-AUC"]
    cols = [c for c in metric_cols if c in df.columns]
    df = df[cols].round(3)


    with metrics_tab:
        st.markdown("#### Metrics")
        st.dataframe(df.round(3), width='stretch', hide_index=True)

        # ---------- Confusion matrix ----------
        cm_fig = metrics.get("Confusion matrix", None)
        column1, column2 = st.columns([1, 1])
        column1.pyplot(cm_fig)

    return column2

def get_telco_rf_metrics_caption(column2):

    column2.caption(body="<br><br>The tuned Random Forest delivers a strong operating balance. On " \
    "the test set it produced 831 true negatives, 204 false positives, 98 false negatives, and 276 " \
    "true positives (positive = churn). This yields accuracy 0.79 overall. Because the data are imbalanced," \
    " balanced accuracy 0.77 is more informative; it reflects a recall of 0.74 for churners and a specificity" \
    " of about 0.80 for non-churners. At the chosen threshold, precision 0.58 indicates that a little over half" \
    " of the customers flagged as churners actually churn, while the F1-score 0.65 summarizes an improved" \
    " precision–recall trade-off at this operating point. The model’s ROC-AUC 0.85 shows good separability " \
    "independent of threshold, supporting its reliability for ranking customers by churn risk.", unsafe_allow_html=True)

def get_internet_rf_metrics_caption(column2):
    column2.caption(body="<br><br>After tuning, the Random Forest achieves strong, well-balanced performance on the test set. " \
    "Overall accuracy is 0.909 and balanced accuracy is 0.907, indicating comparable performance on both classes (little " \
    "bias toward the majority class). The model attains precision = 0.912 and recall = 0.926, meaning over 92% of actual " \
    "churners are correctly identified while about 91% of churn predictions are correct, yielding an F1-score of 0.919. " \
    "The ROC–AUC of 0.963 shows excellent ranking ability—RF separates churners from non-churners very effectively.<br><br>" \
    "From the confusion matrix: TP = 7,417, TN = 5,650, FP = 719 (false-positive rate ≈ 11.3%), and FN = 593 (false-negative " \
    "rate ≈ 7.4%), corresponding to TPR/Recall = 92.6% and TNR = 88.7%. The closeness of accuracy and balanced accuracy " \
    "reflects the near-balanced dataset and similar sensitivity/specificity. Overall, the tuned Random Forest provides a " \
    "strong precision–recall trade-off with higher recall than the logistic model, making it well-suited for catching most " \
    "churners while keeping false alarms moderate.", unsafe_allow_html=True)

