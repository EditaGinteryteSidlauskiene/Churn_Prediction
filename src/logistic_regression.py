from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.metrics import (
    make_scorer, precision_recall_curve,
    classification_report, roc_auc_score, confusion_matrix,
    balanced_accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import altair as alt
from src.helper import render_confusion_matrix


def perform_primary_lg_training(scaled_X_train, y_train, scaled_X_test, y_test, class_weight, pos_label=1):
   # Model
    log_reg = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
        class_weight=class_weight
    )
    log_reg.fit(scaled_X_train, y_train)

    # Predictions
    y_pred = log_reg.predict(scaled_X_test)
    y_proba = log_reg.predict_proba(scaled_X_test)[:, 1]  # for ROC–AUC

    # Metrics (focus on the positive class = churn)
    acc = (y_pred == y_test).mean()
    prec_pos = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rec_pos  = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1_pos   = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    roc_auc  = roc_auc_score(y_test, y_proba)
    bal_acc  = balanced_accuracy_score(y_test, y_pred)

    # Optional: confusion matrix if you want to report it
    # confusion_matrix = confusion_matrix(y_test, y_pred, labels=[pos_label, 1 - pos_label])

    # Return a neat dict/row for your table
    row = {
        "Model": "Logistic Regression",
        "Accuracy": acc,
        "Precision": prec_pos,
        "Recall": rec_pos,
        "F1-score": f1_pos,
        "ROC-AUC": roc_auc,
        "Balanced Acc.": bal_acc
    }

    # Nicely formatted printout
    print("\nLogistic Regression Model Evaluation")
    print("-" * 55)
    for key, value in row.items():
        print(f"{key:<15}: {value}")
    print("-" * 55)

def cross_validate_lg_model(X, y, cv, class_weight, *, pos_label=1,sparse_input=False):
    """
    Cross-validate Logistic Regression with churn-centric scorers.
    Returns a DataFrame (metric, mean, std) and prints a tidy summary.
    """

    def _ix(A, idx):
        return A.iloc[idx] if hasattr(A, "iloc") else A[idx]
    
    records = []
    k = cv.get_n_splits(X, y)  # derive k from the provided splitter

    fold = 0
    for tr_idx, te_idx in cv.split(X, y):
        fold += 1
        X_tr_full, X_te = _ix(X, tr_idx), _ix(X, te_idx)
        y_tr_full, y_te = _ix(y, tr_idx), _ix(y, te_idx)

        scaler = StandardScaler(with_mean=not sparse_input)
        X_tr = scaler.fit_transform(X_tr_full)  # fit on train fold only
        X_ts = scaler.transform(X_te)
            

        lr = LogisticRegression(
                max_iter=1000, solver="lbfgs", class_weight=class_weight, random_state=42
            )

        lr.fit(X_tr, y_tr_full)

        y_pred  = lr.predict(X_ts)
        y_proba = lr.predict_proba(X_ts)[:, 1]

        records.append({
            "fold": fold,
            "accuracy":            float((y_pred == y_te).mean()),
            "balanced_accuracy":   float(balanced_accuracy_score(y_te, y_pred)),
            "precision":           float(precision_score(y_te, y_pred, pos_label=pos_label, zero_division=0)),
            "recall":              float(recall_score(y_te, y_pred, pos_label=pos_label, zero_division=0)),
            "f1":                  float(f1_score(y_te, y_pred, pos_label=pos_label, zero_division=0)),
            "roc_auc":             float(roc_auc_score(y_te, y_proba)),
        })

    per_fold = pd.DataFrame(records)
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = (per_fold[metrics].agg(["mean", "std"]).T
               .reset_index().rename(columns={"index": "metric"})
               .sort_values("metric"))
    
    print(f"\nLogistic Regression Cross-Validation (k={k}, per-fold scaling, no Pipeline)")
    print("-" * 70)
    for _, r in summary.iterrows():
        print(f"{r['metric']:<20}: {r['mean']:.3f} ± {r['std']:.3f}")
    print("-" * 70)

def _ix(A, idx):
    return A.iloc[idx] if hasattr(A, "iloc") else A[idx]

def hyperparameter_tune_lg(X, y, cv, *,                                # required StratifiedKFold (reuse across models)
    C_grid=None,                       # e.g., np.logspace(-3, 3, 13)
    class_weight_grid=(None, "balanced"),
    pos_label=1,
    sparse_input=False,                # True if X is sparse one-hot
    max_iter=2000,
    solver="lbfgs",
    scoring="f1"):
    C_grid = np.logspace(-3, 3, 13)  # 1e-3 … 1e3
    """
    Manual grid search for Logistic Regression WITHOUT Pipeline.
    - Scales inside each fold (fit on train, transform train & val).
    - Optimizes a chosen scorer on the validation fold.
    - Returns best_params, best_cv_score, and a per-config summary DataFrame.
    """
    if C_grid is None:
        C_grid = np.logspace(-3, 3, 13)

    def score_fold(y_true, y_pred, y_proba):
        if scoring == "f1":
            return f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        elif scoring == "recall":
            return recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        elif scoring == "precision":
            return precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        elif scoring == "roc_auc":
            return roc_auc_score(y_true, y_proba)
        else:
            raise ValueError(f"Unsupported scoring='{scoring}'")
    
    records = []
    best_score = -np.inf
    best_params = None
    best_threshold_median = None
    best_thresholds_per_fold = None
    # iterate grid
    for C in C_grid:
        for cw in class_weight_grid:
            fold_scores = []
            fold_metrics = []
            fold_thresholds = []

            for tr_idx, va_idx in cv.split(X, y):
                X_tr_full, X_va = _ix(X, tr_idx), _ix(X, va_idx)
                y_tr, y_va = _ix(y, tr_idx), _ix(y, va_idx)
                # scale INSIDE the fold (no leakage)
                scaler = StandardScaler(with_mean=not sparse_input)
                X_tr = scaler.fit_transform(X_tr_full)
                X_v  = scaler.transform(X_va)
                # fit LR
                lr = LogisticRegression(
                    C=C, class_weight=cw, solver=solver, max_iter=max_iter, random_state=42
                )
                lr.fit(X_tr, y_tr)
                # predict
                y_pred  = lr.predict(X_v)
                y_proba = lr.predict_proba(X_v)[:, 1]

                thr, f1_at_thr = best_threshold_for_f1(y_va, y_proba, pos_label=pos_label)
                y_pred = (y_proba >= thr).astype(int)
                s = f1_at_thr  # use improved F1 as the fold score
                fold_thresholds.append(thr)
                
                # primary score for model selection
                s = score_fold(y_va, y_pred, y_proba)
                fold_scores.append(s)
                # collect extra metrics (optional, helpful to inspect trade-offs)
                fold_metrics.append({
                    "accuracy":           float((y_pred == y_va).mean()),
                    "balanced_accuracy":  float(balanced_accuracy_score(y_va, y_pred)),
                    "precision":          float(precision_score(y_va, y_pred, pos_label=pos_label, zero_division=0)),
                    "recall":             float(recall_score(y_va, y_pred, pos_label=pos_label, zero_division=0)),
                    "f1":                 float(f1_score(y_va, y_pred, pos_label=pos_label, zero_division=0)),
                    "roc_auc":            float(roc_auc_score(y_va, y_proba)),
                })
            mean_score = float(np.mean(fold_scores))
            std_score  = float(np.std(fold_scores))
            mean_thr   = float(np.median(fold_thresholds)) if (scoring == "f1" and fold_thresholds) else None

            # aggregate extra metrics (mean across folds)
            extra_df = pd.DataFrame(fold_metrics)
            extras_mean = extra_df.mean().to_dict()
            records.append({
                "C": C,
                "class_weight": cw,
                "cv_score_mean": mean_score,
                "cv_score_std": std_score,
                "opt_threshold_median": mean_thr,
                **{f"mean_{k}": float(v) for k, v in extras_mean.items()}
            })
            if mean_score > best_score:
                best_score = mean_score
                best_params = {"C": C, "class_weight": cw}

                best_threshold_median = mean_thr
                best_thresholds_per_fold = list(fold_thresholds)

    summary = pd.DataFrame(records).sort_values("cv_score_mean", ascending=False).reset_index(drop=True)

    # pretty print top rows
    print(f"\nLogistic Regression manual grid (scoring='{scoring}', folds={cv.get_n_splits()})")
    print("Top configs:")
    display_cols = ["C", "class_weight", "cv_score_mean", "cv_score_std",
                    "mean_precision", "mean_recall", "mean_f1", "mean_balanced_accuracy", "mean_roc_auc"]
    print(summary[display_cols].head(10).to_string(index=False))
    print("\nBest params:", best_params)
    print(f"Best CV {scoring}: {best_score:.3f}")
    if best_threshold_median is not None:
        print(f"Chosen threshold (median across folds) for best params: {best_threshold_median:.4f}")
        print(f"Per-fold thresholds (best params): {[f'{t:.4f}' for t in best_thresholds_per_fold]}")

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

def retrain_lg_model(
    X_train, y_train, X_test, y_test, 
    threshold, C, *, 
    class_weight="balanced", pos_label=1, n_splits=5, max_iter=2000, random_state=42
):
    # 2) Refit on all training data with tuned hyperparams
    lr = LogisticRegression(
        C=C, class_weight=class_weight, solver="lbfgs",
        max_iter=max_iter, random_state=random_state
    )
    lr.fit(X_train, y_train)

    # 3) Evaluate on test with the chosen operating threshold
    y_proba = lr.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    # Render CM at tuned threshold
    cm_fig = render_confusion_matrix(
        y_true=y_test, y_proba=y_proba, threshold=threshold
    )

    acc     = (y_pred == y_test).mean()
    prec    = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rec     = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1      = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rocauc  = roc_auc_score(y_test, y_proba)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    return {
        "Confusion matrix": cm_fig,
        "Model": "Logistic Regression",
        "C": C,
        "class_weight": class_weight,
        "Threshold": threshold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "ROC-AUC": rocauc,
        "Balanced Acc.": bal_acc
    }, lr

def display_lr_metrics(metrics_tab, metrics):
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

def get_telco_lr_metrics_caption(column2):
    column2.caption(body="<br><br>The logistic regression model demonstrates solid discriminatory " \
        "power (ROC-AUC = 0.84) and a balanced operating point chosen for F1. On the test set " \
        "it produced 838 true negatives, 197 false positives, 126 false negatives, and 248 true " \
        "positives. This yields accuracy 0.77, reflecting overall correctness across both classes. " \
        "Because churn is imbalanced, balanced accuracy 0.74 is more informative; it averages the " \
        "model’s recall for churn (0.66) and specificity for non-churn (~0.81). At the selected threshold, " \
        "precision 0.56 means that 56% of customers flagged as churners actually churn, while 44% are false " \
        "alarms. The F1-score 0.61 summarizes this precision–recall trade-off and indicates a reasonable " \
        "balance between catching churners and limiting false positives. In practice, lowering the threshold " \
        "would increase recall (fewer missed churners) at the cost of precision, whereas raising it would reduce " \
        "false positives but miss more churners.", unsafe_allow_html=True)
    

def get_internet_lr_metrics_caption(column2):
    column2.caption(body="After hyperparameter tuning, the Logistic Regression model (C = 0.01, class_weight = " \
    "'balanced', threshold = 0.408) achieved strong and well-balanced performance on the test set. The model " \
    "reached an accuracy of 0.865 and a balanced accuracy of 0.860, indicating that it performs consistently across " \
    "both classes without bias toward the majority class. With a precision of 0.859 and a recall of 0.906, the model " \
    "correctly identified about 91% of all churners while maintaining a high proportion of true positives among its churn " \
    "predictions, resulting in an F1-score of 0.882. The ROC-AUC of 0.924 confirms strong discriminative power, meaning " \
    "the model effectively separates churners from non-churners.<br><br>The confusion matrix shows 7,258 true positives, " \
    "5,182 true negatives, 1,187 false positives, and 752 false negatives, corresponding to a true positive rate of 90.6% "
    "and a true negative rate of 81.4%. The close values of accuracy and balanced accuracy suggest that the dataset is " \
    "reasonably balanced and the classifier performs similarly on both classes. The slightly lowered decision threshold "
    "(0.408 instead of 0.5) increased recall without significantly harming precision, which is desirable in churn prediction " \
    "where missing a potential churner (false negative) is costlier than issuing a false alert.", unsafe_allow_html=True)
    