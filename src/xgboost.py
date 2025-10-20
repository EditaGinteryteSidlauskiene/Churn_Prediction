from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score
)
import pandas as pd
import numpy as np
from numpy.random import default_rng
from itertools import product
from src.helper import render_confusion_matrix
import streamlit as st

def perform_primary_xgb_training(X_train_encoded, y_train, X_test_encoded, y_test, balance,
    pos_label=1, valid_size=0.15, random_state=42):
    
    # --- validation split from TRAIN (no test leakage)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_encoded, y_train,
        test_size=valid_size, stratify=y_train, random_state=random_state
    )

     # --- DMatrix objects
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test_encoded)

    # --- imbalance-aware weight
    if balance:
        pos = int((y_tr == pos_label).sum())
        neg = int((y_tr != pos_label).sum())
        scale_pos_weight = neg / max(pos, 1)
    else:
        scale_pos_weight = 1.0

    # --- params
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "scale_pos_weight": float(scale_pos_weight),
        "tree_method": "hist",
        "seed": random_state,
        "base_score": 0.5
    }


    # --- train with early stopping
    evals = [(dvalid, "valid")]
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    best_it = getattr(bst, "best_iteration", None)
    if best_it is None:  # fallback if early stopping didn’t trigger
        best_it = bst.num_boosted_rounds() - 1

    # --- predict on TEST using best_iteration
    # --- predict (probabilities for ROC–AUC; 0.5 threshold for labels)
    y_proba = bst.predict(dtest, iteration_range=(0, best_it + 1))
    y_pred  = (y_proba >= 0.5).astype(int)

    # --- metrics (focus on positive "Churn" class)
    acc     = (y_pred == y_test).mean()
    prec    = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rec     = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1      = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # --- table-ready row (rounded)
    row = {
        "Model": "XGBoost",
        "Accuracy": round(float(acc), 3),
        "Precision": round(float(prec), 3),
        "Recall": round(float(rec), 3),
        "F1-score": round(float(f1), 3),
        "ROC-AUC": round(float(roc_auc), 3),
        "Balanced Acc.": round(float(bal_acc), 3)
    }

    # --- neat terminal print
    print("\nXGBoost Model Evaluation")
    print("-" * 55)
    for k, v in row.items():
        print(f"{k:<15}: {v}")
    print(f"{'Best iteration':<15}: {best_it}")
    print("-" * 55)

def cross_validate_xgb_model(X, y, cross_validation, balance, *,
    pos_label=1,
    n_splits=5,
    inner_valid_size=0.15,
    random_state=42,
    num_boost_round=1000,
    early_stopping_rounds=50,
    eta=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist"):

    """
    Stratified K-fold CV for XGBoost matching perform_primary_xgb_training:
      - per-fold inner validation for early stopping (no leakage)
      - per-fold scale_pos_weight from the inner-train split
      - metrics on the outer-fold holdout: Accuracy, Precision/Recall/F1 (pos class),
        ROC-AUC (from probabilities), Balanced Accuracy
    Returns (summary_df, per_fold_df).
    """

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    records = []

    fold_idx = 0
    for train_index, test_index in cross_validation.split(X, y):
        fold_idx += 1
        X_tr_full, X_te = X[train_index], X[test_index]
        y_tr_full, y_te = y[train_index], y[test_index]

        # Inner validation split from training fold for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full,
            test_size=inner_valid_size,
            stratify=y_tr_full,
            random_state=random_state
        )

        # Compute class imbalance weight on inner-train
        if balance:
            pos = int((y_tr == pos_label).sum())
            neg = int((y_tr != pos_label).sum())
            scale_pos_weight = neg / max(pos, 1)
        else:
            scale_pos_weight=1.0

        # DMatrices
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        dtest  = xgb.DMatrix(X_te)

        params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "eta": eta,
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "lambda": reg_lambda,
                "scale_pos_weight": float(scale_pos_weight),
                "tree_method": tree_method,
                "seed": random_state,
            }
        
        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )

        best_it = getattr(bst, "best_iteration", None)
        if best_it is None:
            # Fallback if early stopping did not trigger
            best_it = bst.num_boosted_rounds() - 1

        # Predict on outer-fold holdout
        y_proba = bst.predict(dtest, iteration_range=(0, best_it + 1))
        y_pred  = (y_proba >= 0.5).astype(int)

        # Metrics (focus on positive class for P/R/F1)
        acc     = (y_pred == y_te).mean()
        prec    = precision_score(y_te, y_pred, pos_label=pos_label, zero_division=0)
        rec     = recall_score(y_te, y_pred, pos_label=pos_label, zero_division=0)
        f1      = f1_score(y_te, y_pred, pos_label=pos_label, zero_division=0)
        roc_auc = roc_auc_score(y_te, y_proba)
        bal_acc = balanced_accuracy_score(y_te, y_pred)

        records.append({
            "fold": fold_idx,
            "best_iteration": int(best_it),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "balanced_accuracy": float(bal_acc)
        })

    per_fold_df = pd.DataFrame(records)

    # Summary (mean ± std)
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = (
        per_fold_df[metrics]
        .agg(["mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
        .sort_values("metric")
    )

    # Pretty print
    print("\nXGBoost Cross-Validation Results (k={}, inner_valid={}, seed={})"
          .format(n_splits, inner_valid_size, random_state))
    print("-" * 70)
    for _, r in summary.iterrows():
        print(f"{r['metric']:<20}: {r['mean']:.3f} ± {r['std']:.3f}")
    print("-" * 70)

def _ix(A, idx):
    return A.iloc[idx] if hasattr(A, "iloc") else A[idx]

def _best_threshold_for_f1(y_true, y_scores, pos_label=1, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, 0.0
    for t in grid:
        y_pred = (y_scores >= t).astype(int)
        s = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        if s > best_f1:
            best_f1, best_t = s, t
    return float(best_t), float(best_f1)

def evaluate_xgb_params_cv(
    X, y, cv, params, *,
    pos_label=1,
    inner_valid_size=0.15,
    random_state=42,
    num_boost_round=1000,
    early_stopping_rounds=50,
    optimize_max_f1=True
):
    """
    Evaluates a given XGBoost param dict with outer CV:
      - For each outer fold, make an inner validation split for early stopping
      - Recompute scale_pos_weight on the inner-train
      - Score on the outer test fold
      - Optionally sweep threshold to maximize F1 per fold
    Returns: mean_score, std_score, per_fold_df (metrics + best_iteration)
    """
    skf = cv
    records = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        X_tr_full, X_te = _ix(X, tr_idx), _ix(X, te_idx)
        y_tr_full, y_te = _ix(y, tr_idx), _ix(y, te_idx)

        # Inner validation for early stopping (no leakage)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full, test_size=inner_valid_size,
            stratify=y_tr_full, random_state=random_state
        )

        # Recompute scale_pos_weight on inner-train
        pos = int((y_tr == pos_label).sum())
        neg = int((y_tr != pos_label).sum())
        spw = neg / max(pos, 1)

        # Assemble params (fold-specific spw + housekeeping)
        p = dict(params)  # copy
        p.update({
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": float(spw),
            "tree_method": p.get("tree_method", "hist"),
            "seed": p.get("seed", random_state),
        })

        dtrain = xgb.DMatrix(X_tr,  label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        dtest  = xgb.DMatrix(X_te)

        bst = xgb.train(
            params=p, dtrain=dtrain, num_boost_round=num_boost_round,
            evals=[(dvalid, "valid")], early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
        best_it = getattr(bst, "best_iteration", None)
        if best_it is None:
            best_it = bst.num_boosted_rounds() - 1

        y_proba = bst.predict(dtest, iteration_range=(0, best_it + 1))

        # Optionally pick best threshold per fold for F1 (like your LR logic)
        if optimize_max_f1:
            thr, _ = _best_threshold_for_f1(y_te, y_proba, pos_label=pos_label)
        else:
            thr = 0.5

        y_pred = (y_proba >= thr).astype(int)

        metrics = {
            "fold": fold,
            "best_iteration": int(best_it),
            "thr": float(thr),
            "accuracy": float((y_pred == y_te).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
            "precision": float(precision_score(y_te, y_pred, pos_label=pos_label, zero_division=0)),
            "recall": float(recall_score(y_te, y_pred, pos_label=pos_label, zero_division=0)),
            "f1": float(f1_score(y_te, y_pred, pos_label=pos_label, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_te, y_proba)),
        }
        records.append(metrics)

    per_fold = pd.DataFrame(records)
    mean_f1 = per_fold["f1"].mean()
    std_f1  = per_fold["f1"].std(ddof=0)
    return mean_f1, std_f1, per_fold

def xgb_random_search(
    X, y, cv, *,
    n_iter=40,
    pos_label=1,
    optimize_max_f1=True,
    random_state=42
):
    """
    Broad randomized search across key XGBoost hyperparameters.
    Returns: best_params, results_df (sorted by mean F1), rng_state
    """
    rng = default_rng(random_state)

    def sample_params():
        return {
            "eta": float(10 ** rng.uniform(-2.0, -0.3)),          # ~[0.01, 0.5]
            "max_depth": int(rng.integers(3, 11)),                # 3..10
            "min_child_weight": float(10 ** rng.uniform(-1, 1)),  # ~[0.1, 10]
            "subsample": float(rng.uniform(0.6, 1.0)),            # 0.6..1.0
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),     # 0.6..1.0
            "lambda": float(10 ** rng.uniform(-2, 2)),            # reg_lambda ~[0.01,100]
            "alpha": float(10 ** rng.uniform(-3, 1)),             # reg_alpha ~[0.001,10]
            "tree_method": "hist",
        }

    trials = []
    for i in range(1, n_iter + 1):
        params = sample_params()
        mean_f1, std_f1, per_fold = evaluate_xgb_params_cv(
            X, y, cv, params,
            pos_label=pos_label,
            optimize_max_f1=optimize_max_f1
        )
        row = dict(params)
        row.update({
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_recall": per_fold["recall"].mean(),
            "mean_precision": per_fold["precision"].mean(),
            "mean_balanced_accuracy": per_fold["balanced_accuracy"].mean(),
            "mean_roc_auc": per_fold["roc_auc"].mean(),
            "mean_best_it": per_fold["best_iteration"].mean(),
        })
        trials.append(row)
        print(f"[RandomSearch] {i:>2}/{n_iter} | F1={mean_f1:.3f} "
              f"| md={row['max_depth']}, eta={row['eta']:.3f}, mcw={row['min_child_weight']:.3f}, "
              f"sub={row['subsample']:.2f}, col={row['colsample_bytree']:.2f}, "
              f"lam={row['lambda']:.3f}, alp={row['alpha']:.3f}")

    results = pd.DataFrame(trials).sort_values("mean_f1", ascending=False).reset_index(drop=True)
    best_params = results.iloc[0][["eta","max_depth","min_child_weight","subsample",
                                   "colsample_bytree","lambda","alpha","tree_method"]].to_dict()
    
    _, _, per_fold_best = evaluate_xgb_params_cv(
        X, y, cv, best_params,
        pos_label=pos_label,
        optimize_max_f1=optimize_max_f1
        )
    thr_med = float(per_fold_best["thr"].median())
    thr_list = [float(t) for t in per_fold_best["thr"]]
    print(f"\nChosen threshold (median across outer folds): {thr_med:.4f}")
    print("Per-fold thresholds:", ", ".join(f"{t:.4f}" for t in thr_list))
    return best_params, results, rng.bit_generator.state

def around(val, factors):
    return sorted({float(val * f) for f in factors})

def xgb_grid_refine(
        
    X, y, cv, best_params_stage1, *,
    pos_label=1,
    optimize_max_f1=True
):
    eta_grid   = [max(0.005, e) for e in around(best_params_stage1["eta"],   [0.5, 0.75, 1.0, 1.25, 1.5])]
    md_grid    = sorted(set([max(2, int(best_params_stage1["max_depth"] + d)) for d in (-2,-1,0,1,2)]))
    mcw_grid   = [max(0.05, c) for c in around(best_params_stage1["min_child_weight"], [0.5, 0.75, 1.0, 1.5, 2.0])]
    sub_grid   = [min(1.0, max(0.5, s)) for s in around(best_params_stage1["subsample"], [0.85, 1.0])]
    col_grid   = [min(1.0, max(0.5, c)) for c in around(best_params_stage1["colsample_bytree"], [0.85, 1.0])]
    lam_grid   = [max(0.001, l) for l in around(best_params_stage1["lambda"], [0.5, 1.0, 2.0])]
    alp_grid   = [max(0.0, a) for a in around(best_params_stage1["alpha"],  [0.5, 1.0, 2.0])]

    grid = []
    for eta, md, mcw, sub, col, lam, alp in product(
        eta_grid, md_grid, mcw_grid, sub_grid, col_grid, lam_grid, alp_grid
    ):
        grid.append({
            "eta": eta, "max_depth": md, "min_child_weight": mcw,
            "subsample": sub, "colsample_bytree": col,
            "lambda": lam, "alpha": alp, "tree_method": "hist"
        })

    trials = []
    for i, params in enumerate(grid, start=1):
        mean_f1, std_f1, per_fold = evaluate_xgb_params_cv(
            X, y, cv, params,
            pos_label=pos_label,
            optimize_max_f1=optimize_max_f1
        )
        row = dict(params)
        row.update({
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "mean_recall": per_fold["recall"].mean(),
            "mean_precision": per_fold["precision"].mean(),
            "mean_balanced_accuracy": per_fold["balanced_accuracy"].mean(),
            "mean_roc_auc": per_fold["roc_auc"].mean(),
            "mean_best_it": per_fold["best_iteration"].mean(),
        })
        trials.append(row)

    results = pd.DataFrame(trials).sort_values("mean_f1", ascending=False).reset_index(drop=True)
    best_params = results.iloc[0][["eta","max_depth","min_child_weight","subsample",
                                   "colsample_bytree","lambda","alpha","tree_method"]].to_dict()
    
    # NEW: print threshold for the refined best params
    _, _, per_fold_best = evaluate_xgb_params_cv(
        X, y, cv, best_params,
        pos_label=pos_label,
        optimize_max_f1=optimize_max_f1
    )
    thr_med = float(per_fold_best["thr"].median())
    thr_list = [float(t) for t in per_fold_best["thr"]]
    print(f"\nChosen threshold (median across outer folds): {thr_med:.4f}")
    print("Per-fold thresholds:", ", ".join(f"{t:.4f}" for t in thr_list))
    return best_params, results

def retrain_xgb_model(X_train_encoded, y_train, X_test_encoded, y_test, threshold, eta, max_depth,min_child_weight, subsample,
                      colsample_bytree, lambda_param, alpha,
    pos_label=1, valid_size=0.15, random_state=42):
    # --- validation split from TRAIN (no test leakage)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_encoded, y_train,
        test_size=valid_size, stratify=y_train, random_state=random_state
    )

     # --- DMatrix objects
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test_encoded)

    # --- imbalance-aware weight
    pos = int((y_tr == pos_label).sum())
    neg = int((y_tr != pos_label).sum())
    scale_pos_weight = neg / max(pos, 1)

    # --- params
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": eta,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "lambda": lambda_param,
        "scale_pos_weight": float(scale_pos_weight),
        "tree_method": "hist",
        "seed": random_state,
        "alpha": alpha
    }


    # --- train with early stopping
    evals = [(dvalid, "valid")]
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    best_it = getattr(bst, "best_iteration", None)
    if best_it is None:  # fallback if early stopping didn’t trigger
        best_it = bst.num_boosted_rounds() - 1

    # --- predict on TEST using best_iteration
    # --- predict (probabilities for ROC–AUC; 0.5 threshold for labels)
    y_proba = bst.predict(dtest, iteration_range=(0, best_it + 1))
    y_pred  = (y_proba >= threshold).astype(int)

    # Render CM at tuned threshold
    cm_fig = render_confusion_matrix(
        y_true=y_test, y_proba=y_proba, threshold=threshold
    )

    # --- metrics (focus on positive "Churn" class)
    acc     = (y_pred == y_test).mean()
    prec    = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    rec     = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1      = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # --- table-ready row (rounded)
    row = {
        "Confusion matrix": cm_fig,
        "Model": "XGBoost",
        "Accuracy": round(float(acc), 3),
        "Precision": round(float(prec), 3),
        "Recall": round(float(rec), 3),
        "F1-score": round(float(f1), 3),
        "ROC-AUC": round(float(roc_auc), 3),
        "Balanced Acc.": round(float(bal_acc), 3)
    }

    return row, bst, params

def display_xgb_metrics(metrics_tab, metrics):
    df = pd.DataFrame([metrics])

    metric_cols = ["Accuracy", "Balanced Acc.", "Precision", "Recall", "F1-score", "ROC-AUC"]
    cols = [c for c in metric_cols if c in df.columns]
    df = df[cols].round(3)


    with metrics_tab:
        st.markdown("#### Metrics")
        st.dataframe(df.round(3), use_container_width=True, hide_index=True)

        # ---------- Confusion matrix ----------
        cm_fig = metrics.get("Confusion matrix", None)
        column1, column2 = st.columns([1, 1])
        column1.pyplot(cm_fig)

    return column2

def get_telco_metrics_caption(column2):

    column2.caption(body="<br><br>The tuned XGBoost model shows balanced performance at the " \
    "chosen operating threshold. On the test set it produced 814 true negatives, 221 false " \
    "positives, 105 false negatives, and 269 true positives (positive = churn). This corresponds to " \
    "accuracy 0.77 overall. Given class imbalance, balanced accuracy 0.75 is more informative: it " \
    "reflects recall 0.72 for churners and roughly 0.79 specificity for non-churners. At this operating " \
    "point, precision 0.55 indicates that a little over half of customers flagged as churners actually " \
    "churn, while the F1-score 0.62 summarizes the precision–recall trade-off. The ROC-AUC 0.84 suggests " \
    "good ranking ability independent of threshold, indicating that XGBoost separates churners from non-" \
    "churners reasonably well, though not as strongly as the Random Forest in your results.", unsafe_allow_html=True)

def get_internet_metrics_caption(column2):

    column2.caption(body="<br><br>After tuning, XGBoost reaches accuracy = 0.913 and balanced accuracy = 0.911, showing similar sensitivity to both classes (little " \
    "bias toward the majority class). The model achieves precision = 0.920 and recall = 0.924, so about 92% of actual" \
    " churners are correctly detected while 92% of churn predictions are correct, giving an F1-score of 0.922. Its " \
    "ROC–AUC = 0.965 indicates excellent ranking ability—the strongest discrimination among the models.<br><br>From " \
    "the confusion matrix: TP = 7,402, TN = 5,725, FP = 644 (false-positive rate ≈ 10.1%), and FN = 608 (false-negative " \
    "rate ≈ 7.6%), corresponding to TPR/Recall = 92.4% and TNR = 89.9%. The closeness of accuracy and balanced accuracy " \
    "confirms the dataset is fairly balanced and that TPR and TNR are well matched. Overall, the tuned XGBoost provides a " \
    "slightly better precision–recall trade-off and the highest discriminative power, making it the top choice for churn " \
    "detection in this experiment.", unsafe_allow_html=True)