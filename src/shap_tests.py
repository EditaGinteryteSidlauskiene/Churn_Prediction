import numpy as np, pandas as pd
from typing import Callable, Dict, Iterable, Tuple
from scipy.stats import spearmanr
from sklearn.utils import resample
from sklearn.base import clone


# --- helpers ---------------------------------------------------------------

def _predict_p1(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]

def _rankcorr(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = spearmanr(a, b)
    return float(rho)

def _mean_abs_by_feature(attribs: np.ndarray) -> np.ndarray:
    # attribs shape: [n_rows, n_features]
    return np.mean(np.abs(attribs), axis=0)

def _occlusion_row_drop(
    model, x_row: pd.Series, feats_sorted: list[str], baseline: pd.Series, k_list=(1,3,5)
) -> Dict[int, float]:
    """Return {k: (p_base - p_muted_topk)} for a single row."""
    def _mute(row, cols):
        r = row.copy()
        r[cols] = baseline[cols]
        return r

    x1 = x_row.to_frame().T
    p_base = float(_predict_p1(model, x1)[0])
    out = {}
    for k in k_list:
        cols = feats_sorted[:k]
        xm = _mute(x_row, cols).to_frame().T
        pm = float(_predict_p1(model, xm)[0])
        out[k] = p_base - pm
    return out

# --- FAITHFULNESS ----------------------------------------------------------

def faithfulness_suite(
    model,
    X: pd.DataFrame,                    # model-ready matrix
    get_attribs: Callable[[pd.DataFrame], np.ndarray],  # returns [n_rows, n_features]
    feature_names: Iterable[str],
    k_list=(1,3,5),
    n_rows_local: int = 20,
    baseline_mode: str = "mean",
) -> Dict:
    feats = list(feature_names)
    A = get_attribs(X)                       # [N, F]
    mean_abs_attr = _mean_abs_by_feature(A)  # [F]

    # Global permutation importance vs mean|attr|
    rng = np.random.RandomState(42)
    base_p = _predict_p1(model, X)
    perm_impacts = []
    for j, f in enumerate(feats):
        Xp = X.copy()
        Xp[f] = rng.permutation(Xp[f].values)
        p2 = _predict_p1(model, Xp)
        perm_impacts.append(np.mean(np.abs(base_p - p2)))
    perm_impacts = np.array(perm_impacts)
    rho_global = _rankcorr(mean_abs_attr, perm_impacts)

    # Per-row occlusion on a small sample
    Xs = X.sample(min(n_rows_local, len(X)), random_state=7)
    deltas_k = {k: [] for k in k_list}
    rhos_local = []
    baseline = (X.mean(axis=0) if baseline_mode=="mean"
                else X.median(axis=0) if baseline_mode=="median"
                else pd.Series(0.0, index=X.columns))

    A_s = get_attribs(Xs)
    for i, (_, row) in enumerate(Xs.iterrows()):
        attr_row = A_s[i, :]
        rank = np.argsort(-np.abs(attr_row))
        feats_sorted = [feats[r] for r in rank]

        # single-feature impacts for rank correlation
        p0 = float(_predict_p1(model, row.to_frame().T)[0])
        single_impacts = []
        for f in feats_sorted:
            xmut = row.copy()
            xmut[f] = baseline[f]
            p1 = float(_predict_p1(model, xmut.to_frame().T)[0])
            single_impacts.append(p0 - p1)
        rhos_local.append(_rankcorr(np.abs(attr_row), np.abs(single_impacts)))

        # cumulative top-k
        topk = _occlusion_row_drop(model, row, feats_sorted, baseline, k_list=k_list)
        for k in k_list:
            deltas_k[k].append(topk[k])

    return {
        "global_perm_rho": float(rho_global),
        "global_perm_mean_abs_delta": perm_impacts,
        "mean_abs_attr": mean_abs_attr,
        "local_occlusion_topk_mean": {k: float(np.mean(v)) for k, v in deltas_k.items()},
        "local_rank_rho_mean": float(np.mean(rhos_local)),
        "local_rank_rho_std": float(np.std(rhos_local)),
    }

# --- STABILITY -------------------------------------------------------------

def stability_suite(
    model,
    X: pd.DataFrame,
    get_attribs: Callable[[pd.DataFrame, int | None], np.ndarray],  # accepts optional seed
    seeds: Iterable[int] = (1,2,3,4,5),
    noise_sigma: float = 0.01,
    n_rows: int = 500,
) -> Dict:
    Xs = X.sample(min(n_rows, len(X)), random_state=123)

    # Across seeds
    bases = get_attribs(Xs, None)
    rho_seeds = []
    for s in seeds:
        A = get_attribs(Xs, s)
        rho_seeds.append(_rankcorr(_mean_abs_by_feature(bases), _mean_abs_by_feature(A)))

    # Noise robustness
    rng = np.random.RandomState(9)
    Xn = Xs.copy()
    num_cols = Xn.columns
    Xn[num_cols] = Xn[num_cols] + noise_sigma * rng.randn(*Xn[num_cols].shape)
    A_noise = get_attribs(Xn, None)
    rho_noise = _rankcorr(_mean_abs_by_feature(bases), _mean_abs_by_feature(A_noise))

    # Bootstrap robustness
    rho_boot = []
    for b in range(5):
        Xb = resample(Xs, replace=True, n_samples=len(Xs), random_state=100+b)
        A_b = get_attribs(Xb, None)
        rho_boot.append(_rankcorr(_mean_abs_by_feature(bases), _mean_abs_by_feature(A_b)))

    return {
        "seed_rank_rho_mean": float(np.mean(rho_seeds)),
        "seed_rank_rho_std": float(np.std(rho_seeds)),
        "noise_rank_rho": float(rho_noise),
        "bootstrap_rank_rho_mean": float(np.mean(rho_boot)),
        "bootstrap_rank_rho_std": float(np.std(rho_boot)),
    }

# --- SANITY ---------------------------------------------------------------

def sanity_suite(
    base_model,
    X: pd.DataFrame,
    y: pd.Series,
    get_attribs_for_model: Callable[[object, pd.DataFrame], np.ndarray],
    weak_learner_factory: Callable[[], object] | None = None,
) -> Dict:
    """
    Label randomization sanity via a lightweight surrogate (avoid retraining heavy models).
    Feature randomization sanity on a top feature.
    SHAP additivity residual (if SHAP-like attributions).
    """
    # Baseline attributions
    A0 = get_attribs_for_model(base_model, X)
    mean_abs0 = _mean_abs_by_feature(A0)

    # Label randomization: train a weak surrogate on shuffled y
    if weak_learner_factory is not None:
        y_shuf = y.sample(frac=1.0, random_state=77).reset_index(drop=True)
        X_shuf = X.reset_index(drop=True)
        surrogate = weak_learner_factory()
        surrogate.fit(X_shuf, y_shuf)
        A_shuf = get_attribs_for_model(surrogate, X_shuf)
        rho_label = _rankcorr(mean_abs0, _mean_abs_by_feature(A_shuf))
    else:
        rho_label = np.nan

    # Feature randomization on the strongest feature
    j_top = int(np.argmax(mean_abs0))
    f_top = X.columns[j_top]
    X_perm = X.copy()
    X_perm[f_top] = np.random.RandomState(13).permutation(X_perm[f_top].values)
    A_perm = get_attribs_for_model(base_model, X_perm)
    mean_abs_perm = _mean_abs_by_feature(A_perm)
    drop_ratio = float((mean_abs0[j_top] - mean_abs_perm[j_top]) / (mean_abs0[j_top] + 1e-12))

    # Additivity residual (works for SHAP: base + sum ≈ model output). Here we test prob space.
    p_hat = _predict_p1(base_model, X)
    # Approximate reconstruction quality using a linear probe:
    # correlation between per-row sum(|attr|) and |p_hat - p_hat.mean()|
    s_attr = np.sum(A0, axis=1)
    recon_rho = _rankcorr(np.abs(s_attr), np.abs(p_hat - p_hat.mean()))

    return {
        "label_randomization_rank_rho": float(rho_label),   # should be ~0 if sane
        "feature_randomization_top_feature": f_top,
        "feature_randomization_drop_ratio": drop_ratio,     # large positive drop is good
        "additivity_proxy_rho": float(recon_rho),           # higher is better (proxy)
    }

# --- ORCHESTRATOR ---------------------------------------------------------

def run_explainability_qa(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Iterable[str],
    get_attribs: Callable[[pd.DataFrame, int | None], np.ndarray],
    weak_learner_factory: Callable[[], object] | None = None,
    k_list=(1,3,5),
) -> Dict:
    """Returns a single dict with Faithfulness + Stability + Sanity summaries."""
    # wrappers to satisfy signatures
    def _ga(X_in: pd.DataFrame, seed: int | None = None) -> np.ndarray:
        # your get_attribs can ignore seed if deterministic
        return get_attribs(X_in, seed)

    def _for_model(m, X_in: pd.DataFrame) -> np.ndarray:
        # same attribution function but allowing model override (for surrogate)
        return get_attribs(X_in, None)

    out = {}
    out["faithfulness"] = faithfulness_suite(model, X, lambda Z: _ga(Z, None),
                                            feature_names, k_list=k_list)
    out["stability"]    = stability_suite(model, X, _ga)
    out["sanity"]       = sanity_suite(model, X, y, _for_model, weak_learner_factory)
    return out
