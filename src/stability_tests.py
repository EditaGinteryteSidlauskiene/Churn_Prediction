import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple, Callable
from scipy.stats import kendalltau, spearmanr

def _rank(arr: np.ndarray) -> np.ndarray:
    # higher attribution => higher rank (1 = most important)
    order = (-arr).argsort()
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(arr)+1)
    return ranks

def _topk_overlap(a_names: Iterable[str], b_names: Iterable[str], k: int) -> float:
    return len(set(list(a_names)[:k]).intersection(set(list(b_names)[:k]))) / float(k)

def _build_perturber(
    numeric_cols: Iterable[str],
    binary_cols: Iterable[str],
    onehot_groups: Dict[str, Iterable[str]]
) -> Callable[[pd.Series, np.random.Generator, float, float], pd.Series]:
    num_set = set(numeric_cols)
    bin_set = set(binary_cols)
    group_lists = {g: list(cols) for g, cols in onehot_groups.items()}

    def perturb(x_row: pd.Series, rng: np.random.Generator, noise_scale: float, flip_prob: float) -> pd.Series:
        x = x_row.copy()
        # 1) jitter numeric (small Gaussian)
        if num_set:
            noise = rng.normal(0.0, noise_scale, size=len(num_set))
            x.loc[list(num_set)] = x.loc[list(num_set)].to_numpy() + noise

        # 2) flip independent binaries
        for c in bin_set:
            if rng.random() < flip_prob and c in x.index:
                xv = x[c]
                # clamp to {0,1} if it looks binary
                if xv in (0,1) or (0.0 <= xv <= 1.0):
                    x[c] = 1 - round(float(xv))

        # 3) within each one-hot group, pick a different active category with small prob
        for _, cols in group_lists.items():
            cols = [c for c in cols if c in x.index]
            if not cols:
                continue
            if rng.random() < flip_prob:
                # set all to 0, choose one to 1
                x.loc[cols] = 0.0
                pick = rng.integers(0, len(cols))
                x.loc[cols[pick]] = 1.0

        return x
    return perturb

def local_stability_report_generic(
    get_attr_for_batch: Callable[[pd.DataFrame], np.ndarray],
    X_matrix: pd.DataFrame,
    row_id,
    numeric_cols: Iterable[str],
    binary_cols: Iterable[str],
    onehot_groups: Dict[str, Iterable[str]],
    n_perturb: int = 30,
    noise_scale: float = 0.02,
    flip_prob: float = 0.05,
    k_list: Tuple[int, ...] = (5, 10),
    seed: int = 42
) -> dict:
    """
    Compute local input-based stability for a single row (row_id) in X_matrix.
    - get_attr_for_batch: accepts a DataFrame (n, p) and returns attributions (n, p) aligned to X_matrix.columns
    """
    rng = np.random.default_rng(seed)
    x0 = X_matrix.loc[row_id]
    cols = X_matrix.columns.tolist()

    # baseline attribution on the original row (size 1 batch)
    A0 = get_attr_for_batch(pd.DataFrame([x0], columns=cols))[0]  # (p,)
    base_rank = _rank(A0)
    base_order = pd.Index(cols)[(-A0).argsort()].tolist()

    # build perturber
    perturb = _build_perturber(numeric_cols, binary_cols, onehot_groups)

    # create batch of perturbed rows
    pert_rows = []
    for _ in range(n_perturb):
        pert_rows.append(perturb(x0, rng, noise_scale=noise_scale, flip_prob=flip_prob))
    X_pert = pd.DataFrame(pert_rows, columns=cols)

    # attributions for the whole batch (fast)
    A = get_attr_for_batch(X_pert)  # (n_perturb, p)

    # metrics
    taus, rhos = [], []
    std_vec = A.std(axis=0)
    overlaps = {k: [] for k in k_list}

    for i in range(n_perturb):
        Ai = A[i]
        ri = _rank(Ai)
        # Kendall's tau / Spearman vs baseline ranks
        taus.append(kendalltau(base_rank, ri, nan_policy='omit').correlation)
        rhos.append(spearmanr(base_rank, ri, nan_policy='omit').correlation)

        order_i = pd.Index(cols)[(-Ai).argsort()].tolist()
        for k in k_list:
            overlaps[k].append(_topk_overlap(base_order, order_i, k))

    report = {
        "kendall_tau_mean": float(np.nanmean(taus)),
        "spearman_rho_mean": float(np.nanmean(rhos)),
        "mean_attr_std": float(np.mean(std_vec)),
        "topk_overlap_mean": {k: float(np.mean(v)) for k, v in overlaps.items()},
        "attr_std_by_feature": pd.Series(std_vec, index=cols).sort_values(ascending=False),
        "base_top10": base_order[:10],
    }
    return report



