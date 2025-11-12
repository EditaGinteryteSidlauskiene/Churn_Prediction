from pathlib import Path
import json, numpy as np

ART = Path("artifacts"); ART.mkdir(exist_ok=True)

def _path(dataset: str, model: str, run_id: str = "latest") -> Path:
    p = ART / dataset / model / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_shap(dataset: str, model: str, values: np.ndarray, base_values: np.ndarray,
              feature_names: list, row_index: list, run_id: str = "latest"):
    p = _path(dataset, model, run_id)
    np.savez_compressed(p / "shap.npz", values=values, base_values=base_values)
    (p / "meta.json").write_text(json.dumps({
        "feature_names": list(map(str, feature_names)),
        "row_index": list(map(str, row_index))
    }))

def load_shap(dataset: str, model: str, run_id: str = "latest"):
    p = _path(dataset, model, run_id)
    npz = np.load(p / "shap.npz", allow_pickle=False)
    meta = json.loads((p / "meta.json").read_text())
    return npz["values"], npz["base_values"], meta["feature_names"], meta["row_index"]
