"""Remove sample-level data from the model bundles before publishing.

The training package stores its test set (sample ids, lab values, predictions)
and PLS training scores inside each bundle. The app needs none of it.
Run after every model sync:  python tools/strip_models.py
"""

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # soilspec classes for unpickling

DROP_KEYS = ("ks_train_idx", "ks_test_idx", "test_ids", "test_obs", "test_pred", "test_source")
DROP_ATTRS = ("x_scores_", "y_scores_", "_x_scores", "_y_scores")   # PLS training scores


def strip(path: Path) -> bool:
    b = joblib.load(path)
    if not isinstance(b, dict):
        return False
    changed = [k for k in DROP_KEYS if b.pop(k, None) is not None]
    inner = getattr(b.get("model"), "model", None)
    for a in DROP_ATTRS:
        if inner is not None and a in vars(inner):
            delattr(inner, a)
            changed.append(a)
    if changed:
        joblib.dump(b, path)
    return bool(changed)


if __name__ == "__main__":
    files = sorted((ROOT / "models").glob("*/*.joblib"))
    n = sum(strip(f) for f in files)
    print(f"{n} of {len(files)} bundles stripped")
