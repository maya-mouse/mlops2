import numpy as np
from scipy import stats

class DriftDetector:
    def __init__(self, reference: np.ndarray, feature_names: list[str]):
        if reference.ndim != 2:
            raise ValueError("Reference data must be 2D")
        self.reference = reference
        self.feature_names = feature_names

    def detect(self, current: np.ndarray, alpha: float = 0.05) -> dict:
        if current.shape[1] != self.reference.shape[1]:
            raise ValueError("Feature count mismatch")

        per_feature = {}
        drifted = []

        for i, name in enumerate(self.feature_names):
            ref_col = self.reference[:, i]
            cur_col = current[:, i]
            
            # Тест Колмогорова-Смирнова
            ks_stat, p_value = stats.ks_2samp(ref_col, cur_col)
            is_drift = bool(p_value < alpha)
            
            per_feature[name] = {
                "statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": is_drift
            }
            if is_drift:
                drifted.append(name)

        return {
            "drift_detected": len(drifted) > 0,
            "n_drifted_features": len(drifted),
            "drifted_features": drifted,
            "per_feature": per_feature,
            "n_samples": int(current.shape[0]),
            "alpha": alpha
        }