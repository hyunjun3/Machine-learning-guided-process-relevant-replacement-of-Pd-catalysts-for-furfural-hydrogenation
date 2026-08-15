from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "hyperparameter_tuning" / "output"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


class ModelArtifactTests(unittest.TestCase):
    def test_primary_xgboost_model_is_present(self) -> None:
        model = MODEL_DIR / "xgb_model_seed_23.json"
        self.assertGreater(model.stat().st_size, 1_000_000)
        self.assertFalse(model.read_bytes()[:64].startswith(LFS_HEADER))

    def test_optimized_model_set_is_complete(self) -> None:
        expected = {
            "catboost_model_seed_23.cbm",
            "DT_model_seed23.pkl",
            "lightGBM_model_seed23.txt",
            "RF_model_seed23.pkl",
            "xgb_model_seed_23.json",
            "lr_model_seed_23.pkl",
            "lasso_model_seed_23.pkl",
            "ridge_model_seed_23.pkl",
            "svr_model_seed_23.pkl",
        }
        self.assertTrue(expected.issubset({p.name for p in MODEL_DIR.iterdir()}))


if __name__ == "__main__":
    unittest.main()

