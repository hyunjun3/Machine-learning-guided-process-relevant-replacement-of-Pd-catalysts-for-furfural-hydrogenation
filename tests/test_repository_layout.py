from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_fixed_dataset_files_exist(self) -> None:
        expected = {
            "preprocessed_dataset_for_ML.csv",
            "ML_dataset_final_x_train.csv",
            "ML_dataset_final_y_train.csv",
            "ML_dataset_final_x_test.csv",
            "ML_dataset_final_y_test.csv",
            "molar_mass.pickle",
        }
        self.assertTrue(expected.issubset({p.name for p in (ROOT / "dataset").iterdir()}))

    def test_fixed_split_dimensions(self) -> None:
        x_train = pd.read_csv(ROOT / "dataset" / "ML_dataset_final_x_train.csv")
        x_test = pd.read_csv(ROOT / "dataset" / "ML_dataset_final_x_test.csv")
        y_train = pd.read_csv(ROOT / "dataset" / "ML_dataset_final_y_train.csv")
        y_test = pd.read_csv(ROOT / "dataset" / "ML_dataset_final_y_test.csv")
        self.assertEqual(x_train.shape, (1956, 161))
        self.assertEqual(x_test.shape, (177, 161))
        self.assertEqual(y_train.shape, (1956, 1))
        self.assertEqual(y_test.shape, (177, 1))

    def test_current_figure_numbering(self) -> None:
        main = ROOT / "figures" / "main"
        supplementary = ROOT / "figures" / "supplementary"
        self.assertTrue((main / "figure_06d.py").exists())
        self.assertFalse((supplementary / "figure_s19.py").exists())
        for number in [1, 2, 3, 4, 5, 6, 7, 8, 17, 18]:
            self.assertTrue((supplementary / f"figure_s{number:02d}.py").exists())


if __name__ == "__main__":
    unittest.main()
