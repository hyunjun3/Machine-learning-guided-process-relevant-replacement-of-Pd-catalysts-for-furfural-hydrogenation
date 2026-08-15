"""Shared loading, plotting, and output helpers for all figure scripts."""

from __future__ import annotations

import os
import pickle
import random
import string
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from matplotlib import cm, rcParams
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
TUNED_MODEL_DIR = REPO_ROOT / "hyperparameter_tuning" / "output"
BOOTSTRAP_DIR = REPO_ROOT / "bootstrap_output"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "figures"
SEED = 23

# Legacy plotting sections use repository-relative paths. Normalizing the working
# directory here makes direct execution from any location deterministic.
os.chdir(REPO_ROOT)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def set_seed(seed: int = SEED) -> None:
    """Set the Python and NumPy random seeds used in the manuscript."""

    random.seed(seed)
    np.random.seed(seed)


def load_ml_context(seed: int = SEED) -> dict[str, object]:
    """Load the fixed data split, feature groups, and optimized XGBoost model."""

    set_seed(seed)
    dataset = pd.read_csv(DATASET_DIR / "preprocessed_dataset_for_ML.csv")

    active_metal = dataset.columns[
        dataset.columns.get_loc("Ca") : dataset.columns.get_loc("Zn") + 1
    ].tolist()
    cat_support = dataset.columns[
        dataset.columns.get_loc("Al2O3") : dataset.columns.get_loc("ZrO2") + 1
    ].tolist()
    precursor = dataset.columns[
        dataset.columns.get_loc("(NH4)2PdCl4") : dataset.columns.get_loc("Zn(NO3)2") + 1
    ].tolist()
    preparation = dataset.columns[
        dataset.columns.get_loc("Unknown_preparation") : dataset.columns.get_loc("wet impregnation") + 1
    ].tolist()
    solvent = dataset.columns[
        dataset.columns.get_loc("1,2-dichloroethane") : dataset.columns.get_loc("water") + 1
    ].tolist()

    float_cols = [
        "Reduction_temp",
        "Reduction_time",
        "Calcination_temp",
        "Calcination_time",
        "Furfural (mg)",
        "Catalyst amount (mg)",
        "Operating_temp",
        "Operating_pressure",
        "Operating_time",
        "Stirring rate (rpm)",
        "Substrate to metal ratio (mmol/mmol)",
        "Substrate concentration (mg/ml)",
        "THFA_yield (%)",
    ]

    X_train = pd.read_csv(DATASET_DIR / "ML_dataset_final_x_train.csv")
    y_train = pd.read_csv(DATASET_DIR / "ML_dataset_final_y_train.csv")
    X_test = pd.read_csv(DATASET_DIR / "ML_dataset_final_x_test.csv")
    y_test = pd.read_csv(DATASET_DIR / "ML_dataset_final_y_test.csv")

    model = XGBRegressor(random_state=seed, n_jobs=-1)
    model.load_model(TUNED_MODEL_DIR / f"xgb_model_seed_{seed}.json")
    y_test_pred = np.clip(model.predict(X_test), 0, 100)
    y_train_pred = np.clip(model.predict(X_train), 0, 100)

    return {
        "dataset": dataset,
        "active_metal": active_metal,
        "cat_support": cat_support,
        "precursor": precursor,
        "preparation": preparation,
        "solvent": solvent,
        "float_cols": float_cols,
        "SEED": seed,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "model": model,
        "y_test_pred": y_test_pred,
        "y_train_pred": y_train_pred,
        "XGB_train_r2": r2_score(y_train, y_train_pred),
        "XGB_test_r2": r2_score(y_test, y_test_pred),
        "XGB_train_rmse": root_mean_squared_error(y_train, y_train_pred),
        "XGB_test_rmse": root_mean_squared_error(y_test, y_test_pred),
    }


def save_figure(
    fig: mpl.figure.Figure,
    stem: str,
    *,
    output_dir: str | Path | None = None,
    dpi: int = 600,
) -> tuple[Path, Path]:
    """Save a figure as publication PNG and editable-text PDF."""

    root = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    png_dir = root / "png"
    pdf_dir = root / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    png_path = png_dir / f"{stem}.png"
    pdf_path = pdf_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {png_path.relative_to(REPO_ROOT)}")
    print(f"Saved {pdf_path.relative_to(REPO_ROOT)}")
    return png_path, pdf_path


__all__ = [name for name in globals() if not name.startswith("_")]
