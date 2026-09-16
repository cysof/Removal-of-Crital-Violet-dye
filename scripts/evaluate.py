"""
evaluate.py
-----------
Loads the held-out test set (never seen during training/tuning) and
evaluates all 3 trained models using the four metrics specified in
thesis section 3.3.4: R2, RMSE, MAE, and MAPE.

Also produces a predicted-vs-actual scatter plot per model, saved to
outputs/, useful for the results chapter.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import joblib

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "SVR": "svr.joblib",
    "ANN": "ann.joblib",
}


def load_test_data():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").iloc[:, 0]
    return X_test, y_test


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def plot_predicted_vs_actual(y_true, y_pred, model_name, ax):
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1, label="Ideal (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Removal Efficiency (%)")
    ax.set_ylabel("Predicted Removal Efficiency (%)")
    ax.set_title(model_name)
    ax.legend(fontsize=8)


def main():
    X_test, y_test = load_test_data()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, fname) in zip(axes, MODEL_FILES.items()):
        model_path = MODELS_DIR / fname
        if not model_path.exists():
            print(f"WARNING: {model_path} not found, skipping {name}")
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        metrics["model"] = name
        results.append(metrics)

        plot_predicted_vs_actual(y_test, y_pred, name, ax)

        print(f"\n{name}:")
        for k, v in metrics.items():
            if k != "model":
                print(f"  {k}: {v:.4f}")

    plt.tight_layout()
    fig_path = OUTPUTS_DIR / "predicted_vs_actual.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    results_df = pd.DataFrame(results)[["model", "R2", "RMSE", "MAE", "MAPE (%)"]]
    results_df = results_df.sort_values("R2", ascending=False).reset_index(drop=True)

    metrics_path = OUTPUTS_DIR / "test_set_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    print("\n=== Test Set Performance Summary (sorted by R2) ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {fig_path}")
    print(f"\nBest model: {results_df.iloc[0]['model']}")


if __name__ == "__main__":
    main()