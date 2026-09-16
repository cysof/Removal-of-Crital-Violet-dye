"""
interpret.py
------------
Performs the two interpretability analyses described in thesis
sections 3.3.5 and 3.3.6, on the best-performing model (selected
automatically from outputs/test_set_metrics.csv, expected to be
Random Forest since it's the only tree-ensemble model in this study):

    1. Feature importance via Mean Decrease in Impurity (MDI),
       normalised to sum to 1, extracted directly from the trained
       tree ensemble.
    2. SHAP (SHapley Additive exPlanations) analysis: summary plot,
       a dependence plot for the top feature, and a waterfall plot
       for a single example prediction.

All plots are saved to outputs/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import shap

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# Model file lookup used to resolve which .joblib to load for a given
# "model" name in test_set_metrics.csv
MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "SVR": "svr.joblib",
    "ANN": "ann.joblib",
}

FEATURE_LABELS = {
    "Initial_Concentration_mgL": "Initial Concentration (mg/L)",
    "Adsorbent_Dosage_gL": "Adsorbent Dosage (g/L)",
    "pH": "pH",
    "Contact_Time_min": "Contact Time (min)",
    "Temperature_C": "Temperature (C)",
}


def pick_best_model():
    metrics_path = OUTPUTS_DIR / "test_set_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Run scripts/evaluate.py first to determine the best model.")
    df = pd.read_csv(metrics_path)
    best_row = df.sort_values("R2", ascending=False).iloc[0]
    return best_row["model"]


def feature_importance_analysis(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        print(f"Model of type {type(model).__name__} has no native feature_importances_ "
              f"(only tree ensembles do). Skipping MDI feature importance step.")
        return None

    importances = model.feature_importances_
    importances = importances / importances.sum()  # normalise to sum to 1

    fi_df = pd.DataFrame({
        "feature": [FEATURE_LABELS.get(f, f) for f in feature_names],
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi_df.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="#3B7A57")
    ax.set_xlabel("Normalised Feature Importance (Mean Decrease in Impurity)")
    ax.set_title("Feature Importance — Random Forest")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    print("\nFeature Importance (MDI, normalised):")
    print(fi_df.to_string(index=False))
    return fi_df


def shap_analysis(model, X_train, X_test, feature_names, top_feature):
    display_names = [FEATURE_LABELS.get(f, f) for f in feature_names]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    X_test_display = X_test.copy()
    X_test_display.columns = display_names

    # --- Summary plot (global) ---
    plt.figure()
    shap.summary_plot(shap_values, X_test_display, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Dependence plot for top feature ---
    top_display_name = FEATURE_LABELS.get(top_feature, top_feature)
    plt.figure()
    shap.dependence_plot(
        top_display_name, shap_values, X_test_display, show=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_dependence_top_feature.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Waterfall plot for a single example prediction (first test row) ---
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).flatten()[0]

    expl = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=X_test_display.iloc[0].values,
        feature_names=display_names,
    )
    plt.figure()
    shap.plots.waterfall(expl, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSHAP plots saved: shap_summary.png, shap_dependence_top_feature.png, "
          "shap_waterfall_example.png")


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    best_model_name = pick_best_model()
    print(f"Best-performing model (per test_set_metrics.csv): {best_model_name}")

    model_path = MODELS_DIR / MODEL_FILES[best_model_name]
    model = joblib.load(model_path)

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    feature_names = list(X_train.columns)

    fi_df = feature_importance_analysis(model, feature_names)

    if hasattr(model, "feature_importances_"):
        # SHAP TreeExplainer only applies cleanly to tree-based models
        top_feature = feature_names[np.argmax(model.feature_importances_)]
        shap_analysis(model, X_train, X_test, feature_names, top_feature)
    else:
        print("Skipping SHAP TreeExplainer analysis: best model is not tree-based. "
              "Use shap.KernelExplainer for SVR/ANN if SHAP analysis on those is needed.")


if __name__ == "__main__":
    main()