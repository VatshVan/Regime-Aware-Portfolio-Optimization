from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

META_PREDICTIONS_PATH = PROCESSED_DIR / "meta_models" / "meta_predictions.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "models" / "predictions.parquet"

DIAGNOSTICS_DIR = PROCESSED_DIR / "diagnostics"
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_REPORT_PATH = DIAGNOSTICS_DIR / "final_report.json"
SHAP_SUMMARY_PATH = DIAGNOSTICS_DIR / "shap_summary.csv"

# Optional: Number of samples to compute SHAP values for (to save time)
SHAP_SAMPLES = 5000
