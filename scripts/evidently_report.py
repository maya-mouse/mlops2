import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Шляхи до файлів
ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "reference_stats.joblib"
REPORT_OUTPUT = ROOT / "drift_report.html"

def generate_report():
    if not REFERENCE_PATH.exists():
        print("Помилка: Файл reference_stats.joblib не знайдено. Спочатку запустіть ml/train.py")
        return

    ref_data = joblib.load(REFERENCE_PATH)
    ref_df = pd.DataFrame(ref_data["X"], columns=ref_data["feature_names"])


    rng = np.random.default_rng(42)
    current_df = ref_df.copy().sample(n=100, random_state=42, replace=True)
    current_df["petal_length"] = current_df["petal_length"] + 1.5 
    current_df["sepal_width"] = current_df["sepal_width"] - 0.8

    report = Report(metrics=[
        DataDriftPreset(),
    ])

    print("Генерація звіту... зачекайте.")
    report.run(reference_data=ref_df, current_data=current_df)
    
    report.save_html(str(REPORT_OUTPUT))
    print(f"Звіт успішно збережено: {REPORT_OUTPUT}")

if __name__ == "__main__":
    generate_report()