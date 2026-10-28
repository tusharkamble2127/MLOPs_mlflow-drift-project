
import os
import json

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Create a folder for generated reports
os.makedirs("reports", exist_ok=True)

# 1. Load the dataset
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target

print("Dataset loaded successfully")
print("Number of records:", len(X))
print("Number of features:", X.shape[1])
print("Target classes:", dataset.target_names)

# Save a dataset summary
summary = {
    "dataset_name": "Breast Cancer Wisconsin Diagnostic",
    "records": int(X.shape[0]),
    "features": int(X.shape[1]),
    "target_classes": dataset.target_names.tolist()
}

with open("reports/dataset_summary.json", "w") as file:
    json.dump(summary, file, indent=4)

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Configure MLflow
mlflow.set_experiment("Breast_Cancer_Drift_Experiment")

# 4. Create and train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42)
)

with mlflow.start_run(run_name="baseline_and_drift_analysis"):

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # 5. Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("feature_count", X.shape[1])

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("training_records", len(X_train))
    mlflow.log_metric("test_records", len(X_test))

    print(f"\nModel accuracy: {accuracy:.4f}")
    print(f"Model F1-score: {f1:.4f}")

    # Save the trained model in MLflow
    mlflow.sklearn.log_model(model, artifact_path="model")

    # 6. Create reference and simulated current data
    reference = X_test.copy()
    current = X_test.copy()

    # Deliberately simulate changes in two feature distributions
    current["mean radius"] = current["mean radius"] + 2.0
    current["mean texture"] = current["mean texture"] + 3.0

    # 7. Apply the KS test to every numeric feature
    results = []

    for feature in X.columns:
        statistic, p_value = ks_2samp(
            reference[feature],
            current[feature]
        )

        results.append({
            "feature": feature,
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < 0.05)
        })

    drift_df = pd.DataFrame(results)
    drift_path = "reports/drift_results.csv"
    drift_df.to_csv(drift_path, index=False)

    drift_count = int(drift_df["drift_detected"].sum())

    # 8. Log drift results in MLflow
    mlflow.log_param("drift_test", "Kolmogorov-Smirnov")
    mlflow.log_param("significance_level", 0.05)
    mlflow.log_param("current_data_type", "simulated")

    mlflow.log_metric("features_tested", len(drift_df))
    mlflow.log_metric("features_flagged", drift_count)

    mlflow.log_artifact(drift_path)
    mlflow.log_artifact("reports/dataset_summary.json")

    # 9. Create a drift visualization
    plot_df = drift_df.sort_values("p_value", ascending=True)

    plt.figure(figsize=(10, 9))
    plt.barh(plot_df["feature"], plot_df["p_value"])
    plt.axvline(
        x=0.05,
        linestyle="--",
        label="Significance threshold (0.05)"
    )
    plt.xlabel("KS test p-value")
    plt.ylabel("Feature")
    plt.title("Feature Drift Detection Results")
    plt.legend()
    plt.tight_layout()

    plot_path = "reports/drift_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    mlflow.log_artifact(plot_path)

    # 10. Print the final report
    print("\n--- Drift Analysis Results ---")
    print("Features tested:", len(drift_df))
    print("Features flagged:", drift_count)
    print("\nFlagged features:")

    print(
        drift_df.loc[
            drift_df["drift_detected"],
            ["feature", "ks_statistic", "p_value"]
        ].to_string(index=False)
    )

    print("\nReports saved in the reports folder.")
    print("MLflow experiment run completed successfully.")
