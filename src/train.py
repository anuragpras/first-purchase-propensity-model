"""Train an open-source first-purchase propensity model and export analysis reports."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

HAS_LIGHTGBM = False


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model(random_state: int):
    return RandomForestClassifier(n_estimators=80, max_depth=8, random_state=random_state, n_jobs=-1, class_weight="balanced")


def make_pipeline(X: pd.DataFrame, random_state: int) -> Pipeline:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", get_model(random_state)),
    ])


def safe_qcut(series: pd.Series, q: int = 10, labels=None):
    ranked = series.rank(method="first", ascending=True)
    return pd.qcut(ranked, q=q, labels=labels, duplicates="drop")


def add_score_buckets(df: pd.DataFrame, score_col: str = "probability_score") -> pd.DataFrame:
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    df = df.copy()
    df["score_bucket"] = pd.cut(df[score_col], bins=bins, labels=labels, include_lowest=True)
    return df


def decile_summary(scores: pd.Series, actual: pd.Series | None = None, dataset_name: str = "dataset") -> pd.DataFrame:
    df = pd.DataFrame({"probability_score": scores})
    if actual is not None:
        df["actual"] = actual.values if hasattr(actual, "values") else actual
    df["decile"] = 10 - safe_qcut(df["probability_score"], q=10, labels=False)

    agg = {
        "probability_score": ["count", "min", "max", "mean"],
    }
    if actual is not None:
        agg["actual"] = ["sum", "mean"]

    out = df.groupby("decile", observed=False).agg(agg).reset_index()
    out.columns = ["decile", "users", "min_score", "max_score", "avg_score"] + (["conversions", "conversion_rate"] if actual is not None else [])
    out.insert(0, "dataset", dataset_name)
    return out.sort_values("decile")


def percentile_distribution(scores: pd.Series, dataset_name: str) -> pd.DataFrame:
    percentiles = [0, 1, 5, 10, 20, 25, 50, 75, 80, 90, 95, 99, 100]
    vals = np.percentile(scores, percentiles)
    return pd.DataFrame({"dataset": dataset_name, "percentile": percentiles, "score": np.round(vals, 6)})


def threshold_analysis(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        preds = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, preds),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1_score": f1_score(y_true, preds, zero_division=0),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        })
    return pd.DataFrame(rows)


def feature_importance(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array(X.columns)

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        # Fallback mock-like importance from transformed feature variance when model has no native importance.
        Xt = preprocessor.transform(X)
        importance = np.var(Xt, axis=0)

    out = pd.DataFrame({"feature": feature_names, "importance": importance})
    out["base_feature"] = out["feature"].str.replace(r"^(num|cat)__", "", regex=True).str.split("_").str[0]
    out = out.groupby("base_feature", as_index=False)["importance"].sum()
    total = out["importance"].sum()
    out["importance_percent"] = np.where(total > 0, out["importance"] / total * 100, 0).round(2)
    return out.sort_values("importance_percent", ascending=False).drop(columns="importance")


def feature_bucket_analysis(df: pd.DataFrame, features: list[str], score_col: str) -> pd.DataFrame:
    df = add_score_buckets(df, score_col)
    rows = []
    for feature in features:
        if feature not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[feature]):
            temp = df.groupby("score_bucket", observed=False).agg(
                users=(score_col, "count"),
                avg_score=(score_col, "mean"),
                feature_avg=(feature, "mean"),
                feature_median=(feature, "median"),
            ).reset_index()
            temp.insert(0, "feature", feature)
            temp.insert(1, "analysis_type", "numeric")
        else:
            temp = df.groupby(["score_bucket", feature], observed=False).agg(
                users=(score_col, "count"),
                avg_score=(score_col, "mean"),
            ).reset_index().sort_values(["score_bucket", "users"], ascending=[True, False])
            temp = temp.groupby("score_bucket", observed=False).head(3)
            temp = temp.rename(columns={feature: "top_value"})
            temp.insert(0, "feature", feature)
            temp.insert(1, "analysis_type", "categorical_top_values")
        rows.append(temp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(config_path: str):
    cfg = load_config(config_path)
    random_state = cfg["project"]["random_state"]
    train_path = ROOT / cfg["paths"]["train_data"]
    predict_path = ROOT / cfg["paths"]["predict_data"]
    output_dir = ROOT / cfg["paths"]["output_dir"]
    model_dir = ROOT / cfg["paths"]["model_dir"]
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    user_id_col = cfg["data"]["user_id_col"]
    target_col = cfg["data"]["target_col"]

    train_df = pd.read_csv(train_path)
    predict_df = pd.read_csv(predict_path)

    features = [c for c in train_df.columns if c not in [user_id_col, target_col]]
    X = train_df[features].copy()
    y = train_df[target_col].copy()
    X_predict = predict_df[features].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=random_state, stratify=y
    )

    pipe = make_pipeline(X_train, random_state)
    pipe.fit(X_train, y_train)

    train_scores = pipe.predict_proba(X_train)[:, 1]
    test_scores = pipe.predict_proba(X_test)[:, 1]
    predict_scores = pipe.predict_proba(X_predict)[:, 1]

    auc_score = roc_auc_score(y_test, test_scores)
    metrics = pd.DataFrame([{
        "model": "RandomForest open-source baseline",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "prediction_rows": len(X_predict),
        "test_auc": round(auc_score, 4),
        "test_accuracy_at_0_5": round(accuracy_score(y_test, (test_scores >= 0.5).astype(int)), 4),
        "test_f1_at_0_5": round(f1_score(y_test, (test_scores >= 0.5).astype(int)), 4),
    }])

    all_predictions = predict_df[[user_id_col]].copy()
    all_predictions["probability_score"] = predict_scores.round(6)
    all_predictions = add_score_buckets(all_predictions, "probability_score")
    all_predictions = all_predictions.sort_values("probability_score", ascending=False)

    top_pct = cfg["model"]["top_percent_to_export"]
    top_n = max(1, int(len(all_predictions) * top_pct / 100))
    top_predictions = all_predictions.head(top_n)

    train_dist = percentile_distribution(pd.Series(train_scores), "train")
    test_dist = percentile_distribution(pd.Series(test_scores), "test")
    predict_dist = percentile_distribution(pd.Series(predict_scores), "predict")
    percentile_df = pd.concat([train_dist, test_dist, predict_dist], ignore_index=True)

    train_deciles = decile_summary(pd.Series(train_scores), y_train.reset_index(drop=True), "train")
    test_deciles = decile_summary(pd.Series(test_scores), y_test.reset_index(drop=True), "test")
    predict_deciles = decile_summary(pd.Series(predict_scores), None, "predict")
    decile_df = pd.concat([train_deciles, test_deciles, predict_deciles], ignore_index=True)

    feat_imp = feature_importance(pipe, X_train)

    scored_predict_for_analysis = predict_df.copy()
    scored_predict_for_analysis["probability_score"] = predict_scores
    bucket_analysis = feature_bucket_analysis(scored_predict_for_analysis, features, "probability_score")

    threshold_df = threshold_analysis(y_test.reset_index(drop=True), test_scores)

    all_predictions.to_csv(output_dir / "all_predicted_users.csv", index=False)
    top_predictions.to_csv(output_dir / f"top_{top_pct}pct_users.csv", index=False)
    joblib.dump(pipe, model_dir / "propensity_model.pkl")

    with pd.ExcelWriter(output_dir / "model_analysis_report.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Model_Metrics", index=False)
        feat_imp.to_excel(writer, sheet_name="Feature_Importance", index=False)
        percentile_df.to_excel(writer, sheet_name="Percentile_Distribution", index=False)
        decile_df.to_excel(writer, sheet_name="Decile_Distribution", index=False)
        threshold_df.to_excel(writer, sheet_name="Threshold_Analysis", index=False)
        bucket_analysis.to_excel(writer, sheet_name="Feature_Bucket_Analysis", index=False)

    summary = {
        "model": metrics.iloc[0]["model"],
        "test_auc": float(metrics.iloc[0]["test_auc"]),
        "outputs": [
            "outputs/all_predicted_users.csv",
            f"outputs/top_{top_pct}pct_users.csv",
            "outputs/model_analysis_report.xlsx",
            "models/propensity_model.pkl",
        ],
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Model run complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    main(args.config)
