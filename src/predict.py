"""Score a new dataset using the trained propensity model."""
from pathlib import Path
import argparse
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(input_file: str, model_file: str, output_file: str, user_id_col: str = "user_id"):
    model = joblib.load(model_file)
    df = pd.read_csv(input_file)
    features = [c for c in df.columns if c != user_id_col]
    scores = model.predict_proba(df[features])[:, 1]

    out = df[[user_id_col]].copy() if user_id_col in df.columns else pd.DataFrame(index=df.index)
    out["probability_score"] = scores.round(6)
    out = out.sort_values("probability_score", ascending=False)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    print(f"Saved scored users to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/sample_predict.csv"))
    parser.add_argument("--model", default=str(ROOT / "models/propensity_model.pkl"))
    parser.add_argument("--output", default=str(ROOT / "outputs/scored_new_users.csv"))
    parser.add_argument("--user-id-col", default="user_id")
    args = parser.parse_args()
    main(args.input, args.model, args.output, args.user_id_col)
