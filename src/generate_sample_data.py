"""Generate synthetic train and prediction datasets for the propensity model."""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def make_dataset(n: int, include_target: bool = True, start_id: int = 100000) -> pd.DataFrame:
    countries = ["India", "United States", "United Kingdom", "Canada", "Australia", "UAE"]
    devices = ["Android", "iOS", "Web"]
    sources = ["organic", "google", "meta", "email", "referral", "affiliate"]
    campaigns = ["brand", "generic", "remarketing", "lookalike", "crm_push", "none"]
    cities = ["Delhi", "Mumbai", "Bengaluru", "Pune", "Hyderabad", "Chennai", "London", "Dubai"]

    df = pd.DataFrame({
        "user_id": np.arange(start_id, start_id + n),
        "signup_day": np.random.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n),
        "signup_hour": np.random.randint(0, 24, n),
        "country": np.random.choice(countries, n, p=[0.72, 0.08, 0.05, 0.04, 0.04, 0.07]),
        "city": np.random.choice(cities, n),
        "gender": np.random.choice(["Male", "Female", "Unknown"], n, p=[0.52, 0.44, 0.04]),
        "age": np.clip(np.random.normal(31, 8, n).round(), 18, 65),
        "device_type": np.random.choice(devices, n, p=[0.68, 0.24, 0.08]),
        "acquisition_source": np.random.choice(sources, n, p=[0.28, 0.22, 0.18, 0.14, 0.10, 0.08]),
        "campaign_group": np.random.choice(campaigns, n),
        "lifetime_sessions": np.random.poisson(5, n),
        "sessions_last_7d": np.random.poisson(2, n),
        "avg_session_minutes": np.round(np.random.gamma(2.2, 3.0, n), 2),
        "notification_views": np.random.poisson(3, n),
        "notification_clicks": np.random.binomial(8, 0.12, n),
        "used_free_trial": np.random.binomial(1, 0.42, n),
        "viewed_pricing_page": np.random.binomial(1, 0.30, n),
        "added_payment_method": np.random.binomial(1, 0.12, n),
        "support_interaction": np.random.binomial(1, 0.08, n),
        "positive_interaction": np.random.binomial(1, 0.22, n),
        "negative_interaction": np.random.binomial(1, 0.06, n),
        "wallet_balance": np.round(np.random.exponential(80, n), 2),
    })

    signal = (
        -3.2
        + 0.18 * df["sessions_last_7d"]
        + 0.09 * df["lifetime_sessions"]
        + 0.06 * df["avg_session_minutes"]
        + 1.15 * df["viewed_pricing_page"]
        + 1.35 * df["added_payment_method"]
        + 0.65 * df["used_free_trial"]
        + 0.45 * df["positive_interaction"]
        - 0.55 * df["negative_interaction"]
        + 0.18 * (df["acquisition_source"].isin(["google", "email"]).astype(int))
        + 0.25 * (df["campaign_group"].isin(["remarketing", "crm_push"]).astype(int))
    )
    prob = 1 / (1 + np.exp(-signal))
    if include_target:
        df["converted"] = np.random.binomial(1, prob)
    return df


if __name__ == "__main__":
    train = make_dataset(2500, include_target=True, start_id=100000)
    predict = make_dataset(800, include_target=False, start_id=500000)
    train.to_csv(DATA_DIR / "sample_train.csv", index=False)
    predict.to_csv(DATA_DIR / "sample_predict.csv", index=False)
    print(f"Saved: {DATA_DIR / 'sample_train.csv'}")
    print(f"Saved: {DATA_DIR / 'sample_predict.csv'}")
