# First Purchase Propensity Model

## Overview

This repository contains an open-source machine learning pipeline for predicting which users are most likely to make their first purchase or first recharge.

The model generates user-level probability scores that can be used for CRM targeting, marketing segmentation, remarketing audiences, growth experiments, and conversion analysis.

The project is fully generic and does not depend on any company-specific data or naming.

---

## What This Project Does

The pipeline takes two input files:

1. A training dataset with historical users and a conversion label
2. A prediction dataset with users who need to be scored

It then trains a propensity model and exports:

- All scored users
- Top X% highest-propensity users
- Model performance metrics
- Feature importance report
- Train/Test/Predict percentile distribution
- Train/Test/Predict decile distribution
- Threshold analysis
- Feature-level analysis by probability score bucket
- Saved model file for future scoring

---

## Repository Structure

```text
first-purchase-propensity-model/
│
├── data/
│   ├── sample_train.csv
│   └── sample_predict.csv
│
├── models/
│   └── propensity_model.pkl
│
├── notebooks/
│   └── README.md
│
├── outputs/
│   ├── all_predicted_users.csv
│   ├── top_10pct_users.csv
│   ├── model_analysis_report.xlsx
│   └── run_summary.json
│
├── src/
│   ├── __init__.py
│   ├── generate_sample_data.py
│   ├── train.py
│   └── predict.py
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## Input Data

### Training File

Default path:

```text
data/sample_train.csv
```

Required columns:

```text
user_id
converted
```

`converted` is the target variable.

```text
0 = user did not purchase
1 = user purchased
```

The training file can include any mix of numeric and categorical features.

Example columns:

```text
user_id
signup_day
signup_hour
country
city
gender
age
device_type
acquisition_source
campaign_group
lifetime_sessions
sessions_last_7d
avg_session_minutes
notification_views
notification_clicks
used_free_trial
viewed_pricing_page
added_payment_method
support_interaction
positive_interaction
negative_interaction
wallet_balance
converted
```

---

### Prediction File

Default path:

```text
data/sample_predict.csv
```

Required column:

```text
user_id
```

The prediction file should contain the same feature columns as the training file, except the target column `converted`.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/first-purchase-propensity-model.git
cd first-purchase-propensity-model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Generate Sample Data

This repository includes a script to generate mock sample data.

Run:

```bash
python src/generate_sample_data.py
```

This creates:

```text
data/sample_train.csv
data/sample_predict.csv
```

---

## Train The Model

Run:

```bash
python src/train.py
```

By default, the script reads paths from `config.yaml`:

```yaml
paths:
  train_data: data/sample_train.csv
  predict_data: data/sample_predict.csv
  output_dir: outputs
  model_dir: models
```

After training, the outputs are saved inside:

```text
outputs/
models/
```

---

## Score New Users

After training, use the saved model to score another file:

```bash
python src/predict.py --input data/sample_predict.csv --output outputs/scored_new_users.csv
```

The script uses:

```text
models/propensity_model.pkl
```

by default.

---

## Output Files

### 1. all_predicted_users.csv

Contains every user from the prediction file with a probability score.

Example:

| user_id | probability_score | score_bucket |
|---|---:|---|
| 500101 | 0.9124 | Very High |
| 500102 | 0.7631 | High |
| 500103 | 0.3187 | Low |

---

### 2. top_10pct_users.csv

Contains the top 10% users ranked by probability score.

The export percentage can be changed in `config.yaml`:

```yaml
model:
  top_percent_to_export: 10
```

---

### 3. model_analysis_report.xlsx

The Excel report contains the following sheets:

#### Model_Metrics

High-level model performance summary.

Includes:

- Model name
- Train rows
- Test rows
- Prediction rows
- Test AUC
- Test accuracy at 0.5 threshold
- Test F1 score at 0.5 threshold

#### Feature_Importance

Shows the most important model features and their relative contribution.

#### Percentile_Distribution

Shows score distribution across train, test, and prediction datasets.

Includes:

- 0th percentile
- 1st percentile
- 5th percentile
- 10th percentile
- 20th percentile
- 25th percentile
- 50th percentile
- 75th percentile
- 80th percentile
- 90th percentile
- 95th percentile
- 99th percentile
- 100th percentile

#### Decile_Distribution

Splits users into 10 equal groups based on probability score.

For train and test datasets, it includes actual conversions and conversion rate.

For the prediction dataset, it includes user count and score range by decile.

#### Threshold_Analysis

Shows model performance at different probability thresholds.

Thresholds included:

```text
0.10
0.20
0.30
0.40
0.50
0.60
0.70
0.80
0.90
```

Metrics included:

- Accuracy
- Precision
- Recall
- F1 Score
- True Positives
- False Positives
- True Negatives
- False Negatives

#### Feature_Bucket_Analysis

Analyzes features by probability score bucket.

Score buckets:

```text
Very High: 0.80 - 1.00
High:      0.60 - 0.80
Medium:    0.40 - 0.60
Low:       0.20 - 0.40
Very Low:  0.00 - 0.20
```

For numeric features, it shows average and median values by bucket.

For categorical features, it shows top category values by bucket.

---

## Model Details

The default model is a Scikit-learn Random Forest baseline so the repository runs easily in any Python environment. You can replace it with Random Forest, XGBoost, CatBoost, or any other classifier that supports `predict_proba`.

The preprocessing pipeline handles:

- Missing numeric values using median imputation
- Missing categorical values using `unknown`
- One-hot encoding for categorical features
- Standard scaling for numeric features
- Train/test split with stratification
- Probability scoring

---

## Example Workflow

```text
Raw user data
    ↓
Sample data generation
    ↓
Preprocessing
    ↓
Train/test split
    ↓
Model training
    ↓
Model evaluation
    ↓
Probability scoring
    ↓
User ranking
    ↓
Excel report and CSV exports
```

---

## Business Use Cases

This model can be used for:

- First purchase prediction
- First recharge prediction
- CRM campaign targeting
- Push notification targeting
- Paid remarketing audience creation
- User segmentation
- Conversion funnel analysis
- Growth experiment prioritization

---

## How To Use With Your Own Data

Replace the sample files with your own files:

```text
data/sample_train.csv
data/sample_predict.csv
```

Or update `config.yaml` with your file names:

```yaml
paths:
  train_data: data/your_train_file.csv
  predict_data: data/your_prediction_file.csv
```

Make sure:

1. The training file has a target column.
2. The prediction file does not need the target column.
3. Both files have the same feature columns.
4. The user ID column is present in both files.
5. Column names match the values configured in `config.yaml`.

Target column configuration:

```yaml
data:
  user_id_col: user_id
  target_col: converted
```

---

## Future Improvements

Possible enhancements:

- SHAP explainability
- MLflow experiment tracking
- Hyperparameter tuning
- Model monitoring
- API deployment
- Batch scoring scheduler
- Drift detection
- Champion/challenger model comparison

---

## Disclaimer

The included sample data is synthetic and does not contain real user information.

This project is intended as a reusable open-source machine learning template for propensity modeling and user conversion prediction.

---

## License

MIT License
