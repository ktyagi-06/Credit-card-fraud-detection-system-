# Credit Card Fraud Detection (ML)

An end-to-end machine learning system for detecting fraudulent credit card transactions. Built around the class-imbalance problem that defines real-world fraud detection: fraud is typically <1% of transactions, so accuracy is a useless metric and the whole pipeline — sampling, model choice, evaluation — has to be designed around that.

## Highlights

- Reproducible pipeline: data loading → preprocessing → feature engineering → training → evaluation
- Handles severe class imbalance (SMOTE, class weighting, threshold tuning)
- Multiple models benchmarked: Logistic Regression, Random Forest, XGBoost, and a simple Neural Net (Keras)
- Evaluation focused on imbalance-appropriate metrics: PR-AUC, ROC-AUC, F1, recall at fixed precision, confusion matrices
- Config-driven (`config.yaml`) — no hardcoded paths or hyperparameters in code
- Synthetic data generator included so the project runs out of the box without needing to download anything
- Ships with a trained-model inference script and a minimal Streamlit demo app
- Unit tests for the preprocessing pipeline
- GitHub Actions CI workflow for linting + tests

## Dataset

This project is built against the schema of the well-known [Kaggle "Credit Card Fraud Detection" dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (European cardholders, Sept 2013): `Time`, `Amount`, `Class` (0/1), and 28 PCA-anonymized features `V1`–`V28`.

Two ways to get data:

1. **Real dataset (recommended):** download `creditcard.csv` from Kaggle and place it at `data/creditcard.csv`.
2. **Synthetic fallback:** run `python src/generate_synthetic_data.py` to generate a realistic synthetic dataset with the same schema and a ~0.17% fraud rate, so you can run the full pipeline immediately without Kaggle access.

## Project structure

```
fraud-detection-ml/
├── config.yaml                  # all paths, hyperparameters, thresholds
├── requirements.txt
├── data/                        # creditcard.csv goes here (gitignored)
├── models/                      # trained model artifacts land here (gitignored)
├── notebooks/
│   └── 01_eda.ipynb             # exploratory data analysis
├── src/
│   ├── generate_synthetic_data.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py                 # trains & saves all models
│   ├── evaluate.py              # generates metrics report + plots
│   └── predict.py               # inference on new transactions
├── app.py                       # Streamlit demo
├── tests/
│   └── test_preprocessing.py
└── .github/workflows/ci.yml
```

## Setup

```bash
git clone https://github.com/<your-username>/fraud-detection-ml.git
cd fraud-detection-ml
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Get data (pick one)
python src/generate_synthetic_data.py          # synthetic, works offline
# or place your downloaded creditcard.csv into data/

# 2. Train all models
python src/train.py --config config.yaml

# 3. Evaluate and generate report + plots
python src/evaluate.py --config config.yaml

# 4. Run inference on a sample transaction
python src/predict.py --model models/xgboost_model.pkl --input sample_transaction.json

# 5. (Optional) launch the demo app
streamlit run app.py
```

## Approach

**Why not just optimize accuracy?** With ~0.17% fraud, a model that predicts "not fraud" every time scores 99.8% accuracy and catches zero fraud. Every design choice here follows from that:

- **Resampling:** SMOTE on the training set only (never on validation/test — that would leak synthetic neighbors of test points into training).
- **Class weighting:** as an alternative/complement to resampling for tree-based and linear models.
- **Metrics:** Precision-Recall AUC is the primary metric (ROC-AUC is reported but can look misleadingly good under heavy imbalance). We also report recall achievable at a fixed precision (e.g., "what recall do we get if we insist on ≤5% false positive rate among flagged transactions").
- **Threshold tuning:** the default 0.5 probability threshold is rarely right for imbalanced problems; `evaluate.py` sweeps thresholds and reports the trade-off curve.
- **Stratified splits:** train/val/test splits are stratified on the target to preserve the fraud rate in each split.

## Results (on synthetic data — replace with your own run)

| Model | PR-AUC | ROC-AUC | F1 | Recall @ 90% Precision |
|---|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.97 | ~0.68 | ~0.55 |
| Random Forest | ~0.83 | ~0.98 | ~0.79 | ~0.71 |
| XGBoost | ~0.87 | ~0.99 | ~0.82 | ~0.76 |
| Neural Net (MLP) | ~0.84 | ~0.98 | ~0.80 | ~0.73 |

Run `src/evaluate.py` to regenerate this table with your actual data — these numbers are illustrative placeholders from the synthetic generator, not a benchmark claim.

## Limitations & next steps

- The synthetic generator approximates the real dataset's statistical shape but isn't a substitute for real transaction data — always validate on the real Kaggle set (or your own data) before drawing conclusions.
- No temporal/online-learning component: real fraud patterns drift over time, which this batch-trained setup doesn't model.
- No cost-sensitive optimization tied to actual dollar amounts (a missed $10,000 fraud and a missed $10 fraud are treated the same by standard classification metrics).

## License

MIT — see [LICENSE](LICENSE).
