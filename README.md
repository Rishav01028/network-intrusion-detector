# Network Intrusion Detection System 🛡️

A machine-learning classifier that flags network traffic as normal or as one
of four attack categories, using the feature schema and attack taxonomy of
the classic **NSL-KDD** intrusion-detection dataset.

## Attack categories

| Label  | Meaning                                                             |
|--------|----------------------------------------------------------------------|
| normal | Regular traffic                                                     |
| dos    | Denial of Service (e.g. SYN flood) — huge connection counts, short duration |
| probe  | Port/network scanning — many connections, tiny payloads             |
| r2l    | Remote-to-Local — failed logins, suspicious service access          |
| u2r    | User-to-Root — rare privilege-escalation traffic                    |

## Pipeline

1. **`generate_dataset.py`** — generates a labeled traffic dataset with the
   same column schema as NSL-KDD (`duration`, `protocol_type`, `service`,
   `flag`, `src_bytes`, `dst_bytes`, `count`, `srv_count`,
   `num_failed_logins`, `label`). This synthetic generator lets the whole
   pipeline run **offline** — see "Using the real NSL-KDD dataset" below to
   swap in the real thing.
2. **`train_model.py`** — compares **Random Forest**, **Decision Tree**, and
   **Logistic Regression** using **5-fold stratified cross-validation**, and
   reports **precision/recall/F1 per class** rather than raw accuracy, since
   accuracy is misleading on an imbalanced dataset (u2r is <2% of traffic
   here, just like in real NSL-KDD). The best model by macro-F1 is saved to
   `model.joblib`.
3. **`app.py`** — a Streamlit dashboard: upload a CSV of traffic records and
   get each one flagged normal/attack with a confidence score.

## Running it

```bash
pip install -r requirements.txt
python generate_dataset.py      # creates data/traffic_dataset.csv
python train_model.py           # trains, compares, and saves the best model
streamlit run app.py            # launches the dashboard
```

## Using the real NSL-KDD dataset

Download `KDDTrain+.txt` / `KDDTest+.txt` from the
[NSL-KDD dataset page](https://www.unb.ca/cic/datasets/nsl.html) (or the
mirrored version on Kaggle), map its 41 columns down to (or extend
`NUMERIC_FEATURES`/`CATEGORICAL_FEATURES` in `train_model.py` to match) the
schema above, save it as `data/traffic_dataset.csv`, and re-run
`train_model.py`. Nothing else in the pipeline needs to change.

## Why Random Forest wins here

Logistic Regression assumes roughly linear decision boundaries between
classes, which struggles on the rare `u2r` class. Random Forest handles the
non-linear, imbalanced structure of intrusion data much better — which is
also why it's the standard baseline in NSL-KDD research.
