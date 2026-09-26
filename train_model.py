"""
train_model.py
Trains and compares three classifiers (Random Forest, Decision Tree, Logistic
Regression) on the traffic dataset, using stratified k-fold cross-validation,
then saves the best-performing model + preprocessing pipeline to disk.

We report precision/recall/F1 per class (not just accuracy) because in
intrusion detection missing an attack (false negative) is far more costly
than a false alarm — accuracy alone hides that trade-off, especially with
an imbalanced dataset like this one (u2r is <2% of rows).
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

NUMERIC_FEATURES = ["duration", "src_bytes", "dst_bytes", "count", "srv_count", "num_failed_logins"]
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def build_candidates():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, class_weight="balanced", random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=42
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
    }


def main():
    df = pd.read_csv("data/traffic_dataset.csv")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_name, best_pipeline, best_f1 = None, None, -1

    print("=" * 70)
    print("5-fold cross-validated comparison (on training split)")
    print("=" * 70)

    for name, clf in build_candidates().items():
        pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])
        preds = cross_val_predict(pipeline, X_train, y_train, cv=cv)
        report = classification_report(y_train, preds, output_dict=True, zero_division=0)
        macro_f1 = report["macro avg"]["f1-score"]

        print(f"\n--- {name} (macro F1 = {macro_f1:.3f}) ---")
        print(classification_report(y_train, preds, zero_division=0))

        if macro_f1 > best_f1:
            best_name, best_pipeline, best_f1 = name, pipeline, macro_f1

    print("=" * 70)
    print(f"Best model on cross-validation: {best_name} (macro F1 = {best_f1:.3f})")
    print("=" * 70)

    # Fit the winning pipeline on the full training split, then check held-out test set
    best_pipeline.fit(X_train, y_train)
    test_preds = best_pipeline.predict(X_test)
    print("\nHeld-out test set performance:")
    print(classification_report(y_test, test_preds, zero_division=0))

    joblib.dump(best_pipeline, "model.joblib")
    print(f"\nSaved best model ({best_name}) to model.joblib")


if __name__ == "__main__":
    main()
