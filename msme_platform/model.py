from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN


@dataclass
class GrowthModelArtifacts:
    pipeline: Pipeline
    metrics: dict[str, float]
    classification_report: pd.DataFrame
    feature_importance: pd.DataFrame
    reference_medians: pd.Series
    classes: list[str]


SCORE_WEIGHTS = {
    "Low": 25.0,
    "Moderate": 60.0,
    "High": 95.0,
}


def _base_feature_name(encoded_name: str, categorical_columns: list[str]) -> str:
    if encoded_name.startswith("num__"):
        return encoded_name.replace("num__", "", 1)

    if encoded_name.startswith("cat__"):
        raw_name = encoded_name.replace("cat__", "", 1)
        for col in categorical_columns:
            prefix = f"{col}_"
            if raw_name.startswith(prefix):
                return col
        return raw_name

    return encoded_name


def _aggregate_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["model"]
    encoded_feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame(
        {
            "encoded_feature": encoded_feature_names,
            "importance": classifier.feature_importances_,
        }
    )
    importance_df["feature"] = importance_df["encoded_feature"].apply(
        lambda name: _base_feature_name(name, CATEGORICAL_COLUMNS)
    )

    aggregated = (
        importance_df.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    total = aggregated["importance"].sum()
    aggregated["importance_pct"] = (
        (aggregated["importance"] / total) * 100 if total else 0.0
    )
    return aggregated.reset_index(drop=True)


def train_growth_model(msme_df: pd.DataFrame, random_state: int = 42) -> GrowthModelArtifacts:
    features = msme_df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    target = msme_df[TARGET_COLUMN].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=random_state,
        stratify=target,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=350,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "label"})

    feature_importance = _aggregate_feature_importance(pipeline)

    return GrowthModelArtifacts(
        pipeline=pipeline,
        metrics=metrics,
        classification_report=report_df,
        feature_importance=feature_importance,
        reference_medians=msme_df[NUMERIC_COLUMNS].median(),
        classes=sorted(target.unique().tolist()),
    )


def predict_growth(artifacts: GrowthModelArtifacts, profile_df: pd.DataFrame) -> dict[str, object]:
    x_input = profile_df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    probabilities = artifacts.pipeline.predict_proba(x_input)[0]
    classes = artifacts.pipeline.named_steps["model"].classes_
    predicted = classes[int(np.argmax(probabilities))]

    growth_score = float(
        np.sum([probabilities[i] * SCORE_WEIGHTS.get(label, 50.0) for i, label in enumerate(classes)])
    )
    probability_map = {label: float(probabilities[i]) for i, label in enumerate(classes)}

    return {
        "predicted_category": str(predicted),
        "growth_score": round(growth_score, 2),
        "probabilities": probability_map,
    }


def score_msme_dataset(artifacts: GrowthModelArtifacts, msme_df: pd.DataFrame) -> pd.DataFrame:
    x_input = msme_df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    probabilities = artifacts.pipeline.predict_proba(x_input)
    classes = artifacts.pipeline.named_steps["model"].classes_

    class_weights = np.array([SCORE_WEIGHTS.get(label, 50.0) for label in classes])
    growth_scores = probabilities @ class_weights
    predicted_idx = np.argmax(probabilities, axis=1)
    predicted_labels = [classes[i] for i in predicted_idx]

    scored_df = msme_df.copy()
    scored_df["Predicted_Growth_Category"] = predicted_labels
    scored_df["Growth_Score"] = np.round(growth_scores, 2)
    for idx, label in enumerate(classes):
        scored_df[f"Prob_{label}"] = probabilities[:, idx]
    return scored_df


def generate_local_reasons(profile_row: pd.Series, reference_medians: pd.Series) -> list[str]:
    """Generate transparent rule-based reasons for advisory output."""
    reasons: list[str] = []

    if profile_row["Revenue_Growth_Rate"] >= reference_medians["Revenue_Growth_Rate"] + 2:
        reasons.append("Recent revenue growth is stronger than the dataset median.")
    if profile_row["Profit_Margin"] >= reference_medians["Profit_Margin"] + 1:
        reasons.append("Profit margin is above median, supporting reinvestment capacity.")
    if profile_row["Capacity_Utilization"] >= reference_medians["Capacity_Utilization"] + 3:
        reasons.append("Capacity utilization is high, indicating efficient operations.")
    if profile_row["Technology_Level"] >= reference_medians["Technology_Level"]:
        reasons.append("Technology level is at or above median, improving scalability.")
    if profile_row["GST_Compliance_Score"] >= reference_medians["GST_Compliance_Score"]:
        reasons.append("Compliance profile is strong, reducing execution risk for schemes.")
    if profile_row["Loan_to_Revenue_Ratio"] > reference_medians["Loan_to_Revenue_Ratio"] + 0.12:
        reasons.append("Leverage is relatively high and may moderate near-term growth.")
    if profile_row["Documentation_Readiness_Score"] < reference_medians["Documentation_Readiness_Score"] - 4:
        reasons.append("Documentation readiness is below median and may slow scheme realization.")

    if not reasons:
        reasons.append("Key inputs are close to median levels, resulting in a moderate growth outlook.")

    return reasons[:5]

