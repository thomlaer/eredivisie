from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction.modeling import (
    adjust_draw,
    align,
    base_feature_columns,
    classifier_pipeline,
    tune_no_odds_draw,
)


FEATURES = ROOT / "outputs" / "latest" / "historical_features.csv"
OUTPUT = ROOT / "outputs" / "experiments" / "recency_fixed_split.csv"


def weights(dates: pd.Series, half_life_days: int | None) -> np.ndarray | None:
    if half_life_days is None:
        return None
    parsed = pd.to_datetime(dates, errors="coerce")
    age = (parsed.max() - parsed).dt.days.fillna(half_life_days).clip(lower=0).to_numpy(float)
    result = np.maximum(np.exp(-np.log(2.0) * age / half_life_days), 0.15)
    return result / result.mean()


def fit(frame: pd.DataFrame, numeric: list[str], categorical: list[str], half_life: int | None):
    model = classifier_pipeline(numeric, categorical)
    kwargs = {}
    sample_weight = weights(frame["date"], half_life)
    if sample_weight is not None:
        kwargs["model__sample_weight"] = sample_weight
    model.fit(align(frame, numeric, categorical), frame["target"].to_numpy(), **kwargs)
    return model


def metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> tuple[float, float, float]:
    predicted = probabilities.argmax(axis=1)
    return (
        float(accuracy_score(frame["target"], predicted)),
        float(log_loss(frame["target"], probabilities, labels=[0, 1, 2])),
        float(np.mean(predicted == 1)),
    )


def main() -> None:
    features = pd.read_csv(FEATURES, low_memory=False)
    train = features[features["season_start"].le(2022)].copy()
    validation = features[features["season_start"].isin([2023, 2024])].copy()
    final_train = features[features["season_start"].le(2024)].copy()
    test = features[features["season_start"].eq(2025)].copy()
    numeric, categorical = base_feature_columns(train)
    rows = []
    for half_life in (None, 365, 730, 1460, 2920):
        tuning_model = fit(train, numeric, categorical, half_life)
        validation_base = tuning_model.predict_proba(align(validation, numeric, categorical))
        draw_multiplier, _ = tune_no_odds_draw(validation["target"].to_numpy(), validation_base)
        validation_probabilities = adjust_draw(validation_base, draw_multiplier)

        final_model = fit(final_train, numeric, categorical, half_life)
        test_base = final_model.predict_proba(align(test, numeric, categorical))
        test_probabilities = adjust_draw(test_base, draw_multiplier)
        validation_accuracy, validation_loss, validation_draw_rate = metrics(
            validation, validation_probabilities
        )
        test_accuracy, test_loss, test_draw_rate = metrics(test, test_probabilities)
        rows.append(
            {
                "half_life_days": "none" if half_life is None else half_life,
                "features": len(numeric) + len(categorical),
                "draw_multiplier": draw_multiplier,
                "validation_accuracy": validation_accuracy,
                "validation_log_loss": validation_loss,
                "validation_draw_rate": validation_draw_rate,
                "test_accuracy": test_accuracy,
                "test_log_loss": test_loss,
                "test_draw_rate": test_draw_rate,
            }
        )
    output = pd.DataFrame(rows).sort_values(
        ["validation_accuracy", "validation_log_loss"], ascending=[False, True]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
