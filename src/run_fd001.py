from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.alerts import add_confirmed_alerts
from src.model import (
    BASE_CANDIDATES,
    fit_fd001_temporal_model,
    predict_fd001_history,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "CMAPSSData"
COLUMNS = ["unit_id"] + BASE_CANDIDATES


def load_trajectory(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / filename,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )


def main() -> None:
    train = load_trajectory("train_FD001.txt")
    test = load_trajectory("test_FD001.txt")

    model, feature_columns = fit_fd001_temporal_model(train)
    history = predict_fd001_history(model, feature_columns, test)
    alerts = add_confirmed_alerts(history, threshold=30, consecutive_cycles=3)

    last = (
        alerts.groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    truth = pd.read_csv(
        DATA_DIR / "RUL_FD001.txt",
        sep=r"\s+",
        header=None,
        names=["true_RUL"],
    )
    truth.insert(0, "unit_id", range(1, len(truth) + 1))
    last = last.merge(truth, on="unit_id", how="left", validate="one_to_one")

    if last["true_RUL"].isna().any():
        raise ValueError("Some test engines have no true RUL")

    actual = last["true_RUL"]
    predicted = last["predicted_RUL_capped"]

    print("Model features:", len(feature_columns))
    print("Test engines:", len(last))
    print("Test MAE:", round(mean_absolute_error(actual, predicted), 2))
    print("Test RMSE:", round(mean_squared_error(actual, predicted) ** 0.5, 2))
    print("Test R²:", round(r2_score(actual, predicted), 3))
    print("Engines with a confirmed alert:", int(last["ever_confirmed"].sum()))
    print()
    print(
        last[
            ["unit_id", "cycle", "predicted_RUL_capped",
             "true_RUL", "ever_confirmed"]
        ].head().round(2).to_string(index=False)
    )


if __name__ == "__main__":
    main()
