import pandas as pd

TEMPORAL_FEATURES = (
    "sensor_11_mean_10",
    "sensor_11_trend_5_20",
    "sensor_12_mean_10",
    "sensor_12_trend_5_20",
)


def add_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    required = {"unit_id", "cycle", "sensor_11", "sensor_12"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = data.sort_values(["unit_id", "cycle"]).reset_index(drop=True).copy()

    for sensor in ("sensor_11", "sensor_12"):
        values = result.groupby("unit_id")[sensor]

        mean_5 = values.transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        mean_10 = values.transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        mean_20 = values.transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )

        result[f"{sensor}_mean_10"] = mean_10
        result[f"{sensor}_trend_5_20"] = mean_5 - mean_20

    return result