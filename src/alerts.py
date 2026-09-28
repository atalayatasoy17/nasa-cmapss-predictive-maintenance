import pandas as pd


def add_confirmed_alerts(
    history: pd.DataFrame,
    threshold: float = 30,
    consecutive_cycles: int = 3,
) -> pd.DataFrame:
    """Confirm an alert after consecutive low RUL predictions."""
    required = {"unit_id", "cycle", "predicted_RUL_capped"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if consecutive_cycles < 1:
        raise ValueError("consecutive_cycles must be at least 1")

    result = history.sort_values(
        ["unit_id", "cycle"]
    ).reset_index(drop=True).copy()

    if result.duplicated(["unit_id", "cycle"]).any():
        raise ValueError("Duplicate unit_id and cycle pairs")
    if not result.groupby("unit_id")["cycle"].diff().dropna().eq(1).all():
        raise ValueError("Cycles must be consecutive for each engine")
    if result["predicted_RUL_capped"].isna().any():
        raise ValueError("Predictions contain missing values")

    result["below_threshold"] = (
        result["predicted_RUL_capped"] <= threshold
    )
    result["confirmed_now"] = (
        result.groupby("unit_id")["below_threshold"]
        .transform(
            lambda values: values.rolling(
                window=consecutive_cycles,
                min_periods=consecutive_cycles,
            ).sum().eq(consecutive_cycles)
        )
    )
    result["ever_confirmed"] = (
        result.groupby("unit_id")["confirmed_now"].cummax()
    )

    return result
