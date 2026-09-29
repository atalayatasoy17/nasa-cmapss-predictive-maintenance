import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.alerts import add_confirmed_alerts
from src.model import predict_fd001_history


def infer_fd001_engine_status(
    model: RandomForestRegressor,
    feature_columns: list[str],
    raw_test: pd.DataFrame,
    threshold: float = 30,
    consecutive_cycles: int = 3,
) -> pd.DataFrame:
    """Return the latest RUL prediction and alert status for each engine."""
    history = predict_fd001_history(model, feature_columns, raw_test)
    alerts = add_confirmed_alerts(
        history,
        threshold=threshold,
        consecutive_cycles=consecutive_cycles,
    )

    latest = (
        alerts.groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    return latest[
        ["unit_id", "cycle", "predicted_RUL_capped", "ever_confirmed"]
    ].rename(columns={"cycle": "last_observed_cycle"})
