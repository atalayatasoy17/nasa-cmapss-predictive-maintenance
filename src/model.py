import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.features import TEMPORAL_FEATURES, add_temporal_features


RUL_CAP = 125

BASE_CANDIDATES = (
    ["cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def fit_fd001_temporal_model(
    raw_train: pd.DataFrame,
) -> tuple[RandomForestRegressor, list[str]]:
    """Train the FD001 temporal model on run-to-failure engines."""
    train = add_temporal_features(raw_train)

    train["max_cycle"] = train.groupby("unit_id")["cycle"].transform("max")
    train["RUL"] = train["max_cycle"] - train["cycle"]
    train["RUL_capped"] = train["RUL"].clip(upper=RUL_CAP)

    base_features = [
        column for column in BASE_CANDIDATES
        if train[column].nunique() > 1
    ]
    feature_columns = base_features + list(TEMPORAL_FEATURES)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[feature_columns], train["RUL_capped"])

    return model, feature_columns



def predict_fd001_last_observations(
    model: RandomForestRegressor,
    feature_columns: list[str],
    raw_test: pd.DataFrame,
) -> pd.DataFrame:
    """Predict capped RUL at each test engine's last observed cycle."""
    test = add_temporal_features(raw_test)

    last = (
        test.groupby("unit_id")
        .tail(1)
        .sort_values("unit_id")
        .reset_index(drop=True)
    )

    predictions = last[["unit_id", "cycle"]].rename(
        columns={"cycle": "last_observed_cycle"}
    ).copy()
    predictions["predicted_RUL_capped"] = model.predict(
        last[feature_columns]
    )

    return predictions