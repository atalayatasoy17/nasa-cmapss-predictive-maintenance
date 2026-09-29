from pathlib import Path

import pandas as pd

from src.inference import infer_fd001_engine_status
from src.model import BASE_CANDIDATES, fit_fd001_temporal_model


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
    status = infer_fd001_engine_status(model, feature_columns, test)

    print("Model features:", len(feature_columns))
    print("Engines:", len(status))
    print("Confirmed alerts:", int(status["ever_confirmed"].sum()))
    print()
    print(status.head().round(2).to_string(index=False))
    print()
    print(
        "Alerted unit IDs:",
        status.loc[status["ever_confirmed"], "unit_id"].tolist(),
    )


if __name__ == "__main__":
    main()