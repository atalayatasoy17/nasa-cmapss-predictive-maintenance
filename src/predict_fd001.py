from pathlib import Path

import joblib
import pandas as pd

from src.inference import infer_fd001_engine_status
from src.model import BASE_CANDIDATES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "CMAPSSData"
ARTIFACT_FILE = PROJECT_ROOT / "models" / "fd001_temporal.joblib"
COLUMNS = ["unit_id"] + BASE_CANDIDATES


def main() -> None:
    artifact = joblib.load(ARTIFACT_FILE)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    test = pd.read_csv(
        DATA_DIR / "test_FD001.txt",
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )
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