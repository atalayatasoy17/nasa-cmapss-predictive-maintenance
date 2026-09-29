from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.model import BASE_CANDIDATES, RUL_CAP, fit_fd001_temporal_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_ROOT / "CMAPSSData" / "train_FD001.txt"
ARTIFACT_FILE = PROJECT_ROOT / "models" / "fd001_temporal.joblib"


def main() -> None:
    columns = ["unit_id"] + BASE_CANDIDATES
    train = pd.read_csv(
        TRAIN_FILE,
        sep=r"\s+",
        header=None,
        names=columns,
    )

    model, feature_columns = fit_fd001_temporal_model(train)

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "rul_cap": RUL_CAP,
        "sklearn_version": sklearn.__version__,
    }
    ARTIFACT_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_FILE, compress=3)

    print("Training engines:", train["unit_id"].nunique())
    print("Model features:", len(feature_columns))
    print("Saved model:", ARTIFACT_FILE)


if __name__ == "__main__":
    main()
