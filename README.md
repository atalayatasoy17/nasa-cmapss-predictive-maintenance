# NASA C-MAPSS Predictive Maintenance

A step-by-step study of Remaining Useful Life (RUL) prediction and illustrative maintenance alerts for simulated turbofan engines. The experiments currently focus on the FD001 subset of NASA C-MAPSS.

## Objective

Use an engine's operational settings and sensor measurements to predict the number of operational cycles remaining before failure. Then examine how RUL predictions could support a maintenance alert.

A cycle is an operational observation step in this dataset; it is not directly a number of hours or days.

## Dataset

Each row contains an engine ID, cycle number, three operational settings, and 21 sensor measurements. FD001 has one operating condition and one fault mode.

| FD001 split | Engines | Rows | Observation period |
|---|---:|---:|---|
| Train | 100 | 20,631 | Continues until failure |
| Test | 100 | 13,096 | Stops before failure |

For each train row, `RUL = final_failure_cycle - current_cycle`. The test RUL file gives the true remaining cycles at each test engine's last observed cycle.

The regression experiments use `RUL_capped = min(RUL, 125)` as a modeling choice. This is not a physical limit on engine life: 11 test engines have a true RUL above 125.

**Data source:** [NASA C-MAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

Download and extract the dataset, then place these files in `CMAPSSData/`:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

Raw data is excluded from Git.

## Repository structure

| Path | Purpose |
|---|---|
| `notebooks/01_data_overview.ipynb` | Data checks, RUL construction, sensor exploration, and train/test comparisons |
| `notebooks/02_baseline_model.ipynb` | Dummy, Linear Regression, and Random Forest baselines |
| `notebooks/03_feature_engineering.ipynb` | Temporal features, model comparison, and exploratory maintenance alerts |
| `src/features.py` | Computes four temporal features using each engine's current and earlier cycles |
| `src/model.py` | Trains the FD001 temporal Random Forest and predicts capped RUL for every observed cycle or each test engine's last observation |
| `src/alerts.py` | Applies the illustrative alert rule: predicted RUL at most 30 for three consecutive cycles |
| `src/inference.py` | Combines RUL predictions and alerts into each engine's latest status |
| `src/train_fd001.py` | Trains the FD001 temporal model and saves it in `models/` |
| `src/predict_fd001.py` | Loads the saved model and runs inference without train data or test RUL |
| `src/run_fd001.py` | Evaluates test predictions using the true RUL file and reports alerts |
| `tests/test_alerts.py` | Checks alert confirmation, engine isolation, and missing cycles |
| `reports/figures/` | Reserved for exported figures |
| `requirements.txt` | Python dependencies |

## Getting started

From the project root on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the notebooks in numerical order. Start JupyterLab from the project root so the notebooks can find `CMAPSSData/`:

```bash
jupyter lab
```

Train and save the FD001 model once:

```bash
python3 -m src.train_fd001
```

This creates `models/fd001_temporal.joblib`. The `models/` directory is excluded from Git.

Load the saved model and predict from `test_FD001.txt`:

```bash
python3 -m src.predict_fd001
```

To reproduce the FD001 test evaluation and alert count with `RUL_FD001.txt`:

```bash
python3 -m src.run_fd001
```

To run the alert rule unit tests from the project root:

```bash
python3 -m unittest discover -s tests -v
```

## Methods

- Split validation data by engine with 5-fold `GroupKFold`, so an engine does not appear in both training and validation within a fold.
- Use Random Forest with 18 nonconstant baseline features.
- Add four temporal features: a 10-cycle rolling mean and a 5-cycle minus 20-cycle rolling mean trend for each of `sensor_11` and `sensor_12`. Each feature uses only the same engine's current and earlier observations.
- Evaluate test RUL at one point per engine: its last observed cycle.

## RUL prediction results

| Model | 5-fold CV MAE | 5-fold CV RMSE | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest baseline | 11.42 | 16.69 | 13.51 | 18.02 | 0.812 |
| Random Forest + temporal features | 10.76 | 16.03 | 13.39 | 18.05 | 0.811 |

The cross-validation metrics use all held-out train rows and the capped RUL target. The test metrics use one last observation per engine and the actual, uncapped RUL. These are different evaluation settings and should not be compared directly. Temporal features improved cross-validation MAE in all five folds. On the test engines, the MAE difference was small and RMSE did not improve.

At the engine level, temporal features reduced out-of-fold MAE for 65 of 100 train engines across all observations. In the near-failure period (true RUL 0–50), they reduced MAE for 79 engines and increased it for 21.

## Exploratory maintenance alerts

An example alert rule marks an engine when predicted RUL is at most 30 for three consecutive observed cycles. The rule was examined with a hypothetical goal of warning at least 20 cycles before failure.

On the observed test histories, the baseline model alerted 17 engines and the temporal model alerted 18. Both alerted 15 of the 16 engines whose true RUL was at most 20 at the last observation. Among those 16 engines, 11 received a baseline alert and 13 received a temporal-model alert at least 20 cycles before failure.

For the 21 train engines whose near-failure MAE increased, the temporal model gave the first confirmed alert earlier for 12 engines, later for 7, and at the same cycle for 2. Alerts came at least 20 cycles before failure for 15 engines with the baseline model and 16 with the temporal model. This exploratory comparison shows that higher average prediction error does not necessarily mean a later alert.

These thresholds are examples, not an operational maintenance policy. Test trajectories end before failure, so future alerts for engines without an observed alert cannot be evaluated. The test set was also examined during this exploratory project; these results should not be treated as an untouched final benchmark.

## Reference

Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. Proceedings of the First International Conference on Prognostics and Health Management.
