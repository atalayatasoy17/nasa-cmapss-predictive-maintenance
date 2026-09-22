# NASA C-MAPSS Predictive Maintenance

A step-by-step project for predicting the Remaining Useful Life (RUL) of simulated turbofan engines using the NASA C-MAPSS dataset.

## Objective

Given an engine's operational settings and sensor measurements over time, estimate how many operational cycles remain before failure. This is a regression problem.

## Dataset

Each row represents one engine at one operational cycle. It contains an engine ID, a cycle number, three operational settings, and 21 sensor measurements.

Training trajectories continue until failure. Test trajectories stop before failure; their true RUL values are provided in separate files.

| Subset | Operating conditions | Fault modes |
| --- | ---: | ---: |
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

Source: [NASA C-MAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

The raw data belongs in `CMAPSSData/`. This folder is excluded from Git.

## Repository structure

- `notebooks/`: Exploratory analysis and experiments
- `src/`: Reusable Python code
- `reports/figures/`: Generated charts and figures
- `requirements.txt`: Python dependencies

## Getting started

From the project root on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Place the dataset files in `CMAPSSData/` before opening the notebooks.

## Project status

In progress. The planned steps are data exploration, RUL target construction, model training, and evaluation. Results will be added after the experiments are completed.

## Reference

Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. Proceedings of the First International Conference on Prognostics and Health Management.