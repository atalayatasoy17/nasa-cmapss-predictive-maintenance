# NASA C-MAPSS Predictive Maintenance

A step-by-step project for predicting the Remaining Useful Life (RUL) of simulated turbofan engines using the NASA C-MAPSS dataset.

## Problem

Given an engine's operational settings and sensor measurements over time, predict how many operational cycles remain before failure. This is a regression problem.

## Dataset

The dataset contains four subsets: FD001, FD002, FD003, and FD004. Each row represents one engine at one operational cycle and contains an engine ID, a cycle number, three operational settings, and 21 sensor measurements.

Training trajectories continue until failure. Test trajectories stop before failure, and the corresponding true RUL values are provided separately.

Dataset source: [NASA C-MAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

The raw dataset is kept locally in `CMAPSSData/` and is excluded from this repository.

## Project status

This project is in progress. We will document each step as we explore the data, build models, and evaluate their predictions.

## Reference

Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. Proceedings of the First International Conference on Prognostics and Health Management.
