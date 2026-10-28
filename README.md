# MSFRAM: Multi-label Fracture Risk Assessment Model

# Code accompanying the manuscript "ML-RUSMOTE: An Approach to Address Multi-Label Imbalance for Multi-Site Fracture Risk Assessment Using Deep Learning".

## Overview

The repository provides a reproducible pipeline for:
- Preprocessing NHANES data and constructing an analysis-ready dataset;
- Splitting the dataset into a training set and a hold-out set using stratified sampling;
- Training and evaluating the ML-RUSMOTE algorithm;
- Building and evaluating the MSFRAM model;
- Comparing MSFRAM and MSFRAM variants with FRAX for hip fracture and major osteoporotic fracture prediction.
The raw NHANES data are publicly available from the [CDC NHANES website](https://wwwn.cdc.gov/nchs/nhanes/). This repository provides the derived analysis dataset and the code required to reproduce the main analyses reported in the manuscript.
## Repository Structure
| File | Description |
|---|---|
| `data_preprocessing.R` | R script for NHANES data preprocessing. Variables are selected, cleaned, and merged, and the dataset is split into a training set and a hold-out set using stratified sampling. |
| `ML_RUSMOTE_Function.R` | R implementation of the ML-RUSMOTE algorithm. |
| `ML_RUSMOTE_Training.R` | Script for training and evaluating ML-RUSMOTE on the training set. |
| `ML_RUSMOTE_Holdout.R` | Script for evaluating ML-RUSMOTE on the hold-out set. |
| `MSFRAM.R` | Script for the MSFRAM model, including performance evaluation and comparison with the FRAX model. |
| `NHANES dataset.csv` | Derived analysis-ready NHANES dataset used in the study (N = 1,928). |
| `Related resources (NHANES).txt` | Reference materials and resources used during NHANES data preparation. |
| `Variable description (NHANES).txt` | Description of variables included in the analysis dataset. |


## Requirements

### R Environment
- R version 4.5.0
- Key packages: `car`, `dplyr`, `keras3`, `mldr`, `mldr.resampling`, `pROC`, `PRROC`, `tibble`, `utiml`

### Python Environment
- Python version 3.13.9
- Key packages: `shap`, `numpy`, `pandas`, `matplotlib`, `tensorflow`
