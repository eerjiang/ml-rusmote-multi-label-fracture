# MSFRAM: Multi-label Fracture Risk Assessment Model

# Code accompanying the manuscript "ML-RUSMOTE: An Approach to Address Multi-Label Imbalance for Multi-Site Fracture Risk Assessment Using Deep Learning" submitted to xxx.

## Overview

This repository contains the implementation of:
- **ML-RUSMOTE**: Multi-label resampling strategy for handling class imbalance
- **MSFRAM**: Multi-label deep learning framework for simultaneous fracture risk prediction
- **SHAP analysis**: Model interpretability via SHapley Additive exPlanations

## Requirements

### R Environment
- R version 4.5.0
- Key packages: `car`, `dplyr`, `keras3`, `mldr`, `mldr.resampling`, `pROC`, `PRROC`, `utiml`
- Dependency lock file: `renv.lock`

### Python Environment
- Python version 3.13.9
- Key packages: `shap`, `numpy`, `pandas`, `matplotlib`, `tensorflow`
- Dependency file: `requirements.txt`

## Setup

### R Environment
```r
install.packages("renv")
renv::restore()  # 自动安装 renv.lock 中锁定的所有包及版本
