# Kaggle-Competition-ForestFire
The Machine Learning I build for WiDS Global Datathon 2026 on kaggle.
This repository covers exploratory data analysis (EDA), data cleaning, feature engineering, cross-validation setup, and hyperparameter tuning.

# Project Overview

* **Goal:** Predicting calibrated probability forecasts across multiple time horizons [12H,24H,36H,48H] to support real-world decisions.
* **Evaluation Metric:** LogLoss,ROC-AUC,RMSE,Brier's Loss,Hybrid Score(0.3 x C-index + 0.7 x (1 - Weighted Brier Score)).
* **Validation Strategy:** Stratified 5-10 Fold Cross-Validation.
* **Final Validation Score:**  Hybrid score[XGB_Calibrated : 0.95737,LGBM_Calibrated : 0.95540]

# Exploratory Data Analysis & Key Insights

* **Dataset Profile:** Train Dataset [221 rows x 37]
* **Key Observations:** 
    * High correlation detected between numerical feature clusters.
    * July,August,September have increasing amount of forest fires recorded in the given data.
    * August & September have a sudden peak in the amount of forest fires recorded in the given data.
    * Higher amount of forest fires have been observed from hour 16 - 24 and 1am from the recorded data.

!
