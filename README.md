# Kaggle-Competition-ForestFire
The Machine Learning I build for **[WiDS Global Datathon 2026](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)** on kaggle.
This repository covers exploratory data analysis (EDA), data cleaning, feature engineering, cross-validation setup, and hyperparameter tuning.

# Project Overview

* **Goal:** Predicting calibrated probability forecasts across multiple time horizons [12H,24H,36H,48H] to support real-world decisions.
* **Validation Strategy:** Stratified 5-10 Fold Cross-Validation.
* **Final Validation Score:**  Hybrid score[XGB_Calibrated : 0.95737,LGBM_Calibrated : 0.95540]

![XGB model Traning and validation curve](images/XGBLearningCurve.png)

# Exploratory Data Analysis & Key Insights

* **Dataset Profile:** Train Dataset [221 rows x 37]
* **Key Observations:** 
    * High correlation detected between numerical feature clusters.
    * July,August,September have increasing amount of forest fires recorded in the given data.
    * August & September have a sudden peak in the amount of forest fires recorded in the given data.
    * Higher amount of forest fires have been observed from hour 16 - 24 and 1am from the recorded data.

![Forest Fires Observed](images/ForestFireRecordedTime.png)
![Corelational heatmap of different time horizons and Features](images/CorelationalHeatmapOfTimestampsAndFeatures.png)

* **Evaluation Metric:** 
    * LogLoss,ROC-AUC,RMSE,Brier's Loss,
    * Hybrid Score : a combination of C-index and Brier's Score is taken
        *** (0.3 x C-index + 0.7 x (1 - Weighted Brier Score)) ***

# Key Technical Highlights

* **Data Preprocessing & Scaling:** Applied `Standard Scaler` to use the scaled data for feature selection using L1 lasso and `RobustScaler` to scale the data cleanly without letting extreme outlier values shrink the rest of the features down to zero.
* **Hyperparameter Optimization:** Used `Optuna` with stratified k-fold for cross-validation to minimize logloss and maximize Hybrid score for each time horizon model.

![Optimizing the 24H time horizon XGB model using Optuna](images/XGB_24H_Optuna.png)

* **Feature Selection Iterations:** Constrained feature dimensions based on feature importance ranking using both `L1 lasso` and `permutation importance` then removing the features corelated to other features to maximize generalization on small, high-dimensional validation sets. 

* **Model Calibration:** Both the XGboost and LightGBM models have been clibrated using both `Isotonic calibration` and `Sigmoid (Platt) calibration` during different instances to submit during the competition where Sigmoid performed better.

![Calibration curve for XGB model](images/CalibrationCurve.png)


**NOTE:** All the Hyperparameters and additional info is present in the notebook itself.