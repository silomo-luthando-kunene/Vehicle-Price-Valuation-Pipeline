# 🚗 Used Car Price Prediction 
**End to end machine learning workflow applying Random Forest regression with hyperparameter tuning, feature extraction and sklearn Pipelines.** </br>
![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange)
![Status](https://img.shields.io/badge/Status-Complete-success)
</br>
## 📌 Project Overview 
This project focuses on building a production grade regression system to predict used car prices. Rather than performing manual, one off data cleaning, this project implements a modular pipeline architecture. This ensures that the model can ingest raw, messy technical specifications (like strings containing "cc" or "bhp") and output precise valuations with zero manual intervention. </br>
The system is designed with Industrial Engineering principles in mind: maximizing reliability, minimizing manual "touches," and ensuring robust performance through automated hyperparameter tuning.
</br>
## 🛠️ Engineering Toolkit 
**Architecture:** Scikit-Learn Pipeline & ColumnTransformer. </br>
**Feature Engineering:** Custom FunctionTransformer (Regular Expresssion Extraction). </br>
**Modeling:** Random Forest Regressor. </br>
**Optimization:** GridSearchCV with 5-Fold Cross-Validation (KFold). </br>
**Metrics:** $R^2$, MAE (Mean Absolute Error) and MAPE (Mean Absolute Percentage Error)
