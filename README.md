# 🚗 Used Car Price Prediction 
**End to end machine learning workflow applying Random Forest Regression with Hyperparameter Tuning, Feature Extraction and Sklearn Pipelines.** </br>
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
</br>
## ⚙️ Key Features
- **Data Preprocessing**
  - Handles missing values with median imputation for numerical features and constant filling for categorical features.
  - Standardization of numerical features and one-hot encoding for categorical features.
  - Custom `FunctionTransformer` to extract numerical values from raw strings (e.g., `"1198 cc" → 1198`, `"87 bhp @ 6000 rpm" → Power_Val = 87, Power_RPM = 6000`).

- **Feature Engineering**
  - Extracted **Engine_CC**, **Power_Val**, **Power_RPM**, **Torque_Val**, and **Torque_RPM** from textual specs.
  - Combined with car dimensions, year, mileage, and categorical attributes to form a rich feature set.

- **Modeling**
  - Implemented **RandomForestRegressor** within a Scikit-Learn pipeline.
  - Cross-validation with multiple metrics: R², MAE, RMSE, MAPE.
  - Hyperparameter tuning via **GridSearchCV** for optimal performance.

- **Evaluation & Visualization**
  - Correlation heatmaps to identify relationships between features.
  - Feature importance ranking (Top 15 drivers of car price).
  - Actual vs. Predicted scatter plots for model performance visualization.
  - Tuned model achieved **R² = 0.9765** and **MAE ≈ 111,125**.
## 📊 Dataset

- **Source:** Car Dekho (Indian used car marketplace)
- **Size:** 2,059 vehicles
- **Features:** 20 (technical specs + categorical attributes)
- **Target Variable:** Price (220,000 - 2,650,000)
- **Feature Categories:**
  - **Numerical:** Year, Kilometer, Engine CC, Power (bhp), Torque (Nm), Dimensions, Fuel Tank Capacity
  - **Categorical:** Make, Fuel Type, Transmission, Owner Type, Seller Type, Drivetrain
  - **Text Specs:** "1198 cc", "87 bhp @ 6000 rpm", "109 Nm @ 4500 rpm"
- **Challenge:** Mixed data types (numeric, categorical, text specifications)
## 📈 Performance & Results
The transition from a baseline model to a tuned system yielded a massive improvement in predictive accuracy.

| Metric | Baseline Model | **Tuned Pipeline** |
| :--- | :--- | :--- |
| **R² Score** | 0.8225 | **0.9765** |
| **MAE (Mean Absolute Error)** | 299,060 | **111,125** |

**Interpretation:** The final model explains **97.70%** of the variance in car prices. By reducing the Mean Absolute Error by over 60%, the system provides a highly reliable tool for automated predictive business decisions.
## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**:
  - Data: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - ML: `scikit-learn` (Pipeline, ColumnTransformer, FunctionTransformer, RandomForestRegressor, GridSearchCV)
### **Learning Objectives:**
- Master sklearn pipeline architecture (reproducibility)
- Understand Random Forest hyperparameter impact (n_estimators, max_depth)
- Build deployable ML systems (not just "train a model")
- Think like an ML Engineer (systems thinking, not just algorithms)
---
*Developed by Luthando Silomo Kunene with the help of AI | Aspiring Professional Engineer (Industrial and Systems Engineering) & Data Practitioner (Machine Learning and Data Engineering)* </br>
**Built as part of ALX Africa Machine Learning Course I am enrolled in - the goal is active learning and mastering of ML concepts through project based learning** </br>
**Module:** Week 4 - Random Forest Regressor, Cross Validation & Hyperparameter Tuning  
