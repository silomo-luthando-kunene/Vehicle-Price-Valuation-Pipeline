```python
"""
To build a robust regression pipeline that predicts used car prices by extracting features from technical specifications and categorical data. 
This project focuses on end-to-end reproducibility using Scikit-Learn Pipelines and custom transformers.
"""
```




    '\nTo build a robust regression pipeline that predicts used car prices by extracting features from technical specifications and categorical data. \nThis project focuses on end-to-end reproducibility using Scikit-Learn Pipelines and custom transformers.\n'




```python
# importing relevant libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, train_test_split
```


```python
# importing dataset

df = pd.read_csv(r"car details v4.csv")

print(df.head(10))
print("\n\n")
print(df.describe().T)
print("\n\n")
print(df.shape)
print("\n\n")
print(df.isnull().sum())

```

                Make                                 Model    Price  Year  \
    0          Honda                   Amaze 1.2 VX i-VTEC   505000  2017   
    1  Maruti Suzuki                       Swift DZire VDI   450000  2014   
    2        Hyundai                  i10 Magna 1.2 Kappa2   220000  2011   
    3         Toyota                              Glanza G   799000  2019   
    4         Toyota       Innova 2.4 VX 7 STR [2016-2020]  1950000  2018   
    5  Maruti Suzuki                              Ciaz ZXi   675000  2017   
    6  Mercedes-Benz                  CLA 200 Petrol Sport  1898999  2015   
    7            BMW                  X1 xDrive20d M Sport  2650000  2017   
    8          Skoda  Octavia 1.8 TSI Style Plus AT [2017]  1390000  2017   
    9         Nissan                        Terrano XL (D)   575000  2015   
    
       Kilometer Fuel Type Transmission    Location   Color   Owner Seller Type  \
    0      87150    Petrol       Manual        Pune    Grey   First   Corporate   
    1      75000    Diesel       Manual    Ludhiana   White  Second  Individual   
    2      67000    Petrol       Manual     Lucknow  Maroon   First  Individual   
    3      37500    Petrol       Manual   Mangalore     Red   First  Individual   
    4      69000    Diesel       Manual      Mumbai    Grey   First  Individual   
    5      73315    Petrol       Manual        Pune    Grey   First  Individual   
    6      47000    Petrol    Automatic      Mumbai   White  Second  Individual   
    7      75000    Diesel    Automatic  Coimbatore   White  Second  Individual   
    8      56000    Petrol    Automatic      Mumbai   White   First  Individual   
    9      85000    Diesel       Manual      Mumbai   White   First  Individual   
    
        Engine           Max Power              Max Torque Drivetrain  Length  \
    0  1198 cc   87 bhp @ 6000 rpm       109 Nm @ 4500 rpm        FWD  3990.0   
    1  1248 cc   74 bhp @ 4000 rpm       190 Nm @ 2000 rpm        FWD  3995.0   
    2  1197 cc   79 bhp @ 6000 rpm  112.7619 Nm @ 4000 rpm        FWD  3585.0   
    3  1197 cc   82 bhp @ 6000 rpm       113 Nm @ 4200 rpm        FWD  3995.0   
    4  2393 cc  148 bhp @ 3400 rpm       343 Nm @ 1400 rpm        RWD  4735.0   
    5  1373 cc   91 bhp @ 6000 rpm       130 Nm @ 4000 rpm        FWD  4490.0   
    6  1991 cc  181 bhp @ 5500 rpm       300 Nm @ 1200 rpm        FWD  4630.0   
    7  1995 cc  188 bhp @ 4000 rpm       400 Nm @ 1750 rpm        AWD  4439.0   
    8  1798 cc  177 bhp @ 5100 rpm       250 Nm @ 1250 rpm        FWD  4670.0   
    9  1461 cc   84 bhp @ 3750 rpm       200 Nm @ 1900 rpm        FWD  4331.0   
    
        Width  Height  Seating Capacity  Fuel Tank Capacity  
    0  1680.0  1505.0               5.0                35.0  
    1  1695.0  1555.0               5.0                42.0  
    2  1595.0  1550.0               5.0                35.0  
    3  1745.0  1510.0               5.0                37.0  
    4  1830.0  1795.0               7.0                55.0  
    5  1730.0  1485.0               5.0                43.0  
    6  1777.0  1432.0               5.0                 NaN  
    7  1821.0  1612.0               5.0                51.0  
    8  1814.0  1476.0               5.0                50.0  
    9  1822.0  1671.0               5.0                50.0  
    
    
    
                         count          mean           std      min        25%  \
    Price               2059.0  1.702992e+06  2.419881e+06  49000.0  484999.00   
    Year                2059.0  2.016425e+03  3.363564e+00   1988.0    2014.00   
    Kilometer           2059.0  5.422471e+04  5.736172e+04      0.0   29000.00   
    Length              1995.0  4.280861e+03  4.424585e+02   3099.0    3985.00   
    Width               1995.0  1.767992e+03  1.352658e+02   1475.0    1695.00   
    Height              1995.0  1.591735e+03  1.360740e+02   1165.0    1485.00   
    Seating Capacity    1995.0  5.306266e+00  8.221701e-01      2.0       5.00   
    Fuel Tank Capacity  1946.0  5.200221e+01  1.511020e+01     15.0      41.25   
    
                             50%        75%         max  
    Price               825000.0  1925000.0  35000000.0  
    Year                  2017.0     2019.0      2022.0  
    Kilometer            50000.0    72000.0   2000000.0  
    Length                4370.0     4629.0      5569.0  
    Width                 1770.0     1831.5      2220.0  
    Height                1545.0     1675.0      1995.0  
    Seating Capacity         5.0        5.0         8.0  
    Fuel Tank Capacity      50.0       60.0       105.0  
    
    
    
    (2059, 20)
    
    
    
    Make                    0
    Model                   0
    Price                   0
    Year                    0
    Kilometer               0
    Fuel Type               0
    Transmission            0
    Location                0
    Color                   0
    Owner                   0
    Seller Type             0
    Engine                 80
    Max Power              80
    Max Torque             80
    Drivetrain            136
    Length                 64
    Width                  64
    Height                 64
    Seating Capacity       64
    Fuel Tank Capacity    113
    dtype: int64
    


```python
# further eda

plt.figure(figsize = (12, 8))

numerical_cols = df.select_dtypes(include = [np.number]).columns 
corr_matrix = df[numerical_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


```


    
![png](output_3_0.png)
    



```python
# top numerical features Check
print(f"Correlation of numerical columns in subject of target variable [Price]: \n\n{corr_matrix["Price"].sort_values(ascending = False)}")


```

    Correlation of numerical columns in subject of target variable [Price]: 
    
    Price                 1.000000
    Fuel Tank Capacity    0.584631
    Width                 0.563996
    Length                0.556741
    Year                  0.311400
    Height                0.075080
    Seating Capacity     -0.038524
    Kilometer            -0.150825
    Name: Price, dtype: float64
    


```python
# Preprocessing 
"""
We implement a ColumnTransformer to handle diverse data types. 
Numerical features are imputed with medians and standardized, while categorical features undergo one-hot encoding with unknown-label handling. 
A custom FunctionTransformer is integrated at the start of the pipeline to programmatically clean technical strings (e.g., "cc", "bhp") directly from the raw input.
"""


```




    '\nWe implement a ColumnTransformer to handle diverse data types. \nNumerical features are imputed with medians and standardized, while categorical features undergo one-hot encoding with unknown-label handling. \nA custom FunctionTransformer is integrated at the start of the pipeline to programmatically clean technical strings (e.g., "cc", "bhp") directly from the raw input.\n'




```python
# Extracting Numerical Values from Columns: Max Power and Max Torque

def engineering_specs_transformer(df):
    df = df.copy()
    
    power_regex = r'(\d+\.?\d*)\s*bhp\s*@\s*(\d+\.?\d*)'
    power_data = df['Max Power'].str.extract(power_regex).astype(float)
    df['Power_Val'] = power_data[0]
    df['Power_RPM'] = power_data[1]
    
    torque_regex = r'(\d+\.?\d*)\s*Nm\s*@\s*(\d+\.?\d*)'
    torque_data = df['Max Torque'].str.extract(torque_regex).astype(float)
    df['Torque_Val'] = torque_data[0]
    df['Torque_RPM'] = torque_data[1]

    df['Engine_CC'] = df['Engine'].str.extract(r'(\d+)').astype(float)
    
    cols_to_drop = ['Max Power', 'Max Torque', 'Engine']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df
#print(engineering_specs_transformer(df).head(10))
df_cleaned = engineering_specs_transformer(df)
print(df_cleaned.head(10))
```

                Make                                 Model    Price  Year  \
    0          Honda                   Amaze 1.2 VX i-VTEC   505000  2017   
    1  Maruti Suzuki                       Swift DZire VDI   450000  2014   
    2        Hyundai                  i10 Magna 1.2 Kappa2   220000  2011   
    3         Toyota                              Glanza G   799000  2019   
    4         Toyota       Innova 2.4 VX 7 STR [2016-2020]  1950000  2018   
    5  Maruti Suzuki                              Ciaz ZXi   675000  2017   
    6  Mercedes-Benz                  CLA 200 Petrol Sport  1898999  2015   
    7            BMW                  X1 xDrive20d M Sport  2650000  2017   
    8          Skoda  Octavia 1.8 TSI Style Plus AT [2017]  1390000  2017   
    9         Nissan                        Terrano XL (D)   575000  2015   
    
       Kilometer Fuel Type Transmission    Location   Color   Owner  ...  Length  \
    0      87150    Petrol       Manual        Pune    Grey   First  ...  3990.0   
    1      75000    Diesel       Manual    Ludhiana   White  Second  ...  3995.0   
    2      67000    Petrol       Manual     Lucknow  Maroon   First  ...  3585.0   
    3      37500    Petrol       Manual   Mangalore     Red   First  ...  3995.0   
    4      69000    Diesel       Manual      Mumbai    Grey   First  ...  4735.0   
    5      73315    Petrol       Manual        Pune    Grey   First  ...  4490.0   
    6      47000    Petrol    Automatic      Mumbai   White  Second  ...  4630.0   
    7      75000    Diesel    Automatic  Coimbatore   White  Second  ...  4439.0   
    8      56000    Petrol    Automatic      Mumbai   White   First  ...  4670.0   
    9      85000    Diesel       Manual      Mumbai   White   First  ...  4331.0   
    
        Width  Height  Seating Capacity  Fuel Tank Capacity  Power_Val  Power_RPM  \
    0  1680.0  1505.0               5.0                35.0       87.0     6000.0   
    1  1695.0  1555.0               5.0                42.0       74.0     4000.0   
    2  1595.0  1550.0               5.0                35.0       79.0     6000.0   
    3  1745.0  1510.0               5.0                37.0       82.0     6000.0   
    4  1830.0  1795.0               7.0                55.0      148.0     3400.0   
    5  1730.0  1485.0               5.0                43.0       91.0     6000.0   
    6  1777.0  1432.0               5.0                 NaN      181.0     5500.0   
    7  1821.0  1612.0               5.0                51.0      188.0     4000.0   
    8  1814.0  1476.0               5.0                50.0      177.0     5100.0   
    9  1822.0  1671.0               5.0                50.0       84.0     3750.0   
    
       Torque_Val  Torque_RPM  Engine_CC  
    0    109.0000      4500.0     1198.0  
    1    190.0000      2000.0     1248.0  
    2    112.7619      4000.0     1197.0  
    3    113.0000      4200.0     1197.0  
    4    343.0000      1400.0     2393.0  
    5    130.0000      4000.0     1373.0  
    6    300.0000      1200.0     1991.0  
    7    400.0000      1750.0     1995.0  
    8    250.0000      1250.0     1798.0  
    9    200.0000      1900.0     1461.0  
    
    [10 rows x 22 columns]
    


```python
# Numeric Features and Categorical Features
numeric_features = [
    'Year', 'Kilometer', 'Engine_CC', 
    'Power_Val', 'Power_RPM', 
    'Torque_Val', 'Torque_RPM', 
    'Length', 'Width', 'Height', 'Fuel Tank Capacity'
]

# 2. Categorical Features remain the same
categorical_features = [
    'Make', 'Fuel Type', 'Transmission', 'Owner', 'Seller Type', 'Drivetrain'
]

print(df_cleaned[categorical_features])
print(df_cleaned[numeric_features])
```

                   Make Fuel Type Transmission   Owner Seller Type Drivetrain
    0             Honda    Petrol       Manual   First   Corporate        FWD
    1     Maruti Suzuki    Diesel       Manual  Second  Individual        FWD
    2           Hyundai    Petrol       Manual   First  Individual        FWD
    3            Toyota    Petrol       Manual   First  Individual        FWD
    4            Toyota    Diesel       Manual   First  Individual        RWD
    ...             ...       ...          ...     ...         ...        ...
    2054       Mahindra    Diesel       Manual   First  Individual        FWD
    2055        Hyundai    Petrol       Manual  Second  Individual        FWD
    2056           Ford    Petrol       Manual   First  Individual        FWD
    2057            BMW    Diesel    Automatic   First  Individual        RWD
    2058       Mahindra    Diesel       Manual   First  Individual        RWD
    
    [2059 rows x 6 columns]
          Year  Kilometer  Engine_CC  Power_Val  Power_RPM  Torque_Val  \
    0     2017      87150     1198.0       87.0     6000.0    109.0000   
    1     2014      75000     1248.0       74.0     4000.0    190.0000   
    2     2011      67000     1197.0       79.0     6000.0    112.7619   
    3     2019      37500     1197.0       82.0     6000.0    113.0000   
    4     2018      69000     2393.0      148.0     3400.0    343.0000   
    ...    ...        ...        ...        ...        ...         ...   
    2054  2016      90300     2179.0      138.0     3750.0    330.0000   
    2055  2014      83000      814.0       55.0     5500.0     75.0000   
    2056  2013      73000     1196.0       70.0     6250.0    102.0000   
    2057  2018      60474     1995.0      188.0     4000.0    400.0000   
    2058  2017      72000     1493.0       70.0     3600.0    195.0000   
    
          Torque_RPM  Length   Width  Height  Fuel Tank Capacity  
    0         4500.0  3990.0  1680.0  1505.0                35.0  
    1         2000.0  3995.0  1695.0  1555.0                42.0  
    2         4000.0  3585.0  1595.0  1550.0                35.0  
    3         4200.0  3995.0  1745.0  1510.0                37.0  
    4         1400.0  4735.0  1830.0  1795.0                55.0  
    ...          ...     ...     ...     ...                 ...  
    2054      1600.0  4585.0  1890.0  1785.0                70.0  
    2055      4000.0  3495.0  1550.0  1500.0                32.0  
    2056      4000.0  3795.0  1680.0  1427.0                45.0  
    2057      1750.0  4936.0  1868.0  1479.0                65.0  
    2058      1400.0  3995.0  1745.0  1880.0                 NaN  
    
    [2059 rows x 11 columns]
    


```python
# Pipelines - Preprocessing Step
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])


final_pipeline = Pipeline(steps=[
    ('cleaner', FunctionTransformer(engineering_specs_transformer)),
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

```


```python
X = df.drop(columns = 'Price')
y = df["Price"]

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

scoring = {
    'mae': 'neg_mean_absolute_error',
    'r2': 'r2',
    'mape' : 'neg_mean_absolute_percentage_error',
    'mse' : 'neg_mean_squared_error',
    'rmse' : 'neg_root_mean_squared_error'    
}

cv_results = cross_validate(final_pipeline, X, y, cv = kf, scoring = scoring, n_jobs = -1, return_estimator = True)
mae_scores = -cv_results['test_mae']
r2_scores = cv_results['test_r2']
mape_scores = -cv_results['test_mape']
mse_scores = -cv_results['test_mse']
rmse_scores = -cv_results['test_rmse']

print(f"Average R2 Score: {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
print(f"Average MAE: {mae_scores.mean():,.2f} (+/- {mae_scores.std():,.2f})")
print(f"\nMean Absolute Percentage Error: \n {100*mape_scores.mean():.4f}%")
print(f"\nMean Squared Error: \n {mse_scores.mean():.4f}")
print(f"\nRoot Mean Squared Error: \n {rmse_scores.mean():.4f}")

best_fold_index = np.argmax(cv_results['test_r2'])
best_model = cv_results['estimator'][best_fold_index]

```

    Average R2 Score: 0.8740 (+/- 0.0300)
    Average MAE: 286,876.39 (+/- 24,645.00)
    
    Mean Absolute Percentage Error: 
     17.5106%
    
    Mean Squared Error: 
     757064884703.1758
    
    Root Mean Squared Error: 
     856555.5068
    


```python
final_pipeline.fit(X, y)

ohe_features = final_pipeline.named_steps['preprocessor']\
               .transformers_[1][1]\
               .named_steps['ohe']\
               .get_feature_names_out(categorical_features)

all_feature_names = np.concatenate([numeric_features, ohe_features])

importances = final_pipeline.named_steps['regressor'].feature_importances_

feat_importances = pd.Series(importances, index = all_feature_names)

plt.figure(figsize=(10,6))
feat_importances.nlargest(15).plot(kind='barh', color='teal')
plt.title("Top 15 Drivers of Car Price")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis() # Highest at the top
plt.show()

```


    
![png](output_10_0.png)
    



```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results_df.head(10))


```

           Actual   Predicted
    1298  4800000  4823489.99
    591    825000   902199.99
    1318   695000   559299.99
    1067   950000   994469.99
    29     819999   795549.65
    1058   310000   371790.00
    712   1000000   963659.97
    453   1998999  2626049.98
    1646   850000  1513159.98
    757   3950000  5113609.96
    


```python
plt.figure(figsize = (10, 6))
plt.scatter(y_test, y_pred, alpha = 0.5, color = 'teal', label = 'Predictions')

limit = max(max(y_test), max(y_pred))
plt.plot([0, limit], [0, limit], color='red', linestyle='--', label='Perfect Prediction')

plt.title('Actual vs. Predicted Car Prices', fontsize=14)
plt.xlabel('Actual Price (y_test)', fontsize=12)
plt.ylabel('Predicted Price (y_pred)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R Score: \n{r2:.4f}")
print(f"\nMean Absolute Error: \n{mae:.4f}")
```


    
![png](output_12_0.png)
    


    R Score: 
    0.8225
    
    Mean Absolute Error: 
    299060.6473
    


```python
# Running SearchGridCV - Hyper parameter Tuning
from sklearn.model_selection import GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator = final_pipeline, 
    param_grid = param_grid, 
    cv = kf, 
    scoring = 'r2', 
    n_jobs = -1,
    verbose = 1
)

grid_search.fit(X, y)

print(f"Best Score (R2): {grid_search.best_score_:.4f}")
print(f"Best Parameters: {grid_search.best_params_}")

# 4. Use the best version for your final model
final_tuned_pipeline = grid_search.best_estimator_


```

    Fitting 5 folds for each of 12 candidates, totalling 60 fits
    Best Score (R2): 0.8771
    Best Parameters: {'regressor__max_depth': 20, 'regressor__min_samples_split': 2, 'regressor__n_estimators': 100}
    


```python
# Visualising Predictive Accuracy of GridSearchCV Model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
sns.regplot(x = y_test, y = y_pred_tuned, 
            scatter_kws = {'alpha':0.4, 'color':'teal'}, 
            line_kws = {'color':'red', 'label':'Best Fit Line'})

plt.title('Final Tuned Model: Actual vs. Predicted Prices', fontsize=14)
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Tuned R2 Score: {r2_score(y_test, y_pred_tuned):.4f}")
print(f"Tuned MAE: {mean_absolute_error(y_test, y_pred_tuned):,.2f}")

```


    
![png](output_14_0.png)
    


    Tuned R2 Score: 0.9765
    Tuned MAE: 111,125.31
    
