from ucimlrepo import fetch_ucirepo 
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# fetch dataset, clean date/time into workable data, and drop NA values
individual_household_electric_power_consumption = fetch_ucirepo(id=235)
individual_household_electric_power_consumptionDF = individual_household_electric_power_consumption.data.features.copy()

# EDA
print('Description of Data Set:')
print(individual_household_electric_power_consumptionDF.describe())
print('Information about Data Set:')
individual_household_electric_power_consumptionDF.info()

individual_household_electric_power_consumptionDF['dateTime'] = pd.to_datetime(individual_household_electric_power_consumptionDF['Date'] + ' ' + individual_household_electric_power_consumptionDF['Time'], format='%d/%m/%Y %H:%M:%S')
individual_household_electric_power_consumptionDF = individual_household_electric_power_consumptionDF.drop(columns=['Date','Time'])

for col in individual_household_electric_power_consumptionDF.columns.drop('dateTime'):
    individual_household_electric_power_consumptionDF[col] = pd.to_numeric(individual_household_electric_power_consumptionDF[col],errors='coerce')
individual_household_electric_power_consumptionDF = individual_household_electric_power_consumptionDF.dropna()

individual_household_electric_power_consumptionDF['hour'] = individual_household_electric_power_consumptionDF['dateTime'].dt.hour
individual_household_electric_power_consumptionDF['day'] = individual_household_electric_power_consumptionDF['dateTime'].dt.dayofweek
individual_household_electric_power_consumptionDF['month'] = individual_household_electric_power_consumptionDF['dateTime'].dt.month

individual_household_electric_power_consumptionDF = individual_household_electric_power_consumptionDF.sort_values('dateTime')
y_reg = (individual_household_electric_power_consumptionDF['Global_active_power'])
x_reg = individual_household_electric_power_consumptionDF.drop(columns=['Global_active_power','dateTime','Global_intensity'])
feature_names = [str(x_reg)]

#Splitting data chronologically for stronger results because it is a time series
split = int(len(individual_household_electric_power_consumptionDF)*0.8)
X_train, X_test = x_reg.iloc[:split], x_reg.iloc[split:]
y_train, y_test = y_reg.iloc[:split], y_reg.iloc[split:]

scaler = StandardScaler().fit(X_train)
Z_train = scaler.transform(X_train)
Z_test = scaler.transform(X_test)

#Naive Regression Model (Persistence, Seasonal)
naivePersist = y_reg.shift(1).iloc[split:]
naiveDaily = y_reg.shift(1440).iloc[split:]

#Linear Regression Model
energyLR = LinearRegression()
energyLR.fit(Z_train,y_train)

#Random Forest (Regression) Model
energyRF = RandomForestRegressor(n_estimators=100,max_depth=None,random_state=42,n_jobs=-1)
energyRF.fit(Z_train,y_train)

predictionsLR = energyLR.predict(Z_test)
predictionsRF = energyRF.predict(Z_test)
alphaRidge = 500
ridge = Ridge(alpha=alphaRidge).fit(Z_train,y_train)
predictionsRidge = ridge.predict(Z_test)

print("\nPrediction vs. True (LR)")
for i in range(10):
    print(f"{predictionsLR[i]} --- {y_test.iloc[i]}")
print("\nPrediction vs. True (RF)")
for i in range(10):
    print(f"{predictionsRF[i]} --- {y_test.iloc[i]}")

print('Model Comparisons')
print("-" * 20)
print("Naive Persistence (Minute by Minute)")
print("-" * 20)
print("\nMean Absolute Error (Naive Persistence):", mean_absolute_error(y_test,naivePersist))
print("\nRMSE (Naive Persistence):", np.sqrt(mean_squared_error(y_test,naivePersist)))
print("\nR2 Score (Naive Persistence):", r2_score(y_test,naivePersist))
print("-" * 20)
print("Naive Daily (Day by Day)")
print("-" * 20)
print("\nMean Absolute Error (Naive Daily):", mean_absolute_error(y_test,naiveDaily))
print("\nRMSE (Naive Daily):", np.sqrt(mean_squared_error(y_test,naiveDaily)))
print("\nR2 Score (Naive Daily):", r2_score(y_test,naiveDaily))
print("-" * 20)
print("Linear Regression")
print("-" * 20)
print("\nMean Absolute Error (LR):", mean_absolute_error(y_test,predictionsLR))
print("\nRMSE (LR):", np.sqrt(mean_squared_error(y_test,predictionsLR)))
print("\nR2 Score (LR):", r2_score(y_test,predictionsLR))
print("-" * 20)
print("Random Forest (Regression)")
print("-" * 20)
print("\nMean Absolute Error (RF):", mean_absolute_error(y_test,predictionsRF))
print("\nRMSE (RF):", np.sqrt(mean_squared_error(y_test,predictionsRF)))
print("\nR2 Score (RF):", r2_score(y_test,predictionsRF))
print("-" * 20)
print("Ridge")
print("-" * 20)
print("\nMean Absolute Error (Ridge):", mean_absolute_error(y_test,predictionsRidge))
print("\nRMSE (Ridge):", np.sqrt(mean_squared_error(y_test,predictionsRidge)))
print("\nR2 Score (Ridge):", r2_score(y_test,predictionsRidge))