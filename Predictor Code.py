# Imports
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Import Check
print("Check 1")

# Load Suicide Data
data = pd.read_csv('Final Data Set - Big Data Project.csv')

# Load Prediction Data, Replace "testing_data.csv" With Actual Filename
testing_data = pd.read_csv('testing_data.csv')

# Data Check
print("Check 2")

# Seperate Variables (X) and Suicide Rate (Y)
X = data.drop('suicides100kpop', axis=1)
Y = data['suicides100kpop']

# Variable Name Check
print("Check 3")
print(testing_data.columns)
print(data.columns)

# Input Desired Values Into Prediction File.
# Age(y), Sex (0 = Male and 1 = Female), HDI (0 to 1), GDP per capita ($USD)

# Isolates Input Variables
X_test = testing_data.drop('suicides100kpop', axis=1)

# Random Forest
random_forest = RandomForestRegressor()
random_forest.fit(X, Y)

# Predictions
rf_pred = random_forest.predict(X_test)
print("Random Forest Predictions:", rf_pred)
