import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Loading the provided dataset
df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/Oasis Infobytes Internship Linear Regression/Housing.csv')

# Checking for missing data
print(df.isnull().sum())

# Converting categorical variables to dummy variables
df = pd.get_dummies(df, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], drop_first=True)

# Defining the feature set and target variable
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad_yes', 'guestroom_yes', 'basement_yes', 'hotwaterheating_yes', 'airconditioning_yes', 'parking', 'prefarea_yes', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']]
y = df['price']

# Creating and fitting the model
reg = linear_model.LinearRegression()
reg.fit(X, y)

# Displaying the coefficients and intercept
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

# Making predictions
y_pred = reg.predict(X)

# Calculating accuracy metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plotting the results
plt.scatter(y, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

'''
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Loading the provided dataset
df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/Oasis Infobytes Internship/Housing.csv')

# Checking for missing data
print(df.isnull().sum())

# Converting categorical variables to dummy variables
df = pd.get_dummies(df, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], drop_first=True)

# Defining the feature set and target variable
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad_yes', 'guestroom_yes', 'basement_yes', 'hotwaterheating_yes', 'airconditioning_yes', 'parking', 'prefarea_yes', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']]
y = df['price']

# Creating and fitting the model
reg = linear_model.LinearRegression()
reg.fit(X, y)

# Displaying the coefficients and intercept
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

# Making predictions
y_pred = reg.predict(X)

# Calculating accuracy metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plotting the results
plt.scatter(y, y_pred, color='blue')
plt.plot(y, y, color='red', linewidth=2)  # Line of best fit
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
'''
'''
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats

# Loading the provided dataset
df = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/Oasis Infobytes Internship/Housing.csv')

# Converting categorical variables to dummy variables
df = pd.get_dummies(df, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], drop_first=True)

# Ensuring all columns are numeric and filling NaN values with column mean
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean())

# Defining the feature set and target variable
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y = df['price']

# Identifying outliers using Z-score
z_scores = np.abs(stats.zscore(X))
outliers = (z_scores > 3).any(axis=1)

# Removing outliers
df_clean = df[~outliers]

# Defining the cleaned feature set and target variable
X_clean = df_clean[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
y_clean = df_clean['price']

# Creating and fitting the model with cleaned data
reg = linear_model.LinearRegression()
reg.fit(X_clean, y_clean)

# Making predictions
y_pred = reg.predict(X_clean)

# Calculating accuracy metrics
mse = mean_squared_error(y_clean, y_pred)
r2 = r2_score(y_clean, y_pred)
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plotting the results
plt.scatter(y_clean, y_pred, color='blue')
plt.plot(y_clean, y_clean, color='red', linewidth=2)  # Line of best fit
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Cleaned Data)")
plt.show()'''



