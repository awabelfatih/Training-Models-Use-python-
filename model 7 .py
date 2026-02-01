from pandas import DataFrame
import pandas as pd
import scipy as sp 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
# Sample data
us = pd.DataFrame({
    'info': ['data', 'source', 'modules'],
    'Date': ['2002-03-11', '2020-02-15', '2000-04-05'],
    'confirmation': [20000, 30000, 15000],
    'Mileage': [15000, 20000, 5000],
    'Color': ['red', 'blue', 'green'],
    'Condition': ['new', 'used', 'new'],
    'Location': ['NY', 'CA', 'TX'],
})

# Convert categorical columns to numeric
us_encoded = pd.get_dummies(us, columns=['info', 'Color', 'Condition', 'Location'])
us_encoded['Date'] = pd.to_datetime(us_encoded['Date']).astype(int) / 10**9  # Convert date to timestamp

# Features and target
X = us_encoded.drop('confirmation', axis=1)
y = us_encoded['confirmation']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
# Metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


