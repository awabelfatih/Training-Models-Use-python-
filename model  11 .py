import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Prepare data
df = pd.DataFrame({
    'column': [89990, 89970, 877889],
    'price': [1000, 2000, 2120]
})

X = df[['column']]
y = df['price']

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
predictions = model.predict(x_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(x_test, predictions, color='red', label='Prediction')
plt.x_label('Column')
plt.y_label('Price')
plt.title('Linear Regression: Price vs Column')
plt.legend()
plt.show()