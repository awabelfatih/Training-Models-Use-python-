from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Create sample data
point = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'label': [0, 1, 0, 1, 0]
})

chunk_point = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': [5, 4, 3],
    'label': [0, 1, 0]
})
# Clean data
chunk_point = chunk_point.dropna().reset_index(drop=True)
columns = ['feature1', 'feature2', 'label']

# Visualize the data
plt.figure(figsize=(10, 6))
sns.scatterplot(x='feature1', y='feature2', hue='label', data=chunk_point, palette='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Features')
plt.legend(title='Label')
plt.show()

# Train/test split (not used further, but shown for completeness)
# X_train, X_test, y_train, y_test = train_test_split(chunk_point[['feature1', 'feature2']], chunk_point['label'], test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(chunk_point[['feature1', 'feature2']], chunk_point['label'])

# Make predictions
predictions = model.predict(chunk_point[['feature1', 'feature2']])

# Evaluate
accuracy = accuracy_score(chunk_point['label'], predictions)
conf_matrix = confusion_matrix(chunk_point['label'], predictions)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Remove duplicates and visualize again
chunk_point = chunk_point.drop_duplicates().reset_index(drop=True)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='feature1', y='feature2', hue='label', data=chunk_point, palette='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Features (No Duplicates)')
plt.legend(title='Label')
plt.show()