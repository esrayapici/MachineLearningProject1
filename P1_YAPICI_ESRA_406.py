### Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import VotingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

### Data Loading
train = pd.read_csv('/content/drive/MyDrive/esraOdev/train.csv')
val = pd.read_csv('/content/drive/MyDrive/esraOdev/val.csv')
test = pd.read_csv('/content/drive/MyDrive/esraOdev/test.csv')

### Data Visualization: Initial Exploration
# General overview of datasets
print("Train Set Shape:", train.shape)
print("Validation Set Shape:", val.shape)
print("Test Set Shape:", test.shape)

# Missing values visualization
plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull(), cbar=False, cmap="viridis", yticklabels=False)
plt.title("Missing Values in Training Data")
plt.show()

# Target distribution in train set
plt.figure(figsize=(8, 6))
sns.histplot(train['score'], kde=True, bins=20, color='blue')
plt.title("Score Distribution in Training Data")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()

# Text length distribution in train set
train['text_length'] = train['text'].fillna('').apply(len)
plt.figure(figsize=(8, 6))
sns.histplot(train['text_length'], bins=30, color='green', kde=True)
plt.title("Text Length Distribution in Training Data")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.show()

### Data Preprocessing
train['text'] = train['text'].fillna('missing')
val['text'] = val['text'].fillna('missing')
test['text'] = test['text'].fillna('missing')

vectorizer = TfidfVectorizer(max_features=2000)
X_train_full = vectorizer.fit_transform(train['text']).toarray()
y_train_full = train['score']
X_val = vectorizer.transform(val['text']).toarray()
y_val = val['score']
X_test = vectorizer.transform(test['text']).toarray()

X_train, X_val_split, y_train, y_val_split = train_test_split(X_train_full, y_train_full,
                                                              test_size=0.2, random_state=42)

### Hyperparameter Optimization
param_grid_gb = {
    'n_estimators': [50, 100],
    'max_depth': [5, 7],
    'learning_rate': [0.05, 0.1]
}
grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_gb.fit(X_train, y_train)
best_gb = grid_gb.best_estimator_

# Plotting hyperparameter tuning results
cv_results = pd.DataFrame(grid_gb.cv_results_)
plt.figure(figsize=(10, 6))
sns.barplot(data=cv_results, x='param_n_estimators', y='mean_test_score', hue='param_learning_rate')
plt.title("Mean Test Scores Across Different Hyperparameters (Gradient Boosting)")
plt.xlabel("n_estimators")
plt.ylabel("Mean Test Score (neg MSE)")
plt.legend(title='learning_rate')
plt.show()

### Additional Models
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 4]
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

### Neural Network
cnn_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='linear')
])
cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
cnn_model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_val_split, y_val_split))
cnn_predictions = cnn_model.predict(X_val_split).flatten()

### Ensemble Model
ensemble = VotingRegressor(estimators=[
    ('gb', best_gb),
    ('rf', best_rf)
])
ensemble.fit(X_train, y_train)
ensemble_predictions = ensemble.predict(X_val_split)

### Model Evaluation
metrics = ["MSE", "MAE", "Spearman", "Kendall"]
results = {metric: [] for metric in metrics}

predictions = {
    "Gradient Boosting": best_gb.predict(X_val_split),
    "Random Forest": best_rf.predict(X_val_split),
    "CNN": cnn_predictions,
    "Ensemble": ensemble_predictions
}

for name, pred in predictions.items():
    results["MSE"].append(mean_squared_error(y_val_split, pred))
    results["MAE"].append(mean_absolute_error(y_val_split, pred))
    results["Spearman"].append(spearmanr(y_val_split, pred)[0])
    results["Kendall"].append(kendalltau(y_val_split, pred)[0])

results_df = pd.DataFrame(results, index=predictions.keys())
print("\nModel Evaluation Results:")
print(results_df)

# Plot the evaluation metrics
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=results_df.index, y=results_df[metric])
    plt.title(f"Model Performance Comparison ({metric})")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.show()

### Submission File
final_predictions = ensemble.predict(X_test)
submission = pd.DataFrame({
    'id': test['id'],
    'score': final_predictions
})
submission.to_csv('/content/drive/MyDrive/esraOdev/submission_3_new.csv', index=False)
print("Submission file created successfully!")
