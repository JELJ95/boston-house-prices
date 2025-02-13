import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load the dataset and manually assign column names.
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dataset = pd.read_csv('housing.csv', header=None, names=column_names, sep='\s+')

# Feature Engineering: Create new features
dataset['TAX_per_ROOM'] = dataset['TAX'] / dataset['RM']    # TAX_per_ROOM = Property tax divided by number of rooms
dataset['AGE_per_DIS'] = dataset['AGE'] / dataset['DIS']    # Age of buildings relative to distance to workplaces
dataset['LSTAT_squared'] = dataset['LSTAT'] ** 2            # Square LSTAT to capture non-linear patterns
dataset['log_CRIM'] = np.log1p(dataset['CRIM'])             # Log transformation of CRIM

# Check dataset after Feature Engineering
print(dataset[['TAX_per_ROOM', 'AGE_per_DIS', 'LSTAT_squared', 'log_CRIM']].head())

# Visualize correlation using a heatmap to determine if more features should be removed
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Heatmap of Feature Correlations (after Feature Engineering)")
plt.show()

# Remove unnecessary features based on correlation analysis
dataset.drop(columns=['RAD', 'CRIM'], inplace=True)  # Removed `RAD` (high correlation with `TAX`) and `CRIM` (replaced with `log_CRIM`)

# Standardize dataset (excluding 'MEDV')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(dataset.drop(columns=['MEDV']))
scaled_dataset = pd.DataFrame(scaled_features, columns=dataset.drop(columns=['MEDV']).columns)

# Add back 'MEDV' (target variable)
scaled_dataset['MEDV'] = dataset['MEDV']

# Split data into training and test sets
X = scaled_dataset.drop(columns=['MEDV'])
y = scaled_dataset['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Create Grid Search for XGBoost
xgb_model = XGBRegressor(random_state=0)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=2)

# Train the model
grid_search.fit(X_train, y_train)

# Retrieve best hyperparameters and train the best model
print("Best hyperparameters:", grid_search.best_params_)
best_xgb = grid_search.best_estimator_

# Make predictions and evaluate the model
y_pred = best_xgb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Optimized XGBoost with Feature Engineering:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")