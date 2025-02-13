import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Last inn datasettet og legg til kolonner manulet.
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

dataset = pd.read_csv('housing.csv', header=None, names=column_names, sep='\s+')

# Feature Engineering: Lag nye features
dataset['TAX_per_ROOM'] = dataset['TAX'] / dataset['RM']    # TAX_per_ROOM = Eiendomsskatt delt på antall rom
dataset['AGE_per_DIS'] = dataset['AGE'] / dataset['DIS']    # Hvor gamle bygningene er i forhold til avstand til arbeidsplasser
dataset['LSTAT_squared'] = dataset['LSTAT'] ** 2            # Kvadrere LSTAT for å fange ikke-lineære mønstre
dataset['log_CRIM'] = np.log1p(dataset['CRIM'])             # Log-transformasjon av CRIM

# Sjekk datasettet etter Feature Engineering er gjennomført
print(dataset[['TAX_per_ROOM', 'AGE_per_DIS', 'LSTAT_squared', 'log_CRIM']].head())

# Visualiser korrelasjon med et heatmap for å se om det bør fjernes flere features
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Heatmap of Feature Correlations (after Feature Engineering)")
plt.show()

# Fjern unødvendige features basert på korrelasjon
dataset.drop(columns=['RAD', 'CRIM'], inplace=True) # Fjernet `RAD` (sterk korrelasjon med `TAX`) og `CRIM` (erstattet med `log_CRIM`)

# Standardiser datasettet (uten 'MEDV')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(dataset.drop(columns=['MEDV']))
scaled_dataset = pd.DataFrame(scaled_features, columns=dataset.drop(columns=['MEDV']).columns)

# Legg tilbake 'MEDV' (målvariablen)
scaled_dataset['MEDV'] = dataset['MEDV']

# Splitt dataen i trenings- og testsett
X = scaled_dataset.drop(columns=['MEDV'])
y = scaled_dataset['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Definer parametergrid for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Opprett Grid Search for XGBoost
xgb_model = XGBRegressor(random_state=0)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=2)

# Tren modellen
grid_search.fit(X_train, y_train)

# Hent de beste hyperparametrene og tren den beste modellen
print("Beste hyperparametere:", grid_search.best_params_)
best_xgb = grid_search.best_estimator_

# Lag prediksjoner og evaluer modellen
y_pred = best_xgb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print resultatene
print(f"Optimalisert XGBoost med Feature Engineering:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.2f}")