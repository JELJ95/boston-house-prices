# Boston House Price Prediction 🏡

## 📌 Prosjektbeskrivelse

Dette prosjektet bruker maskinlæring for å predikere boligpriser i Boston. Jeg tester flere modeller og optimaliserer ytelsen ved bruk av Feature Engineering og hyperparameter-tuning. Du finner en engelsk versjon av prosjektet under mappen "English".

## 📂 Datasett

- **Boston House Prices** (Kaggle) https://www.kaggle.com/datasets/vikrishnan/boston-house-prices
- Datasettet inneholder boligpriser og variabler som blant annet kriminalitet, antall rom, skatter osv.

## 🛠️ Teknologier og verktøy

- **Python 🐍**
- **Biblioteker:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost

## 🔬 Feature Engineering
For å forbedre modellen, lagde jeg nye features og transformerte eksisterende data:
- **TAX_per_ROOM** = `TAX` (eiendomsskatt) delt på `RM` (antall rom)  
- **AGE_per_DIS** = `AGE` (bygningens alder) delt på `DIS` (avstand til jobber)  
- **LSTAT_squared** = `LSTAT` (andel lavinntektsfamilier) **i andre potens**  
- **log_CRIM** = Log-transformasjon av `CRIM` (kriminalitet per område)  

## 🔧 Modelloptimalisering
- **Hyperparameter-tuning med Grid Search**  
- **Testet Random Forest og XGBoost**  
- **Valgte XGBoost som beste modell etter Feature Engineering**  


## 🏆 Modeller og Resultater

| Modell                     | MAE  | MSE   | R²-score |
| -------------------------- | ---- | ----- | -------- |
| **Linear Regression**      | 3.96 | 35.75 | 0.56     |
| **Random Forest**          | 2.89 | 26.42 | 0.68     |
| **Standard XGBoost**       | 2.84 | 22.67 | 0.72     |
| **XGBoost + Feature Eng.** | 2.68 | 19.60 | 0.76     |

XGBoost med Feature Engineering ga **den beste ytelsen** med lavest feil og høyest R²-score.

## 🚀 Hvordan kjøre prosjektet

1. **Installer nødvendige pakker**
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost
   ```
2. **Last ned datasettet**
3. **Kjør Python-skriptet eller åpne Jupyter Notebook**

## 📊 Hva lærte jeg?

- **Feature Engineering forbedret ytelsen**
- **XGBoost var den beste modellen**
- **Hyperparameter-tuning hjalp med å finne optimale verdier**

## 📧 Kontakt

Hvis du har spørsmål, ta kontakt med Julie Jansen på julie_emmy_95@hotmail.com eller www.linkedin.com/in/julie-jansen-a73232138.
