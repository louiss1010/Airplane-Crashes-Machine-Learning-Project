# Airplane Crash Fatality Prediction

Regression project predicting the number of fatalities in historical airplane crashes.

## Task

Predict `Fatalities` for each crash record, then generate predictions on an unseen holdout set. Evaluated on RMSE.

## Dataset

Historical aviation accident records (~4,700 crashes), including date, operator, aircraft type, registration, number of people aboard, ground fatalities, and a free text crash summary. Source: [Kaggle, Airplane Crashes and Fatalities](https://www.kaggle.com/datasets/thedevastator/airplane-crashes-and-fatalities).

## Approach

The key idea is reframing the target. Instead of predicting `Fatalities` directly, the model predicts the fatality rate (`Fatalities / Aboard`), then multiplies back by `Aboard` to get a count. This handles the fact that fatalities scale with aircraft size in a way a single linear model cannot represent well.

Final model: Lasso regression with engineered features including aircraft registration country, a year by aircraft size interaction term, and TF-IDF features from the crash summary text. Validated across 11 random seeds with 5 fold cross validation, using leakage safe preprocessing (categories and interaction terms recomputed per fold).

Ridge regression was also tested as a comparison model.

## Results

Mean RMSE approximately 16.8, R squared approximately 0.73. This compares to RMSE of approximately 20.8 when predicting fatalities directly rather than as a rate, confirming the target reframing as the single largest driver of model performance.

## Files

- `louis_strehlow_airplane_crashes_model.ipynb`: full analysis, from data cleaning through to unseen set predictions
- `louis_strehlow_airplane_crashes.csv`: final predictions on the unseen set
