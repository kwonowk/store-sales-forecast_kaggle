# store-sales-forecast_kaggle

https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

-Objective

Time series

-Data

-Approach
1.Hard coding
LSTM

2.Time series packages
Auto_TS
Darts
ETNA
Pycaret

3.Pre trained models
-TFT, what else?


- First submission (0.057679): Setup for modeling per family in each stores, with a blunt model (RandomForestRegressor)
- Second submission (0.46133): Auto-arima applied to individual product family in each store (untreated 0 values at the beginning)
- Third submission (0.45510, 157/627) : Auto-arima applied to individual product family in each store, with promotion and national holidays as exogenous variables
- Fourth submission (1.90147) : HistGradientBoostingRegressor with dates tranformed using teigonometric features
- Fifth submission (1.79006) : HistGradientBoostingRegressor with lag features and window rolling prediction
