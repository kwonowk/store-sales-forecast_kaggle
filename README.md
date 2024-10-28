# Kaggle competition : Store Sales - Time Series Forecasting

## Overview
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

## Objective
Become familiar in working with time series data, and practice time-based forecasting

## Data
Time series data tracking product sales across stores of a supermarket chain, categorized by store location and product family in each store. On-going promotions are also provided in the saled data. Detailed store information, list of holidays and events, as well as oil price are provided as separate data.

## Methodology and results

1. Univariate forecast by using SARIMA
- `Auto-ARIMA` applied to individual product family in each store.
-  Submission score 0.45655
2. Multivariate forecast by using SARIMAX
- `Auto-ARIMA` applied to individual product family in each store, with promotion, national holidays, and oil price as exogenous variables.
- Submission score 0.47397
3. Machine Learning
- HistGradientBoostingRegressor used for prediction, with promotion, national holidays, and oil price as variables.
- Dates tranformed using teigonometric features to indicate seasonal and cyclical characteristic of data, which are 1)Day in a given year 2)Day in a given month and 3)Day in a given week
- Lag features of 3-day lags used with window rolling prediction
- Submission score 3.28478

## Takeaways and limitation


## Next steps


3.Pre trained models
-TFT, what else?


- First submission (0.45655): Auto-arima applied to individual product family in each store (untreated 0 values at the beginning)
- Second submission (0.47397) : Auto-arima applied to individual product family in each store, with promotion, national holidays, and oil price as exogenous variables
- Third submission (1.90147) :


For later:

LSTM / Facebook prophet
