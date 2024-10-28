# Kaggle competition : Store Sales - Time Series Forecasting



## Overview
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data


## Objective
Build proficiency in working with time series data by practicing forecasting techniques to improve store sales predictions


## Data
- Time series data tracking product sales across stores of a supermarket chain, categorized by store location and product family in each store. On-going promotions are also provided in the saled data.
- Detailed store information
- List of holidays and events
- Oil price


## Methodology and results
Tools used are highlighted in `pre-formatting`
0. Exploratory Data Analysis
-  Using `pandas` and `matplotlib`
1. Univariate forecast by using SARIMA
- Auto-ARIMA (in `pmdarima`) applied to individual product family in each store.
-  Submission score 0.45655
2. Multivariate forecast by using SARIMAX
- Auto-ARIMA (in `pmdarima`) applied to individual product family in each store, with promotion, national holidays, and oil price as exogenous variables.
- Submission score 0.47397
3. Machine Learning technique
- `HistGradientBoostingRegressor` (in `scikit-learn`) used for prediction, with promotion, national holidays, and oil price as variables.
- Dates tranformed using teigonometric features to indicate seasonal and cyclical characteristic of data, which are :
  1)Day in a given year
  2)Day in a given month and
  3)Day in a given week
- Lag features of 3-day lags used with window rolling prediction
- Submission score 3.28478


## Takeaways and limitation
- May have spent too much time on the exercise than what I should have. This was mainly due to constant kernel crashes blocking advancements. Crashed were frequent when fitting the model, which I suspect is linked to the size of the data being around 2 GB after the preprocessing. Also used Kaggle's online notebook to run the code, and it also runs without any indication. As this exercise was coded from scratch without consulting others' codes, building on already functioning notebook could also be an option to take.
- Related to the point above, one of the results obtained with ML technique performed worse than more traditional ARIMA, so there might have been features that are more of a noise.


## Future experimentation
- Work with less complex data
- Use deep learning models and pre-trained models (notably Facebook Prophet)
