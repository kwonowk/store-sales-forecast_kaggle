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
  3.1. `HistGradientBoostingRegressor` (in `scikit-learn`) as estimator, with promotion, national holidays, and oil price as variables
  - Dates tranformed to cyclical encoding using trigonometric functions to indicate seasonal and cyclical characteristic of data, which are :
    1) Day in a given year
    2) Day in a given month and
    3) Day in a given week
  - Lag features of 3-day lags used with window rolling prediction, predicting per row
  - Submission score 3.28478
  3.2. `LightGBMRegressor` (in `LightGBM`) as estimator, with promotion, national holidays, and oil price as variables
  - Used cyclical encoding, but lag-feature was not used as it's incompatible with training and predicting in batches
  - Trained and predicted on data per product category, in other words, executed the process for 33 times
  - Conducted parameter search on 20% of each product data with limited grid options for efficient use of memory
  - Submission score 1.82569


## Takeaways and limitation
- May have spent too much time on the exercise than what I should have. This was mainly due to constant kernel crashes blocking advancements, when using HistGradientBoostingRegressor. Crashes were frequent when fitting the model, which I suspect is linked to the size of the data being around 2 GB after the preprocessing (123 columns). Also used Kaggle's online notebook to run the code, but it also runs for indefiniate amount of time, without any indication. As this exercise was coded from scratch without consulting others' codes, building on already functioning notebook could also be an option to take. Using less teigonometric features could also lessen the load.
  - [Update (with 3.2.)] As training in batches seemed to be more appropriate for the task, tried making predictions by product groups (i.e. in 33 batches), which stopped crashes. Number of features decreased, as there's no more need to have product columns. A lesson learned.
- Related to the point above, one of the results obtained with ML technique performed worse than more traditional ARIMA, so there might have been features that are more of noise.
- For large dataset, conducting estimator parameter search on smaller set of dataset could be an option when efficiency and time saving is important. There are always trade-offs to be made when making methodological decisions.


## Future experimentation
- Focus also on making features more precise, such as differentiating between national and local events, and generating additional features, including events
- Work with less complex data to properly apply time series-focused feature engineering
- Consider using other packages (such as dart), deep learning models (LSTM) and pre-trained models (notably Facebook Prophet)
