# Statistics_Arbitray_Backtest In Taiwan Stock Market
```
Using mathematics to trade at stock market
In test folder:

Index_data folder: including some of the Taiwan Stock Market Data

getdata.py: using for getting the data via API

mean_reversion.py: caculating the sprad of two assets, while they spread over the mean, Long A Short B
=> It's a basic strategy of mean revesrion.

model.py: Some of the Math model which can use to evaluate the assets, and It's can also predict the stock by using some of the model
=> Including
Evaluate: ADF, Hurst, Kpass ... etc
Math/MachineLearning Model: VECP, OLS, ARIMA, GARCH
DeepLearning Model: LSTM

yf_mean_reverion.py:
Using yahoo Y-fiance stock data, to do a long-term mean reversion strtegy backtest
same as the mean-reversion.py, caculating the sprad of two assets, while they spread over the mean, Long A Short B
```
