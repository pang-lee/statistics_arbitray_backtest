import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 定义股票代码和日期范围
tickers = ['6016.TWO', '6026.TWO']
start_date = '2024-01-01'
end_date = '2024-10-11'

# 获取数据
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# 将数据放入 pandas DataFrame
df = pd.DataFrame(data)
df['6016_diff'] = df['6016.TWO'].diff().dropna()
df['6026_diff'] = df['6026.TWO'].diff().dropna()

def plot_two_trend():
    # 绘图
    plt.figure(figsize=(10, 6))

    # 绘制每只股票的价格
    for ticker in tickers:
        plt.plot(data[ticker], label=ticker)

    plt.title('Stock Prices of 6016.TWO and 6026.TWO (2024-01-01 to 2024-10-11)')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid()
    plt.show()
    
# plot_two_trend()


def plot_mean(timeseries, col, label):
    plt.figure(figsize=(12, 6))

    for col, label in zip(col, label):
        # 计算 z-score： (值 - 均值) / 标准差
        col_zscore = (col - np.mean(col)) / np.std(col)

        # 绘制 z-score 标准化后的时间序列数据
        plt.plot(timeseries.index, col_zscore, label=label, alpha=0.7, color=np.random.rand(3,))

    # 绘制均值为0的水平线
    plt.axhline(y=0, color='black', label='Mean = 0')

    # z-score 标准化后的数据标准差为1
    std_val = 1

    # 添加标准差范围的水平线
    plt.axhline(0 - 1 * std_val, color='g', linestyle='--', label=f'Mean ± 1 * STD: {round(std_val, 2)}')
    plt.axhline(0 + 1 * std_val, color='g', linestyle='--')

    plt.axhline(0 - 2 * std_val, color='orange', label=f'Mean ± 2 * STD: {round(2 * std_val, 2)}')
    plt.axhline(0 + 2 * std_val, color='orange')

    plt.axhline(0 - 3 * std_val, color='red', label=f'Mean ± 3 * STD: {round(3 * std_val, 2)}')
    plt.axhline(0 + 3 * std_val, color='red')

    plt.legend()
    plt.title('Z-score Normalization of diff_pct and index_diff_pct')
    plt.xlabel('Date')
    plt.ylabel('Z-score')
    plt.show()
    
# 使用函數繪圖
# plot_mean(df, [df['6016.TWO'], df['6026.TWO']], ['6016.TWO', '6026.TWO'])

def pair_trade(data, s_col1, s_col2, t_col3, t_col4, threshold=1, **params):  
    # 计算价差
    spread = data[s_col1] - data[s_col2]

    # 计算价差的均值和标准差
    mean_spread = spread.mean()
    std_spread = spread.std()

    # 计算Z-score
    z_score = (spread - mean_spread) / std_spread
 
    # 设定阈值
    upper_threshold = threshold  # Z-score 上限
    lower_threshold = -threshold  # Z-score 下限
    
    # 初始化交易信号
    signal = pd.Series(0, index=z_score.index)

    # 生成交易信号：当 Z-score > upper_threshold，做空；Z-score < lower_threshold，做多
    signal[z_score > upper_threshold] = -1  # 做空
    signal[z_score < lower_threshold] = 1   # 做多

    # 初始化买入和卖出信号
    buy_signals = signal[signal == 1].index  # 做多 A 做空 B
    sell_signals = signal[signal == -1].index  # 做空 A 做多 B
    
    # 回测参数
    initial_capital = params.get('initial_capital', 100000)  # 初始资金
    commission = params.get('commission', 0.001)  # 手续费比例
    cash = initial_capital
    shares_a = 0  # 股票 A 的持仓
    shares_b = 0  # 股票 B 的持仓
    total_profit_loss = 0  # 总盈亏
    
    # 记录交易
    trades = []
    capital_over_time = []
    
    # 遍历信号，执行交易逻辑
    for i in range(1, len(signal)):
        current_signal = signal.iloc[i]
        prev_signal = signal.iloc[i-1]
        
        # 获取当前的价格
        price_a = data[t_col3].iloc[i]
        price_b = data[t_col4].iloc[i]

        # 如果当前信号不同于之前信号，表示产生了交易信号
        if current_signal != prev_signal:
            if current_signal == 1:
                # 做多：买入股票 A，卖出股票 B
                shares_a += 1  # 买入 A
                shares_b -= 1  # 卖出 B
                cash -= price_a  # 支出买入 A 的资金
                cash += price_b  # 收入卖出 B 的资金
                trades.append(f"Buy {t_col3} at {price_a}, Sell {t_col4} at {price_b}")

            elif current_signal == -1:
                # 做空：卖出股票 A，买入股票 B
                shares_a -= 1  # 卖出 A
                shares_b += 1  # 买入 B
                cash += price_a  # 收入卖出 A 的资金
                cash -= price_b  # 支出买入 B 的资金
                trades.append(f"Sell {t_col3} at {price_a}, Buy {t_col4} at {price_b}")

        # 计算当前的资产价值
        current_value = cash + shares_a * price_a + shares_b * price_b
        capital_over_time.append(current_value)

    # 记录回测的总盈亏
    total_profit_loss = capital_over_time[-1] - initial_capital
    
    
    if params.get('plot_singal', None):
        # 绘制价差和交易信号
        plt.figure(figsize=(10, 6))
        
        # 绘制价差
        plt.plot(spread.index, spread, label='Spread', color=np.random.rand(3,), linestyle='-', alpha=0.7)

        # 绘制 Z-score
        plt.plot(spread.index, z_score, label='Z-Score', color='blue')

        # 绘制均值
        plt.axhline(mean_spread, color='black', label='Mean Spread')

        # 绘制上、下阈值线 (交易用 threshold)
        plt.axhline(upper_threshold, color='red', label='Threshold ({}x Std)'.format(threshold))
        plt.axhline(lower_threshold, color='red')

        # 绘制1倍标准差线2倍、3倍标准差线
        plt.axhline(1, color='purple', linestyle='--', label='1x Std')
        plt.axhline(-1, color='purple', linestyle='--')
        plt.axhline(2, color='orange', linestyle=':', label='2x Std')
        plt.axhline(-2, color='orange', linestyle=':')
        plt.axhline(+3, color='g', linestyle='-.', label='3x Std')
        plt.axhline(-3, color='g', linestyle='-.')
        
        # 绘制买入信号 (做多A做空B)
        plt.scatter(buy_signals, z_score.loc[buy_signals], marker='^', color='red', label='Buy Signal', s=100)
        plt.scatter(sell_signals, z_score.loc[sell_signals], marker='v', color='green', label='Sell Signal', s=100)

        # 图表信息
        plt.title('Spread and Z-Score between {} and {} (Threshold = {})'.format(t_col3, t_col4, threshold))
        plt.xlabel('Date')
        plt.ylabel('Spread / Z-Score')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    if params.get('plot_trade', None):
        # 绘制价格和进出场信号
        plt.figure(figsize=(10, 6))

        # 绘制两只股票的价格
        plt.plot(data[t_col3], label=f'{t_col3} Price', color='blue')
        plt.plot(data[t_col4], label=f'{t_col4} Price', color='orange')

        # 绘制买入信号 (做多A做空B)
        plt.scatter(buy_signals, data.loc[buy_signals, t_col3], marker='^', color='red', label=f'Long {t_col3}', s=100)
        plt.scatter(buy_signals, data.loc[buy_signals, t_col4], marker='v', color='green', label=f'Short {t_col4}', s=100)

        # 绘制卖出信号 (做空A做多B)
        plt.scatter(sell_signals, data.loc[sell_signals, t_col3], marker='v', color='green', label=f'Short {t_col3}', s=100)
        plt.scatter(sell_signals, data.loc[sell_signals, t_col4], marker='^', color='red', label=f'Long {t_col4}', s=100)

        plt.title('Stock Prices and Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()

params = {
    'plot_singal': True,
    'plot_trade': False,
    'initial_capital': 100000,
    'commission': 0.001
}

pair_trade(df, '6016_diff', '6026_diff', '6016.TWO', '6026.TWO', threshold=2, **params)