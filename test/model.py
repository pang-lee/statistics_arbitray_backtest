import numpy as np
import pandas as pd
import talib, math
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, kpss, coint, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from filterpy.kalman import KalmanFilter
from scipy import stats
from scipy.stats import boxcox
from statsmodels.tsa.api import VAR, VECM, VARMAX
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from scipy.stats import jarque_bera, ttest_rel
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_arch

# ------------------------ 加載資料 ------------------------

def prepare_data(type_of_data, data_name):
    result = type_of_data.split('/')[0]
    tmp = pd.read_csv(f'./index_data/{type_of_data}/{data_name}.csv')
    if result == 'shioaji':
        tmp['ts'] = pd.to_datetime(tmp['ts'])
    else:
        tmp['ts'] = pd.to_datetime(tmp['datetime'])

    return tmp

df1 = prepare_data('fugle/2024_0708', '00631L')
df2 = prepare_data('fugle/2024_0708', '00663L')
df3 = prepare_data('shioaji/2024_0708', '2330')


# Fugle資料合併兩張表
# merge1 = pd.merge(df1, df2, on='ts', how='outer', suffixes=('_stock1', '_stock2'))[['ts', 'close_stock1', 'close_stock2']]
# 若使用指數資料, 可使用下
# df3.set_index('ts', inplace=True)
# df3 = df3.resample('min').last()
# # 计算每分钟的变化率
# df3['index'] = df3['close'].ffill()

# -------------------------- 資料前處理 -------------------------

# 合併數據，按時間戳對齊
df1_renamed = df1.rename(columns={'close': 'close_stock1', 'open': 'open_stock1'})
df3_renamed = df3.rename(columns={'Close': 'close_stock3'})
mer_ori_data = pd.merge(df1_renamed, df3_renamed, on='ts', how='outer')[['ts', 'open_stock1', 'close_stock1', 'close_stock3']]

mer_ori_data['diff'] = mer_ori_data['close_stock1'].diff()
mer_ori_data['diff3'] = mer_ori_data['close_stock3'].diff()
mer_ori_data['diff3'] = mer_ori_data['diff3'].shift(1)

# 使用前一笔的数据填充 NaN
mer_ori_data.ffill(inplace=True)

# 設置時間戳列為索引
mer_ori_data.set_index('ts', inplace=True)

# 删除含有NaN的行
mer_ori_data.dropna(inplace=True)

# -------------------------- 使用指定的日期 -------------------------

# 提取特定日期的数据
start_date = '2024-08-05'
end_date = '2024-08-05'
merged_data = mer_ori_data.loc[start_date:end_date]

# 按时间段筛选该日期从 9:02 到 13:24 的数据
merged_data = merged_data.between_time('09:02', '13:24')

# 以分钟为单位重采样，并用前一笔的数值填充缺失值
merged_data = merged_data.resample('min').ffill()

# 删除含有NaN的行
merged_data.dropna(inplace=True)

# -------------------------- 資料轉換方法 -------------------------

# Box-Cox 轉換(數據不能負數)
# def box_cox(merged_data, cols, plot=False):
#     # 檢查是否有負值或零值
#     if (merged_data[cols] <= 0).any():
#         # 如果有負值或零值，將數據加上一個常數（例如 1.01）使其全部為正
#         shifted_data = merged_data[cols] + abs(merged_data[cols].min()) + 1.01
#     else:
#         shifted_data = merged_data[cols]

#     # 進行Box-Cox轉換
#     boxcox_data, lambda_value = boxcox(shifted_data)

#     # 將轉換後的數據存入新的列
#     merged_data['boxcox'] = boxcox_data

#     # 打印最佳λ值
#     print(f"Box-Cox 轉換的最佳 λ 值為: {lambda_value}")

#     if plot is True:
#         # 可視化轉換前後的數據分佈
#         plt.figure(figsize=(12,6))

#         # 原始數據的直方圖
#         plt.subplot(1, 2, 1)
#         plt.hist(merged_data[cols].dropna(), bins=30)
#         plt.title('Original Distribution')

#         # Box-Cox轉換後的直方圖
#         plt.subplot(1, 2, 2)
#         plt.hist(merged_data['boxcox'].dropna(), bins=30)
#         plt.title('Box-Cox Distribution')

#         plt.show()

# box_cox(merged_data, 'close_stock1')

# 使用 Yeo-Johnson 轉換数据(數據可以負值)
# pt = PowerTransformer(method='yeo-johnson')
# merged_data['yj_transform'] = pt.fit_transform(merged_data[['diff']])


# def kalman_filter(merged_data):
#     merged_data.dropna(inplace=True)
#     kf = KalmanFilter(dim_x=1, dim_z=1)
#     kf.x = np.array([[merged_data.iloc[0]]])  # 初始状态，使用 .iloc 获取第一个元素
#     kf.F = np.array([[1]])              # 状态转移矩阵
#     kf.H = np.array([[1]])              # 观测矩阵
#     kf.P *= 1000                        # 初始协方差
#     kf.R = 1                            # 尝试减小观测噪声
#     kf.Q = 1                            # 尝试调整过程噪声

#     filtered_data = []
#     for i in range(len(merged_data)):
#         measurement = merged_data.iloc[i]  # 使用 .iloc 获取位置索引
#         if pd.notna(measurement):   # 检查测量值是否为 NaN
#             kf.predict()
#             kf.update(np.array([[measurement]]))
#             filtered_data.append(kf.x[0, 0])
#         else:
#             filtered_data.append(np.nan)  # 如果测量值是 NaN，添加 NaN

#     return pd.Series(filtered_data, index=merged_data.index[:len(filtered_data)])

# mer_ori_data['close_stock1'] = kalman_filter(mer_ori_data['close_stock1'])

# ------------------- 開始計算是否有均值回歸 -------------------

# 格蘭傑因果檢驗
def cause_result(data, x, y, max_lag):
    results = grangercausalitytests(data[[x, y]], max_lag)
    # 輸出每個滯後期的檢驗結果
    for lag, res in results.items():
        print(f"Lag {lag}:")
        print(f"  F-statistic: {res[0]['ssr_ftest'][0]}")
        print(f"  p-value: {res[0]['ssr_ftest'][1]}")
    
# cause_result(merged_data, 'diff3', 'diff', 5)


def test_co(timeseries1, timeseries2):
    # 进行协整检验
    coint_test = coint(timeseries1, timeseries2)

    # 获取结果
    t_stat = coint_test[0]
    p_value = coint_test[1]
    critical_values = coint_test[2]

    print(f"协整检验统计量 (t-statistic): {t_stat}")
    print(f"p 值: {p_value}")
    print(f"临界值 (Critical values): {critical_values}\n")
    
    # 设置条件判断并给出总结
    if p_value < 0.05:
        print("基于 p 值：存在协整关系，意味着它们具有长期均衡关系。")
    elif p_value < 0.10:
        print("基于 p 值：存在协整关系，但在 10% 显著性水平下。")
    else:
        print("基于 p 值：不存在协整关系。")

    # 也可以基于 t_stat 和 critical_values 进行判断
    if t_stat < critical_values[0]:  # 比如使用 1% 的置信水平
        print("基于1% 置信水平：协整关系非常显著。")
    elif t_stat < critical_values[1]:  # 5% 的置信水平
        print("基于5% 置信水平：协整关系显著。")
    elif t_stat < critical_values[2]:  # 10% 的置信水平
        print("基于10% 置信水平：协整关系较为显著。")
    else:
        print("基于t檢驗: 不存在显著的协整关系。")
        
    # 2. 计算相关性
    correlation = timeseries1.corr(timeseries2)

    # 输出相关性结果
    print(f"\n相关性系数: {correlation:.2f}")

    # 相关性总结
    if correlation > 0.8:
        print("存在很强的正相关性。")
    elif correlation > 0.5:
        print("存在较强的正相关性。")
    elif correlation > 0:
        print("存在弱正相关性。")
    elif correlation == 0:
        print("没有相关性。")
    elif correlation > -0.5:
        print("存在弱负相关性。")
    elif correlation > -0.8:
        print("存在较强的负相关性。")
    else:
        print("存在很强的负相关性。")

# test_co(merged_data['close_stock1'], merged_data['close_stock3'])


def plot_scatter(timeseries1, label1, timeseries2, label2):
    plt.scatter(timeseries1, timeseries2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title(f'{label1} vs {label2}')
    plt.show()

# plot_scatter(merged_data['close_stock1'], 'close_stock1', merged_data['close_stock3'], 'close_stock3')


def plot_data(timeseries, col1, label1, col2=None, label2=None, col3=None, label3=None):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(timeseries[col1], label=label1, color='blue')
    if col2 is not None:
        ax1.plot(timeseries[col2], label=label2, color='green')
    ax1.set_ylabel('Stock Prices')
    ax1.legend(loc='upper left')

    if col3 is not None:
        # 使用次坐標軸
        ax2 = ax1.twinx()
        ax2.plot(timeseries[col3], label=label3, color='red')
        ax2.set_ylabel('Difference')
        ax2.legend(loc='upper right')

    plt.show()
    
# plot_data(merged_data, col1='close_stock1', label1='stock1', col3='close_stock3', label3='close_stock3')


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
# plot_mean(merged_data, [merged_data['close_stock1'], merged_data['close_stock3']], ['stock1', 'stock3'])


def qq_plot(timeseries, title, line):
    sm.qqplot(timeseries, line=line)
    plt.title(f'QQ plot of {title}')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()
    
# qq_plot(merged_data['yj_transform'], 'close1 - close2', '45')

# Shapiro-Wilk正態分佈檢驗
def test_shapiro(timeseries):
    if len(timeseries) >= 5000:
        print('數據量過大，無法進行 Shapiro-Wilk 檢驗')
        return
    
    # 进行 Shapiro-Wilk 检验
    shapiro_test = stats.shapiro(timeseries)
    p_value = shapiro_test.pvalue

    # 输出 p 值
    print(f'Shapiro-Wilk Test p-value: {p_value:.3f}')
    alpha = 0.05  # 显著性水平
    if p_value < alpha:
        print("数据不是正态分布")
    else:
        print("数据可能是正态分布")
        
# test_shapiro(merged_data['yj_transform'])

# 使用 ADF 測試進行均值回歸檢驗
def adf_test(timeseries, title):
    try:
        result = adfuller(timeseries.dropna(), autolag='AIC')  # 確保數據沒有 NaN 值
        print(f'Augmented Dickey-Fuller Test: {title}')
        print(f'ADF Test Statistic       {result[0]}')
        print(f'p-value                   {result[1]}')
        print(f'# Lags Used               {result[2]}')
        print(f'# Observations Used     {result[3]}')
        for key, value in result[4].items():
            print(f'Critical Value ({key})      {value}')
        print('=> 平穩' if result[1] <= 0.05 else '=> 非平穩')
        return result[1]  # 返回p值
    except Exception as e:
        print(f"Error in ADF test: {e}")
        return None

# print('-------------------------- ADF計算 --------------------------')
# adf_test(merged_data['yj_transform'], title='Close1 - Close2')

# KPSS 檢驗
def kpss_test(timeseries):
    ts = timeseries.dropna()
    # 进行KPSS检验
    kpss_stat, p_value, lags, critical_values = kpss(ts)
    print('-------------------------- KPSS結果 --------------------------')
    print(f'KPSS统计量: {kpss_stat}')
    print(f'P值: {p_value}')
    print(f'lags值: {lags}')
    print(f'临界值: {critical_values}')
    
    # 添加平稳性判断
    if kpss_stat > critical_values['5%'] and p_value < 0.05:
        print("时间序列非平稳")
    else:
        print("时间序列平稳")
# print('-------------------------- kpss計算 --------------------------')
# kpss_test(merged_data['yj_transform'])

# Hurst 指數計算  
def hurst_exponent(ts):
    # 确保输入序列为有效数据
    ts = ts.dropna()
    max_lag = min(len(ts), 100)
    lags = range(2, max_lag)
    tau = []
    
    for lag in lags:
        # 计算滞后期的差值
        diff = np.subtract(ts[lag:].values, ts[:-lag].values)
        std_dev = np.std(diff)

        if std_dev > 0:  # 只保留标准差大于零的值
            tau.append(std_dev)
        else:
            print(f'Lag: {lag}, Std Dev: 0. 计算滞后期的标准差为0，跳过此滞后期。')

    # 如果tau为空，说明没有有效的滞后期
    if len(tau) == 0:
        print("没有有效的滞后期，无法计算Hurst指数。")
        return np.nan

    log_lags = np.log(lags[:len(tau)])
    log_tau = np.log(tau)
    
    # 线性回归
    coeff = np.polyfit(log_lags, log_tau, 1)
    print(f'Hurst指数: {coeff[0]}')
    # 判断趋势
    if coeff[0] < 0.5:
        print("时间序列呈现有回归到均值的趋势")
    elif coeff[0] == 0.5:
        print("时间序列是随机游走")
    else:
        print("时间序列具有长期记忆或趋势")
    return coeff[0]

# 计算Hurst指数
# print('-------------------------- HRUST計算 --------------------------')
# hurst_exponent(merged_data['boxcox'])

# BDS 检验(判斷是否非線性)
def bds_test(ts, m=2, epsilon=0.1, alpha=0.05):
    n = len(ts.dropna())
    count = 0
    
    # 计算符合条件的对数
    for i in range(n - m):
        for j in range(i + 1, n - m):
            if abs(ts.iloc[i] - ts.iloc[j]) < epsilon:  # 使用 .iloc 访问元素
                count += 1
                
    # 计算 BDS 统计量
    bds_stat = count / (n * (n - m))

    # 计算 BDS 统计量的标准误差
    std_error = np.sqrt(bds_stat * (1 - bds_stat) / n)

    # 计算 Z 统计量
    z_stat = (bds_stat - 0.5) / std_error

    # 计算 p 值
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # 双侧检验

    # 输出结果
    if p_value < alpha:
        print("BDS檢驗: 拒绝零假设, 时间序列具有非线性依赖性")
    else:
        print("BDS檢驗:未拒绝零假设, 时间序列是随机的")

# 使用差分后的时间序列进行 BDS 检验
# print('-------------------------- BDS計算 --------------------------')
# bds_test(merged_data['yj_transform'])

# HetARCH 检验(判斷是否非線性)
def het(ts, alpha=0.05):
    # 执行 ARCH 检验
    arch_test = het_arch(ts)

    # 打印检验结果
    lm_statistic = arch_test[0]
    p_value = arch_test[1]
    f_value = arch_test[2]
    f_p_value = arch_test[3]

    print("Lagrange multiplier statistic:", lm_statistic)
    print("p-value:", p_value)
    print("f-value:", f_value)
    print("f p-value:", f_p_value)
    
    if p_value < alpha:
        print("拒绝原假设：时间序列具有非线性依赖性（存在条件异方差）。")
        print("建议使用非线性模型（如 GARCH）进行进一步分析。")
    else:
        print("无法拒绝原假设：时间序列没有明显的非线性依赖性（没有条件异方差）。")
        print("可以考虑使用线性模型（如 ARIMA）进行分析。")

# print('-------------------------- HetARCH計算 --------------------------')
# het(merged_data['yj_transform'])

# VIF 檢驗
def check_vif(data, features):
    """
    Calculate VIF for each feature and provide a summary on multicollinearity.
    """
    X = sm.add_constant(data[features])
    vif_data = pd.DataFrame()
    vif_data["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data["features"] = X.columns
    
    # Display VIF result
    print('VIF輸出結果:')
    print(f'{vif_data}\n')
    
    # VIF 结果总结
    for index, row in vif_data.iterrows():
        feature = row['features']
        vif_value = row['VIF Factor']
        
        if vif_value > 10:
            print(f"警告：'{feature}' 的 VIF 值为 {vif_value:.2f}，表示存在潜在的多重共线性。")
        elif 5 < vif_value <= 10:
            print(f"'{feature}' 的 VIF 值为 {vif_value:.2f}，表明可能存在一些多重共线性。")
        else:
            print(f"'{feature}' 的 VIF 值为 {vif_value:.2f}，表明没有显著的多重共线性。")
    
    return vif_data

# check_vif(merged_data, ['spread', 'spread2'])

# ------------------- 數學機器模型檢驗 -------------------

# OLS回歸
def OLS_trend(merged_data, x, y, plot=False):
    merged_data.dropna(inplace=True)
    train_size = int(len(merged_data) * 0.8)
    train_data = merged_data.iloc[:train_size]
    test_data = merged_data.iloc[train_size:]

    # 准备训练数据
    x_train = sm.add_constant(train_data[x])  # 添加常数项
    y_train = train_data[y]
    
    # 使用 OLS 拟合训练数据
    model_with_const = sm.OLS(y_train, x_train).fit()

    # 从训练数据中获取趋势
    train_trend_with_const = model_with_const.predict(x_train)

    # 从训练数据中去除趋势
    train_detrended_diff_with_const = train_data[y] - train_trend_with_const

    # 在 merged_data 中记录训练集的去趋势结果
    merged_data['detrended_train_diff_with_const'] = np.nan
    merged_data.iloc[:train_size, merged_data.columns.get_loc('detrended_train_diff_with_const')] = train_detrended_diff_with_const

    # 从测试数据中准备预测
    x_test = sm.add_constant(test_data[x])  # 添加常数项
    test_trend_with_const = model_with_const.predict(x_test)

    # 从测试数据中去除趋势
    test_detrended_diff_with_const = test_data[y] - test_trend_with_const

    # 在 merged_data 中记录测试集的去趋势结果
    merged_data['detrended_test_diff_with_const'] = np.nan
    merged_data.iloc[train_size:, merged_data.columns.get_loc('detrended_test_diff_with_const')] = test_detrended_diff_with_const

    if plot is True:
        # 创建第一个 Y 轴
        ax1 = plt.gca()  # 获取当前的轴

        # 绘制原始差额和去趋势的数据
        ax1.plot(merged_data.index, merged_data[y], label='Original', alpha=0.5)
        ax1.plot(merged_data.index, merged_data['detrended_train_diff_with_const'], label='Detrended Train with Const', color='orange')
        ax1.plot(merged_data.index, merged_data['detrended_test_diff_with_const'], label='Detrended Test with Const', color='green')

        # 设置第一个 Y 轴的标签和标题
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Original Diff and Detrended Values')
        ax1.set_title('Original Diff and Detrended Results with OLS Trends (with Constant)')
        ax1.legend(loc='upper left', framealpha=0.5)

        # 创建第二个 Y 轴
        ax2 = ax1.twinx()  # 共享 x 轴

        # 这里假设你有一个要绘制在第二个 Y 轴的数据，比如 'some_other_data'
        # 替换 'some_other_data' 为你要绘制的实际数据
        ax2.plot(merged_data.index[:train_size], train_trend_with_const, label='Train Trend (OLS with Const)', linestyle='--', color='red')
        ax2.plot(merged_data.index[train_size:], test_trend_with_const, label='Test Trend (Predicted with Const)', linestyle='--', color='purple')

        # 设置第二个 Y 轴的标签
        ax2.set_ylabel('Test Data Values')

        # 如果需要，也可以添加第二个图例
        ax2.legend(loc='upper right', framealpha=0.5)

        # 显示图表
        plt.show()

        # 检查拟合结果
        print('OLS回归拟合结果 (带常数项):')
        print(model_with_const.summary())

        # 评估模型性能：例如计算均方误差
        mse_with_const = np.mean((test_data[y] - test_trend_with_const) ** 2)
        print(f'Mean Squared Error (MSE) on Test Set (with Const): {mse_with_const}')
        
    return model_with_const

# OLS_trend(merged_data, 'yj_transform_shift', 'yj_transform', plot=True)

# 多項式回歸
def poly_trend(data, x, y, d=2, plot=False):
    X = data[x].values.reshape(-1, 1)
    y = data[y].values  # 因变量

    # 使用时间序列分割数据
    split_index = int(len(X) * 0.8)  # 80% 的索引位置
    X_train, X_test = X[:split_index], X[split_index:]  # 训练集和测试集
    y_train, y_test = y[:split_index], y[split_index:]


    # 生成多项式特征
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # 进行线性回归
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # 进行预测
    y_pred = model.predict(X_test_poly)

    if plot is True:
        # 可视化结果
        plt.scatter(X, y, color='blue', label='Data Points')
        plt.scatter(X_test, y_pred, color='red', label='Predicted Points')
        plt.title('Polynomial Regression')
        plt.xlabel('diff_pct')
        plt.ylabel('index_diff_pct')
        plt.legend()
        plt.show()
        
        # 残差分析
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, color='orange')
        plt.axhline(0, color='black', lw=2, linestyle='--')
        plt.title('Residuals vs Predicted')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()
        
        
        # 输出系数
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
        
        # 评估模型
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')
        
        rmse = np.sqrt(mse)
        print(f'Root Mean Squared Error: {rmse}')
        
        mae = mean_absolute_error(y_test, y_pred)
        print(f'Mean Absolute Error: {mae}')
        
        scores = cross_val_score(model, poly.fit_transform(X), y, cv=5, scoring='neg_mean_squared_error')
        print(f'Cross-Validated MSE: {-scores.mean()}')
    
    return model

# poly_trend(merged_data, 'index_pct', 'diff_pct', d=2, plot=True)


# VAR脈衝模型
def var_model(all_df, x, y, maxlags=10, plot=False):
    all_df.dropna()
    data = all_df[[x, y]]
    
    # 檢查平穩性
    for col in data.columns:
        result = adfuller(data[col])
        print(f"{col}: ADF Statistic = {result[0]}, p-value = {result[1]}")
        if result[1] > 0.05:
            print(f"{col} 非平穩，需要差分。")
            data.loc[:, col] = data[col].diff().dropna()
        else:
            print(f"{col} 是平穩的。\n")


    # 划分训练集和测试集 (80% 训练, 20% 测试)
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 建立 VAR 模型
    model = VAR(train_data)

    try: # 選擇最佳的滯後期數 (lag order)
        lag_order = model.select_order(maxlags=maxlags)
        print(lag_order.summary())
    except Exception as e:
        print("Error occurred while selecting order:", str(e))
        return

    # 最佳滯後期數
    optimal_lag = lag_order.aic
    print(f'最佳滯後期數：{optimal_lag}\n')
    
    # 擬合模型
    var_result = model.fit(optimal_lag)
    print(var_result.summary())

    # 檢查殘差是否為白噪聲
    residuals = var_result.resid
    print("\n檢查殘差是否為白噪聲:")
    for col in residuals.columns:
        print(f'殘差 {col}:')
        lb_test = acorr_ljungbox(residuals[col], lags=[10], return_df=True)
        print(lb_test)

    # 使用训练集进行预测 (预测未来 test_data 长度的数据)
    prediction = var_result.forecast(train_data.values[-optimal_lag:], steps=len(test_data))
    
    # 将预测结果转为 DataFrame，和实际的测试集对比
    forecast_index = test_data.index
    forecast_df = pd.DataFrame(prediction, index=forecast_index, columns=data.columns)
    
    # 打印预测结果与实际值
    print("\nTest Set Predictions vs Actual Values:")
    print('Predict:', forecast_df.head(5).values.tolist())
    print('Test:', test_data.head(5).values.tolist())
    
    # 计算均方误差 (MSE) 用于验证
    mse = ((forecast_df - test_data) ** 2).mean()
    print(f"\nMean Squared Error (MSE) on Test Set: {mse}")
    
    # 预测误差评估
    actual_values = test_data  # 使用测试集的实际值进行评估
    mse_diff = mean_squared_error(actual_values[x], forecast_df['diff_pct'])

    print(f'Mean Squared Error for diff_pct: {mse_diff}')

    mae_diff = mean_absolute_error(actual_values['diff_pct'], forecast_df['diff_pct'])

    print(f'Mean Absolute Error for y: {mae_diff}')
    
    # 5. Granger 因果檢驗
    cause_result(data, x, y, optimal_lag)

    # 6. 模型的穩定性檢查
    stability_test = var_result.is_stable(verbose=True)
    if stability_test:
        print("模型是穩定的，所有特徵根都在單位圓內。\n")
    else:
        print("模型不穩定，有些特徵根超出了單位圓。\n")
    
    if plot is True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_data.index, test_data[y], label="Actual diff_pct", color="blue")
        plt.plot(forecast_df.index, forecast_df[y], label="Predicted diff_pct", color="red")
        plt.legend()
        plt.title("Actual vs Predicted diff_pct")
        plt.show()

        # 1. 脈衝響應函數
        irf = var_result.irf(10)
        irf.plot(orth=False)
        plt.title("Impulse Response Function (IRF)")
        plt.show()

        # 2. 方差分解
        fevd = var_result.fevd(10)
        fevd.plot()
        plt.title("Forecast Error Variance Decomposition (FEVD)")
        plt.show()

        # 7. AIC/BIC分數趨勢圖
        aic_values = []
        bic_values = []
        for i in range(1, maxlags+1):
            result = model.fit(i)
            aic_values.append(result.aic)
            bic_values.append(result.bic)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, maxlags+1), aic_values, label='AIC', marker='o')
        plt.plot(range(1, maxlags+1), bic_values, label='BIC', marker='o')
        plt.title('AIC and BIC vs Lag Length')
        plt.xlabel('Lag Length')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

# var_model(merged_data, 'diff3', 'diff', maxlags=10, plot=False)

# SVAR模型
def svar_model(data, x, y, p, q, plot=False):
    # 检查平稳性
    adf_test = lambda series: adfuller(series)[1]
    print(f"Testing stationarity for {x} and {y}:")
    p_value_y1 = adf_test(data[x])
    p_value_y2 = adf_test(data[y])
    
    if p_value_y1 >= 0.05:
        print(f'{x}序列不平稳，进行差分')
        data[x] = data[x].diff().dropna()
    if p_value_y2 >= 0.05:
        print(f'{y}序列不平稳，进行差分')
        data[y] = data[y].diff().dropna()
        
    
    # 拟合SVAR模型
    model = VARMAX(data[[x, y]], order=(p, q))
    results = model.fit(disp=False)
    
    print(results.summary())

    # 检查稳定性（特征根）
    ar_params = results.params[:p]  # 前p个参数是AR部分的系数
    ar_poly = np.r_[1, -ar_params]  # 特征多项式系数，添加1表示常数项
    roots = np.roots(ar_poly)  # 计算特征根
    print("Roots of the characteristic polynomial:")
    print(roots)
    
    if np.all(np.abs(roots) < 1):
        print("Model is stable (all roots are inside the unit circle).")
    else:
        print("Model is not stable (some roots are outside the unit circle).")

    # 残差自相关检验
    lb_test = acorr_ljungbox(results.resid.iloc[:, 0], lags=[1, 2, 3], return_df=True)  # 选择第一列残差
    print("Ljung-Box test results:")
    print(lb_test)

    # 脉冲响应分析
    if plot:
        irf = results.irf(10)
        irf.plot(orth=False)
        plt.title('Impulse Response Functions')
        plt.show()

        # 方差分解
        fevd = results.fevd(10)
        fevd.plot()
        plt.title('Forecast Error Variance Decomposition')
        plt.show()

    return results

# svar_model(merged_data, 'diff3', 'diff', 1, 0, plot=False)


# VECM模型
def vecm_model(data, x, y):
    # 检查缺失值
    data.dropna(inplace=True)
    
    adf_test = lambda series: adfuller(series)[1]
    p_value_y1 = adf_test(data[x])
    p_value_y2 = adf_test(data[y])
    print(f'{x} 平稳性检验 p-value: {p_value_y1}')
    print(f'{y} 平稳性检验 p-value: {p_value_y2}')

    # 如果序列不平稳，则进行差分
    if p_value_y1 >= 0.05:
        print(f'{x}序列不平稳，进行差分')
        data[x] = data[x].diff().dropna()
    if p_value_y2 >= 0.05:
        print(f'{y}序列不平稳，进行差分')
        data[y] = data[y].diff().dropna()

    # 重新检查缺失值
    data.dropna(inplace=True)
    
    # 进行协整检验
    score, p_value, _ = coint(data[x], data[y])
    print(f'协整检验统计量: {score}, p-value: {p_value}')

    # 如果 p-value < 0.05，则存在协整关系
    if p_value < 0.05:
        print("拒绝零假设，表示两个时间序列是协整的。")
        
        # 构建 VECM 模型
        model = VECM(data[[x, y]], k_ar_diff=1, coint_rank=1)  # 1 表示协整秩
        vecm_result = model.fit()

        print(vecm_result.summary())
    else:
        print("无法拒绝零假设，表示两个时间序列不协整。")

# vecm_model(merged_data, 'diff3', 'diff')


# 使用ARIMA模型
def plot_func(timeseries, p, d, q, params):
    # 根据传入的 p 和 q 确定范围
    p_values = range(0, p + 1)  # AR 参数的范围
    q_values = range(0, q + 1)  # MA 参数的范围
    aic_values = np.zeros((len(p_values), len(q_values)))
    
    for i, p_val in enumerate(p_values):
        for j, q_val in enumerate(q_values):
            try:
                model = ARIMA(timeseries, order=(p_val, d, q_val), exog=params.get('exog', None))
                result = model.fit()
                aic_values[i, j] = result.aic
            except Exception as e:
                aic_values[i, j] = np.nan  # 如果模型拟合失败，记录为 NaN
                print(f"Error fitting ARIMA({p_val}, 0, {q_val}): {e}")

    # 替换 NaN 值为很大的数
    aic_values = np.nan_to_num(aic_values, nan=np.inf)

    p_grid, q_grid = np.meshgrid(p_values, q_values)

    # 创建 3D 图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(p_grid, q_grid, aic_values, cmap='viridis')

    # 设置标签
    ax.set_xlabel('AR Parameter (p)')
    ax.set_ylabel('MA Parameter (q)')
    ax.set_zlabel('AIC Value')
    ax.set_title('AIC Values of ARIMA Model')

    # 显示图形
    plt.show()

def arima_model(timeseries, p, d, q, plot=False, **parmas):
    if timeseries.isna().sum() > 0:
        print(f'數據缺失: {timeseries.isna().sum()}個, 進行向後填充')
        # 创建完整的时间索引
        full_range = pd.date_range(start=timeseries.index.min(), end=timeseries.index.max(), freq='min')
        timeseries = timeseries.reindex(full_range)

        # 前向填充缺失值
        timeseries.ffill(inplace=True)
        
    ts = timeseries.dropna()
    
    if params.get('sarima', False) is True:
        model = SARIMAX(ts, order=(p, d, q),
                      exog=parmas.get('exog', None))
        fit = model.fit(disp=False)
    else:
        model = ARIMA(ts, order=(p, d, q),
                  exog=parmas.get('exog', None),
                  trend=parmas.get('trend', None),
                  freq=parmas.get('freq', None))
        
        fit_params = params.get('fit_params', {})
        
        fit = model.fit(**fit_params)
        
        # 获取自回归系数 beta
        beta = fit.params['ar.L1']

        # 计算半衰期
        half_life = -np.log(2) / np.log(beta)
        print(f"半衰期: {round(half_life, 2)} 个时间单位")
    
    if plot is True:
        plot_func(ts, p, d, q, params)
        
        # 获取残差
        residuals = fit.resid
        
        # 绘制残差图
        plt.figure(figsize=(12, 8))

        # 残差图
        plt.subplot(3, 2, 1)
        plt.plot(residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residuals')
        plt.xlabel('Time')
        plt.ylabel('Residuals')

        # 残差自相关图
        plt.subplot(3, 2, 2)
        sm.graphics.tsa.plot_acf(residuals, lags=20, ax=plt.gca())
        plt.title('ACF of Residuals')

        # 残差偏自相关图
        plt.subplot(3, 2, 3)
        sm.graphics.tsa.plot_pacf(residuals, lags=20, ax=plt.gca())
        plt.title('PACF of Residuals')

        # QQ图
        plt.subplot(3, 2, 4)
        sm.qqplot(residuals, line='s', ax=plt.gca())
        plt.title('QQ Plot of Residuals')

        # 残差的直方图
        plt.subplot(3, 2, 5)
        plt.hist(residuals, bins=20, density=True, alpha=0.6, color='g')
        plt.title('Histogram of Residuals')

        # 拟合值图
        plt.subplot(3, 2, 6)
        plt.plot(merged_data.index, fit.fittedvalues, color='orange', label='Fitted')
        plt.plot(merged_data.index, merged_data, color='blue', label='Original')
        plt.title('Fitted vs Original')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
        print(f"{'SARIMA' if params.get('sarima', False) else 'ARIMA'}模型輸出結果:")
        print(fit.summary())
        
    return fit


# print('-------------------------- ARIMA計算 --------------------------')
# (lambda data: (plot_acf(data.dropna()), plot_pacf(data.dropna()), plt.show()))(merged_data['diff'])

params = {
    'sarima': False,
    'exog': merged_data['diff3'],
    'freq': 'min',
    'trend': None,
    'fit': {}
}
# arima = arima_model(merged_data['diff'], 2, 0, 2, plot=True, **params)


# def half_life (data, edge=1):
#     data.dropna(inplace=True)
#     # 对数据进行 Z-score 标准化
#     scaler = StandardScaler()
#     data_z_score = scaler.fit_transform(data.values.reshape(-1, 1))
    
#     # 转换为 Pandas Series
#     data_z = pd.Series(data_z_score.flatten(), index=data.index, name='z-score')
    
#     best_aic = np.inf  # 初始化 AIC 为一个很大的值
#     best_bic = np.inf  # 初始化 BIC 为一个很大的值
#     best_lag_aic = 1   # 初始化最优滞后项（根据AIC）
#     best_lag_bic = 1   # 初始化最优滞后项（根据BIC）
#     maxlag = 10  # 设定最大滞后项

#     # 遍历不同的滞后项，找出 AIC 和 BIC 最小的滞后项
#     for lag in range(1, maxlag + 1):
#         model = sm.tsa.AutoReg(data_z, lags=lag)
#         result = model.fit()

#         aic_value = result.aic
#         bic_value = result.bic

#         print(f"滞后项 {lag} 的 AIC 值: {aic_value}, BIC 值: {bic_value}")

#         # 更新 AIC 的最佳滞后项
#         if aic_value < best_aic:
#             best_aic = aic_value
#             best_lag_aic = lag

#         # 更新 BIC 的最佳滞后项
#         if bic_value < best_bic:
#             best_bic = bic_value
#             best_lag_bic = lag

#     print(f"选择的最优滞后项（AIC）: {best_lag_aic}, 对应的 AIC: {best_aic}")
#     print(f"选择的最优滞后项（BIC）: {best_lag_bic}, 对应的 BIC: {best_bic}")

#     # 最终选择滞后项，通常可以根据你的分析目标决定使用 AIC 还是 BIC
#     # 例如，如果你更注重模型简洁性，可以选择 BIC，或者考虑 AIC 和 BIC 的折衷
#     final_lag = best_lag_bic  # 可以选择 BIC 最优滞后项，也可以改为 AIC
#     model = sm.tsa.AutoReg(data_z, lags=final_lag)
#     result = model.fit()

#     # 均值回归系数
#     speed_of_mean_reversion = result.params.iloc[1]
    
#     # 半衰期公式
#     if speed_of_mean_reversion >= 1:
#         print(f"均值回归速度过高({round(speed_of_mean_reversion, 2)})，无法计算半衰期。")
#         half_life = np.inf
#     elif speed_of_mean_reversion <= 0:
#         print(f"均值回归速度为负或零({round(speed_of_mean_reversion, 2)})，无法计算半衰期。")
#         half_life = np.inf
#     else:
#         # 半衰期公式
#         half_life = -np.log(2) / np.log(1 - speed_of_mean_reversion)
#         print(f"半衰期: {half_life:.2f} 分鐘")

#     # 判断价差是否接近半衰期
#     current_spread = data_z.iloc[-1] - data_z.mean()  # 使用 .iloc 正确访问最后一项
#     threshold = data_z.std() * edge  # 设定标准差阈值

#     if abs(current_spread) < threshold:
#         print(f"当前spread: {data_z.iloc[-1].round(2)}, 小於{edge}倍標準差, 接近均值, 可能接近回归。")
#     else:
#         print(f"当前spread: {data_z.iloc[-1].round(2)}, 大於{edge}倍標準差, 仍然远离均值。")

#     return half_life

# half_life(merged_data['spread3'])

# GRAHM-NGARCH 模型
def garch_model(timeseries, vol, p, q, plot=False, **params):
    ts = timeseries.dropna()
    
    if params.get('auotfit', False) is True:
        
        best_aic = float('inf')
        best_order = (None, None)
        best_model = None
    
        # Generate all combinations of p and q
        for p, q in itertools.product(params.get('p_range'), params.get('q_range')):
            try:
                # Fit the GARCH model
                model = arch_model(y=ts, vol=vol, p=p, q=q, 
                                   x=params.get('exog', None), 
                                   mean=params.get('mean', 'Constant'),
                                   dist=params.get('dist', 'normal'))

                model_fit = model.fit(disp='off')
                current_aic = model_fit.aic

                # Check if this model is better (lower AIC)
                if current_aic < best_aic:
                    best_aic = current_aic
                    best_order = (p, q)
                    best_model = model_fit

                # Optionally plot the results for the best model
                if plot and (p, q) == best_order:
                    model_fit.plot()

            except Exception as e:
                print(f"Error fitting GARCH({p}, {q}): {e}")
        
        print(f"Best GARCH model: GARCH({best_order[0]}, {best_order[1]}) with AIC: {best_aic}")
    else:    
         # Fit the GARCH model
        model = arch_model(y=ts, vol=vol, p=p, q=q, 
                           x=params.get('exog', None), 
                           mean=params.get('mean', 'Constant'),
                           dist=params.get('dist', 'normal'))

        best_model = model.fit(disp='off')
    
    results = best_model
    
    if plot is True:
        print(f'{vol}模型輸出結果:')
        print(results.summary())
        
        # 提取模型输出
        residuals = results.resid
        conditional_volatility = results.conditional_volatility

        # 绘制时间序列图
        plt.figure(figsize=(12, 6))

        # 残差图
        plt.subplot(3, 1, 1)
        plt.plot(residuals, label='Residuals', color='blue')
        plt.title('Residuals of GARCH Model')
        plt.axhline(0, color='red', linestyle='--')
        plt.legend()

        # 条件波动性图
        plt.subplot(3, 1, 2)
        plt.plot(conditional_volatility, label='Conditional Volatility', color='orange')
        plt.title('Conditional Volatility of GARCH Model')
        plt.legend()

        # QQ图
        plt.subplot(3, 1, 3)
        sm.qqplot(residuals, line='s', ax=plt.gca())
        plt.title('QQ Plot of Residuals')

        plt.tight_layout()
        plt.show()
    
    return results
    
# print('-------------------------- ARCH計算 --------------------------')
params = {
    'auotfit': False,
    'p_range': range(1, 6),
    'q_range': range(0, 6),
    # 'exog': merged_data['diff3'],
    'dist': 'skewstudent',
    'mean': 'Zero',
}

# grach = garch_model(arima.resid, 'HARCH', 3, 0, plot=False, **params)

# --------------------------- 進行模型預測 ------------------------------- #

def model_evaluation(test_data, forecast_arima, forecast_garch, residuals, garch_model=None, arima=True):
    """
    评估ARIMA和GARCH模型的预测效果，包含多种检验方法

    参数:
    - test_data: 实际数据
    - forecast_arima: ARIMA模型的预测结果
    - forecast_garch: GARCH模型的波动性预测结果
    - residuals: 模型的残差（ARIMA残差或GARCH残差）
    - garch_model: GARCH模型对象，包含fit后的模型，用于计算QML Loss（如果评估GARCH）
    - arima: 是否评估ARIMA模型，若为False则主要评估GARCH模型

    输出:
    - 各类误差指标和检验结果
    """
    if forecast_arima is not None:
        print("\nARIMA Model Evaluation")
    else:
        print("\nGARCH Model Evaluation")
    
    results = {}
    
    # 1. 计算 MSE, MAE, R², SMAPE, MAPE
    mse = mean_squared_error(test_data, forecast_arima)
    mae = mean_absolute_error(test_data, forecast_arima)
    r2 = r2_score(test_data, forecast_arima)
    smape = 100 * np.mean(2 * np.abs(forecast_arima - test_data) / (np.abs(test_data) + np.abs(forecast_arima)))
    mape = 100 * np.mean(np.abs((test_data - forecast_arima) / test_data))
    
    results['MSE'] = mse
    results['MAE'] = mae
    results['R²'] = r2
    results['SMAPE'] = smape
    results['MAPE'] = mape
    
    # 2. Jarque-Bera Test for Normality (正态性检验)
    jb_stat, jb_pvalue = jarque_bera(residuals)
    results['Jarque-Bera Statistic'] = jb_stat
    results['Jarque-Bera p-value'] = jb_pvalue
    
    # 3. Durbin-Watson Test (检查自相关)
    dw_stat = durbin_watson(residuals)
    results['Durbin-Watson Statistic'] = dw_stat
    
    # 4. Ljung-Box Test (自相关性检验)
    lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=[10])
    results['Ljung-Box Statistic'] = lb_stat[-1]
    results['Ljung-Box p-value'] = lb_pvalue[-1]
    
    # 5. Theil's U (衡量预测与实际数据的差异)
    u_stat = np.sqrt(np.sum((forecast_arima - test_data)**2)) / (np.sqrt(np.sum(test_data**2)) + np.sqrt(np.sum(forecast_arima**2)))
    results['Theil\'s U'] = u_stat
    
    # 6. QML Loss (使用GARCH模型结果)
    if garch_model is not None:
        qml_loss_value = -garch_model.loglikelihood
        results['QML Loss'] = qml_loss_value
    
    # 7. Engle's ARCH LM Test (仅针对GARCH模型，异方差检验)
    if not arima:
        arch_lm_stat, arch_lm_pvalue, _, _ = het_arch(residuals)
        results['ARCH LM Statistic'] = arch_lm_stat
        results['ARCH LM p-value'] = arch_lm_pvalue
    
    # 8. Kupiec Proportion of Failures Test (风险管理测试)
    if not arima:
        actual_in_band = (test_data >= (forecast_arima - forecast_garch)) & \
                         (test_data <= (forecast_arima + forecast_garch))
        failures = np.sum(~actual_in_band)
        pof_stat = failures / len(test_data)
        results['Kupiec Proportion of Failures'] = pof_stat
    
    # 打印结果
    for key, value in results.items():
        print(f"{key}: {value}")
    
    return results



# def forecast(df, test_date, forecast_steps, plot=False, simulate=False):
#     # 提取特定日期的数据
#     test_data = df.loc[test_date]

#     # 按时间段筛选该日期从 9:02 到 13:24 的数据
#     test_data = test_data.between_time('09:02', '13:24')

#     # 删除含有NaN的行
#     test_data.dropna(inplace=True)

#     arima_forecast = arima.forecast(steps=forecast_steps, exog=test_data['diff3'].iloc[:forecast_steps])
#     # arima_forecast = arima.forecast(steps=forecast_steps, exog=None)
    
#     volatility_forecast = []  # 存储未来的波动性预测
    
#     # 进行多步预测
#     for i in range(forecast_steps):
    
#         # grach_forecast = grach.forecast(horizon=1, x=test_data['diff3'].iloc[i+1].reshape(1, -1))  # 一步预测
#         grach_forecast = grach.forecast(horizon=1, x=None)  # 一步预测
#         predicted_volatility = np.sqrt(grach_forecast.variance.values[-1, -1])  # 提取预测的波动性
#         volatility_forecast.append(predicted_volatility)  # 存储预测结果
        
#     # 将预测结果转换为 Pandas Series
#     volatility_forecast = np.array(volatility_forecast)

#     # 确保时间索引匹配# 将预测结果添加到 test_data 中
#     forecast_index = pd.date_range(start=test_data.index[0], periods=forecast_steps, freq='min')
#     test_data['forecast_arima'] = pd.Series(index=forecast_index, data=arima_forecast.values)
#     test_data['forecast_grach'] = pd.Series(index=forecast_index, data=volatility_forecast)
    
    
#     if plot is True and simulate is False:
#         # 合併變化率到一個 DataFrame
#         comparison_df = pd.DataFrame({
#             'arima':  test_data['forecast_arima'],
#             'actual': test_data['diff']
#         })

#         # 移除包含NaN的行
#         comparison_df.dropna(inplace=True)

#         # 繪製變化率的比較圖
#         plt.figure(figsize=(12, 6))
#         plt.plot(comparison_df['arima'], label='ARIMA Forecast Change', color='orange')
#         plt.plot(comparison_df['actual'], label='Actual Change', color='blue')
#         plt.title('Comparison of Change Rates')
#         plt.xlabel('Index')
#         plt.ylabel('Change Rate')
#         plt.legend()
#         plt.show()

#         # 進行配對 t 檢驗
#         t_stat, p_value = ttest_rel(comparison_df['arima'].dropna(), comparison_df['actual'].dropna())
#         print(f'ttest_rel檢驗結果, T: {t_stat}, P: {p_value}')
#         if p_value < 0.05:
#             print("ARIMA模型预测与实际变化率「有差异」\n")
#         else:
#             print("ARIMA模型预测与实际变化率之间「没有差异」\n")
        
#         # 可视化实际结果、ARIMA预测和GARCH预测波动性
#         plt.figure(figsize=(12, 6))

#         # 绘制实际数据
#         plt.plot(test_data.index, test_data['diff'], label='Actual Data', color='blue')

#         # 绘制ARIMA预测数据
#         plt.plot(test_data.index, test_data['forecast_arima'], label='ARIMA Forecast', color='orange')

#         # 绘制GARCH预测波动性 (可以作为误差范围)
#         plt.fill_between(test_data.index, 
#                          test_data['forecast_arima'] - test_data['forecast_grach'],
#                          test_data['forecast_arima'] + test_data['forecast_grach'],
#                          color='gray', alpha=0.3, label='GARCH Volatility Forecast (1 Std Dev)')

#         plt.title('ARIMA and GARCH Forecast vs Actual Data')
#         plt.xlabel('Time')
#         plt.ylabel('Log Price / Volatility')
#         plt.legend()
#         plt.grid()
#         plt.show()
    
    
#         # 開始進行檢驗, 清理数据，去除包含 NaN 的行
#         test_data_clean = test_data.dropna(subset=['diff', 'forecast_arima', 'forecast_grach'])

#         # 如果你的 residuals 可能也含有 NaN，可以在此处进行相应处理
#         residuals_arima = arima.resid.dropna()
#         residuals_garch = grach.resid.dropna()

#         # 确保 residuals 的长度与测试数据一致
#         min_length = min(len(test_data_clean), len(residuals_arima), len(residuals_garch))

#         # 提取实际值和预测值
#         actual = test_data_clean['diff'].iloc[:min_length]
#         forecast_arima = test_data_clean['forecast_arima'].iloc[:min_length]
#         forecast_garch = test_data_clean['forecast_grach'].iloc[:min_length]

#         # 进行 ARIMA 模型的评估
#         model_evaluation(actual, 
#                         forecast_arima=forecast_arima, 
#                         forecast_garch=None, 
#                         residuals=residuals_arima.iloc[:min_length],  # 只使用有效的 residuals
#                         arima=True)

#         # 进行 GARCH 模型的评估
#         model_evaluation(actual, 
#                         forecast_arima=forecast_arima, 
#                         forecast_garch=forecast_garch, 
#                         residuals=residuals_garch.iloc[:min_length],  # 只使用有效的 residuals
#                         garch_model=grach,  # GARCH 模型对象
#                         arima=False)
#     elif simulate is True: # 模型模擬交易策略
#         print("ARIMA模型進行交易預測")
#         test_arima = test_data.iloc[:forecast_steps].copy()
        
#         buy_signals = []
#         sell_signals = []
#         trade_results = []

#         # 在策略判断之前，计算变化率, 繪圖可看到預測結果, 可再進行調整
#         test_arima['pct_change_arima'] = test_arima['forecast_arima'].pct_change(fill_method=None)

#         capital = 100000  # 初始資金
#         position = 0  # 持倉狀態
#         shares_per_trade = 1000  # 每次交易1000股

#         # r×P×1000−(P×1000×0.001425+(P+r×P)×1000×0.001425+(P+r×P)×1000×0.003)≥1.0(由此公式推導出, 固定購買股數1000股)
#         # 設定變化百分比閾值(P:股票價錢, desired_profit:目標利潤=>1代表要賺到1塊錢時, 需要設定多少%數)
#         threshold = (lambda P, desired_profit=1.5: 5.022e-5 * ((117.0 * P + 20.0) / P) * (desired_profit / 1.0))(test_arima['forecast_arima'].iloc[0], 1.5)

#         print(f'本次交易的變化率閾值：{(threshold * 100):.2f}%')

#         # 初始化费用记录
#         total_fees = 0  # 交易中产生的手续费和税费

#         # 根據變化率來判斷進出場
#         for i in range(1, forecast_steps):
#             actual_change = test_arima['pct_change_arima'].iloc[i]
#             price = test_arima['forecast_arima'].iloc[i]
            
#             # 判斷做空信號：當變化率大於threshold，做空
#             if actual_change > threshold and position == 0:
#                 # 买入成本计算：手续费
#                 buy_fee = math.ceil(price * shares_per_trade * 0.001425)
#                 total_fees += buy_fee  # 累加手续费
#                 print(f"做空手续费：{buy_fee:.2f}")

#                 # 做空头寸
#                 position = -shares_per_trade  # 持仓为负表示做空
#                 buy_price = price  # 记录买入价格
#                 sell_stop_loss = buy_price * 1.01  # 止损价(1%亏损)
#                 sell_take_profit = buy_price * 0.97  # 止盈价(3%盈利)
#                 buy_signals.append(i)
#                 print(f"做空：價格={price}, 變化率={actual_change:.2%}, 持倉={position}股\n")

#             # 判斷做多信號：當變化率小於threshold，做多
#             elif actual_change < -threshold and position == 0:
#                 # 买入成本计算：手续费
#                 buy_fee = math.ceil(price * shares_per_trade * 0.001425)
#                 total_fees += buy_fee  # 累加手续费
#                 print(f"做多手续费：{buy_fee:.2f}")

#                 # 做多头寸
#                 position = shares_per_trade  # 持仓为正表示做多
#                 buy_price = price  # 记录买入价格
#                 buy_stop_loss = buy_price * 0.99  # 止损价(1%亏损)
#                 buy_take_profit = buy_price * 1.03  # 止盈价(3%盈利)
#                 buy_signals.append(i)
#                 print(f"做多：價格={price}, 變化率={actual_change:.2%}, 持倉={position}股\n")

#             # 仓位控制
#             if position > 0:  # 如果是做多仓位
#                 if price <= buy_stop_loss:  # 达到止损，直接平仓
#                     print(f"做多平仓：达到止损价格 {price}")
#                     sell_fee = math.ceil(price * shares_per_trade * 0.001425)
#                     sell_tax = math.ceil(price * shares_per_trade * 0.003)
#                     total_fees += sell_fee + sell_tax
    
#                     # 计算盈亏
#                     profit_loss = (price - buy_price) * position
#                     net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
#                     trade_results.append(net_profit_loss)
#                     capital += net_profit_loss
                    
#                     print(f"平仓：價格={price}, 變化率={actual_change:.2%}, 交易含稅結果{'盈利' if net_profit_loss > 0 else '亏损'}, 本次(手續費+稅){(buy_fee + sell_fee + sell_tax)}, 盈虧(未扣稅+手續費){profit_loss:.2f}，净盈亏={net_profit_loss:.2f}\n")
#                     print(f"平倉後總資產：{capital:.2f}\n")
                    
#                     position = 0  # 清空仓位
                    
#                     if capital < 0:
#                         buy_signals.append(i)  # 做空的平仓记录为买入信号
#                     else:
#                         sell_signals.append(i)  # 做多的平仓记录为卖出信号
    
#                 elif price >= buy_take_profit:  # 达到止盈，不平仓，更新止盈止损
#                     print(f"做多未平仓：达到止盈价格 {price}, 更新止盈止损")
#                     # 以当前价格重新设定新的止盈止损
#                     buy_take_profit = price * 1.03  # 新的止盈价为当前价格上方3%
#                     buy_stop_loss = price * 0.99  # 新的止损价为当前价格下方1%
#                     print(f"新的止盈价格设定为 {buy_take_profit:.2f}, 新的止损价格设定为 {buy_stop_loss:.2f}\n")
    
#             elif position < 0:  # 如果是做空仓位
#                 if price >= sell_stop_loss:  # 达到止损，直接平仓
#                     print(f"做空平仓：达到止损价格 {price}")
#                     sell_fee = math.ceil(price * shares_per_trade * 0.001425)
#                     sell_tax = math.ceil(price * shares_per_trade * 0.003)
#                     total_fees += sell_fee + sell_tax
    
#                     # 计算盈亏
#                     profit_loss = (buy_price - price) * -position
#                     net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
#                     trade_results.append(net_profit_loss)
#                     capital += net_profit_loss
#                     print(f"平仓：價格={price}, 變化率={actual_change:.2%}, 交易含稅結果{'盈利' if net_profit_loss > 0 else '亏损'}, 本次(手續費+稅){(buy_fee + sell_fee + sell_tax)}, 盈虧(未扣稅+手續費){profit_loss:.2f}，净盈亏={net_profit_loss:.2f}\n")
#                     print(f"平倉後總資產：{capital:.2f}\n")
                    
#                     position = 0  # 清空仓位
                    
#                     if capital < 0:
#                         buy_signals.append(i)  # 做空的平仓记录为买入信号
#                     else:
#                         sell_signals.append(i)  # 做多的平仓记录为卖出信号
    
#                 elif price <= sell_take_profit:  # 达到止盈，不平仓，更新止盈止损
#                     print(f"做空未平仓：达到止盈价格 {price}, 更新止盈止损")
#                     # 以当前价格重新设定新的止盈止损
#                     sell_take_profit = price * 0.97  # 新的止盈价为当前价格下方3%
#                     sell_stop_loss = price * 1.01  # 新的止损价为当前价格上方1%
#                     print(f"新的止盈价格设定为 {sell_take_profit:.2f}, 新的止损价格设定为 {sell_stop_loss:.2f}\n")


#         # 如果还持有头寸，到最后时平仓
#         if position != 0:
#             price = test_arima['forecast_arima'].iloc[-1]

#             # 卖出成本计算：手续费和税, 累加手续费和税费
#             sell_fee = price * shares_per_trade * 0.001425
#             sell_tax = price * shares_per_trade * 0.003
#             total_fees += sell_fee + sell_tax

#             # 计算盈亏（不包括手续费和税费）
#             profit_loss = (price - buy_price) * position

#             # 计算净盈亏（扣除手续费和税费）
#             net_profit_loss = profit_loss - (buy_fee + sell_fee + sell_tax)
#             trade_results.append(net_profit_loss)
            
#             if capital < 0:
#                 buy_signals.append(i)  # 做空的平仓记录为买入信号
#             else:
#                 sell_signals.append(i)  # 做多的平仓记录为卖出信号

#             print(f"最后平仓：价格={round(price, 2)}, 最终资产={capital}")
#             print(f"最后交易净盈亏：{net_profit_loss:.2f}\n")

#         # 输出总手续费和税费
#         print(f"总手续费和税费：{total_fees:.2f}")
#         print(f"最终总资产：{capital - total_fees:.2f}\n")

#         # 输出每次交易的盈亏结果
#         for idx, result in enumerate(trade_results):
#             print(f"第 {idx + 1} 次交易净盈亏：{result:.2f}")

#         if plot is True:
#             # 可视化实际结果、ARIMA预测和GARCH预测波动性
#             plt.figure(figsize=(12, 6))

#              # 绘制实际数据
#             plt.plot(test_arima.index, test_arima['forecast_arima'], label='ARIMA Forecast Data', color='blue')

#          # 绘制进出场信号
#             plt.scatter(test_arima.index[buy_signals], test_arima['forecast_arima'].iloc[buy_signals], 
#                     marker='^', color='red', label='Buy Signal', s=100)  # 进场信号

#             plt.scatter(test_arima.index[sell_signals], test_arima['forecast_arima'].iloc[sell_signals], 
#                     marker='v', color='green', label='Sell Signal', s=100)  # 出场信号

#             plt.title('ARIMA with Trading Signals')
#             plt.xlabel('Time')
#             plt.ylabel('Log Price / Volatility')
#             plt.legend()
#             plt.grid()
#             plt.show()

# forecast(mer_ori_data, '2024-08-06', 200, plot=True, simulate=False)



# --------------------------- 深度學習LSTM模型 ------------------------------- #
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GRU, Dense, Dropout, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore

scaler = MinMaxScaler()
# mer_ori_data[['diff', 'diff3']] = scaler.fit_transform(mer_ori_data[['diff', 'diff3']])
# mer_ori_data[['close_stock1', 'close_stock3']] = scaler.fit_transform(mer_ori_data[['close_stock1', 'close_stock3']])


# 创建输入和输出序列
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 1])
    return np.array(X), np.array(y)

# 绘制预测结果的函数
def plot_predictions(y_test, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual', color='blue')
    plt.plot(predicted, label='Predicted', color='orange')
    plt.title('Compare Actual and Predicted Values')
    plt.xlabel('No.')
    plt.ylabel('diff')
    plt.legend()
    plt.show()
    
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='train loss', color='blue')
    plt.plot(history.history['val_loss'], label='validate loss', color='orange')
    plt.title('loss curve')
    plt.xlabel('train epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model(model, history, X_test, y_test):
    # 绘制损失曲线
    plot_loss(history)
    
    # 预测
    predicted = model.predict(X_test)
    # 计算和显示性能指标
    mse = mean_squared_error(y_test, predicted)
    rmse = np.sqrt(mse)
    print(f'均方误差 (MSE): {mse:.4f}')
    print(f'均方根误差 (RMSE): {rmse:.4f}')
    
    # 可视化预测结果
    plot_predictions(y_test, predicted)
    
    print(model.summary())

def train_gru_model(data, x, y, time_step=60, epochs=100, batch_size=8, learning_rate=0.0001):
    # 准备数据
    train = data[[x, y]].values  # 获取需要的列
    
    X, y = create_dataset(train, time_step)

    # 将输入转换为 [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 划分训练和测试集
    train_size = int(len(X) * 0.8)  # 80% 训练集
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 步骤 2: 构建 GRU 模型
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))  # 使用 Input 对象
    model.add(GRU(256, return_sequences=True))  # 第一层 GRU
    model.add(Dropout(0.3))  # 添加 Dropout 层，丢弃 20% 的神经元
    model.add(GRU(32))  # 第二层 GRU
    model.add(Dropout(0.1))  # 添加 Dropout 层，丢弃 20% 的神经元
    model.add(Dense(1))  # 输出层，预测 y 的值

    # 可视化模型结构
    plot_model(model, to_file='./gru_model.png', show_shapes=True, show_layer_names=True)

    # 编译模型
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # 设置 Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    # 设置学习率调度器
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    # 步骤 3: 训练模型
    fit = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])
    
    evaluate_model(model=model, history=fit, X_test=X_test, y_test=y_test)
    
    return model


# 使用示例
# model = train_gru_model(mer_ori_data, 'close_stock3', 'close_stock1')
