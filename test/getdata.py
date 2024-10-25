import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
import os
import shioaji as sj
from fugle_marketdata import WebSocketClient, RestClient
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

# https://ithelp.ithome.com.tw/articles/10280898 教學參考
def get_shioaji_index_data(begin, end, id, type='1k'):
    
    # 將 begin 和 end 轉換為所需的格式 2024_0708
    begin_month = begin[:7].replace('-', '')  # 取得2024-07部分，變成202407
    end_month = end[:7].replace('-', '')      # 取得2024-08部分，變成202408
    folder_name = f"{begin[:4]}_{begin_month[4:6]}{end_month[4:6]}"  # 2024_0708
    
    # 定義路徑
    folder_path = f"./index_data/shioaji/{folder_name}/"
    
    # 如果資料夾不存在，則創建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 定義CSV檔案的名稱
    file_name = f"{id}.csv"
    file_path = os.path.join(folder_path, file_name)
    
    api = sj.Shioaji()
    api.login(
        api_key=os.getenv('API_KEY'),
        secret_key=os.getenv('SECRET_KEY'),
        fetch_contract=False,
    )
    api.fetch_contracts(contract_download=True)
    #如果要查看合約代號可以直接print(api.Contracts), 查看特定的股票可以print(api.Contracts.Stocks.TSE['2330'])
    
    # 創建一個空的 DataFrame 來存儲所有數據
    all_ticks = pd.DataFrame()

    if type == '1k':
        print('Remain usage API usage:', api.usage())
            
        # 調用 API 獲取當天的數據
        kbars = api.kbars(
            contract=api.Contracts.Stocks.TSE[id],
            start=begin,
            end=end,
        )
            
        df = pd.DataFrame({**kbars})
        all_ticks = pd.concat([all_ticks, df], ignore_index=True)
        
        print(f"Processed data from {begin} to {end}")
        
        all_ticks['ts'] = pd.to_datetime(all_ticks['ts'])

        all_ticks.to_csv(file_path, index=False)
        print(f'Data saved to {file_path}')
        
    else:
        # 設置開始日期和結束日期
        start_date, end_date = datetime.strptime(begin, "%Y-%m-%d"), datetime.strptime(end, "%Y-%m-%d")
        
        # 遍歷日期範圍，按天調用 API
        current_date = start_date
        
        while current_date <= end_date:
            print('Remain usage API usage:', api.usage())

            # 格式化日期為字符串
            date_str = current_date.strftime("%Y-%m-%d")

            # 調用 API 獲取當天的數據 TSE001(大盤), TSE099(台灣50)
            ticks = api.ticks(
                contract=api.Contracts.Indexs.TSE.TSE001,
                date=date_str
            )

            df = pd.DataFrame({**ticks})
            all_ticks = pd.concat([all_ticks, df], ignore_index=True)

            print(f"Processed data for {date_str}")

            current_date += timedelta(days=1)

        all_ticks = all_ticks[['ts', 'close']]
        all_ticks['ts'] = pd.to_datetime(all_ticks['ts'])

        all_ticks.to_csv(file_path, index=False)
        print(f'Data saved to {file_path}')

# get_shioaji_index_data('2023-12-30', '2024-07-30', '00631L')


#使用Fugle API獲得資料
def get_fugle_data(stock_list, data_folder):
    folder = f'./index_data/fugle/{data_folder}'

    client = RestClient(api_key = os.getenv('FUGLE'))
    
    # 遍历股票代码列表
    for stock_code in stock_list:
        stock = client.stock.historical.candles(**{"symbol": f'{stock_code}',  "timeframe": 1})
    
        if 'symbol' not in stock:
            raise ValueError(stock['message'])

        data = pd.DataFrame(stock['data'])
        data.rename(columns={'date': 'datetime'}, inplace=True)
        data['datetime'] = pd.to_datetime(data['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        data.set_index('datetime', inplace=True)
        reversed_df = data.iloc[::-1]
        reversed_df.to_csv(f'{folder}/{data_folder}/{stock_code}.csv')
        
    print(f'Data All safet to {folder}/{data_folder}')

# 獲取ETF
stock_list = ["00631L", "00632R", "00663L", "00664R", "00675L", "00676R", "00685L", "00686R", 
              "00633L", "00634R", "00637L", "00638R", "00655L", "00656R", "00753L", "00650L",
              "00651R", "00665L", "00666R", "00640L", "00641R", "00654R", "00647L", "00648R",
              "00852L", "00669R", "00670L", "00671R", "00683L", "00684R", "00706L", "00707R",
              "00708L", "00674R", "00673R", "00715L", "00680L", "00681R", "00688L", "00689R",
              "2330"
            ]


#每個月要更換一次月份代號(2024_0708 => 2024_0809)
# get_fugle_data(stock_list=stock_list, data_folder='2024_0809')


def get_heatmap(stock_list, start, end):
    data = yf.download(stock_list, start=start, end=end)['Adj Close']
    
    # 计算股票的日收益率
    returns = data.pct_change()
    
    # 计算相关性矩阵
    correlation_matrix = returns.corr()

    # 使用seaborn绘制相关性矩阵的热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5,
                xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.index,  # 显示代号
                cbar_kws={"shrink": .8})  # 调整颜色条大小
    plt.title('Stock Correlation Heatmap')
    plt.xlabel('Stocks')  # 添加x轴标签
    plt.ylabel('Stocks')  # 添加y轴标签
    plt.xticks(rotation=45)  # 使x轴标签倾斜，便于阅读
    plt.yticks(rotation=0)  # 保持y轴标签水平
    plt.show()
    
# heatmap_stock_list = ["2312.TW","2454.TW", "3231.TW", "2377.TW", "3661.TW", "3227.TW", "2353.TW", "2382.TW",
#                       "2330.TW", "3029.TW", "2357.TW", "3443.TW", "2449.TW", "6841.TW", "2317.TW", "3169.TW",
#                       "2356.TW", "6412.TW", "6414.TW", "6166.TW", '6214.TW', "2453.TW", "6902.TW", "5351.TW",
#                       "2395.TW", "2308.TW", "2439.TW"
#                       ]

heatmap_stock_list = ['6023.TWO', '6015.TWO', '6020.TWO', '6026.TWO', '6021.TWO', '5878.TWO', '5864.TWO', '6016.TWO']

# get_heatmap(heatmap_stock_list, '2024-01-01', '2024-10-01')