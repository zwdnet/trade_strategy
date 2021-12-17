# 实现经典量化策略
# 配对交易 Pairs Trading
# 参考 ZuraKakushadze,JuanAndrésSerur. 151 Trading Strategies.


import tradesys as ts
import run
import sys
import akshare as ak
import efinance as ef
import pandas as pd
import numpy as np
import os
import datetime


# 策略类
class PAIRStrategy(ts.Strategy):
    """
    N,交易股票只数
    period, 调仓周期
    bprint, 是否输出交易过程
    """
    params = (("N", 1),
              ("bprint", False),)
    def __init__(self, refresh = False):
        super(PAIRStrategy, self).__init__()
        self.bIn = False
        self.begin = []
        self.begin.append(self.datas[0].close[0])
        self.begin.append(self.datas[1].close[0])
        self.ret = [list(), list()]
        self.ret_sub = []
        self.ret_sub_std = []
        
    # 交易数量取整
    def downcast(self, amount, lot): 
        return abs(amount//lot*lot)
        
    def next(self):
        # print("日期", self.datas[0].datetime.date(0))
        for i, d in enumerate(self.datas):
            # print("股票", i, d.close[0])
            temp_ret = d.close[0]/self.begin[i] - 1.0
            self.ret[i].append(temp_ret)
        self.ret_sub.append(self.ret[0][-1] - self.ret[1][-1])
        self.ret_sub_std.append(np.std(list(map(abs, self.ret_sub))))
        bBuy = self.ret_sub[-1] >  self.p.N * self.ret_sub_std[-1]
        if self.bIn == False:
            # 判断买入条件
            if bBuy == True:
                i = -1
                if self.ret_sub[-1] > 0:
                    i = 1
                else:
                    i = 0
                # 计算买入数量
                cash = self.broker.get_cash()
                price = self.datas[i].close[0]
                amount = self.downcast(cash*0.9/price, 100)
                # print("现金", cash, "股价", price, "交易量", amount)
                self.buy(data = self.datas[i], size = amount)
                self.bIn = True
        else:
            # 判断卖出条件
            if bBuy == False:
                for i, d in enumerate(self.datas):
                    data = self.datas[i]
                    pos = self.getposition(data).size
                    if pos != 0:
                        # print("卖出", i, pos)
                        self.close(data = data)
                        self.bIn = False
        
    def is_lastday(self,data): 
        try: 
            next_next_close = data.close[2]
        except IndexError: 
            return True 
        except: 
            print("发生其它错误")
            return False
        
        
# 形成股票池
@run.change_dir
def make_pool(refresh = False):
    data = pd.DataFrame()
    path = "./datas/"
    stockfile = path + "stocks.csv"
    if os.path.exists(stockfile) and refresh == False:
        data = pd.read_csv(stockfile, dtype = {"code":str, "昨日收盘":np.float64})
    else:
        stock_zh_a_spot_df = ak.stock_zh_a_spot()
        stock_zh_a_spot_df.to_csv(stockfile)
        data = stock_zh_a_spot_df
    codes = select(data)
    return codes
        

# 对股票数据进行筛选
def select(data, highprice = sys.float_info.max, lowprice = 0.0):
    # 对股价进行筛选
    smalldata = data[(data.最高 < highprice) & (data.最低 > lowprice)]
    # 排除ST个股
    smalldata = smalldata[~ smalldata.名称.str.contains("ST")]
    # 排除要退市个股
    smalldata = smalldata[~ smalldata.名称.str.contains("退")]

    codes = []
    for code in smalldata.代码.values:
        codes.append(code[2:])
    
    return codes
    
    
# 下载数据并计算收益方差
def make_data(codes, start_date, end_date, refresh = False):
    rets = pd.Series()
    n = len(codes)
    i = 0
    start = np.datetime64(datetime.datetime.strptime(start_date, "%Y%m%d"))
    end = np.datetime64(datetime.datetime.strptime(end_date, "%Y%m%d"))
    for code in codes:
        print("下载数据进度", i/n)
        i += 1
        stock_data = ts.get_data(code = code, 
        start_date = start_date, 
        end_date = end_date,
        adjust = "qfq", 
        period = "daily",
        refresh = refresh)
        if len(stock_data) == 0:
            continue
        date = stock_data.日期.values
        start_gap = gap_days(start, date[0])
        end_gap = gap_days(end, date[-1])
        if start_gap == 0 and end_gap == 0:
            # 生成累积收益率数据
            stock_data["每日收益"] = stock_data["收盘"] - stock_data["收盘"].shift(1)
            rets[code] = stock_data["每日收益"]
    return rets
    
    
# 计算每日收益率序列方差
@run.change_dir
def get_top10(rets, retry = False):
    datafile = "./datas/retvar.csv"
    if os.path.exists(datafile) and retry == False:
        results = pd.read_csv(datafile)
        results.日期 = pd.to_datetime(results.日期)
        results.set_index("日期", drop = True, inplace = True)
        return results
    returns = pd.DataFrame()
    temp = pd.Series()
    n = len(rets)
    m = len(rets[0].index)
    j = 0
    # print(m, n)
    # input("按任意键继续")
    for date in rets[0].index:
        # print(date)
        i = 0
        temp["日期"] = date
        # temp["股票代码"] = rets.index[i]
        temp_data = pd.DataFrame()
        temp_code = []
        temp_var = []
        for stock in rets:
            j += 1
            print("计算收益方差序列进度:", j/(m*n))
            
            ret = stock[stock.index < date].values
            if len(ret) == 0 or len(ret) == 1:
                temp_var.append(0.0)
            else:
                temp_var.append(np.var(ret[1:]))
            temp_code.append(rets.index[i])
            i += 1
        # temp["收益方差"] = temp_var
        # temp["股票代码"] = temp_code
        temp_data["收益方差"] = temp_var
        temp_data["股票代码"] = temp_code
        # print("测试a", date, temp_data)
        temp["收益方差"] = temp_data
        returns = returns.append(temp, ignore_index = True)
    
    # 找到每个交易日收益波动最小的十只股票
    top10 = pd.DataFrame(columns = ["日期", "股票代码"])
    dates = rets[0].index
    temp_data = []
    m = len(returns)
    n = len(returns.iloc[i, :].values[0])
    t = 0
    for i in range(len(returns)):
        retvar = returns.iloc[i, :].values[0]
        # print(dates[i], retvar, type(retvar))
        topvar = []
        topcode = []
        min_index_list = []
        min_index = -1
        for k in range(10):
            min_var = float("inf")
            for j in range(len(retvar)):
                t += 1
                print("准备数据进程", t/(m*n*10))
                if j in min_index_list:
                    continue
                var_value = retvar.iloc[j, :].收益方差
                if min_var > var_value:
                    min_var = var_value
                    min_index = j
            topvar.append(min_var)
            min_index_list.append(min_index)
            topcode.append(retvar.iloc[min_index, :].股票代码)
        data = pd.Series(topcode, name = dates[i])
        temp_data.append(data)
    top10["日期"] = dates
    top10["股票代码"] = temp_data
    top10.set_index("日期", drop = True, inplace = True)

    top10.to_csv(datafile)
        
        
# 两个日期之间相差的天数
def gap_days(date1, date2):
    return (date1 - date2)/np.timedelta64(1, 'D')
    
    
# 寻找相关性最高的两只股票
def find_cov(rets, start_date = "20100108", end_date = "20201231"):
    codes = rets.index.values
    n = len(codes)
    i = 0
    max_codes = None
    max_r = -100.0
    for codeA in codes:
        for codeB in codes:
            print("找股票进程", i/(n*n))
            i += 1
            if codeA == codeB:
                continue
            r = cal_cov(rets[codeA], rets[codeB], start_date = start_date, end_date = "20151231")
            if r > max_r:
                max_r = r
                max_codes = (codeA, codeB)
                print("找到候选股票", max_codes, max_r)
    
    return max_codes, max_r
    
    
# 计算相关系数
def cal_cov(x_val, y_val, start_date = "20100108", end_date = "20201231"):
    x = []
    y = []
    for date in pd.date_range(start = start_date, end = end_date):
        try:
            a = x_val[date]
        except KeyError:
            continue
        try:
            b = y_val[date]
        except KeyError:
            continue
        if np.isnan(a) or np.isnan(b):
            continue
        x.append(a)
        y.append(b)
    return np.corrcoef(x, y)[0, 1]
        

# 重新计算数据
def init_data(start_date = "20100108", end_date = "20201231", retry = False):
    ts.init_display()
    codes = make_pool()
    rets = make_data(codes, start_date, end_date, refresh = retry)
    # print("测试", rets.head())
    # print(rets.index)
    # print(rets[0].index)
    # input("按任意键继续")
    
    codes = rets.index.values
    codes, r = find_cov(rets)
    return codes
    
    
# 测试读取数据
@run.change_dir
def test_read_data():
    print("测试读取数据")
    datafile = "./datas/retvar.csv"
    data = pd.read_csv(datafile)
    data.日期 = pd.to_datetime(data.日期)
    data.set_index("日期", drop = True, inplace = True)
    print(data.info(), data.index)
    print(data.iloc[100, :].values[0])
    print(type(data.iloc[100, :].values[0]))


@run.change_dir
def pair():
    ts.init_display()
    start_date = "20160101"
    end_date = "20201231"
    codes = [
    ["601186", "601390"],
    ["600837", "600030"],
    ["600029", "600115"],
    ["600000", "601166"],
    ["600000", "600036"],
    ["600000", "600015"]
    ]
    for code in codes:
        backtest = ts.BackTest(
            strategy = PAIRStrategy, 
            codes = code, 
            bk_code = "000300",
            start_date = start_date, 
            end_date = end_date, 
            rf = 0.03, 
            start_cash = 10000000,
            stamp_duty=0.005, 
            commission=0.0001, 
            adjust = "hfq", 
            period = "daily", 
            refresh = False, 
            bprint = False, 
            bdraw = False)
        results = backtest.run()
        print(code, "回测结果\n", results.年化收益率)
    
    
# 测试日期索引
@run.change_dir
def test_index():
    datafile = "./datas/cumreturn.csv"
    cumreturns = pd.read_csv(datafile)
    cumreturns.日期 = pd.to_datetime(cumreturns.日期)
    cumreturns.set_index("日期", drop = True, inplace = True)
    print(cumreturns.info())
    print(cumreturns.head())
    x = cumreturns.loc["2010-01-08"]
    date = datetime.date(2010, 1, 8)
    y = cumreturns.loc[str(date)]
    print(x)
    

if __name__ == "__main__":
    # test_index()
    # init_data(retry = False)
    pair()
