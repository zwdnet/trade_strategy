# 实现经典量化策略
# 价量策略Price Momentum
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
class PMStrategy(ts.Strategy):
    """
    N,交易股票只数
    period, 调仓周期
    bprint, 是否输出交易过程
    """
    params = (("N", 10), 
              ("period", 20),
              ("bprint", False),)
    def __init__(self, refresh = False):
        super(PMStrategy, self).__init__()
        datafile = "./datas/cumreturn.csv"
        self.cumreturns = pd.read_csv(datafile)
        self.cumreturns.日期 = pd.to_datetime(self.cumreturns.日期)
        self.cumreturns.set_index("日期", drop = True, inplace = True)
        self.bIn = False
        self.bstart = True
        self.days = 0 # 记录交易天数
        self.bookmarker = pd.DataFrame()
        
    # 数据转换
    def transform(self, date):
        stock_list = self.cumreturns.loc[str(date)]
        s = stock_list.values[0][1:-1]
        s = s.replace("'", "")
        s_list = s.split()
        return s_list
        
    # 交易数量取整
    def downcast(self, amount, lot): 
        return abs(amount//lot*lot)
        
    def next(self):
        s_list = self.transform(self.datas[0].datetime.date(0))
        if self.bIn == False:
            cash = self.broker.get_cash()/self.p.N
            # 如果是第一次交易，直接使用排序结果
            if self.bstart:
                self.bookmarker["股票代码"] = s_list
                self.bookmarker["买入价"] = 0.0
                self.bstart = False
            for stock in self.bookmarker["股票代码"].values:
                data = self.getdatabyname(stock)
                pos = self.getposition(data).size
                # 计算交易数量
                amount = self.downcast(cash*0.9/data.close[0], 100)
                if not pos:
                    self.buy(data = data, size = amount)
                    self.p.N -= 1
                
                self.bookmarker.买入价[self.bookmarker.股票代码 == stock] = data.close[0]
                if self.is_lastday(data = data):
                    self.close(data = data)
            self.bIn = True
        # 到达交易天数
        elif self.days == self.p.period:
            self.bookmarker["现价"] = 0.0
            self.bookmarker["累积收益率"] = 0.0
            for code in self.bookmarker.股票代码.values:
                data = self.getdatabyname(code)
                self.bookmarker.现价[self.bookmarker.股票代码 == code] = data.close[0]
                self.bookmarker.累积收益率[self.bookmarker.股票代码 == code] = data.close[0]/self.bookmarker[self.bookmarker.股票代码 == code].买入价 - 1.0
            
            # 找出累积收益率最低的股票，卖出
            min_code = self.bookmarker[self.bookmarker.累积收益率 == self.bookmarker.min().累积收益率].股票代码.values[0]
            min_data = self.getdatabyname(min_code)
            self.close(data = min_data)
            self.bookmarker = self.bookmarker[self.bookmarker.股票代码 != min_code]
            self.p.N += 1
            self.bIn = False
            # 放入累积收益最高的股票
            self.bookmarker = self.bookmarker.append({"股票代码":s_list[0], "买入价":0.0, "现价":0.0, "累积收益率":0.0}, ignore_index = True)
            self.days = 0
        else:
            self.days += 1
        
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
    
    
# 下载数据并形成累积收益率
def make_data(codes, start_date, end_date, refresh = False):
    cumret = pd.Series()
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
            stock_data["累积收益率"] = stock_data["收盘"]/stock_data["收盘"][0] - 1.0
            cumret[code] = stock_data["累积收益率"]
    return cumret
    
    
# 计算每日累积收益率
@run.change_dir
def get_top10(cumret, retry = False):
    datafile = "./datas/cumreturn.csv"
    if os.path.exists(datafile) and retry == False:
        results = pd.read_csv(datafile)
        results.日期 = pd.to_datetime(results.日期)
        results.set_index("日期", drop = True, inplace = True)
        return results
    cumreturn = pd.DataFrame()
    temp = pd.Series()
    n = len(cumret)
    m = len(cumret[0].index)
    j = 0
    print(m, n)
    # input("按任意键继续")
    for date in cumret[0].index:
        # print(date)
        i = 0
        for stock in cumret:
            j += 1
            print("计算累积收益率进度:", j/(m*n))
            temp["日期"] = date
            temp["股票代码"] = cumret.index[i]
            ret = stock[stock.index == date].values
            if len(ret) == 0:
                temp["累积收益率"] = np.NaN
            else:
                temp["累积收益率"] = stock[stock.index == date].values[0]
            # print(temp["累积收益率"])
            cumreturn = cumreturn.append(temp, ignore_index = True)
            i += 1
    # print(cumreturn)
    results = pd.DataFrame()
    top10 = []
    for date in cumret[0].index:
        temp = cumreturn[cumreturn.日期 == date]
        temp = temp.sort_values(by = "累积收益率", ascending = False)
        top10.append(temp.loc[:, ["股票代码"]].values[:10].T[0])
        # results.append(temp)
        # results = results.append({"日期": date, "累积收益率": temp}, ignore_index = True)
    results["日期"] = cumret[0].index
    results["累积收益率"] = top10
    results.set_index("日期", drop = True, inplace = True)
    results.to_csv(datafile)
    print(results.info(), results.head())
    return results
        
        
# 两个日期之间相差的天数
def gap_days(date1, date2):
    return (date1 - date2)/np.timedelta64(1, 'D')
    

# 重新计算数据
def init_data(start_date = "20100108", end_date = "20201231", retry = False):
    codes = make_pool()
    cumret = make_data(codes, start_date, end_date, refresh = retry)
    codes = cumret.index.values
    if retry == True:
        results = get_top10(cumret, retry = False)
    return codes


@run.change_dir
def pm():
    ts.init_display()
    start_date = "20100108"
    end_date = "20201231"
    codes = init_data(start_date = start_date, end_date = end_date, retry = False)
    backtest = ts.BackTest(
        strategy = PMStrategy, 
        codes = codes, 
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
    print("回测结果", results[:-2])
    
    
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
    pm()
    # test_index()
