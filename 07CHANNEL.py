# 实现经典量化策略
# 通道策略 Channel。
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
import backtrader as bt


# 策略类
class CHAStrategy(ts.Strategy):
    """
    bprint, 是否输出交易过程
    """
    params = (
              ("bprint", False),
              ("T", 20), )
    def __init__(self):
        self.close = self.datas[0].close
        self.H = bt.ind.Highest(self.datas[0].high, period =  self.p.T)
        self.L = bt.ind.Highest(self.datas[0].high, period =  self.p.T)
        self.order = None
        self.price = 0.0
        # 测试用
        # print("参数", self.p.T1, self.p.T2, self.p.stoprate)
        
    def next(self):
        if self.order:
            return
            
        if not self.position:
            if self.close[0] <= self.L[0]:
                cash = self.broker.get_cash()
                amount = self.downcast(cash*0.9/self.close[0], 100)
                self.order = self.buy(size = amount)
                self.price = self.close[0]
        else:
            pos = self.getposition()
            if self.close[0] >= self.H[0]:
                self.order = self.sell(size = pos.size)
                self.price = 0.0
            if self.is_lastday(data = self.datas[0]):
                self.close()
    

# 主函数                
@run.change_dir
def channel():
    ts.init_display()
    start_date = "20100108"
    end_date = "20201231"
    # codes = init_data(start_date = start_date, end_date = end_date, retry = False)
    codes = ["000100"]
    backtest = ts.BackTest(
        strategy = CHAStrategy, 
        codes = codes, 
        bk_code = "000001",
        start_date = start_date, 
        end_date = end_date, 
        rf = 0.03, 
        start_cash = 10000000,
        stamp_duty=0.005, 
        commission=0.0001, 
        adjust = "hfq", 
        period = "daily", 
        refresh = True, 
        bprint = True, 
        bdraw = True, 
        T = 20, )
    results = backtest.run()
    print("回测结果", results[:-2])
    
    
# 调参试试                
@run.change_dir
def opt_channel():
    ts.init_display()
    start_date = "20100108"
    end_date = "20201231"
    # codes = init_data(start_date = start_date, end_date = end_date, retry = False)
    codes = ["000100"]
    backtest = ts.OptStrategy(
        strategy = CHAStrategy, 
        codes = codes, 
        bk_code = "000001",
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
        bdraw = False,
        T = range(10, 60))
    results = backtest.run()
    print("回测结果", results.loc[:,["参数", "年化收益率"]])
    
    
# 对整个市场回测                
@run.change_dir
def research_channel():
    ts.init_display()
    start_date = "20100108"
    end_date = "20201231"
    # codes = init_data(start_date = start_date, end_date = end_date, retry = False)
    backtest = ts.Research(
        strategy = CHAStrategy, 
        bk_code = "000001",
        start_date = start_date, 
        end_date = end_date, 
        start_cash = 10000000,
        min_len = 2000,
        adjust = "hfq", 
        period = "daily", 
        refresh = True, 
        bprint = False,
        retest = True,
        T = 17)
    results = backtest.run()
    # print("测试3")
    # print(results.info())
    results.sort_values(by = "年化收益率", inplace = True, ascending = False)
    
    print("回测结果", results.loc[:, ["年化收益率"]])


if __name__ == "__main__":
    # channel()
    # opt_channel()
    # research_channel()
    pass