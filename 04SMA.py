# 实现经典量化策略
# 单移动均线 Signal Moving Average。
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
class SMAStrategy(ts.Strategy):
    """
    period, 调仓周期
    bprint, 是否输出交易过程
    """
    params = (("maperiod", 20),
              ("bprint", False),)
    def __init__(self):
        self.close = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period = self.p.maperiod)
        self.order = None
        
    def next(self):
        if self.order:
            return
            
        if not self.position:
            if self.close[0] > self.sma[0]:
                cash = self.broker.get_cash()
                amount = self.downcast(cash*0.9/self.close[0], 100)
                self.order = self.buy(size = amount)
        else:
            pos = self.getposition()
            if self.close[0] < self.sma[0]:
                self.order = self.sell(size = pos.size)
            if self.is_lastday(data = self.datas[0]):
                self.close()
    

# 主函数                
@run.change_dir
def sma():
    ts.init_display()
    start_date = "20100108"
    end_date = "20201231"
    # codes = init_data(start_date = start_date, end_date = end_date, retry = False)
    codes = ["000100"]
    backtest = ts.BackTest(
        strategy = SMAStrategy, 
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
        bprint = False, 
        bdraw = True)
    results = backtest.run()
    print("回测结果", results[:-2])
    
    
# 调参试试                
@run.change_dir
def opt_sma():
    ts.init_display()
    start_date = "20100108"
    end_date = "20201231"
    # codes = init_data(start_date = start_date, end_date = end_date, retry = False)
    codes = ["000100"]
    backtest = ts.OptStrategy(
        strategy = SMAStrategy, 
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
        maperiod = range(10, 30))
    results = backtest.run()
    print("回测结果", results.loc[:,["参数", "年化收益率"]])
    
    
# 对整个市场回测                
@run.change_dir
def research_sma():
    ts.init_display()
    start_date = "20100108"
    end_date = "20201231"
    # codes = init_data(start_date = start_date, end_date = end_date, retry = False)
    backtest = ts.Research(
        strategy = SMAStrategy, 
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
        maperiod = 25)
    results = backtest.run()
    # print("测试3")
    # print(results.info())
    results.sort_values(by = "年化收益率", inplace = True, ascending = False)
    
    print("回测结果", results.loc[:, ["年化收益率"]])


if __name__ == "__main__":
    # sma()
    # opt_sma()
    research_sma()