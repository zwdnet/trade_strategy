# 实现经典量化策略
# 支撑与阻挡策略 Support and Resistance。
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
class SRStrategy(ts.Strategy):
    """
    bprint, 是否输出交易过程
    """
    params = (
              ("bprint", False),)
    def __init__(self):
        self.close = self.datas[0].close
        self.H = self.datas[0].high
        self.L = self.datas[0].low
        self.C = (self.H + self.L + self.close)/3
        self.R = 2.0*self.C - self.L
        self.order = None
        self.price = 0.0
        # 测试用
        # print("参数", self.p.T1, self.p.T2, self.p.stoprate)
        
    def next(self):
        if self.order:
            return
            
        if not self.position:
            if self.close[0] > self.C[0]:
                cash = self.broker.get_cash()
                amount = self.downcast(cash*0.9/self.close[0], 100)
                self.order = self.buy(size = amount)
                self.price = self.close[0]
        else:
            pos = self.getposition()
            if self.H[0] >= self.R[0]:
                self.order = self.sell(size = pos.size)
                self.price = 0.0
            if self.is_lastday(data = self.datas[0]):
                self.close()
    

# 主函数                
@run.change_dir
def sr():
    ts.init_display()
    start_date = "20100108"
    end_date = "20201231"
    # codes = init_data(start_date = start_date, end_date = end_date, retry = False)
    codes = ["000100"]
    backtest = ts.BackTest(
        strategy = SRStrategy, 
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
    
    
# 对整个市场回测                
@run.change_dir
def research_sr():
    ts.init_display()
    start_date = "20100108"
    end_date = "20201231"
    # codes = init_data(start_date = start_date, end_date = end_date, retry = False)
    backtest = ts.Research(
        strategy = SRStrategy, 
        bk_code = "000001",
        start_date = start_date, 
        end_date = end_date, 
        start_cash = 10000000,
        min_len = 2000,
        adjust = "hfq", 
        period = "daily", 
        refresh = True, 
        bprint = False,
        retest = True)
    results = backtest.run()
    # print("测试3")
    # print(results.info())
    results.sort_values(by = "年化收益率", inplace = True, ascending = False)
    
    print("回测结果", results.loc[:, ["年化收益率"]])


if __name__ == "__main__":
    sr()
    # research_sr()