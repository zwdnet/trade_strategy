# 实现经典量化策略
# 双移动均线 Two Moving Average。
# 用ALMA版本
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


# 定义ALMA，参考https://community.backtrader.com/topic/3262/alma-arnaud-legoux-moving-average
class ALMA(bt.Indicator):
    lines = ("alma", )
    
    params = dict(
            period = 40, 
            sigma = 6,
            offset = 1,
    )
    
    def __init__(self):
        self.asize = self.p.period - 1
        self.m = self.p.offset * self.asize
        self.s = self.p.period / self.p.sigma
        self.dss = 2 * self.s * self.s
        
    def next(self):
        try:
            wtd_sum = 0
            self.l.alma[0] = 0
            if len(self) >= self.asize:
                for i in range(self.p.period):
                    im = i - self.m
                    wtd = np.exp( -(im * im) / self.dss)
                    self.l.alma[0] += self.data[0 - self.p.period + i] * wtd
                    wtd_sum += wtd
                self.l.alma[0] = self.l.alma[0] / wtd_sum
                # print(self.l.alma[0])
                
        except TypeError:
            self.l.alma[0] = 0
            return
                


# 策略类
class SMAStrategy(ts.Strategy):
    """
    T1, 短周期
    T2, 长周期
    stoprate, 止损位
    bprint, 是否输出交易过程
    """
    params = (("T1", 10),
              ("T2", 20),
              ("stoprate", 0.05),
              ("bprint", False),)
    def __init__(self):
        self.close = self.datas[0].close
        # self.sma1 = bt.indicators.SimpleMovingAverage(self.datas[0], period = self.p.T1)
        # self.sma2 = bt.indicators.SimpleMovingAverage(self.datas[0], period = self.p.T2)
        self.sma1 = ALMA(self.close, period = self.p.T1)
        self.sma2 = ALMA(self.close, period = self.p.T2)
        self.order = None
        self.price = 0.0
        # 测试用
        # print("参数", self.p.T1, self.p.T2, self.p.stoprate)
        
    def next(self):
        if self.order:
            return
            
        print(self.sma1[0], self.sma2[0])
        if not self.position:
            if self.sma1 > self.sma2:
                cash = self.broker.get_cash()
                amount = self.downcast(cash*0.9/self.close[0], 100)
                self.order = self.buy(size = amount)
                self.price = self.close[0]
        else:
            pos = self.getposition()
            if self.sma1 < self.sma2 or self.close[0] < self.price*(1-self.p.stoprate):
                self.order = self.sell(size = pos.size)
                self.price = 0.0
            if self.is_lastday(data = self.datas[0]):
                self.close()
    

# 主函数                
@run.change_dir
def tma():
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
def opt_tma():
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
        num_params = 3,
        T1 = range(10, 20),
        T2 = range(30, 60),
        stoprate = np.arange(0.01, 0.1, 0.01))
    results = backtest.run()
    print("回测结果", results.loc[:,["参数", "年化收益率"]])
    
    
# 对整个市场回测                
@run.change_dir
def research_tma():
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
        refresh = False, 
        bprint = False,
        retest = True,
        T1 = 16,
        T2 = 55,
        stoprate = 0.08)
    results = backtest.run()
    # print("测试3")
    # print(results.info())
    results.sort_values(by = "年化收益率", inplace = True, ascending = False)
    
    print("回测结果", results.loc[:, ["年化收益率"]])


if __name__ == "__main__":
    tma()
    # opt_tma()
    # research_tma()