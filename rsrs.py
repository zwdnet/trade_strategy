# 计算rsrs
# 来自https://gitee.com/xinyangq/open_test/blob/master/get_rsrs.py

from scipy.stats import linregress 
import tradesys as ts

# get slope by ordinary least squares(ols) 
# x, narray or series 
# y, narray or series 
# y, narray or series 
# slope, float 
def get_slope(ind, df): 
    slope, *args = linregress(df.loc[ind, 'low'], df.loc[ind, 'high']) 
    return slope 

# get z-score results 
# my_series, series, initial data 
# ser, float, single result 
def get_zscore(my_series): 
    my_series = (my_series - my_series.mean()) / my_series.std() 
    ser = my_series.iloc[-1] 
    return ser 

# get rsrs indicator 
# df, DataFrame, including low and high data 
# window, int, the size of the window 
# rsrs, Series, the initial rsrs indicator 
def get_rsrs(df_ini, win_1, win_2): 
    df=df_ini.copy() 
    df['ind'] = df.index 
    print("测试", df)
    # get initial rsrs indicator 
    rsrs_ini = df['ind'].rolling(window=win_1).apply(get_slope, args=(df,)) 
    # get rsrs indicator 
    rsrs = rsrs_ini.rolling(window=win_2).apply(get_zscore) 
    return rsrs 


if __name__ == '__main__': 
    my_daily = ts.get_data('510300') 
    my_slope = get_rsrs(df_ini=my_daily, win_1=18, win_2=25) 
    print(my_slope)

