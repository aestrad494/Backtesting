#!/usr/bin/env python
# coding: utf-8

# Backtesting 

# Importing Packages
import pandas as pd
import numpy as np
import datetime
import calendar
import time
from IPython.display import clear_output
from nested_lookup import nested_lookup
import pytz
import smtplib
import math
from datetime import timedelta
import matplotlib.pyplot as plt


# Definning variables
exit_target_sell = False
exit_range_sell = False
exit_hour_sell = False
exit_target_buy = False
exit_range_buy = False
exit_hour_buy = False
in_dd = False

instrument = 'UNH'
hora_ini = '09:30:00'
hora_fin = '16:00:00'
client = 100
target = 1.27
tempo = 5
num_bars = 1
tempo_h = 1/12
num_bars_h = int((num_bars*tempo)/tempo_h)

account = 20000
risk = 0.01
profit_buy_pos = 0
profit_sell_pos = 0
profit_buy_neg = 0
profit_sell_neg = 0
exit_buy = 0
exit_sell = 0
total = []

# Commision Calculation
def calc_commission(shares):
    commission = shares * 0.005
    if (commission < 1):
        commission = 1
    return(commission)

# Progress Bar
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

# Histoical Data to Evaluate
historical = pd.read_csv('data/UNH_5secs.csv',parse_dates=['date'],index_col='date')#.set_index('date')

# Setting the initial and final date to gaet days of evaluation
initial_date = '2018/06/08'
final_date = '2019/06/05'

d0 = pd.to_datetime(initial_date)
d1 = pd.to_datetime(final_date)
delta = d1 - d0
delta = delta.days + 1
print(delta)

dates = []
for i in range(delta):
    date_n = pd.to_datetime(initial_date) + timedelta(days=i)
    date = str(date_n.strftime("%Y/%m/%d" ))
    dates += [date]


# Main Code to calculate the backtesting results
total = pd.DataFrame(total)
for i in range(delta):    
    #Getting date
    date = dates[i]

    #Getting the historical piece of data to evaluate
    hist = historical.loc[date,:]

    #Getting the max and mix from historical data. Calculating lots and target
    if (hist.empty == False):
        maximum = hist.high.rolling(num_bars_h).max()[num_bars_h-1]
        minimum = hist.low.rolling(num_bars_h).min()[num_bars_h-1]
        range_tam = round(maximum - minimum,2)
        target = 1.27
        lots = math.floor((account*risk)/(maximum-minimum)) 
    else:
        maximum = minimum = 0
        range_tam = target = lots = 0
        max_high = min_low = 0
        exit_buy = exit_sell = 0
        calc_sell = calc_buy = False

    #When to buy and sell
    hist['in_buy'] = hist['high'] > maximum
    hist['in_sell'] = hist['low'] < minimum

    if (hist.in_sell.sum() > 0):
        calc_sell = True
        price_sell = round(minimum - 0.02,2)
        in_sell_bar = list(hist['in_sell'])
        in_sell_bar = in_sell_bar.index(True)
        highs_sell = hist.iloc[in_sell_bar:,1]
        lows_sell = hist.iloc[in_sell_bar:,2]
    else:
        calc_sell = False
        price_sell = 0

    if (hist.in_buy.sum() > 0):
        calc_buy = True
        price_buy = round(maximum + 0.02,2)
        in_buy_bar = list(hist['in_buy'])
        in_buy_bar = in_buy_bar.index(True)
        highs_buy = hist.iloc[in_buy_bar:,1]
        lows_buy = hist.iloc[in_buy_bar:,2]
    else:
        calc_buy = False
        price_buy = 0

    #Determining when to exit
    ##sells
    if (calc_sell == True):    
        for k in range(len(lows_sell)):
            if (k == 0):
                new_high_sells = highs_sell[k]
                new_low_sells = lows_sell[k]
            if (k > 0):
                if (highs_sell[k] > new_high_sells):
                    new_high_sells = highs_sell[k]
                if (lows_sell[k] < new_low_sells):
                    new_low_sells = lows_sell[k]

            profit_sell_pos = round(price_sell - new_low_sells ,2)
            profit_sell_neg = round(price_sell - new_high_sells , 2)
            if (profit_sell_neg < -range_tam ):
                profit_sell_neg = -range_tam

            if (profit_sell_pos > target):
                exit_target_sell = True
                exit_sell = target
            if (exit_target_sell == False) and (new_high_sells > maximum):
                exit_range_sell = True
                exit_sell = round(price_sell - maximum,2)
            if (exit_target_sell == False) and (exit_range_sell == False) and (k == len(lows_sell)-1):
                exit_hour_sell = True
                exit_sell = round(price_sell - hist.iloc[-1,3],2)
    else:
        exit_sell = 0
        profit_sell_pos = 0
        profit_sell_neg = 0

    ##buys
    if(calc_buy == True):    
        for j in range(len(highs_buy)):
            if (j == 0):
                new_high_buys = highs_buy[j]
                new_low_buys = lows_buy[j]
            if(j > 0):
                if (highs_buy[j] > new_high_buys):
                    new_high_buys = highs_buy[j]
                if(lows_buy[j] < new_low_buys):
                    new_low_buys = lows_buy[j]

            profit_buy_pos = round(new_high_buys - price_buy ,2)
            profit_buy_neg = round(new_low_buys - price_buy, 2)
            if (profit_buy_neg < -range_tam ):
                profit_buy_neg = -range_tam

            if (profit_buy_pos > target):
                exit_target_buy = True
                exit_buy = target
            if (exit_target_buy == False) and (new_low_buys < minimum):
                exit_range_buy = True
                exit_buy = round(minimum - price_buy,2)
            if (exit_target_buy == False) and (exit_range_buy == False) and (j == len(highs_buy)-1):
                exit_hour_buy = True
                exit_buy = round(hist.iloc[-1,3] - price_buy,2)
    else:
        exit_buy = 0
        profit_buy_pos = 0
        profit_buy_neg = 0

    #Getting results
    results = [date, exit_buy, exit_sell, profit_buy_pos, profit_sell_pos, 
               profit_buy_neg, profit_sell_neg, lots]
    results = pd.DataFrame(results).T.set_index(0)

    #Appending results
    total = pd.concat([total,results])

    #restart variables in each iteration
    profit_buy_pos = profit_sell_pos = 0
    profit_buy_neg = profit_sell_neg = 0
    exit_buy = exit_sell = 0
    exit_target_sell = exit_range_sell = False
    exit_hour_sell = exit_target_buy = False
    exit_range_buy = exit_hour_buy = False
    calc_buy = calc_sell = False
    
    printProgressBar(i + 1, delta   , prefix = 'Progress:', suffix = 'Complete', length = 50)

# Organizing total table with columns
total.index.names = ['date']
total.columns = ['final profit buy', 'final profit sell', 
                 'max profit buy', 'max profit sell', 
                 'min profit buy', 'min profit sell', 'lots']


# Backtesting Results -----------------------------

# Number of entry days
entry = total['final profit buy'] + total['final profit sell']
entry_days = (entry != 0).sum()

# Calculating profits in ticks and usd

pf_buy_ticks = round(total['final profit buy'].sum(),2)
pf_sell_ticks = round(total['final profit sell'].sum(),2)
total_pf_ticks = round(pf_buy_ticks + pf_sell_ticks,2)
print('Profit Buy (ticks): ',pf_buy_ticks, ' Profit Sell (ticks): ',pf_sell_ticks, ' Total Profit(ticks): ', total_pf_ticks)

pf_buy_usd = round((total['final profit buy'] * total['lots']).sum(),2)
pf_sell_usd = round((total['final profit sell'] * total['lots']).sum(),2)
total_profit_usd = round(pf_buy_usd + pf_sell_usd,2)
print('profit buy(usd): ',pf_buy_usd, 'profit sell(usd): ',pf_sell_usd, 'total profit(usd): ', total_profit_usd)


profit_per_day = round(total_profit_usd/entry_days,2)
print('profit per day(usd): ', profit_per_day)


max_profit_buy  = total['max profit buy']
max_profit_buy = max_profit_buy.loc[(max_profit_buy != 0)]
profit_buy_mean = round(max_profit_buy.mean(),2)
print('profit buy mean(ticks): ', profit_buy_mean)


max_profit_sell  = total['max profit sell']
max_profit_sell = max_profit_sell.loc[(max_profit_sell != 0)]
profit_sell_mean = round(max_profit_sell.mean(),2)
print('profit sell mean(ticks): ',profit_sell_mean)


print('profit mean (ticks): ', (profit_buy_mean + profit_sell_mean)/2)


# Calculating the profit by day
total['profit usd'] = (total['final profit buy']+total['final profit sell'])*total['lots']


# Calculate commissions
commissions_per_trade = total['lots'].apply(calc_commission)

trades_per_day = [0]*delta
for i in range(delta):
    entry_buys = total['final profit buy'] != 0
    entry_sells = total['final profit sell'] != 0
    trades_per_day[i] = entry_buys[i] * 2 + entry_sells[i] * 2


total['commissions'] = commissions_per_trade * trades_per_day
sum_commissions = total['commissions'].sum()

# Net profit
total['net profit'] = total['profit usd'] - total['commissions']

# Acumulated Profit
total['accumulated profit'] = 0
ind_net = total.columns.get_loc("net profit")
ind_acc = total.columns.get_loc("accumulated profit")

for j in range(delta):
    if (j == 0):
        total.iloc[j,ind_acc] = account + total.iloc[j,ind_net]
    if (j > 0):
        total.iloc[j,ind_acc] = total.iloc[j-1,ind_acc] + total.iloc[j,ind_net]


total['max profit'] = 0
ind_max = total.columns.get_loc("max profit")

for i in range(delta):
    if (i == 0):
        total.iloc[i,ind_max] = total.iloc[i,ind_acc]
    if (i > 0):
        if (total.iloc[i,ind_acc] > total.iloc[i-1,ind_max]):
            total.iloc[i,ind_max] = total.iloc[i,ind_acc]
        else:
            total.iloc[i,ind_max] = total.iloc[i-1,ind_max]

#total[['accumulated profit','max profit']].plot(figsize=(20, 10))


# Maximal Drawdown
drawdown = total['max profit'] - total['accumulated profit']
max_drawdown = drawdown.max()
print('maximal drawdown(usd): ', round(max_drawdown,2))


# Relative Drawdown
relative_drawdown = max_drawdown/account
print('Relative drawdown: ', round(relative_drawdown*100,2),'%')

# Absolute Drawdown
if (total['accumulated profit'].min() < account):
    absolute_drawdown = account - total['accumulated profit'].min()
else:
    absolute_drawdown = 0
print('Absolute Drawdown(usd):', round(absolute_drawdown,2))

# Date of maximal drawdown
draw_index = list(drawdown).index(max_drawdown)
max_draw_date = dates[draw_index]
print('The maximal drawdown was in ', max_draw_date)

# Percent Profitable

## In long
longs = total['final profit buy'] != 0
number_longs = longs.sum()
pos_longs = total['final profit buy'] > 0
number_pos_longs = pos_longs.sum()
percent_longs = number_pos_longs/number_longs
print('total buy trades: ', number_longs, ' positive longs: ',number_pos_longs,  ' negative longs: ',number_longs - number_pos_longs,
      '\nPercent profitable longs: ', round(percent_longs*100,2),'%')

## in short
shorts = total['final profit sell'] != 0
number_shorts = shorts.sum()
pos_shorts = total['final profit sell'] > 0
number_pos_shorts = pos_shorts.sum()
percent_shorts = number_pos_shorts/number_shorts
print('total sell trades: ', number_shorts, ' positive shorts: ',number_pos_shorts, ' negative shorts: ',number_shorts - number_pos_shorts,
      '\nPercent profitable shorts: ', round(percent_shorts*100,2),'%')


## total
total_trades = number_longs + number_shorts
total_positive = number_pos_longs + number_pos_shorts
percent_total = total_positive/total_trades
print('total trades: ', total_trades, ' positive trades: ',total_positive, ' negative trades: ', total_trades - total_positive,
      '\nTotal Percent profitable: ', round(percent_total*100,2),'%')

# Gross Profit

## Longs
results_long = total['final profit buy']*total['lots']
profit_long = results_long[results_long > 0].sum()

## Shorts
results_short = total['final profit sell']*total['lots']
profits_short = results_short[results_short > 0].sum()

## Total
total_profits = profit_long + profits_short
print('Gross Profit(usd): ', round(total_profits,2))


# Gross Loss

## Longs
losses_long = results_long[results_long < 0].sum()


## Shorts
losses_short = results_short[results_short < 0].sum()

## Total
total_losses = losses_long + losses_short
print('Gross Loss(usd): ', round(total_losses,2))

# Expected Payoff
expected_payoff = (total_profit_usd - sum_commissions) / total_trades
print('Expected Payoff: ', round(expected_payoff,2))


# Greater profitable transaction 
max_profit_long = results_long[results_long > 0].max()
max_profit_short = results_short[results_short > 0].max()
max_profit = max(max_profit_long, max_profit_short)
print('Greater Profitable transaction: ' , round(max_profit,2))

# Greater non profitable transaction
max_loss_long = results_long[results_long < 0].min()
max_loss_short = results_short[results_short < 0].min()
max_loss = min(max_loss_long, max_loss_short)
print('Greater non Profitable transaction: ' , round(max_loss,2))

# Average profitable transaction
profit_list = pd.concat([results_long[results_long > 0],results_short[results_short > 0]])
ave_profit = profit_list.mean()
print('Average profitable transaction: ', round(ave_profit,2))


# Average non profitable transaction
loss_list = pd.concat([results_long[results_long < 0],results_short[results_short < 0]])
ave_loss = loss_list.mean()
print('Average non profitable transaction: ', round(ave_loss,2))

# Profit Factor
profit_factor = abs(total_profits/total_losses)
print('Profit Factor: ', round(profit_factor,2))

# Maximum number of Drawdown days

## Finding dates where there was DD
dates_dd = []
for i in range(delta):
    if (i < 1):
        if(total.iloc[i,10] < account):
            date_dd = total.index[i]
            dates_dd.append(date_dd)
            in_dd = True
    if (i >= 1):
        if (in_dd == False):
            if(total.iloc[i,10] < total.iloc[i-1,11]):
                date_dd = total.index[i-1]
                dates_dd.append(date_dd)
                in_dd = True
        if (in_dd == True):
            if(total.iloc[i,10] > total.iloc[i-1,11]):
                date_dd = total.index[i]
                dates_dd.append(date_dd)
                in_dd = False

len_dates_dd = len(dates_dd)

if len_dates_dd % 2 == 0:
    dates_dd 
else:
    dates_dd.append(total.index[-1])

## Organizing dates of DD in pairs
dates_if = []
for i in range(0,len(dates_dd),2):
    date_i = dates_dd[i]
    date_f = dates_dd[i+1]
    date_if = [date_i,date_f]
    dates_if.append(date_if)


#total[['accumulated profit','max profit']].plot(figsize=(20, 10))


## Getting the number of days in DD
len_dates = len(dates_dd)
delta_dates = []
for i in range(0,len_dates,2):
    date_1 = pd.to_datetime(dates_dd[i])
    date_2 = pd.to_datetime(dates_dd[i+1])
    
    delta_date = date_2 - date_1
    delta_date = delta_date.days + 1
    delta_dates.append(delta_date)

## Getting the max. number of days in DD
max_dd_days = max(delta_dates)
ind_max_dd_days = delta_dates.index(max_dd_days)
dates_max_dd = dates_if[ind_max_dd_days]
print('Max. number of days in DD: ', max_dd_days, ', between days: ', dates_max_dd)

# Final Metrics Resume
final_metrics = {}
final_metrics = {'Initial Account (usd)': account, 'Total profit (usd)': total_profit_usd, 'Total Commissions (usd)': sum_commissions, 'Net Profit (usd)': total_profit_usd - sum_commissions,
                'Gross Profit (usd)': round(total_profits,2), 'Gross Loss (usd)': round(total_losses,2), 'Profit Factor': round(profit_factor,2), 'Expected Payoff (usd)': round(expected_payoff,2),
                'Absolute Drawdown (usd)': round(absolute_drawdown,2), 'Maximal Drawdown (usd)': round(max_drawdown,2), 'Relative Drawdown (%)': round(relative_drawdown*100,2),
                'Maximal Drawdown Date': max_draw_date, 'Maximal period in Drawdown (days)': max_dd_days, 'Dates of max. DD period': str(dates_max_dd), 
                'Total Transactions': total_trades, 'Short Positions': number_shorts, 
                'Winning Shorts': number_pos_shorts, 'Winning Shorts (%)': round(percent_shorts*100,2),
                'Long Positions': number_longs, 'Winning Longs': number_pos_longs, 'Winning Longs (%)': round(percent_longs*100,2),
                'Winning Trades': total_positive, 'Winning Trades (%)': round(percent_total*100,2),
                'Losing Trades': total_trades - total_positive, 'Losing Trades (%)': round(100-(percent_total*100),2),
                'Greater Profitable Transaction (usd)': max_profit, 'Greater non Profitable Transaction (usd)': max_loss,
                'Profitable Transaction Average (usd)': round(ave_profit,2), 'Non Profitable Transaction Average (usd)': round(ave_loss,2)}

final_metrics = pd.DataFrame(final_metrics, index = [0]).T
final_metrics.index.names = ['Metric']
final_metrics.columns = ['Values']
final_metrics


# Exporting Final Metrics to Excel
final_metrics.to_excel('UNH_5Min.xlsx')

# Exporting Total table to Excel
total.to_excel('total_UNH_5Min.xlsx')