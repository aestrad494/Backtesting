import pandas as pd
import numpy as np

class Backtesting_Strategy():
    def __init__(self,df):
        self.df = df
    
    def calc_commission(self, shares):
        commission = shares * 0.005
        if (commission < 1):
            commission = 1
        return(commission)
    
    def final_profit_usd(self):
        total_profit_usd = round((self.df['profit usd']).sum(),2)
        return(total_profit_usd)
    
    def total_commissions(self):
        commissions_per_trade = self.df['lots'].apply(self.calc_commission)
        trades_per_day = (self.df['final profit buy'] != 0)*2 + (self.df['final profit sell'] != 0)*2
        total_commissions = (commissions_per_trade * trades_per_day).sum()
        return(round(total_commissions,2))
    
    def gross_profit_and_loss(self):
        results_long = self.df['final profit buy'] * self.df['lots']
        results_short = self.df['final profit sell'] * self.df['lots']
        total_profits = round(results_long[results_long > 0].sum() + results_short[results_short > 0].sum(),2)
        total_losses = round(results_long[results_long < 0].sum() + results_short[results_short < 0].sum(),2)
        return([total_profits,total_losses])
    
    def max_drawdown(self):
        drawdown = self.df['max profit'] - self.df['accumulated profit']
        max_drawdown = drawdown.max()
        draw_index = list(drawdown).index(max_drawdown)
        max_draw_date = self.df.index[draw_index]
        return([round(max_drawdown,2),max_draw_date])
    
    def absolute_drawdown(self, account):
        if (self.df['accumulated profit'].min() < account):
            absolute_drawdown = account - self.df['accumulated profit'].min()
        else:
            absolute_drawdown = 0
        return(round(absolute_drawdown,2))
    
    def max_drawdown_date(self, account):
        dates_dd = []
        in_dd = False
        delta = self.df.shape[0]
        for i in range(delta):
            if (i < 1):
                if(self.df['accumulated profit'][i] < account):
                    date_dd = self.df.index[i]
                    dates_dd.append(date_dd)
                    in_dd = True
            if (i >= 1):
                if (in_dd == False):
                    if(self.df['accumulated profit'][i] < self.df['max profit'][i-1]):
                        date_dd = self.df.index[i-1]
                        dates_dd.append(date_dd)
                        in_dd = True
                if (in_dd == True):
                    if(self.df['accumulated profit'][i] > self.df['max profit'][i-1]):
                        date_dd = self.df.index[i]
                        dates_dd.append(date_dd)
                        in_dd = False

        len_dates_dd = len(dates_dd)

        if len_dates_dd % 2 == 0:
            dates_dd 
        else:
            dates_dd.append(self.df.index[-1])
        
        dates_if = []
        for i in range(0,len(dates_dd),2):
            date_i = dates_dd[i]
            date_f = dates_dd[i+1]
            date_if = [date_i,date_f]
            dates_if.append(date_if)
            
        len_dates = len(dates_dd)
        delta_dates = []
        for i in range(0,len_dates,2):
            date_1 = pd.to_datetime(dates_dd[i])
            date_2 = pd.to_datetime(dates_dd[i+1])

            delta_date = date_2 - date_1
            delta_date = delta_date.days + 1
            delta_dates.append(delta_date)
            
        max_dd_days = max(delta_dates)
        ind_max_dd_days = delta_dates.index(max_dd_days)
        dates_max_dd = dates_if[ind_max_dd_days]
        
        return([max_dd_days, dates_max_dd])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        