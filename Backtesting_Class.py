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