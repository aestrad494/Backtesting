import pandas as pd
import numpy as np

class Backtesting_Strategy():
    def __init__(self,df):
        self.df = df
        
    def entry_days(self):
        entry_days = ((self.df['final profit buy'] + self.df['final profit sell']) != 0).sum()
        return(entry_days)
        
        
        