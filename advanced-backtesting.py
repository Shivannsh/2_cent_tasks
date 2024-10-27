import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib

class AdvancedCryptoStrategy(Strategy):
    # Define strategy parameters
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    ema_short = 10
    ema_long = 50
    atr_period = 14
    risk_per_trade = 0.02

    def init(self):
        # Calculate indicators
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.ema_short = self.I(talib.EMA, self.data.Close, self.ema_short)
        self.ema_long = self.I(talib.EMA, self.data.Close, self.ema_long)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)

    def next(self):
        # Check for buy signal
        if crossover(self.ema_short, self.ema_long) and self.rsi < self.rsi_oversold:
            # Calculate position size
            price = self.data.Close[-1]
            atr = self.atr[-1]
            stop_loss = price - 2 * atr
            risk_amount = self.risk_per_trade * self.equity
            position_size = risk_amount / (price - stop_loss)

            # Enter long position
            self.buy(size=position_size, sl=stop_loss)

        # Check for sell signal
        elif crossover(self.ema_long, self.ema_short) and self.rsi > self.rsi_overbought:
            # Exit long position
            self.position.close()

# Load data
data = pd.read_csv('ETHUSDT_1d.csv', index_col='timestamp', parse_dates=True)

# Run backtest
bt = Backtest(data, AdvancedCryptoStrategy, cash=10000, commission=.002)
stats = bt.run()
print(stats)

# Optimize strategy
optimized = bt.optimize(
    rsi_period=range(10, 30, 5),
    rsi_overbought=range(60, 90, 5),
    rsi_oversold=range(10, 40, 5),
    ema_short=range(5, 20, 5),
    ema_long=range(30, 100, 10),
    atr_period=range(10, 30, 5),
    risk_per_trade=[0.01, 0.02, 0.03, 0.04],  # Changed to a list
    maximize='Sharpe Ratio',
    constraint=lambda param: param.ema_short < param.ema_long
)

print(optimized)

# Plot results
bt.plot()