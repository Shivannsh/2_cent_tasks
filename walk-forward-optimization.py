import pandas as pd
import numpy as np
from backtesting import Backtest
from crypto_trading_strategy import AdvancedCryptoStrategy
from sklearn.model_selection import TimeSeriesSplit

def optimize_parameters(train_data, strategy):
    bt = Backtest(train_data, strategy, cash=10000, commission=0.002)
    optimized = bt.optimize(
        rsi_period=range(10, 30, 5),
        rsi_overbought=range(60, 90, 5),
        rsi_oversold=range(10, 40, 5),
        ema_short=range(5, 20, 5),
        ema_long=range(30, 100, 10),
        atr_period=range(10, 30, 5),
        risk_per_trade=np.arange(0.01, 0.05, 0.01),
        maximize='Sharpe Ratio',
        constraint=lambda param: param.ema_short < param.ema_long
    )
    return optimized._strategy

def walk_forward_optimization(data, strategy, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for train_index, test_index in tscv.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # Optimize parameters on training data
        optimized_strategy = optimize_parameters(train_data, strategy)

        # Backtest on test data
        bt = Backtest(test_data, optimized_strategy, cash=10000, commission=0.002)
        test_results = bt.run()

        results.append({
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'sharpe_ratio': test_results['Sharpe Ratio'],
            'return': test_results['Return [%]'],
            'max_drawdown': test_results['Max. Drawdown [%]'],
            'win_rate': test_results['Win Rate [%]'],
            'parameters': optimized_strategy.params
        })

    return pd.DataFrame(results)

# Example usage
data = pd.read_csv('BTC_USD.csv', index_col='timestamp', parse_dates=True)
wfo_results = walk_forward_optimization(data, AdvancedCryptoStrategy)

print(wfo_results)

# Visualize walk-forward optimization results
plt.figure(figsize=(12, 8))
plt.plot(wfo_results['test_end'], wfo_results['sharpe_ratio'], marker='o')
plt.title('Walk-Forward Optimization: Sharpe Ratio')
plt.xlabel('Test Period End Date')
plt.ylabel('Sharpe Ratio')
plt.grid(True)
plt.show()

# Average performance across all folds
print("Average Performance:")
print(f"Sharpe Ratio: {wfo_results['sharpe_ratio'].mean():.2f}")
print(f"Return: {wfo_results['return'].mean():.2f}%")
print(f"Max Drawdown: {wfo_results['max_drawdown'].mean():.2f}%")
print(f"Win Rate: {wfo_results['win_rate'].mean():.2f}%")
