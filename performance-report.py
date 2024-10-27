import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from backtesting import Backtest
from crypto_trading_strategy import AdvancedCryptoStrategy

def generate_performance_report(data, strategy):
    bt = Backtest(data, strategy, cash=10000, commission=0.002)
    results = bt.run()
    
    # Create a DataFrame with daily returns
    equity_curve = pd.Series(results._equity_curve['Equity'], index=data.index)
    daily_returns = equity_curve.pct_change()

    # Calculate additional metrics
    sharpe_ratio = results['Sharpe Ratio']
    sortino_ratio = results['Sortino Ratio']
    max_drawdown = results['Max. Drawdown [%]']
    win_rate = results['Win Rate [%]']
    
    # Visualizations
    plt.figure(figsize=(20, 15))
    
    # 1. Equity Curve
    plt.subplot(3, 2, 1)
    equity_curve.plot()
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    
    # 2. Drawdown
    plt.subplot(3, 2, 2)
    (1 - equity_curve / equity_curve.cummax()).plot()
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    
    # 3. Daily Returns Distribution
    plt.subplot(3, 2, 3)
    sns.histplot(daily_returns, kde=True)
    plt.title('Daily Returns Distribution')
    plt.xlabel('Daily Return')
    
    # 4. Monthly Returns Heatmap
    plt.subplot(3, 2, 4)
    monthly_returns = equity_curve.resample('M').last().pct_change()
    sns.heatmap(monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).mean().unstack(),
                cmap='RdYlGn', center=0, annot=True, fmt='.2%')
    plt.title('Monthly Returns Heatmap')
    
    # 5. Rolling Sharpe Ratio
    plt.subplot(3, 2, 5)
    rolling_sharpe = daily_returns.rolling(window=252).mean() / daily_returns.rolling(window=252).std() * np.sqrt(252)
    rolling_sharpe.plot()
    plt.title('Rolling Sharpe Ratio (252-day)')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    
    # 6. Underwater Plot
    plt.subplot(3, 2, 6)
    underwater = equity_curve / equity_curve.cummax() - 1
    underwater.plot()
    plt.title('Underwater Plot')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")

# Example usage
data = pd.read_csv('ETHUSDT_1d.csv', index_col='timestamp', parse_dates=True)
generate_performance_report(data, AdvancedCryptoStrategy)
