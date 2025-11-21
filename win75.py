import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np

class Improved_MACD_Trend_Strategy(bt.Strategy):
    params = (
        ('ema_period', 200),       # Trend Filter
        ('ema_fast', 50),          # Additional trend confirmation
        ('macd1', 12),             # Fast MA
        ('macd2', 26),             # Slow MA
        ('macdsig', 9),            # Signal MA
        ('atr_period', 14),        # For Stop Loss
        ('stop_loss_mult', 2.5),   # Tighter initial stop
        ('trail_mult', 2.0),       # Trailing stop multiplier
        ('profit_target_mult', 1.5), # Take profit at 1.5x ATR
        ('rsi_period', 14),        # RSI for overbought/oversold
        ('volume_ema', 20),        # Volume filter
        ('position_size_pct', 0.90), # Slightly reduced position size
    )

    def __init__(self):
        # 1. Multiple Trend Filters
        self.ema200 = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_period)
        self.ema50 = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_fast)
        
        # 2. MACD with histogram strength
        self.macd = bt.indicators.MACD(
            self.data.close, 
            period_me1=self.params.macd1, 
            period_me2=self.params.macd2, 
            period_signal=self.params.macdsig
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.macd_histogram = self.macd.macd - self.macd.signal
        
        # 3. Momentum and Volume Confirmation
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.volume_ema = bt.indicators.ExponentialMovingAverage(self.data.volume, period=self.params.volume_ema)
        
        # 4. ATR for Stops
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        
        # Tracking variables
        self.stop_price = None
        self.entry_price = None
        self.profit_target = None
        self.trade_count = 0
        self.win_count = 0

    def next(self):
        # Wait for indicators to load
        if len(self) < self.params.ema_period: 
            return

        # --- ENTRY LOGIC (More Selective) ---
        if not self.position:
            # Enhanced Entry Conditions:
            conditions = [
                # Primary trend: Price above 200 EMA AND 50 EMA above 200 EMA (strong trend)
                self.data.close[0] > self.ema200[0],
                self.ema50[0] > self.ema200[0],
                
                # MACD crossover with histogram confirmation
                self.crossover[0] > 0,
                self.macd_histogram[0] > self.macd_histogram[-1],  # Histogram increasing
                self.macd_histogram[0] > 0,  # Histogram positive
                
                # Momentum not overbought
                self.rsi[0] < 65,
                
                # Volume above average (confirmation)
                self.data.volume[0] > self.volume_ema[0] * 0.8,
            ]
            
            if all(conditions):
                size = int((self.broker.get_cash() * self.params.position_size_pct) / self.data.close[0])
                self.buy(size=size)
                
                self.entry_price = self.data.close[0]
                atr_value = self.atr[0]
                
                # Tighter initial stop
                self.stop_price = self.data.close[0] - (atr_value * self.params.stop_loss_mult)
                
                # Profit target
                self.profit_target = self.data.close[0] + (atr_value * self.params.profit_target_mult)
                
                self.trade_count += 1
                print(f"🚀 ENTRY @ {self.data.close[0]:.2f} | Stop: {self.stop_price:.2f} | Target: {self.profit_target:.2f}")

        # --- EXIT LOGIC (Multiple Conditions) ---
        else:
            current_price = self.data.close[0]
            atr_value = self.atr[0]
            
            # 1. Update Trailing Stop (more aggressive)
            new_stop = current_price - (atr_value * self.params.trail_mult)
            if new_stop > self.stop_price:
                self.stop_price = new_stop
            
            # 2. Check Profit Target
            if current_price >= self.profit_target:
                self.close()
                self.win_count += 1
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
                print(f"🎯 TARGET HIT @ {current_price:.2f} | Profit: {profit_pct:.2f}%")
                return
            
            # 3. Check Stop Loss
            if current_price <= self.stop_price:
                self.close()
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
                print(f"❌ STOP LOSS @ {current_price:.2f} | P&L: {profit_pct:.2f}%")
                return
            
            # 4. Emergency Exit if Trend Reverses
            if (current_price < self.ema200[0] or 
                self.macd_histogram[0] < 0 and self.macd_histogram[0] < self.macd_histogram[-1]):
                self.close()
                profit_pct = (current_price - self.entry_price) / self.entry_price * 100
                print(f"⚠️ TREND REVERSAL EXIT @ {current_price:.2f} | P&L: {profit_pct:.2f}%")

    def stop(self):
        # Performance summary
        if self.trade_count > 0:
            win_rate = (self.win_count / self.trade_count) * 100
            print(f"\n📊 STRATEGY PERFORMANCE SUMMARY")
            print(f"Total Trades: {self.trade_count}")
            print(f"Winning Trades: {self.win_count}")
            print(f"Win Rate: {win_rate:.1f}%")

class DynamicPositionSizing(bt.Sizer):
    params = (('risk_per_trade', 0.02),)  # Risk 2% per trade
    
    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            # Calculate position size based on ATR stop distance
            strategy = self.strategy
            atr_value = strategy.atr[0]
            stop_distance = atr_value * strategy.params.stop_loss_mult
            risk_amount = cash * self.params.risk_per_trade
            size = int(risk_amount / stop_distance)
            return min(size, int(cash * 0.95 / data.close[0]))
        return 0

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Improved_MACD_Trend_Strategy)
    
    # Add dynamic position sizing
    cerebro.addsizer(DynamicPositionSizing)

    print("Downloading Data (QQQ - Past 2 Years)...")
    df = yf.download('QQQ', period='2y', interval='1d', auto_adjust=True, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    # Wallet Setup
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # Additional Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Months, _name='monthly_returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print(f'Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}')
    
    # Run
    results = cerebro.run()
    strat = results[0]
    
    print(f'Final Portfolio Value: ${cerebro.broker.getvalue():.2f}')
    
    # Profit Calculation
    pnl = cerebro.broker.getvalue() - 100000
    print(f"Total Profit: ${pnl:.2f} ({pnl/1000:.1f}%)")

    # Additional Metrics
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    
    print(f"\n📈 RISK-ADJUSTED METRICS")
    print(f"Sharpe Ratio: {sharpe['sharperatio']:.2f}" if 'sharperatio' in sharpe else "Sharpe Ratio: N/A")
    print(f"Max Drawdown: {drawdown['max']['drawdown']:.1f}%")

    print("-" * 30)
    print("📅 MONTHLY PERFORMANCE REPORT")
    print("-" * 30)
    
    returns_dict = strat.analyzers.monthly_returns.get_analysis()
    for date, value in returns_dict.items():
        month_name = date.strftime("%Y-%m")
        pct_value = value * 100
        if pct_value != 0:
            icon = "🟢" if pct_value > 0 else "🔴"
            print(f"{month_name}: {icon} {pct_value:.2f}%")