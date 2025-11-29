import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np

class AdaptiveTrendReversalStrategy(bt.Strategy):
    params = (
        ('weekly_trend_ema', 50),
        ('daily_ema_fast', 20),
        ('daily_ema_slow', 50),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('atr_period', 14),
        ('volatility_period', 20),
        ('min_volatility_threshold', 0.8),  # Avoid low-vol traps
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('volume_ema_period', 20),
        ('risk_per_trade', 0.015),  # 1.5% risk per trade (dollar risk)
        ('pyramid_enabled', True),
        ('pyramid_max_adds', 2),
        ('pyramid_threshold', 1.5),
        ('base_stop_mult', 2.0),
        ('trail_mult', 1.8),
        ('profit_target_base', 3.0),
        ('adaptive_target', True),
    )

    def __init__(self):
        # Weekly trend filter
        self.weekly_ema50 = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.weekly_trend_ema)

        # Daily trend & momentum
        self.ema20 = bt.indicators.EMA(self.data.close, period=self.p.daily_ema_fast)
        self.ema50 = bt.indicators.EMA(self.data.close, period=self.p.daily_ema_slow)
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        self.macd_hist = self.macd.macd - self.macd.signal
        self.rsi = bt.indicators.RSI(period=self.p.rsi_oversold + 10)
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=self.p.volume_ema_period)
        self.atr = bt.indicators.ATR(period=self.p.atr_period)
        self.volatility = bt.indicators.StdDev(self.data.close, period=self.p.volatility_period)
        self.avg_volatility = bt.indicators.SMA(self.volatility, period=self.p.volatility_period)

        # Tracking
        self.entry_price = None
        self.initial_stop = None
        self.trailing_stop = None
        self.profit_target = None
        self.adds = 0
        self.trade_count = 0
        self.win_count = 0
        self.in_uptrend = False

    def next(self):
        if len(self) < 60:  # Warmup
            return

        current_price = self.data.close[0]
        atr_val = self.atr[0]
        vol = self.volatility[0]
        avg_vol = self.avg_volatility[0]

        # Weekly trend filter
        self.in_uptrend = current_price > self.weekly_ema50[0]

        # Volatility filter
        if len(self.avg_volatility) < self.p.volatility_period:
            return
        if vol < avg_vol * self.p.min_volatility_threshold:
            return

        # --- PYRAMIDING ---
        if self.position and self.p.pyramid_enabled and self.adds < self.p.pyramid_max_adds:
            if current_price > self.entry_price + (self.p.pyramid_threshold * atr_val):
                add_size = self.position.size // 3
                # Allow pyramiding without cash check (margin covers it)
                if add_size > 0:
                    self.buy(size=add_size)
                    self.adds += 1
                    print(f"🔺 PYRAMID ADD @ {current_price:.2f} | Adds: {self.adds}")

        # --- EXIT LOGIC ---
        if self.position:
            new_trail = current_price - (self.p.trail_mult * atr_val)
            if self.trailing_stop is None or new_trail > self.trailing_stop:
                self.trailing_stop = new_trail

            if current_price <= self.trailing_stop:
                self.close()
                self._record_trade(current_price, "TRAILING STOP")
                return

            if current_price >= self.profit_target:
                self.close()
                self._record_trade(current_price, "PROFIT TARGET")
                return

            if (self.rsi[0] > self.p.rsi_overbought and
                self.macd_hist[0] < self.macd_hist[-1] and
                self.macd_hist[0] < 0):
                self.close()
                self._record_trade(current_price, "WEAKNESS EXIT")
                return

        # --- ENTRY ---
        if not self.position:
            trend_ok = (
                self.in_uptrend and
                current_price > self.ema20[0] > self.ema50[0]
            )
            macd_ok = (
                self.macd_hist[0] > 0 and
                self.macd_hist[0] > self.macd_hist[-1] and
                self.macd.macd[0] > self.macd.signal[0]
            )
            momentum_ok = self.rsi[0] < self.p.rsi_overbought - 5  # <65
            volume_ok = self.data.volume[0] > self.volume_ma[0] * 0.9

            if trend_ok and macd_ok and momentum_ok and volume_ok:
                risk_amount = self.broker.getvalue() * self.p.risk_per_trade
                stop_dist = self.p.base_stop_mult * atr_val
                size = int(risk_amount / stop_dist)

                # 🔥 APPLY TRUE 10X LEVERAGE
                size = size * 10

                if size <= 0:
                    return

                self.buy(size=size)
                self.entry_price = current_price
                self.initial_stop = current_price - stop_dist
                self.trailing_stop = self.initial_stop

                target_mult = self.p.profit_target_base
                if self.p.adaptive_target and vol > avg_vol * 1.2:
                    target_mult *= 1.3

                self.profit_target = current_price + (target_mult * atr_val)
                self.adds = 0
                self.trade_count += 1
                print(f"🚀 ENTRY @ {current_price:.2f} | Stop: {self.initial_stop:.2f} | Target: {self.profit_target:.2f}")

    def _record_trade(self, exit_price, reason):
        pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        if pnl_pct > 0:
            self.win_count += 1
            icon = "✅"
        else:
            icon = "❌"
        print(f"{icon} {reason} @ {exit_price:.2f} | PnL: {pnl_pct:.2f}%")

    def stop(self):
        if self.trade_count > 0:
            win_rate = self.win_count / self.trade_count * 100
            print(f"\n📊 FINAL PERFORMANCE")
            print(f"Total Trades: {self.trade_count}")
            print(f"Win Rate: {win_rate:.1f}%")


class RiskBasedSizer(bt.Sizer):
    params = (('risk', 0.015),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            return self.broker.getposition(data).size
        return 0


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AdaptiveTrendReversalStrategy)
    cerebro.addsizer(RiskBasedSizer)

    print("📥 Downloading QQQ (2 Years)...")
    df = yf.download('QQQ', period='10y', interval='1d', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    cerebro.broker.setcash(100000.0)
    # Enable 10x leverage for margin accounting
    cerebro.broker.setcommission(commission=0.001, leverage=10)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print(f'💵 Starting Portfolio: ${cerebro.broker.getvalue():,.2f}')
    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    pnl = final_value - 100000
    print(f'💰 Final Portfolio: ${final_value:,.2f}')
    print(f'📈 Total Return: ${pnl:,.2f} ({pnl/100000*100:.2f}%)')

    # Metrics
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()
    print(f"\n🎯 RISK METRICS")
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A'):.2f}")
    print(f"Max Drawdown: {dd['max']['drawdown']:.2f}%")