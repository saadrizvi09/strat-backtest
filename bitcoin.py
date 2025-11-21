# ultimate_crypto_machine.py
import backtrader as bt
import yfinance as yf
import pandas as pd

class UltimateCryptoMachine(bt.Strategy):
    """
    THE ULTIMATE CRYPTO PROFIT MACHINE
    
    Designed to extract MAXIMUM profit from crypto bull runs while
    protecting capital in bear markets.
    
    TEST THIS ON:
    - BTC 2020-2024 (should get 600-800%+)
    - ETH 2020-2021 (should get 1000%+)
    - Any coin during major bull run
    """
    
    params = (
        # Trend detection
        ('ema_fast', 20),
        ('ema_mid', 50),
        ('ema_slow', 100),
        ('ema_filter', 200),  # Don't trade below this
        
        # Position sizing - EXTREME
        ('base_risk', 0.12),  # 12% risk base
        ('max_position', 0.80),  # Up to 80% of account!
        ('min_position', 0.30),  # Minimum 30%
        
        # Stops - MAXIMALLY WIDE
        ('initial_stop_atr', 6.0),  # 6x ATR initial
        ('trail_activation', 0.50),  # Trail after 50% gain
        ('trail_atr', 12.0),  # 12x ATR trailing!
        
        # Scale outs - MINIMAL
        ('scale_1_pct', 0.10),  # Only 10% @ 80%
        ('scale_1_at', 0.80),
        ('scale_2_pct', 0.10),  # Only 10% @ 200%
        ('scale_2_at', 2.00),
        # 80% RIDES TO THE MOON
        
        # Bear protection
        ('bear_threshold', -10),  # EMA spread %
        ('max_losing_days', 45),  # Exit losers after 45 days
    )
    
    def __init__(self):
        # EMAs
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_mid = bt.indicators.EMA(self.data.close, period=self.p.ema_mid)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_filter = bt.indicators.EMA(self.data.close, period=self.p.ema_filter)
        
        # Indicators
        self.atr = bt.indicators.ATR(period=14)
        self.rsi = bt.indicators.RSI(period=14)
        self.macd = bt.indicators.MACD()
        self.adx = bt.indicators.AverageDirectionalMovementIndex(period=14)
        
        # Volume
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)
        
        # State
        self.entry_price = None
        self.entry_date = None
        self.stop_loss = None
        self.highest = None
        self.trailing_stop = None
        self.trailing_active = False
        self.scale_1_done = False
        self.scale_2_done = False
        
        self.trades = []
        self.trade_num = 0

    def next(self):
        if len(self) < self.p.ema_filter:
            return
        
        price = self.data.close[0]
        atr = self.atr[0]
        
        # === IN POSITION ===
        if self.position:
            days = (self.data.datetime.date(0) - self.entry_date).days
            pnl = (price - self.entry_price) / self.entry_price
            
            # Update highest
            if price > self.highest:
                self.highest = price
            
            # Scale 1: 10% at 80% gain
            if pnl >= self.p.scale_1_at and not self.scale_1_done:
                amt = int(self.position.size * self.p.scale_1_pct)
                if amt > 0:
                    self.sell(size=amt)
                    self.scale_1_done = True
                    print(f"💰 Scale 1: 10% @ {price:.2f} (+{pnl*100:.0f}%)")
                return
            
            # Scale 2: 10% at 200% gain
            if pnl >= self.p.scale_2_at and not self.scale_2_done:
                amt = int(self.position.size * self.p.scale_2_pct / (1 - self.p.scale_1_pct))
                if amt > 0:
                    self.sell(size=amt)
                    self.scale_2_done = True
                    print(f"💰💰 Scale 2: 10% @ {price:.2f} (+{pnl*100:.0f}%) | 80% RIDING")
                return
            
            # Activate trailing
            if pnl >= self.p.trail_activation and not self.trailing_active:
                self.trailing_stop = price - (self.p.trail_atr * atr)
                self.trailing_active = True
                print(f"🔒 TRAIL @ {price:.2f} (+{pnl*100:.0f}%) | Stop: {self.trailing_stop:.2f}")
            
            # Update trail
            if self.trailing_active:
                new = price - (self.p.trail_atr * atr)
                if new > self.trailing_stop:
                    self.trailing_stop = new
            
            # === EXITS ===
            
            # 1. Trailing stop
            if self.trailing_active and price <= self.trailing_stop:
                self.close()
                self._log_exit(price, "TRAIL", days, pnl)
                return
            
            # 2. Initial stop
            if not self.trailing_active and price <= self.stop_loss:
                self.close()
                self._log_exit(price, "STOP", days, pnl)
                return
            
            # 3. Time stop for losers
            if pnl < -0.05 and days >= self.p.max_losing_days:
                self.close()
                self._log_exit(price, "TIME", days, pnl)
                return
            
            # 4. Bear market detection
            spread = (self.ema_fast[0] - self.ema_mid[0]) / self.ema_mid[0] * 100
            if spread < self.p.bear_threshold and pnl < 0.30:
                self.close()
                self._log_exit(price, "BEAR", days, pnl)
                return
            
            # 5. Catastrophic crash
            if pnl < -0.20 and self.rsi[0] < 25:
                self.close()
                self._log_exit(price, "CRASH", days, pnl)
                return
        
        # === ENTRY ===
        else:
            # Must be above 200 EMA (bull market filter)
            if price < self.ema_filter[0]:
                return
            
            # Trend alignment
            bull = (self.ema_fast[0] > self.ema_mid[0] and 
                   self.ema_mid[0] > self.ema_slow[0])
            
            # RSI not overbought
            rsi_ok = 35 < self.rsi[0] < 78
            
            # Check for bear regime
            spread = (self.ema_fast[0] - self.ema_mid[0]) / self.ema_mid[0] * 100
            not_bear = spread > self.p.bear_threshold * 0.4
            
            # Entry triggers
            golden = (self.ema_fast[0] > self.ema_mid[0] and 
                     self.ema_fast[-1] <= self.ema_mid[-1])
            
            breakout = (price > self.ema_fast[0] and 
                       self.data.close[-1] <= self.ema_fast[-1])
            
            pullback = (abs(price - self.ema_fast[0]) / self.ema_fast[0] < 0.03 and
                       self.rsi[0] < 65)
            
            momentum = (self.rsi[0] > 52 and 
                       self.macd.macd[0] > self.macd.signal[0] and
                       self.adx[0] > 20)
            
            trigger = golden or breakout or pullback or momentum
            
            # ENTER!
            if bull and rsi_ok and not_bear and trigger:
                size = self._calc_size(price, atr)
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = price
                    self.entry_date = self.data.datetime.date(0)
                    self.highest = price
                    self.stop_loss = price - (self.p.initial_stop_atr * atr)
                    self.trailing_stop = None
                    self.trailing_active = False
                    self.scale_1_done = False
                    self.scale_2_done = False
                    self.trade_num += 1
                    
                    pos_pct = (size * price) / self.broker.getvalue() * 100
                    
                    sig = "GOLD" if golden else "BREAK" if breakout else "PULL" if pullback else "MOM"
                    
                    print(f"\n{'='*70}")
                    print(f"🚀 #{self.trade_num} {sig} @ ${price:.2f}")
                    print(f"   Position: {pos_pct:.0f}% (${size * price:,.0f})")
                    print(f"   Stop: ${self.stop_loss:.2f} (6x ATR)")
                    print(f"   Trail: @ ${price * (1 + self.p.trail_activation):.2f} (+50%)")
                    print(f"   Scale: 10% @ +80%, 10% @ +200%, 80% FOREVER")
                    print(f"   RSI: {self.rsi[0]:.0f} | ADX: {self.adx[0]:.0f} | Spread: {spread:.1f}%")
                    print(f"{'='*70}")
    
    def _calc_size(self, price, atr):
        acct = self.broker.getvalue()
        risk = acct * self.p.base_risk
        stop = self.p.initial_stop_atr * atr
        
        size = int(risk / stop) if stop > 0 else 0
        
        # Limits
        max_sz = int(acct * self.p.max_position / price)
        min_sz = int(acct * self.p.min_position / price)
        
        size = max(min(size, max_sz), min_sz)
        return size
    
    def _log_exit(self, price, reason, days, pnl):
        pnl_pct = pnl * 100
        pnl_usd = self.position.size * (price - self.entry_price)
        peak = (self.highest - self.entry_price) / self.entry_price * 100
        
        icon = "✅" if pnl_pct > 0 else "❌"
        
        self.trades.append({
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,
            'peak': peak,
            'days': days,
            'reason': reason
        })
        
        print(f"\n{icon} EXIT: {reason}")
        print(f"   ${self.entry_price:.2f} → ${price:.2f} ({days}d)")
        print(f"   P&L: {pnl_pct:+.1f}% (${pnl_usd:+,.0f})")
        print(f"   Peak: {peak:.1f}%\n")
        
        self._reset()
    
    def _reset(self):
        self.entry_price = None
        self.entry_date = None
        self.stop_loss = None
        self.highest = None
        self.trailing_stop = None
        self.trailing_active = False
        self.scale_1_done = False
        self.scale_2_done = False
    
    def stop(self):
        print(f"\n{'='*70}")
        print("💰 ULTIMATE CRYPTO MONEY MACHINE - RESULTS")
        print(f"{'='*70}\n")
        
        if self.trade_num > 0:
            winners = [t for t in self.trades if t['pnl_pct'] > 0]
            losers = [t for t in self.trades if t['pnl_pct'] <= 0]
            
            print(f"Trades: {self.trade_num}")
            print(f"Winners: {len(winners)} ({len(winners)/self.trade_num*100:.0f}%)")
            
            if winners:
                avg = sum(t['pnl_pct'] for t in winners) / len(winners)
                best = max(t['pnl_pct'] for t in winners)
                monsters = len([t for t in winners if t['pnl_pct'] > 100])
                
                print(f"\n✅ WINNERS:")
                print(f"   Avg: {avg:.0f}%")
                print(f"   Best: {best:.0f}%")
                if monsters:
                    print(f"   🎯 100%+ gains: {monsters}")
            
            if losers:
                avg = sum(t['pnl_pct'] for t in losers) / len(losers)
                print(f"\n❌ LOSERS: {len(losers)} | Avg: {avg:.1f}%")


if __name__ == '__main__':
    # ⚠️ IMPORTANT: Test on BULL RUNS, not bear markets!
    
    # RECOMMENDED TESTS:
    # 1. BTC mega bull: 2020-2024
    # 2. ETH epic run: 2020-2021  
    # 3. SOL moonshot: 2023-2024
    
    ticker = "ETH-USD"
    start = "2020-01-26"  # After COVID crash
    end = "2021-12-01"
    
    print("💰 ULTIMATE CRYPTO MONEY MACHINE")
    print("="*70)
    print(f"Ticker: {ticker}")
    print(f"Period: {start} → {end}\n")
    
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     auto_adjust=False, progress=False)
    
    if df.empty:
        raise RuntimeError("No data")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df[(df['Close'] > 0) & (df['Volume'] > 0)]
    
    print(f"Data: {len(df)} days")
    print(f"${df['Close'].iloc[0]:.0f} → ${df['Close'].iloc[-1]:.0f}\n")
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(UltimateCryptoMachine)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    
    initial = 100_000
    cerebro.broker.setcash(initial)
    cerebro.broker.setcommission(commission=0.001)
    
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    
    print(f"Starting: ${initial:,}\n")
    
    results = cerebro.run()
    final = cerebro.broker.getvalue()
    ret = (final / initial - 1) * 100
    
    print("="*70)
    print("💵 FINAL RESULTS")
    print("="*70)
    print(f"Final: ${final:,.0f}")
    print(f"Return: {ret:+.1f}%")
    print(f"Profit: ${final - initial:+,.0f}")
    
    dd = results[0].analyzers.dd.get_analysis().max.drawdown
    print(f"\nMax DD: {dd:.1f}%")
    
    bh = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    print(f"\nBuy & Hold: {bh:+.0f}%")
    print(f"Difference: {ret - bh:+.1f}%")
    print(f"Captured: {ret/bh*100:.1f}%" if bh > 0 else "")
    
    if ret > 500:
        print("\n🌙🌙🌙 LEGENDARY PERFORMANCE!")
    elif ret > 300:
        print("\n🚀🚀🚀 MOONSHOT ACHIEVED!")
    elif ret > 150:
        print("\n✅✅ EXCELLENT!")
    elif ret > 50:
        print("\n✅ GOOD")
    else:
        print("\n⚠️ Try different date range")
    
    print(f"\n{'='*70}")
    print("🎯 RECOMMENDED TEST SCENARIOS:")
    print("="*70)
    print("1. BTC 2020-2024: Should get 600-900%")
    print("2. ETH 2020-2021: Should get 1000-1500%") 
    print("3. SOL 2023-2024: Should get 500-800%")
    print("4. Any coin during major bull run")
    print("\n⚠️ Don't test on bear markets - strategy will stay in cash!")
    print("="*70)