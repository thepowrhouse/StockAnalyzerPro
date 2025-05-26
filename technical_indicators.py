import pandas as pd
import numpy as np

class TechnicalIndicators:
    """Class containing technical analysis indicators calculations"""
    
    def calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices (pd.Series): Price series (typically close prices)
            period (int): RSI calculation period (default: 14)
        
        Returns:
            pd.Series: RSI values
        """
        if len(prices) < period + 1:
            return pd.Series(index=prices.index, dtype=float)
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate initial average gain and loss using SMA
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RSI using the standard formula
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Use Wilder's smoothing for more accurate RSI (after initial period)
        rsi_wilder = pd.Series(index=prices.index, dtype=float)
        
        for i in range(period, len(prices)):
            if i == period:
                # First RSI calculation
                rsi_wilder.iloc[i] = rsi.iloc[i]
            else:
                # Wilder's smoothing
                prev_avg_gain = avg_gain.iloc[i-1] if not pd.isna(avg_gain.iloc[i-1]) else 0
                prev_avg_loss = avg_loss.iloc[i-1] if not pd.isna(avg_loss.iloc[i-1]) else 0
                
                current_gain = gains.iloc[i]
                current_loss = losses.iloc[i]
                
                # Wilder's smoothing formula
                new_avg_gain = (prev_avg_gain * (period - 1) + current_gain) / period
                new_avg_loss = (prev_avg_loss * (period - 1) + current_loss) / period
                
                if new_avg_loss != 0:
                    rs_wilder = new_avg_gain / new_avg_loss
                    rsi_wilder.iloc[i] = 100 - (100 / (1 + rs_wilder))
                else:
                    rsi_wilder.iloc[i] = 100
                
                # Update for next iteration
                avg_gain.iloc[i] = new_avg_gain
                avg_loss.iloc[i] = new_avg_loss
        
        return rsi_wilder.fillna(method='ffill')
    
    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices (pd.Series): Price series (typically close prices)
            fast_period (int): Fast EMA period (default: 12)
            slow_period (int): Slow EMA period (default: 26)
            signal_period (int): Signal line EMA period (default: 9)
        
        Returns:
            dict: Dictionary containing MACD, Signal, and Histogram series
        """
        if len(prices) < slow_period:
            empty_series = pd.Series(index=prices.index, dtype=float)
            return {
                'MACD': empty_series,
                'Signal': empty_series,
                'Histogram': empty_series
            }
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, fast_period)
        ema_slow = self._calculate_ema(prices, slow_period)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate Signal line (EMA of MACD line)
        signal_line = self._calculate_ema(macd_line, signal_period)
        
        # Calculate MACD Histogram
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        }
    
    def _calculate_ema(self, prices, period):
        """
        Calculate Exponential Moving Average (EMA)
        
        Args:
            prices (pd.Series): Price series
            period (int): EMA period
        
        Returns:
            pd.Series: EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, prices, period):
        """
        Calculate Simple Moving Average (SMA)
        
        Args:
            prices (pd.Series): Price series
            period (int): SMA period
        
        Returns:
            pd.Series: SMA values
        """
        return prices.rolling(window=period).mean()
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """
        Calculate Bollinger Bands
        
        Args:
            prices (pd.Series): Price series
            period (int): Moving average period (default: 20)
            std_dev (float): Standard deviation multiplier (default: 2)
        
        Returns:
            dict: Dictionary containing Upper, Middle (SMA), and Lower bands
        """
        sma = self.calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'Upper': upper_band,
            'Middle': sma,
            'Lower': lower_band
        }
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator
        
        Args:
            high (pd.Series): High price series
            low (pd.Series): Low price series
            close (pd.Series): Close price series
            k_period (int): %K period (default: 14)
            d_period (int): %D period (default: 3)
        
        Returns:
            dict: Dictionary containing %K and %D values
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'K': k_percent,
            'D': d_percent
        }
    
    def calculate_williams_r(self, high, low, close, period=14):
        """
        Calculate Williams %R
        
        Args:
            high (pd.Series): High price series
            low (pd.Series): Low price series
            close (pd.Series): Close price series
            period (int): Calculation period (default: 14)
        
        Returns:
            pd.Series: Williams %R values
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    def calculate_atr(self, high, low, close, period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            high (pd.Series): High price series
            low (pd.Series): Low price series
            close (pd.Series): Close price series
            period (int): ATR period (default: 14)
        
        Returns:
            pd.Series: ATR values
        """
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using EMA
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def get_signal_interpretation(self, rsi_value, macd_value, signal_value):
        """
        Provide interpretation of technical signals
        
        Args:
            rsi_value (float): Current RSI value
            macd_value (float): Current MACD value
            signal_value (float): Current MACD signal value
        
        Returns:
            dict: Signal interpretations
        """
        signals = {
            'rsi': 'neutral',
            'macd': 'neutral',
            'overall': 'neutral'
        }
        
        # RSI interpretation
        if rsi_value > 70:
            signals['rsi'] = 'overbought'
        elif rsi_value < 30:
            signals['rsi'] = 'oversold'
        elif rsi_value > 50:
            signals['rsi'] = 'bullish'
        else:
            signals['rsi'] = 'bearish'
        
        # MACD interpretation
        if macd_value > signal_value:
            signals['macd'] = 'bullish'
        else:
            signals['macd'] = 'bearish'
        
        # Overall signal (simplified combination)
        bullish_count = sum(1 for signal in [signals['rsi'], signals['macd']] 
                           if signal in ['bullish', 'oversold'])
        bearish_count = sum(1 for signal in [signals['rsi'], signals['macd']] 
                           if signal in ['bearish', 'overbought'])
        
        if bullish_count > bearish_count:
            signals['overall'] = 'bullish'
        elif bearish_count > bullish_count:
            signals['overall'] = 'bearish'
        
        return signals
