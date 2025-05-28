import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

class StockDataFetcher:
    """Class to handle stock data fetching from Indian markets using yfinance"""
    def __init__(self):
        self.nse_suffix = ".NS"
        self.bse_suffix = ".BO"

    def format_symbol(self, symbol, exchange):
        """Format stock symbol with appropriate exchange suffix"""
        symbol = symbol.upper().strip()

        # Remove existing suffixes if any
        if symbol.endswith(self.nse_suffix):
            symbol = symbol[:-3]
        elif symbol.endswith(self.bse_suffix):
            symbol = symbol[:-3]

        # Add appropriate suffix based on exchange
        exchange = exchange.upper()  # Ensure uppercase comparison
        if exchange == "NSE":
            return f"{symbol}{self.nse_suffix}"
        elif exchange == "BSE":
            return f"{symbol}{self.bse_suffix}"
        else:
            return symbol  # For NASDAQ and others, return without suffix

    def get_stock_data(self, symbol, exchange, period):
        """Fetch historical stock data"""
        try:
            formatted_symbol = self.format_symbol(symbol, exchange)
            ticker = yf.Ticker(formatted_symbol)
            data = ticker.history(period=period)

            if data.empty:
                st.error(f"No data for {formatted_symbol}")
                return None

            return data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

    def get_stock_info(self, symbol, exchange):
        """
        Fetch current stock information

        Args:
            symbol (str): Stock symbol without suffix
            exchange (str): Exchange - "NSE" or "BSE"

        Returns:
            dict: Stock information including current price, market cap, etc.
        """
        try:
            formatted_symbol = self.format_symbol(symbol, exchange)
            ticker = yf.Ticker(formatted_symbol)

            # Get stock info
            info = ticker.info

            # Special handling for BSE stocks
            if exchange == "BSE":
                # Try to get BSE-specific data
                info['currentPrice'] = info.get('currentPrice') or np.nan
                info['trailingPE'] = info.get('trailingPE') or info.get('pegRatio') or np.nan
                info['stock_name'] = info.get('shortName') or info.get('longName') or symbol
            else:
                # Get stock name
                info['stock_name'] = info.get('longName', symbol)  # Use symbol as fallback

            # Get current price from recent data if not in info
            if 'currentPrice' not in info or info['currentPrice'] is None:
                recent_data = ticker.history(period="1d", interval="1m")
                if not recent_data.empty:
                    info['currentPrice'] = recent_data['Close'].iloc[-1]

            return info

        except Exception as e:
            st.warning(f"Could not fetch detailed info for {symbol}: {str(e)}")
            return {
                'currentPrice': np.nan,
                'trailingPE': np.nan,
                'priceToBook': np.nan,
                'stock_name': symbol  # Fallback to symbol
            }

    def get_current_price(self, symbol, exchange):
        """Get current market price for a symbol"""
        try:
            formatted_symbol = self.format_symbol(symbol, exchange)
            ticker = yf.Ticker(formatted_symbol)

            # Special handling for BSE
            if exchange == "BSE":
                # Get intraday data for BSE
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    return data['Close'].iloc[-1]
                return ticker.info.get('regularMarketPrice', 0)

            # Get the latest data (last 1 minute of trading)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return data['Close'].iloc[-1]

            # Fallback to daily data
            data = ticker.history(period="1d")
            if not data.empty:
                return data['Close'].iloc[-1]

            return 0
        except Exception as e:
            print(f"Error getting current price for {symbol}: {str(e)}")
            return 0

    def get_current_prices(self, symbols, exchange):
        """Get current market prices for symbols"""
        prices = {}
        for symbol in symbols:
            try:
                # Format symbol with exchange suffix
                formatted_symbol = self.format_symbol(symbol, exchange)
                # Fetch using formatted symbol
                info = self.get_stock_info(formatted_symbol, exchange)
                prices[symbol] = info.get('currentPrice', np.nan)
            except Exception as e:
                print(f"Error fetching price for {symbol}: {str(e)}")
                prices[symbol] = np.nan
        return prices
    
    def get_intraday_data(self, symbol, exchange, interval="5m"):
        """
        Fetch intraday stock data
        
        Args:
            symbol (str): Stock symbol without suffix
            exchange (str): Exchange - "NSE" or "BSE"
            interval (str): Data interval - "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"
        
        Returns:
            pandas.DataFrame: Intraday stock data
        """
        try:
            formatted_symbol = self.format_symbol(symbol, exchange)
            ticker = yf.Ticker(formatted_symbol)
            
            # Fetch today's intraday data
            data = ticker.history(period="1d", interval=interval)
            
            return data
        
        except Exception as e:
            st.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            return None
    
    def is_market_open(self):
        """
        Check if Indian stock market is currently open
        
        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            now = datetime.now()
            
            # Indian stock market hours: Monday to Friday, 9:15 AM to 3:30 PM IST
            market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            is_weekday = now.weekday() < 5
            
            # Check if current time is within market hours
            is_market_hours = market_open_time <= now <= market_close_time
            
            return is_weekday and is_market_hours
        
        except Exception:
            return False
    
    def validate_symbol(self, symbol, exchange):
        """
        Validate if a stock symbol exists
        
        Args:
            symbol (str): Stock symbol without suffix
            exchange (str): Exchange - "NSE" or "BSE"
        
        Returns:
            bool: True if symbol is valid, False otherwise
        """
        try:
            formatted_symbol = self.format_symbol(symbol, exchange)
            ticker = yf.Ticker(formatted_symbol)
            
            # Try to fetch minimal data to validate
            data = ticker.history(period="5d")
            
            return not data.empty
        
        except Exception:
            return False
