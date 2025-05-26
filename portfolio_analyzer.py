import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from stock_data import StockDataFetcher
from technical_indicators import TechnicalIndicators

class PortfolioAnalyzer:
    """Class to analyze portfolio holdings and provide recommendations"""
    
    def __init__(self):
        self.data_fetcher = StockDataFetcher()
        self.tech_indicators = TechnicalIndicators()
    
    def validate_csv_format(self, df):
        """
        Validate the uploaded CSV format
        Expected columns: Symbol, Quantity, Buy_Price, Buy_Date
        """
        required_columns = ['Symbol', 'Quantity', 'Buy_Price', 'Buy_Date']
        
        # Check if all required columns exist (case-insensitive)
        df_columns_lower = [col.lower() for col in df.columns]
        missing_columns = []
        
        for col in required_columns:
            if col.lower() not in df_columns_lower:
                missing_columns.append(col)
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'symbol' in col_lower:
                column_mapping[col] = 'Symbol'
            elif 'quantity' in col_lower or 'qty' in col_lower:
                column_mapping[col] = 'Quantity'
            elif 'price' in col_lower and 'buy' in col_lower:
                column_mapping[col] = 'Buy_Price'
            elif 'date' in col_lower and 'buy' in col_lower:
                column_mapping[col] = 'Buy_Date'
        
        df = df.rename(columns=column_mapping)
        
        # Validate data types
        try:
            df['Quantity'] = pd.to_numeric(df['Quantity'])
            df['Buy_Price'] = pd.to_numeric(df['Buy_Price'])
            df['Buy_Date'] = pd.to_datetime(df['Buy_Date'], format='%Y-%m-%d', errors='coerce')
        except Exception as e:
            return False, f"Data type conversion error: {str(e)}"
        
        return True, df
    
    def get_current_prices(self, symbols):
        """Fetch current prices for all symbols in portfolio"""
        current_prices = {}
        
        for symbol in symbols:
            try:
                stock_info = self.data_fetcher.get_stock_info(symbol, "NSE")
                current_price = stock_info.get('currentPrice', 0)
                
                # If current price not available, try getting from recent data
                if current_price == 0:
                    recent_data = self.data_fetcher.get_stock_data(symbol, "NSE", "1d")
                    if recent_data is not None and not recent_data.empty:
                        current_price = recent_data['Close'].iloc[-1]
                
                current_prices[symbol] = current_price
                
            except Exception as e:
                st.warning(f"Could not fetch price for {symbol}: {str(e)}")
                current_prices[symbol] = 0
        
        return current_prices
    
    def calculate_portfolio_metrics(self, df, current_prices):
        """Calculate portfolio metrics including P&L, CAGR"""
        portfolio_data = []
        
        for _, row in df.iterrows():
            symbol = row['Symbol']
            quantity = row['Quantity']
            buy_price = row['Buy_Price']
            buy_date = row['Buy_Date']
            
            current_price = current_prices.get(symbol, 0)
            
            # Calculate metrics
            invested_amount = quantity * buy_price
            current_value = quantity * current_price
            pnl = current_value - invested_amount
            pnl_percentage = (pnl / invested_amount) * 100 if invested_amount > 0 else 0
            
            # Calculate holding period in years
            holding_period_days = (datetime.now() - buy_date).days
            holding_period_years = holding_period_days / 365.25
            
            # Calculate CAGR
            cagr = 0
            if holding_period_years > 0 and buy_price > 0:
                cagr = ((current_price / buy_price) ** (1 / holding_period_years) - 1) * 100
            
            portfolio_data.append({
                'Symbol': symbol,
                'Quantity': quantity,
                'Buy_Price': buy_price,
                'Current_Price': current_price,
                'Buy_Date': buy_date,
                'Invested_Amount': invested_amount,
                'Current_Value': current_value,
                'PnL': pnl,
                'PnL_Percentage': pnl_percentage,
                'Holding_Period_Days': holding_period_days,
                'CAGR': cagr
            })
        
        return pd.DataFrame(portfolio_data)
    
    def calculate_xirr(self, cash_flows, dates):
        """
        Calculate XIRR (Extended Internal Rate of Return)
        cash_flows: list of cash flows (negative for investments, positive for current value)
        dates: list of corresponding dates
        """
        try:
            # Simple approximation of XIRR using numpy
            # For more accurate XIRR, you might want to use scipy.optimize
            
            if len(cash_flows) != len(dates) or len(cash_flows) < 2:
                return 0
            
            # Convert dates to days from first date
            first_date = min(dates)
            days = [(date - first_date).days for date in dates]
            
            # Use numpy's IRR approximation
            # This is a simplified version - for production, consider using scipy
            def npv(rate, cash_flows, days):
                return sum(cf / (1 + rate) ** (day / 365.25) for cf, day in zip(cash_flows, days))
            
            # Binary search for IRR
            low, high = -0.99, 10.0
            mid = 0.0
            for _ in range(100):  # Max iterations
                mid = (low + high) / 2
                if abs(npv(mid, cash_flows, days)) < 1e-6:
                    return mid * 100
                elif npv(mid, cash_flows, days) > 0:
                    low = mid
                else:
                    high = mid
            
            return mid * 100
        
        except Exception:
            return 0
    
    def get_technical_recommendations(self, symbol):
        """Get buy/sell recommendation based on technical indicators"""
        try:
            # Fetch recent data for technical analysis
            stock_data = self.data_fetcher.get_stock_data(symbol, "NSE", "3mo")
            
            if stock_data is None or stock_data.empty:
                return "NO_DATA", "Insufficient data for analysis"
            
            # Calculate technical indicators
            rsi = self.tech_indicators.calculate_rsi(stock_data['Close'])
            macd_data = self.tech_indicators.calculate_macd(stock_data['Close'])
            
            # Get latest values
            latest_rsi = rsi.iloc[-1] if not rsi.empty else 50
            latest_macd = macd_data['MACD'].iloc[-1] if not macd_data['MACD'].empty else 0
            latest_signal = macd_data['Signal'].iloc[-1] if not macd_data['Signal'].empty else 0
            
            # Simple recommendation logic
            signals = []
            
            # RSI signals
            if latest_rsi > 70:
                signals.append("SELL")  # Overbought
            elif latest_rsi < 30:
                signals.append("BUY")   # Oversold
            else:
                signals.append("HOLD")  # Neutral
            
            # MACD signals
            if latest_macd > latest_signal:
                signals.append("BUY")   # Bullish crossover
            else:
                signals.append("SELL")  # Bearish crossover
            
            # Combine signals
            buy_signals = signals.count("BUY")
            sell_signals = signals.count("SELL")
            
            if buy_signals > sell_signals:
                recommendation = "BUY"
                reason = f"RSI: {latest_rsi:.2f}, MACD above signal line"
            elif sell_signals > buy_signals:
                recommendation = "SELL"
                reason = f"RSI: {latest_rsi:.2f}, MACD below signal line"
            else:
                recommendation = "HOLD"
                reason = f"Mixed signals - RSI: {latest_rsi:.2f}"
            
            return recommendation, reason
        
        except Exception as e:
            return "ERROR", f"Analysis error: {str(e)}"
    
    def generate_portfolio_summary(self, portfolio_df):
        """Generate overall portfolio summary"""
        if portfolio_df.empty:
            return {}
        
        total_invested = portfolio_df['Invested_Amount'].sum()
        total_current_value = portfolio_df['Current_Value'].sum()
        total_pnl = portfolio_df['PnL'].sum()
        overall_pnl_percentage = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
        
        # Calculate portfolio XIRR
        cash_flows = []
        dates = []
        
        # Add all investments as negative cash flows
        for _, row in portfolio_df.iterrows():
            cash_flows.append(-row['Invested_Amount'])
            dates.append(row['Buy_Date'])
        
        # Add current portfolio value as positive cash flow
        cash_flows.append(total_current_value)
        dates.append(datetime.now())
        
        portfolio_xirr = self.calculate_xirr(cash_flows, dates)
        
        # Calculate average CAGR (weighted by investment amount)
        total_weighted_cagr = 0
        for _, row in portfolio_df.iterrows():
            weight = row['Invested_Amount'] / total_invested
            total_weighted_cagr += row['CAGR'] * weight
        
        return {
            'total_invested': total_invested,
            'total_current_value': total_current_value,
            'total_pnl': total_pnl,
            'overall_pnl_percentage': overall_pnl_percentage,
            'portfolio_xirr': portfolio_xirr,
            'weighted_avg_cagr': total_weighted_cagr,
            'number_of_stocks': len(portfolio_df)
        }