import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf

from scipy.optimize import newton

from stock_data import StockDataFetcher
from technical_indicators import TechnicalIndicators

class PortfolioAnalyzer:
    """Class to analyze portfolio holdings and provide recommendations"""

    def __init__(self):
        self.tech_indicators = TechnicalIndicators()

    def validate_symbol(self, symbol):
        """
        Validate and format stock symbol
        """
        try:
            # Remove any whitespace and convert to uppercase
            symbol = symbol.strip().upper()

            # Check common exchanges
            exchanges = {
                'NSE': '.NS',
                'BSE': '.BO',
                'NYSE': '',
                'NASDAQ': ''
            }

            # If no exchange suffix is present, try to determine the correct one
            if '.' not in symbol:
                # Try yfinance symbol lookup
                import yfinance as yf
                try:
                    stock = yf.Ticker(f"{symbol}.NS")
                    info = stock.info
                    if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                        return f"{symbol}.NS"
                except:
                    pass

                try:
                    stock = yf.Ticker(f"{symbol}.BO")
                    info = stock.info
                    if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                        return f"{symbol}.BO"
                except:
                    pass

                # If still no match, try without suffix
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                        return symbol
                except:
                    pass

            return symbol

        except Exception as e:
            print(f"Error validating symbol {symbol}: {str(e)}")
            return None

    def validate_csv_format(self, df):
        """
        Validate the uploaded CSV format
        Expected columns: Symbol, Quantity, Buy_Price, Buy_Date
        """
        required_columns = ['Symbol', 'Quantity', 'Buy_Price', 'Buy_Date']

        # In validate_csv_format method
        try:
            df['Buy_Date'] = pd.to_datetime(df['Buy_Date'], errors='coerce')
            invalid_dates = df[df['Buy_Date'].isna()]
            if not invalid_dates.empty:
                return False, "Invalid date format in Buy_Date column"
        except Exception as e:
            return False, f"Date parsing error: {str(e)}"
        
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

    def calculate_portfolio_metrics(self, df, current_prices):
        """Calculate portfolio metrics including P&L, CAGR"""
        portfolio_data = []

        for _, row in df.iterrows():
            symbol = row['Symbol']
            quantity = row['Quantity']
            buy_price = row['Buy_Price']
            buy_date = row['Buy_Date']

            # Get current price - handle missing values
            current_price = current_prices.get(symbol, 0)
            if pd.isna(current_price) or current_price <= 0:
                current_price = 0

            # Calculate metrics
            invested_amount = quantity * buy_price
            current_value = quantity * current_price
            pnl = current_value - invested_amount
            pnl_percentage = (pnl / invested_amount) * 100 if invested_amount > 0 else 0

            # Calculate holding period in years
            holding_period_days = (datetime.now() - buy_date).days
            holding_period_years = max(holding_period_days / 365.25, 0.001)  # Avoid division by zero

            # Calculate CAGR
            cagr = 0
            if buy_price > 0 and current_price > 0:
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

    def calculate_xirr(self, cashflows):
        """Calculate XIRR given a list of (date, amount) tuples"""
        if len(cashflows) < 2:
            return None

        # Convert to ordinal dates for numerical calculation
        dates, amounts = zip(*cashflows)
        dates = [d.toordinal() for d in dates]

        # Define the XIRR function
        def xirr_func(rate):
            return sum(amt / (1 + rate) ** ((d - dates[0]) / 365) for d, amt in zip(dates, amounts))

        try:
            # Use Newton-Raphson method to solve
            return newton(xirr_func, 0.1) * 100  # Convert to percentage
        except:
            return None

    def _xirr_fallback(self, cash_flows, years):
        """Fallback XIRR calculation method"""
        try:
            # Initialize bounds
            low, high = -0.99, 5.0  # -99% to 500%
            tolerance = 1e-6

            # Define NPV function
            def npv(rate):
                return sum(cf / ((1 + rate) ** year) for cf, year in zip(cash_flows, years))

            # Use bisection method
            for _ in range(100):
                mid = (low + high) / 2
                npv_mid = npv(mid)

                if abs(npv_mid) < tolerance:
                    return mid * 100

                npv_low = npv(low)
                npv_high = npv(high)

                # Check if solution is bracketed
                if npv_low * npv_high > 0:
                    return 0

                if npv_mid * npv_low < 0:
                    high = mid
                else:
                    low = mid

            return mid * 100
        except Exception:
            return 0
    
    def generate_portfolio_summary(self, portfolio_df):
        """Generate overall portfolio summary"""
        if portfolio_df.empty:
            return {}
        
        total_invested = portfolio_df['Invested_Amount'].sum()
        total_current_value = portfolio_df['Current_Value'].sum()
        total_pnl = portfolio_df['PnL'].sum()
        overall_pnl_percentage = (total_pnl / total_invested) * 100 if total_invested > 0 else 0

        # Calculate XIRR
        cashflows = []
        today = datetime.today().date()

        for _, row in portfolio_df.iterrows():
            try:
                buy_date = pd.to_datetime(row['Buy_Date']).date()
                # Add investment as negative cash flow
                cashflows.append((
                    buy_date,
                    -float(row['Quantity'] * row['Buy_Price'])
                ))
            except:
                continue

        # Add current value as positive cash flow on today's date
        cashflows.append((today, float(portfolio_df['Current_Value'].sum())))

        # Calculate XIRR
        try:
            portfolio_xirr = self.calculate_xirr(cashflows)
        except Exception as e:
            st.error(f"XIRR calculation failed: {str(e)}")
            portfolio_xirr = None

        if portfolio_xirr is None:
            # Convert cashflows to (amount, years) format
            min_date = min(d for d, _ in cashflows)
            cash_flows = [amt for _, amt in cashflows]
            years = [(d - min_date).days / 365.25 for d, _ in cashflows]
            portfolio_xirr = self._xirr_fallback(cash_flows, years)
        
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