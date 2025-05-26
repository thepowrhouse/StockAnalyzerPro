import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import numpy as np
from stock_data import StockDataFetcher
from technical_indicators import TechnicalIndicators
from portfolio_analyzer import PortfolioAnalyzer

# Configure page
st.set_page_config(
    page_title="Indian Stock Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = None

def main():
    st.title("💼 Portfolio Manager & Stock Analyzer")
    st.markdown("Upload your portfolio CSV to get live prices, CAGR, XIRR, and buy/sell recommendations")
    
    # Initialize portfolio analyzer
    portfolio_analyzer = PortfolioAnalyzer()
    
    # Main portfolio analysis
    analyze_portfolio(portfolio_analyzer)



def analyze_portfolio(portfolio_analyzer):
    """Portfolio analysis functionality"""
    st.header("💼 Portfolio Analysis & Recommendations")
    st.markdown("Upload your portfolio CSV to get live prices, CAGR, XIRR, and buy/sell recommendations")
    
    # File upload section
    st.subheader("📁 Upload Portfolio Data")
    uploaded_file = st.file_uploader(
        "Choose your portfolio CSV file",
        type=['csv'],
        help="CSV should contain columns: Symbol, Quantity, Buy_Price, Buy_Date"
    )
    
    # Display expected format
    with st.expander("📋 Expected CSV Format"):
        st.markdown("Your CSV file should have these columns:")
        sample_data = {
            'Symbol': ['RELIANCE', 'TCS', 'INFY'],
            'Quantity': [10, 25, 50],
            'Buy_Price': [2150.00, 3200.00, 1450.00],
            'Buy_Date': ['2023-01-15', '2023-03-20', '2023-06-10']
        }
        st.dataframe(pd.DataFrame(sample_data))
        st.markdown("**Note:** Symbol should be NSE symbols (without .NS suffix)")
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV file uploaded successfully!")
            
            # Validate CSV format
            is_valid, result = portfolio_analyzer.validate_csv_format(df)
            
            if not is_valid:
                st.error(f"❌ {result}")
                st.stop()
            
            # Use validated dataframe
            df = result
            
            # Display uploaded data
            st.subheader("📊 Your Portfolio Holdings")
            st.dataframe(df)
            
            # Fetch current prices
            with st.spinner("🔄 Fetching live prices and calculating metrics..."):
                symbols = df['Symbol'].unique().tolist()
                current_prices = portfolio_analyzer.get_current_prices(symbols)
                
                # Calculate portfolio metrics
                portfolio_df = portfolio_analyzer.calculate_portfolio_metrics(df, current_prices)
                
                # Generate portfolio summary
                summary = portfolio_analyzer.generate_portfolio_summary(portfolio_df)
            
            # Display portfolio summary
            st.subheader("📈 Portfolio Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Invested", 
                    f"₹{summary['total_invested']:,.2f}"
                )
            
            with col2:
                st.metric(
                    "Current Value", 
                    f"₹{summary['total_current_value']:,.2f}",
                    f"₹{summary['total_pnl']:,.2f}"
                )
            
            with col3:
                pnl_color = "normal" if summary['overall_pnl_percentage'] >= 0 else "inverse"
                st.metric(
                    "Overall P&L", 
                    f"{summary['overall_pnl_percentage']:.2f}%"
                )
            
            with col4:
                st.metric(
                    "Portfolio XIRR", 
                    f"{summary['portfolio_xirr']:.2f}%"
                )
            
            # Detailed holdings with recommendations
            st.subheader("📋 Detailed Holdings & Recommendations")
            
            # Get recommendations for each stock
            recommendations = {}
            with st.spinner("🤖 Analyzing technical indicators for recommendations..."):
                for symbol in symbols:
                    recommendation, reason = portfolio_analyzer.get_technical_recommendations(symbol)
                    recommendations[symbol] = {'action': recommendation, 'reason': reason}
            
            # Create enhanced portfolio dataframe with recommendations
            portfolio_display = portfolio_df.copy()
            portfolio_display['Recommendation'] = portfolio_display['Symbol'].map(
                lambda x: recommendations.get(x, {}).get('action', 'N/A')
            )
            portfolio_display['Reason'] = portfolio_display['Symbol'].map(
                lambda x: recommendations.get(x, {}).get('reason', 'N/A')
            )
            
            # Format display columns
            portfolio_display['Buy_Price'] = portfolio_display['Buy_Price'].apply(lambda x: f"₹{x:.2f}")
            portfolio_display['Current_Price'] = portfolio_display['Current_Price'].apply(lambda x: f"₹{x:.2f}")
            portfolio_display['Invested_Amount'] = portfolio_display['Invested_Amount'].apply(lambda x: f"₹{x:,.2f}")
            portfolio_display['Current_Value'] = portfolio_display['Current_Value'].apply(lambda x: f"₹{x:,.2f}")
            portfolio_display['PnL'] = portfolio_display['PnL'].apply(lambda x: f"₹{x:,.2f}")
            portfolio_display['PnL_Percentage'] = portfolio_display['PnL_Percentage'].apply(lambda x: f"{x:.2f}%")
            portfolio_display['CAGR'] = portfolio_display['CAGR'].apply(lambda x: f"{x:.2f}%")
            
            # Display the enhanced portfolio table
            st.dataframe(
                portfolio_display[['Symbol', 'Quantity', 'Buy_Price', 'Current_Price', 
                                'Invested_Amount', 'Current_Value', 'PnL', 'PnL_Percentage', 
                                'CAGR', 'Recommendation', 'Reason']],
                use_container_width=True
            )
            
            # Recommendation summary
            st.subheader("🎯 Recommendation Summary")
            
            buy_stocks = [symbol for symbol, rec in recommendations.items() if rec['action'] == 'BUY']
            sell_stocks = [symbol for symbol, rec in recommendations.items() if rec['action'] == 'SELL']
            hold_stocks = [symbol for symbol, rec in recommendations.items() if rec['action'] == 'HOLD']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"🟢 **BUY Recommendations ({len(buy_stocks)})**")
                if buy_stocks:
                    for stock in buy_stocks:
                        st.write(f"• {stock}")
                else:
                    st.write("No buy recommendations")
            
            with col2:
                st.warning(f"🔴 **SELL Recommendations ({len(sell_stocks)})**")
                if sell_stocks:
                    for stock in sell_stocks:
                        st.write(f"• {stock}")
                else:
                    st.write("No sell recommendations")
            
            with col3:
                st.info(f"🟡 **HOLD Recommendations ({len(hold_stocks)})**")
                if hold_stocks:
                    for stock in hold_stocks:
                        st.write(f"• {stock}")
                else:
                    st.write("No hold recommendations")
            
            # Portfolio allocation chart
            st.subheader("📊 Portfolio Allocation")
            fig = go.Figure(data=[go.Pie(
                labels=portfolio_df['Symbol'],
                values=portfolio_df['Current_Value'],
                hole=0.3
            )])
            fig.update_layout(title="Portfolio Allocation by Current Value")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error processing portfolio: {str(e)}")
            st.info("Please check your CSV format and try again.")

if __name__ == "__main__":
    main()
