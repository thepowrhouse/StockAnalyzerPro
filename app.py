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
    st.title("📈 Indian Stock Market Analyzer & Portfolio Manager")
    st.markdown("Real-time technical analysis for NSE and BSE stocks with portfolio recommendations")
    
    # Initialize data fetcher, technical indicators, and portfolio analyzer
    data_fetcher = StockDataFetcher()
    tech_indicators = TechnicalIndicators()
    portfolio_analyzer = PortfolioAnalyzer()
    
    # Main navigation
    tab1, tab2 = st.tabs(["📊 Stock Analysis", "💼 Portfolio Analysis"])
    
    with tab1:
        analyze_individual_stock(data_fetcher, tech_indicators)
    
    with tab2:
        analyze_portfolio(portfolio_analyzer)

def analyze_individual_stock(data_fetcher, tech_indicators):
    """Individual stock analysis functionality"""
    # Sidebar for controls
    with st.sidebar:
        st.header("Stock Selection")
        
        # Stock symbol input
        symbol_input = st.text_input(
            "Enter Stock Symbol",
            placeholder="e.g., RELIANCE, TCS, INFY",
            help="Enter NSE symbol without .NS suffix"
        ).upper().strip()
        
        # Exchange selection
        exchange = st.selectbox(
            "Select Exchange",
            ["NSE", "BSE"],
            index=0,
            help="Choose between National Stock Exchange (NSE) or Bombay Stock Exchange (BSE)"
        )
        
        # Time period selection
        period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Select historical data period"
        )
        
        # Technical indicator parameters
        st.header("Technical Indicators")
        rsi_period = st.slider("RSI Period", min_value=5, max_value=50, value=14)
        macd_fast = st.slider("MACD Fast Period", min_value=5, max_value=30, value=12)
        macd_slow = st.slider("MACD Slow Period", min_value=20, max_value=50, value=26)
        macd_signal = st.slider("MACD Signal Period", min_value=5, max_value=20, value=9)
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        
        # Manual refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    # Auto-refresh logic
    if auto_refresh:
        if (st.session_state.last_refresh is None or 
            (datetime.now() - st.session_state.last_refresh).seconds >= 30):
            st.session_state.last_refresh = datetime.now()
            time.sleep(1)  # Brief delay to prevent excessive requests
            st.rerun()
    
    # Main content area
    if symbol_input:
        # Format symbol with exchange suffix
        full_symbol = data_fetcher.format_symbol(symbol_input, exchange)
        
        try:
            # Fetch stock data
            with st.spinner(f"Fetching data for {full_symbol}..."):
                stock_data = data_fetcher.get_stock_data(symbol_input, exchange, period)
                stock_info = data_fetcher.get_stock_info(symbol_input, exchange)
            
            if stock_data is not None and not stock_data.empty:
                st.session_state.stock_data = stock_data
                st.session_state.current_symbol = full_symbol
                
                # Display stock information
                display_stock_info(stock_info, full_symbol)
                
                # Calculate technical indicators
                rsi_data = tech_indicators.calculate_rsi(stock_data['Close'], rsi_period)
                macd_data = tech_indicators.calculate_macd(
                    stock_data['Close'], macd_fast, macd_slow, macd_signal
                )
                
                # Create charts
                create_charts(stock_data, rsi_data, macd_data, full_symbol)
                
                # Display technical analysis summary
                display_technical_summary(stock_data, rsi_data, macd_data)
                
            else:
                st.error(f"❌ No data found for symbol: {symbol_input}")
                st.info("Please check the symbol and try again. Make sure to use the correct NSE/BSE symbol.")
        
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.info("This could be due to:")
            st.info("• Invalid stock symbol")
            st.info("• Network connectivity issues")
            st.info("• Market closure (Indian markets operate Mon-Fri, 9:15 AM - 3:30 PM IST)")
    
    else:
        # Welcome screen
        st.info("👆 Enter a stock symbol in the sidebar to begin analysis")
        
        # Popular stocks examples
        st.subheader("Popular Indian Stocks")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Large Cap**")
            st.markdown("• RELIANCE")
            st.markdown("• TCS")
            st.markdown("• INFY")
            st.markdown("• HDFCBANK")
        
        with col2:
            st.markdown("**Mid Cap**")
            st.markdown("• BAJAJFINSV")
            st.markdown("• GODREJCP")
            st.markdown("• PAGEIND")
            st.markdown("• PIDILITIND")
        
        with col3:
            st.markdown("**IT Sector**")
            st.markdown("• WIPRO")
            st.markdown("• TECHM")
            st.markdown("• HCLTECH")
            st.markdown("• LTI")

def display_stock_info(stock_info, symbol):
    """Display current stock information"""
    st.subheader(f"📊 {symbol}")
    
    if stock_info:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = stock_info.get('currentPrice', 0)
            st.metric("Current Price", f"₹{current_price:.2f}")
        
        with col2:
            prev_close = stock_info.get('previousClose', 0)
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close > 0 else 0
            st.metric("Change", f"₹{change:.2f}", f"{change_pct:.2f}%")
        
        with col3:
            volume = stock_info.get('volume', 0)
            st.metric("Volume", f"{volume:,}")
        
        with col4:
            market_cap = stock_info.get('marketCap', 0)
            if market_cap > 0:
                market_cap_cr = market_cap / 10000000  # Convert to crores
                st.metric("Market Cap", f"₹{market_cap_cr:.0f}Cr")
            else:
                st.metric("Market Cap", "N/A")

def create_charts(stock_data, rsi_data, macd_data, symbol):
    """Create interactive charts with technical indicators"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=('Price Chart', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price chart (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Price",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # RSI chart
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=rsi_data,
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ),
        row=2, col=1
    )
    
    # RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
    
    # MACD chart
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=macd_data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=macd_data['Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    # MACD histogram
    histogram_colors = ['green' if val >= 0 else 'red' for val in macd_data['Histogram']]
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=macd_data['Histogram'],
            name='Histogram',
            marker_color=histogram_colors,
            opacity=0.6
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Technical Analysis",
        xaxis_title="Date",
        height=800,
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def display_technical_summary(stock_data, rsi_data, macd_data):
    """Display technical analysis summary"""
    st.subheader("📈 Technical Analysis Summary")
    
    # Get latest values
    latest_rsi = rsi_data.iloc[-1] if not rsi_data.empty else 0
    latest_macd = macd_data['MACD'].iloc[-1] if not macd_data['MACD'].empty else 0
    latest_signal = macd_data['Signal'].iloc[-1] if not macd_data['Signal'].empty else 0
    latest_histogram = macd_data['Histogram'].iloc[-1] if not macd_data['Histogram'].empty else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**RSI Analysis**")
        if latest_rsi > 70:
            st.warning(f"🔴 RSI: {latest_rsi:.2f} - Overbought territory")
        elif latest_rsi < 30:
            st.success(f"🟢 RSI: {latest_rsi:.2f} - Oversold territory")
        else:
            st.info(f"🟡 RSI: {latest_rsi:.2f} - Neutral zone")
    
    with col2:
        st.markdown("**MACD Analysis**")
        if latest_macd > latest_signal:
            st.success(f"🟢 MACD above Signal line - Bullish")
        else:
            st.warning(f"🔴 MACD below Signal line - Bearish")
        
        if latest_histogram > 0:
            st.info(f"📈 Histogram: {latest_histogram:.4f} - Positive momentum")
        else:
            st.info(f"📉 Histogram: {latest_histogram:.4f} - Negative momentum")

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
