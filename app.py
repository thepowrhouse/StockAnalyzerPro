import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
from stock_data import StockDataFetcher
from technical_indicators import TechnicalIndicators

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
    st.title("📈 Indian Stock Market Analyzer")
    st.markdown("Real-time technical analysis for NSE and BSE stocks")
    
    # Initialize data fetcher and technical indicators
    data_fetcher = StockDataFetcher()
    tech_indicators = TechnicalIndicators()
    
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

if __name__ == "__main__":
    main()
