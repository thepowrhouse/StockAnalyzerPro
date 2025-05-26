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
    page_title="Portfolio Manager",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = None

def main():
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .recommendation-buy {
        background: linear-gradient(135deg, #81C784 0%, #4CAF50 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .recommendation-sell {
        background: linear-gradient(135deg, #E57373 0%, #F44336 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .recommendation-hold {
        background: linear-gradient(135deg, #FFB74D 0%, #FF9800 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>💼 Smart Portfolio Manager</h1>
        <p style="font-size: 1.2rem; margin: 0;">Upload your holdings and get intelligent buy/sell recommendations with live market data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize portfolio analyzer
    portfolio_analyzer = PortfolioAnalyzer()
    
    # Main portfolio analysis
    analyze_portfolio(portfolio_analyzer)



def analyze_portfolio(portfolio_analyzer):
    """Portfolio analysis functionality"""
    
    # File upload section with enhanced styling
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📁 Upload Your Portfolio")
        st.markdown("Select your CSV file containing your stock holdings to get started with the analysis.")
        
        uploaded_file = st.file_uploader(
            "Choose your portfolio CSV file",
            type=['csv'],
            help="CSV should contain columns: Symbol, Quantity, Buy_Price, Buy_Date",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 📋 Required Format")
        st.markdown("""
        **Columns needed:**
        - **Symbol**: NSE stock symbol (e.g., RELIANCE)
        - **Quantity**: Number of shares
        - **Buy_Price**: Purchase price per share
        - **Buy_Date**: Date of purchase (YYYY-MM-DD)
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display expected format in an expandable section
    with st.expander("💡 See Sample CSV Format", expanded=False):
        sample_data = {
            'Symbol': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK'],
            'Quantity': [10, 25, 50, 15],
            'Buy_Price': [2150.00, 3200.00, 1450.00, 1650.00],
            'Buy_Date': ['2023-01-15', '2023-03-20', '2023-06-10', '2023-08-05']
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        st.info("💡 Tip: Save your data in this exact format as a CSV file for best results.")
    
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
            
            # Fetch current prices
            with st.spinner("🔄 Fetching live prices and calculating metrics..."):
                symbols = df['Symbol'].unique().tolist()
                current_prices = portfolio_analyzer.get_current_prices(symbols)
                
                # Calculate portfolio metrics
                portfolio_df = portfolio_analyzer.calculate_portfolio_metrics(df, current_prices)
                
                # Generate portfolio summary
                summary = portfolio_analyzer.generate_portfolio_summary(portfolio_df)
            
            # Display portfolio summary with enhanced styling
            st.markdown("---")
            st.markdown("### 📈 Portfolio Performance Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                delta_color = "normal"
                st.metric(
                    "💰 Total Invested", 
                    f"₹{summary['total_invested']:,.0f}",
                    help="Total amount invested across all holdings"
                )
            
            with col2:
                pnl_delta = f"₹{summary['total_pnl']:,.0f}"
                delta_color = "normal" if summary['total_pnl'] >= 0 else "inverse"
                st.metric(
                    "💎 Current Value", 
                    f"₹{summary['total_current_value']:,.0f}",
                    delta=pnl_delta,
                    help="Current market value of your portfolio"
                )
            
            with col3:
                pnl_delta = f"{summary['overall_pnl_percentage']:.1f}%"
                delta_color = "normal" if summary['overall_pnl_percentage'] >= 0 else "inverse"
                st.metric(
                    "📊 Overall Return", 
                    pnl_delta,
                    help="Total profit/loss percentage"
                )
            
            with col4:
                xirr_delta = f"{summary['portfolio_xirr']:.1f}%"
                delta_color = "normal" if summary['portfolio_xirr'] >= 0 else "inverse"
                st.metric(
                    "⚡ XIRR", 
                    xirr_delta,
                    help="Extended Internal Rate of Return (annualized)"
                )
            
            # Detailed holdings with recommendations
            st.markdown("---")
            st.markdown("### 📋 Detailed Holdings & Investment Analysis")
            
            # Get recommendations and technical indicators for each stock
            recommendations = {}
            technical_data = {}
            
            with st.spinner("🤖 Analyzing technical indicators and generating recommendations..."):
                for symbol in symbols:
                    recommendation, reason = portfolio_analyzer.get_technical_recommendations(symbol)
                    recommendations[symbol] = {'action': recommendation, 'reason': reason}
                    
                    # Get current RSI and MACD values
                    try:
                        stock_data = portfolio_analyzer.data_fetcher.get_stock_data(symbol, "NSE", "3mo")
                        if stock_data is not None and not stock_data.empty:
                            rsi = portfolio_analyzer.tech_indicators.calculate_rsi(stock_data['Close'])
                            macd_data = portfolio_analyzer.tech_indicators.calculate_macd(stock_data['Close'])
                            
                            current_rsi = rsi.iloc[-1] if not rsi.empty else 0
                            current_macd = macd_data['MACD'].iloc[-1] if not macd_data['MACD'].empty else 0
                            
                            technical_data[symbol] = {
                                'RSI': current_rsi,
                                'MACD': current_macd
                            }
                        else:
                            technical_data[symbol] = {'RSI': 0, 'MACD': 0}
                    except:
                        technical_data[symbol] = {'RSI': 0, 'MACD': 0}
            
            # Create enhanced portfolio dataframe with recommendations and technical indicators
            portfolio_display = portfolio_df.copy()
            portfolio_display['Recommendation'] = portfolio_display['Symbol'].map(
                lambda x: recommendations.get(x, {}).get('action', 'N/A')
            )
            portfolio_display['Reason'] = portfolio_display['Symbol'].map(
                lambda x: recommendations.get(x, {}).get('reason', 'N/A')
            )
            portfolio_display['RSI'] = portfolio_display['Symbol'].map(
                lambda x: technical_data.get(x, {}).get('RSI', 0)
            )
            portfolio_display['MACD'] = portfolio_display['Symbol'].map(
                lambda x: technical_data.get(x, {}).get('MACD', 0)
            )
            
            # Add styling function for recommendations
            def style_recommendation(val):
                if val == 'BUY':
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                elif val == 'SELL':
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                elif val == 'HOLD':
                    return 'background-color: #fff3cd; color: #856404; font-weight: bold'
                return ''
            
            def style_rsi(val):
                if val > 70:
                    return 'background-color: #f8d7da; color: #721c24'  # Overbought - red
                elif val < 30:
                    return 'background-color: #d4edda; color: #155724'  # Oversold - green
                return ''
            
            def style_pnl(val):
                if '+' in str(val) or (isinstance(val, (int, float)) and val > 0):
                    return 'color: #28a745; font-weight: bold'
                elif '-' in str(val) or (isinstance(val, (int, float)) and val < 0):
                    return 'color: #dc3545; font-weight: bold'
                return ''
            
            # Format display columns for better readability
            portfolio_display_formatted = portfolio_display.copy()
            portfolio_display_formatted['Buy_Price'] = portfolio_display_formatted['Buy_Price'].apply(lambda x: f"₹{x:.2f}")
            portfolio_display_formatted['Current_Price'] = portfolio_display_formatted['Current_Price'].apply(lambda x: f"₹{x:.2f}")
            portfolio_display_formatted['Invested_Amount'] = portfolio_display_formatted['Invested_Amount'].apply(lambda x: f"₹{x:,.0f}")
            portfolio_display_formatted['Current_Value'] = portfolio_display_formatted['Current_Value'].apply(lambda x: f"₹{x:,.0f}")
            portfolio_display_formatted['PnL'] = portfolio_display_formatted['PnL'].apply(lambda x: f"₹{x:,.0f}")
            portfolio_display_formatted['PnL_Percentage'] = portfolio_display_formatted['PnL_Percentage'].apply(lambda x: f"{x:.1f}%")
            portfolio_display_formatted['CAGR'] = portfolio_display_formatted['CAGR'].apply(lambda x: f"{x:.1f}%")
            portfolio_display_formatted['RSI'] = portfolio_display_formatted['RSI'].apply(lambda x: f"{x:.1f}")
            portfolio_display_formatted['MACD'] = portfolio_display_formatted['MACD'].apply(lambda x: f"{x:.3f}")
            
            # Create styled dataframe
            styled_df = portfolio_display_formatted[['Symbol', 'Quantity', 'Buy_Price', 'Current_Price', 
                                                   'Invested_Amount', 'Current_Value', 'PnL', 'PnL_Percentage', 
                                                   'CAGR', 'RSI', 'MACD', 'Recommendation', 'Reason']].style.apply(
                lambda x: [style_recommendation(x['Recommendation'])] * len(x), axis=1
            ).apply(
                lambda x: [style_rsi(float(x['RSI'])) if x['RSI'] != 'N/A' else ''] * len(x), axis=1
            ).apply(
                lambda x: [style_pnl(x['PnL_Percentage'])] * len(x), axis=1
            )
            
            # Display the enhanced portfolio table with sorting capability
            st.markdown("**💡 Tip:** Click on column headers to sort the data")
            
            # Create a sortable dataframe display
            sort_by = st.selectbox(
                "Sort by:",
                ['Symbol', 'PnL_Percentage', 'CAGR', 'Current_Value', 'RSI', 'MACD', 'Recommendation'],
                index=1,  # Default to PnL_Percentage
                key="sort_selector"
            )
            
            ascending = st.checkbox("Ascending order", value=False, key="sort_order")
            
            # Sort the dataframe
            portfolio_sorted = portfolio_display_formatted.sort_values(
                by=sort_by.replace('_', ' ') if sort_by in ['PnL_Percentage'] else sort_by,
                ascending=ascending
            )
            
            # Display the table
            st.dataframe(
                portfolio_sorted[['Symbol', 'Quantity', 'Buy_Price', 'Current_Price', 
                               'Invested_Amount', 'Current_Value', 'PnL', 'PnL_Percentage', 
                               'CAGR', 'RSI', 'MACD', 'Recommendation', 'Reason']],
                use_container_width=True,
                height=400
            )
            
            # Enhanced recommendation summary
            st.markdown("---")
            st.markdown("### 🎯 AI-Powered Investment Recommendations")
            
            buy_stocks = [symbol for symbol, rec in recommendations.items() if rec['action'] == 'BUY']
            sell_stocks = [symbol for symbol, rec in recommendations.items() if rec['action'] == 'SELL']
            hold_stocks = [symbol for symbol, rec in recommendations.items() if rec['action'] == 'HOLD']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="recommendation-buy">
                    <h4>🟢 BUY Signals ({len(buy_stocks)})</h4>
                    <p>Stocks showing bullish momentum</p>
                </div>
                """, unsafe_allow_html=True)
                
                if buy_stocks:
                    for stock in buy_stocks:
                        reason = recommendations[stock]['reason']
                        st.success(f"**{stock}** - {reason}")
                else:
                    st.info("No buy recommendations at this time")
            
            with col2:
                st.markdown(f"""
                <div class="recommendation-sell">
                    <h4>🔴 SELL Signals ({len(sell_stocks)})</h4>
                    <p>Stocks showing bearish momentum</p>
                </div>
                """, unsafe_allow_html=True)
                
                if sell_stocks:
                    for stock in sell_stocks:
                        reason = recommendations[stock]['reason']
                        st.error(f"**{stock}** - {reason}")
                else:
                    st.info("No sell recommendations at this time")
            
            with col3:
                st.markdown(f"""
                <div class="recommendation-hold">
                    <h4>🟡 HOLD Signals ({len(hold_stocks)})</h4>
                    <p>Stocks in neutral territory</p>
                </div>
                """, unsafe_allow_html=True)
                
                if hold_stocks:
                    for stock in hold_stocks:
                        reason = recommendations[stock]['reason']
                        st.warning(f"**{stock}** - {reason}")
                else:
                    st.info("No hold recommendations at this time")
            
            # Enhanced portfolio allocation chart
            st.markdown("---")
            st.markdown("### 📊 Portfolio Allocation Breakdown")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create a more visually appealing pie chart
                colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
                
                fig = go.Figure(data=[go.Pie(
                    labels=portfolio_df['Symbol'],
                    values=portfolio_df['Current_Value'],
                    hole=0.4,
                    marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
                    textinfo='label+percent',
                    textfont_size=12,
                    pull=[0.05 if i == 0 else 0 for i in range(len(portfolio_df))]  # Pull out the largest slice
                )])
                
                fig.update_layout(
                    title={
                        'text': "Current Portfolio Distribution",
                        'y': 0.95,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 16, 'color': '#333'}
                    },
                    font=dict(size=12),
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.01
                    ),
                    margin=dict(t=50, b=0, l=0, r=0),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Top Performers")
                
                # Sort by P&L percentage and show top performers
                top_performers = portfolio_df.nlargest(3, 'PnL_Percentage')[['Symbol', 'PnL_Percentage']]
                
                for idx, row in top_performers.iterrows():
                    pnl = row['PnL_Percentage']
                    symbol = row['Symbol']
                    
                    if pnl >= 0:
                        st.success(f"🚀 **{symbol}**: +{pnl:.1f}%")
                    else:
                        st.error(f"📉 **{symbol}**: {pnl:.1f}%")
                
                st.markdown("#### 💡 Quick Stats")
                profitable_stocks = len(portfolio_df[portfolio_df['PnL_Percentage'] > 0])
                total_stocks = len(portfolio_df)
                
                st.info(f"📊 {profitable_stocks}/{total_stocks} stocks in profit")
                
                avg_cagr = portfolio_df['CAGR'].mean()
                st.info(f"📈 Average CAGR: {avg_cagr:.1f}%")
            
        except Exception as e:
            st.error(f"❌ Error processing portfolio: {str(e)}")
            st.info("Please check your CSV format and try again.")

if __name__ == "__main__":
    main()
