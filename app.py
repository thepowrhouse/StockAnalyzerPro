import sys
import urllib.error
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from pip._internal.utils import subprocess
#from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from portfolio_analyzer import PortfolioAnalyzer

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    install_package('scikit-learn')
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

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


def format_indian_currency(amount):
    """Format numbers as Indian currency with proper comma placement"""
    if pd.isna(amount):
        return "₹0"
    if amount >= 10000000:  # 1 crore and above
        return f"₹{amount / 10000000:.2f} Cr"
    elif amount >= 100000:  # 1 lakh and above
        return f"₹{amount / 100000:.2f} L"
    else:
        return f"₹{amount:,.0f}"


class AIPortfolioAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'logistic_regression': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42))
            ]),
            'gradient_boosting': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', HistGradientBoostingClassifier(random_state=42))
            ])
        }
        # Define numerical features for training
        self.numerical_features = [
            'RSI_1Y', 'RSI_5Y', 'MACD_1Y', 'MACD_5Y',
            'MA20', 'MA50', 'MA200', 'Momentum', 'Volatility',
            'MA20_Cross_MA50', 'MA50_Cross_MA200', 'Volume_Ratio'
        ]

    def initialize_models(self, stock_data, symbol):
        """Initialize and train the models with the provided stock data"""
        try:
            X_train, y_train = self.create_training_data(stock_data, symbol)
            if len(X_train) == 0 or len(y_train) == 0:
                st.warning(f"Not enough training data for {symbol}")
                return False
            imputer = SimpleImputer(strategy='median')
            X_train_imputed = imputer.fit_transform(X_train)
            for name, model in self.models.items():
                model.fit(X_train_imputed, y_train)
            return True
        except Exception as e:
            st.warning(f"Error initializing models for {symbol}: {str(e)}")
            return False

    def _calculate_rsi(self, prices, timeframe):
        """
        Calculate RSI with different periods based on timeframe
        """
        try:
            # Define periods based on timeframe
            if timeframe == "1y":
                period = 14  # Standard 14-day RSI for 1 year
            elif timeframe == "5y":
                period = 50  # Longer period for 5 year analysis
            else:
                period = 14  # Default to standard period

            # Calculate price changes
            delta = prices.diff()

            # Separate gains and losses
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)

            # Calculate rolling averages
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()

            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Get the latest RSI value
            current_rsi = rsi.iloc[-1]

            # Generate signals
            signals = []
            if not np.isnan(current_rsi):
                if current_rsi < 30:
                    signals.append(f"{timeframe}_RSI_Oversold")
                elif current_rsi > 70:
                    signals.append(f"{timeframe}_RSI_Overbought")

            return float(current_rsi) if not np.isnan(current_rsi) else 50.0, signals

        except Exception as e:
            print(f"Error in RSI calculation ({timeframe}): {str(e)}")
            return 50.0, []

    def _calculate_macd(self, prices, timeframe):
        """
        Calculate MACD with different periods based on timeframe
        """
        try:
            if timeframe == "1y":
                # Standard MACD parameters for 1 year
                fast_period = 12
                slow_period = 26
                signal_period = 9
            elif timeframe == "5y":
                # Longer periods for 5 year analysis
                fast_period = 24
                slow_period = 52
                signal_period = 18
            else:
                # Default parameters
                fast_period = 12
                slow_period = 26
                signal_period = 9

            # Calculate EMAs
            exp1 = prices.ewm(span=fast_period, adjust=False).mean()
            exp2 = prices.ewm(span=slow_period, adjust=False).mean()

            # Calculate MACD line
            macd_line = exp1 - exp2

            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

            # Calculate MACD histogram
            macd_histogram = macd_line - signal_line

            # Get the latest values
            current_macd = macd_histogram.iloc[-1]

            # Generate signals
            signals = []
            if not np.isnan(current_macd):
                if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                    signals.append(f"{timeframe}_MACD_Crossover_Bullish")
                elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
                    signals.append(f"{timeframe}_MACD_Crossover_Bearish")

            return float(current_macd) if not np.isnan(current_macd) else 0.0, signals

        except Exception as e:
            print(f"Error in MACD calculation ({timeframe}): {str(e)}")
            return 0.0, []

    def calculate_advanced_features(self, data, mode="REAL_TIME"):
        """Calculate technical indicators and features for ML models"""
        try:
            if data.empty:
                return None

            features = {}

            # Pre-initialize all features with 0.0 as float64
            numerical_features = [
                'RSI_1Y', 'RSI_5Y', 'MACD_1Y', 'MACD_5Y',
                'MA20', 'MA50', 'MA200', 'Momentum', 'Volatility',
                'MA20_Cross_MA50', 'MA50_Cross_MA200', 'Volume_Ratio'
            ]

            # Initialize all features with scalar values
            for feature in numerical_features:
                features[feature] = np.float64(0.0)

            close_prices = data['Close'].astype(np.float64)

            # RSI Calculations
            try:
                rsi_1y_value, rsi_1y_signals = self._calculate_rsi(close_prices, timeframe="1y")
                rsi_5y_value, rsi_5y_signals = self._calculate_rsi(close_prices, timeframe="5y")

                # Ensure scalar values
                features['RSI_1Y'] = np.float64(rsi_1y_value if np.isscalar(rsi_1y_value) else 0.0)
                features['RSI_5Y'] = np.float64(rsi_5y_value if np.isscalar(rsi_5y_value) else 0.0)
            except Exception as e:
                print(f"RSI calculation error: {e}")

            # MACD Calculations
            try:
                macd_1y_value, macd_1y_signals = self._calculate_macd(close_prices, timeframe="1y")
                macd_5y_value, macd_5y_signals = self._calculate_macd(close_prices, timeframe="5y")

                # Ensure scalar values
                features['MACD_1Y'] = np.float64(macd_1y_value if np.isscalar(macd_1y_value) else 0.0)
                features['MACD_5Y'] = np.float64(macd_5y_value if np.isscalar(macd_5y_value) else 0.0)
            except Exception as e:
                print(f"MACD calculation error: {e}")

            # Moving Averages
            try:
                ma20 = close_prices.rolling(window=20).mean().iloc[-1]
                ma50 = close_prices.rolling(window=50).mean().iloc[-1]
                ma200 = close_prices.rolling(window=200).mean().iloc[-1]

                features['MA20'] = np.float64(ma20 if not pd.isna(ma20) else 0.0)
                features['MA50'] = np.float64(ma50 if not pd.isna(ma50) else 0.0)
                features['MA200'] = np.float64(ma200 if not pd.isna(ma200) else 0.0)
            except Exception as e:
                print(f"Moving averages calculation error: {e}")

            # Momentum
            try:
                current_price = close_prices.iloc[-1]
                price_5_days_ago = close_prices.iloc[-6] if len(close_prices) > 5 else close_prices.iloc[0]
                momentum = ((current_price - price_5_days_ago) / price_5_days_ago) * 100
                features['Momentum'] = np.float64(momentum if not pd.isna(momentum) else 0.0)
            except Exception as e:
                print(f"Momentum calculation error: {e}")

            # Volatility
            try:
                returns = close_prices.pct_change()
                volatility = returns.std() * np.sqrt(252)
                features['Volatility'] = np.float64(volatility if not pd.isna(volatility) else 0.0)
            except Exception as e:
                print(f"Volatility calculation error: {e}")

            # Moving Average Crossovers
            try:
                features['MA20_Cross_MA50'] = np.float64(1.0 if features['MA20'] > features['MA50'] else 0.0)
                features['MA50_Cross_MA200'] = np.float64(1.0 if features['MA50'] > features['MA200'] else 0.0)
            except Exception as e:
                print(f"MA crossover calculation error: {e}")

            # Volume Ratio
            try:
                if 'Volume' in data.columns:
                    avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
                    current_volume = data['Volume'].iloc[-1]
                    features['Volume_Ratio'] = np.float64(current_volume / avg_volume if avg_volume != 0 else 1.0)
            except Exception as e:
                print(f"Volume ratio calculation error: {e}")

            # Collect signals in a separate list
            signals = []
            if 'rsi_1y_signals' in locals(): signals.extend(rsi_1y_signals)
            if 'rsi_5y_signals' in locals(): signals.extend(rsi_5y_signals)
            if 'macd_1y_signals' in locals(): signals.extend(macd_1y_signals)
            if 'macd_5y_signals' in locals(): signals.extend(macd_5y_signals)

            # Store signals separately
            features['Signals'] = signals

            # Final validation to ensure all numerical features are scalar float64
            for key in numerical_features:
                if not np.isscalar(features[key]):
                    features[key] = np.float64(0.0)
                features[key] = np.float64(features[key])

            return features

        except Exception as e:
            print(f"Error calculating features: {str(e)}")
            return None

    def create_training_data(self, stock_data, symbol):
        """Create training data for ML models based on historical patterns"""
        try:
            features_list = []
            labels_list = []
            for i in range(44, len(stock_data) - 5):
                current_data = stock_data.iloc[:i + 1]
                features = self.calculate_advanced_features(current_data, symbol)
                if not features:
                    continue

                # Select only numerical features for training
                feature_values = [features.get(key, 0.0) for key in self.numerical_features]

                # Validate that all feature values are scalars
                if any(not np.isscalar(val) for val in feature_values):
                    print(f"Non-scalar feature detected for {symbol} at index {i}: {feature_values}")
                    continue

                # Ensure all values are float64 and handle NaN/inf
                feature_values = [np.float64(val) if not pd.isna(val) and not np.isinf(val) else 0.0 for val in
                                  feature_values]

                if all(val == 0.0 for val in feature_values):  # Skip if all features are zero
                    continue

                current_price = stock_data['Close'].iloc[i]
                future_price = stock_data['Close'].iloc[i + 5]
                if pd.isna(current_price) or pd.isna(future_price):
                    continue
                price_change = (future_price - current_price) / current_price
                if price_change < -0.02:
                    label = 0  # SELL
                elif price_change > 0.02:
                    label = 2  # BUY
                else:
                    label = 1  # HOLD
                features_list.append(feature_values)
                labels_list.append(label)

            if not features_list:
                print(f"No valid features for {symbol}")
                return np.array([]), np.array([])

            # Convert to NumPy arrays
            X = np.array(features_list, dtype=np.float64)
            y = np.array(labels_list, dtype=np.int32)

            print(f"Training data shape for {symbol}: {X.shape}")
            return X, y

        except Exception as e:
            st.warning(f"Error creating training data for {symbol}: {str(e)}")
            return np.array([]), np.array([])

    def train_ensemble_models(self, features_data, labels_data):
        """Train ensemble of ML models"""
        if len(features_data) == 0 or len(labels_data) == 0:
            return False
        try:
            for name, model in self.models.items():
                model.fit(features_data, labels_data)
            return True
        except Exception as e:
            st.warning(f"Error training models: {str(e)}")
            return False

    def get_ai_recommendation(self, features):
        """Get AI-powered recommendation using ensemble of models"""
        try:
            if not features or not all(key in features for key in self.numerical_features):
                return "HOLD", "Insufficient data for AI analysis", 0.33

            # Prepare feature array with only numerical features
            feature_array = np.array([[features.get(key, 0.0) for key in self.numerical_features]], dtype=np.float64)

            predictions = {}
            probabilities = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(feature_array)[0]
                    prob = model.predict_proba(feature_array)[0]
                    predictions[name] = pred
                    probabilities[name] = prob
                except Exception as e:
                    print(f"Error in model {name} prediction: {str(e)}")
                    predictions[name] = 1  # Default to HOLD
                    probabilities[name] = [0.33, 0.34, 0.33]

            ensemble_prob = np.mean(list(probabilities.values()), axis=0)
            ensemble_pred = np.argmax(ensemble_prob)
            confidence = np.max(ensemble_prob)
            recommendation_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            recommendation = recommendation_map[ensemble_pred]
            reason = self._generate_ai_reason(features, recommendation, confidence, ensemble_prob)
            return recommendation, reason, confidence
        except Exception as e:
            return "HOLD", f"AI analysis error: {str(e)}", 0.33

    def _generate_ai_reason(self, features, recommendation, confidence, probabilities):
        """Generate detailed AI reasoning"""
        reasons = []
        rsi_1y = features.get('RSI_1Y', 50)
        rsi_5y = features.get('RSI_5Y', 50)
        macd_1y = features.get('MACD_1Y', 0)
        macd_5y = features.get('MACD_5Y', 0)
        ma20 = features.get('MA20', 0)
        ma50 = features.get('MA50', 0)

        # RSI analysis
        if rsi_1y > 70:
            reasons.append("Overbought conditions (RSI 1Y > 70)")
        elif rsi_1y < 30:
            reasons.append("Oversold conditions (RSI 1Y < 30)")
        if rsi_5y > 70:
            reasons.append("Overbought conditions (RSI 5Y > 70)")
        elif rsi_5y < 30:
            reasons.append("Oversold conditions (RSI 5Y < 30)")

        # MACD analysis
        if macd_1y > 0:
            reasons.append("Bullish MACD momentum (1Y)")
        elif macd_1y < 0:
            reasons.append("Bearish MACD momentum (1Y)")
        if macd_5y > 0:
            reasons.append("Bullish MACD momentum (5Y)")
        elif macd_5y < 0:
            reasons.append("Bearish MACD momentum (5Y)")

        # Moving Average analysis
        if ma20 > ma50:
            reasons.append("Price above MA20/MA50 (bullish)")
        elif ma20 < ma50:
            reasons.append("Price below MA20/MA50 (bearish)")

        buy_prob, hold_prob, sell_prob = probabilities[2], probabilities[1], probabilities[0]
        reason_text = f"AI Confidence: {confidence:.1%} | "
        if reasons:
            reason_text += ", ".join(reasons[:4])
        else:
            reason_text += "Neutral signals detected"
        reason_text += f" (Buy:{buy_prob:.1%}, Hold:{hold_prob:.1%}, Sell:{sell_prob:.1%})"
        return reason_text

    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def _calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent

    def _calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    def _calculate_support_resistance(self, data, lookback=20):
        """Calculate the next support and resistance levels based on recent highs and lows"""
        try:
            if data.empty or len(data) < lookback:
                return 0.0, 0.0

            recent_data = data.tail(lookback)
            current_price = data['Close'].iloc[-1]

            # Calculate support (lowest low) and resistance (highest high)
            support_level = recent_data['Low'].min()
            resistance_level = recent_data['High'].max()

            # Validate levels are within 20% of current price to ensure relevance
            if not pd.isna(support_level) and support_level > current_price * 0.8:
                support_level = np.float64(support_level)
            else:
                support_level = np.float64(current_price)

            if not pd.isna(resistance_level) and resistance_level < current_price * 1.2:
                resistance_level = np.float64(resistance_level)
            else:
                resistance_level = np.float64(current_price)

            return support_level, resistance_level

        except Exception as e:
            print(f"Error calculating support/resistance: {str(e)}")
            return np.float64(0.0), np.float64(0.0)


def main():
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
    st.markdown("""
    <div class="main-header">
        <h1>💼 Smart Portfolio Manager</h1>
        <p style="font-size: 1.2rem; margin: 0;">Upload your holdings and get intelligent buy/sell recommendations with live market data</p>
    </div>
    """, unsafe_allow_html=True)
    portfolio_analyzer = PortfolioAnalyzer()
    analyze_portfolio(portfolio_analyzer)


def analyze_portfolio(portfolio_analyzer):
    """Portfolio analysis functionality"""
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
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV file uploaded successfully!")
            is_valid, result = portfolio_analyzer.validate_csv_format(df)
            if not is_valid:
                st.error(f"❌ {result}")
                st.stop()
            df = result
            with st.spinner("🔄 Fetching live prices and calculating metrics..."):
                symbols = df['Symbol'].unique().tolist()
                current_prices = portfolio_analyzer.get_current_prices(symbols)
                portfolio_df = portfolio_analyzer.calculate_portfolio_metrics(df, current_prices)
                summary = portfolio_analyzer.generate_portfolio_summary(portfolio_df)
            st.markdown("---")
            st.markdown("### 📈 Portfolio Performance Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "💰 Total Invested",
                    format_indian_currency(summary['total_invested']),
                    help="Total amount invested across all holdings"
                )
            with col2:
                pnl_delta = format_indian_currency(summary['total_pnl'])
                delta_color = "normal" if summary['total_pnl'] >= 0 else "inverse"
                st.metric(
                    "💎 Current Value",
                    format_indian_currency(summary['total_current_value']),
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
                xirr_value = summary.get('portfolio_xirr', None)
                xirr_display = "N/A" if xirr_value is None or pd.isna(xirr_value) else f"{xirr_value:.1f}%"
                delta_color = "normal" if xirr_value and xirr_value >= 0 else "inverse"
                st.metric(
                    "⚡ XIRR",
                    xirr_display,
                    help="Extended Internal Rate of Return (annualized)" if xirr_display != "N/A" else "XIRR calculation failed due to insufficient data or invalid cash flows"
                )
                if xirr_display == "N/A":
                    st.warning(
                        "⚠️ XIRR calculation failed. Ensure buy dates and prices are valid and include sufficient transaction history.")
            st.markdown("---")
            st.markdown("### 📋 Detailed Holdings & Investment Analysis")
            recommendations = {}
            technical_data = {}
            ai_analyzer = AIPortfolioAnalyzer()
            # Initialize model_trained to False before the loop
            model_trained = False
            with st.spinner("🤖 Training AI models and analyzing technical indicators..."):
                all_training_features = []
                all_training_labels = []
                valid_symbols = []

                for symbol in symbols:
                    # Fetch Stock info for processing
                    try:
                        stock_data = portfolio_analyzer.data_fetcher.get_stock_data(symbol, "NSE", "5y")
                        stock_data_1y = portfolio_analyzer.data_fetcher.get_stock_data(symbol, "NSE", "1y")
                        if stock_data is not None and not stock_data.empty:
                            try:
                                rsi_5y = portfolio_analyzer.tech_indicators.calculate_rsi(stock_data['Close'])
                                macd_data_5y = portfolio_analyzer.tech_indicators.calculate_macd(stock_data['Close'])
                                rsi_1y = portfolio_analyzer.tech_indicators.calculate_rsi(stock_data_1y['Close'])
                                macd_data_1y = portfolio_analyzer.tech_indicators.calculate_macd(stock_data_1y['Close'])
                                sma_44 = stock_data['Close'].rolling(44).mean()
                                info = portfolio_analyzer.data_fetcher.get_stock_info(symbol, "NSE")
                                support_level, resistance_level = ai_analyzer._calculate_support_resistance(stock_data)
                            except Exception as e:
                                st.warning(f"Error in fetching Stock INFO {symbol}: {str(e)}")
                                support_level, resistance_level = 0.0, 0.0

                            if model_trained:
                                features = ai_analyzer.calculate_advanced_features(stock_data)
                                if features:
                                    ai_recommendation, ai_reason, confidence = ai_analyzer.get_ai_recommendation(features)
                                    recommendations[symbol] = {'action': ai_recommendation, 'reason': ai_reason}
                                else:
                                    recommendations[symbol] = {'action': 'HOLD', 'reason': 'Insufficient data for AI analysis'}
                            else:
                                try:
                                    pe_ratio = info.get('trailingPE', np.nan)
                                    pb_ratio = info.get('priceToBook', np.nan)
                                    current_price = stock_data['Close'].iloc[-1]
                                    current_sma44 = sma_44.iloc[-1] if not sma_44.empty else current_price
                                    action = "HOLD"
                                    reasons = []
                                    if current_price > current_sma44 * 1.05:
                                        reasons.append("Price above 44 SMA (bullish)")
                                    elif current_price < current_sma44 * 0.95:
                                        reasons.append("Price below 44 SMA (bearish)")
                                    if not pd.isna(pe_ratio):
                                        if pe_ratio < 15:
                                            reasons.append("Low P/E ratio (undervalued)")
                                        elif pe_ratio > 30:
                                            reasons.append("High P/E ratio (overvalued)")
                                    if not pd.isna(pb_ratio):
                                        if pb_ratio < 1:
                                            reasons.append("Low P/B ratio (undervalued)")
                                        elif pb_ratio > 3:
                                            reasons.append("High P/B ratio (overvalued)")
                                    if rsi_5y.iloc[-1] > 70:
                                        reasons.append("Overbought (RSI 5Y > 70)")
                                    elif rsi_5y.iloc[-1] < 30:
                                        reasons.append("Oversold (RSI 5Y < 30)")
                                    if rsi_1y.iloc[-1] > 70:
                                        reasons.append("Overbought (RSI 1Y > 70)")
                                    elif rsi_1y.iloc[-1] < 30:
                                        reasons.append("Oversold (RSI 1Y < 30)")
                                    if macd_data_5y['MACD'].iloc[-1] > macd_data_5y['Signal'].iloc[-1]:
                                        reasons.append("Bullish MACD crossover (5Y)")
                                    elif macd_data_5y['MACD'].iloc[-1] < macd_data_5y['Signal'].iloc[-1]:
                                        reasons.append("Bearish MACD crossover (5Y)")
                                    if macd_data_1y['MACD'].iloc[-1] > macd_data_1y['Signal'].iloc[-1]:
                                        reasons.append("Bullish MACD crossover (1Y)")
                                    elif macd_data_1y['MACD'].iloc[-1] < macd_data_1y['Signal'].iloc[-1]:
                                        reasons.append("Bearish MACD crossover (1Y)")
                                    if sum(1 for r in reasons if "bullish" in r.lower() or "undervalued" in r.lower()) >= 3:
                                        action = "BUY"
                                    elif sum(1 for r in reasons if "bearish" in r.lower() or "overvalued" in r.lower()) >= 3:
                                        action = "SELL"
                                    reason = ", ".join(reasons) if reasons else "Neutral signals"
                                    recommendations[symbol] = {'action': action, 'reason': reason}
                                except Exception as e:
                                    st.warning(f"Error in traditional analysis for {symbol}: {str(e)}")
                                    recommendations[symbol] = {'action': 'HOLD', 'reason': 'Analysis failed due to data issues'}
                            try:
                                current_rsi_5y = rsi_5y.iloc[-1] if not rsi_5y.empty else 0
                                current_macd_5y = macd_data_5y['MACD'].iloc[-1] if not macd_data_5y['MACD'].empty else 0
                                current_rsi_1y = rsi_1y.iloc[-1] if not rsi_1y.empty else 0
                                current_macd_1y = macd_data_1y['MACD'].iloc[-1] if not macd_data_1y['MACD'].empty else 0
                                current_sma44 = sma_44.iloc[-1] if not sma_44.empty else 0
                                pe_ratio = info.get('trailingPE', 0)
                                pb_ratio = info.get('priceToBook', 0)
                                technical_data[symbol] = {
                                    'RSI_5Y': current_rsi_5y,
                                    'MACD_5Y': current_macd_5y,
                                    'RSI_1Y': current_rsi_1y,
                                    'MACD_1Y': current_macd_1y,
                                    'SMA_44': current_sma44,
                                    'PE_Ratio': pe_ratio,
                                    'PB_Ratio': pb_ratio,
                                    'Support_Level': support_level,
                                    'Resistance_Level': resistance_level
                                }
                            except Exception as e:
                                st.warning(f"Error calculating indicators for {symbol}: {str(e)}")
                                technical_data[symbol] = {
                                    'RSI_5Y': 0,
                                    'MACD_5Y': 0,
                                    'RSI_1Y': 0,
                                    'MACD_1Y': 0,
                                    'SMA_44': 0,
                                    'PE_Ratio': 0,
                                    'PB_Ratio': 0,
                                    'Support_Level': 0.0,
                                    'Resistance_Level': 0.0
                                }
                        else:
                            recommendations[symbol] = {'action': 'HOLD', 'reason': 'Insufficient 5-year data for analysis'}
                            technical_data[symbol] = {
                                'RSI_5Y': 0,
                                'MACD_5Y': 0,
                                'RSI_1Y': 0,
                                'MACD_1Y': 0,
                                'SMA_44': 0,
                                'PE_Ratio': 0,
                                'PB_Ratio': 0,
                                'Support_Level': 0.0,
                                'Resistance_Level': 0.0
                            }
                    except (ValueError, KeyError, urllib.error.URLError) as e:
                        st.warning(f"Error processing data for {symbol}: {str(e)}")
                        recommendations[symbol] = {'action': 'HOLD', 'reason': f'Error fetching data: {str(e)}'}
                        technical_data[symbol] = {
                            'RSI_5Y': 0,
                            'MACD_5Y': 0,
                            'RSI_1Y': 0,
                            'MACD_1Y': 0,
                            'SMA_44': 0,
                            'PE_Ratio': 0,
                            'PB_Ratio': 0,
                            'Support_Level': 0.0,
                            'Resistance_Level': 0.0
                        }

                # Train AI models after collecting data for all symbols
                for symbol in symbols:
                    try:
                        stock_data = portfolio_analyzer.data_fetcher.get_stock_data(symbol, "NSE", "5y")
                        stock_data_1y = portfolio_analyzer.data_fetcher.get_stock_data(symbol, "NSE", "1y")
                        if stock_data is not None and not stock_data.empty and len(stock_data) > 50:
                            features, labels = ai_analyzer.create_training_data(stock_data, symbol)
                            if len(features) > 0:
                                all_training_features.extend(features)
                                all_training_labels.extend(labels)
                                valid_symbols.append(symbol)
                    except (ValueError, KeyError, urllib.error.URLError) as e:
                        st.warning(f"Skipping training for {symbol}: {str(e)}")

                # Train the models only if sufficient data is available
                if len(all_training_features) > 100 and valid_symbols:
                    model_trained = ai_analyzer.train_ensemble_models(
                        np.array(all_training_features),
                        np.array(all_training_labels)
                    )
                    if model_trained:
                        st.success("✅ AI models trained successfully with ensemble learning!")
                        # Recompute AI recommendations for valid symbols
                        for symbol in valid_symbols:
                            try:
                                stock_data = portfolio_analyzer.data_fetcher.get_stock_data(symbol, "NSE", "5y")
                                if stock_data is not None and not stock_data.empty:
                                    features = ai_analyzer.calculate_advanced_features(stock_data)
                                    if features:
                                        ai_recommendation, ai_reason, confidence = ai_analyzer.get_ai_recommendation(features)
                                        recommendations[symbol] = {'action': ai_recommendation, 'reason': ai_reason}
                                    else:
                                        recommendations[symbol] = {'action': 'HOLD', 'reason': 'Insufficient data for AI analysis'}
                            except (ValueError, KeyError, urllib.error.URLError) as e:
                                st.warning(f"Error recomputing AI recommendation for {symbol}: {str(e)}")
                                recommendations[symbol] = {'action': 'HOLD', 'reason': f'Error in AI analysis: {str(e)}'}
                    else:
                        st.warning("⚠️ AI model training failed, using traditional analysis")
                else:
                    st.info("ℹ️ Insufficient data for AI training, using traditional technical analysis")

            portfolio_display = portfolio_df.copy()
            portfolio_display['Recommendation'] = portfolio_display['Symbol'].map(
                lambda x: recommendations.get(x, {}).get('action', 'N/A')
            )
            portfolio_display['Reason'] = portfolio_display['Symbol'].map(
                lambda x: recommendations.get(x, {}).get('reason', 'N/A')
            )
            portfolio_display['RSI_1Y'] = portfolio_display['Symbol'].map(
                lambda x: technical_data.get(x, {}).get('RSI_1Y', 0)
            )
            portfolio_display['MACD_1Y'] = portfolio_display['Symbol'].map(
                lambda x: technical_data.get(x, {}).get('MACD_1Y', 0)
            )
            portfolio_display['SMA_44'] = portfolio_display['Symbol'].map(
                lambda x: technical_data.get(x, {}).get('SMA_44', 0)
            )
            portfolio_display['PE_Ratio'] = portfolio_display['Symbol'].map(
                lambda x: technical_data.get(x, {}).get('PE_Ratio', 0)
            )
            portfolio_display['PB_Ratio'] = portfolio_display['Symbol'].map(
                lambda x: technical_data.get(x, {}).get('PB_Ratio', 0)
            )
            portfolio_display['Support_Level'] = portfolio_display['Symbol'].map(
                lambda x: technical_data.get(x, {}).get('Support_Level', 0)
            )
            portfolio_display['Resistance_Level'] = portfolio_display['Symbol'].map(
                lambda x: technical_data.get(x, {}).get('Resistance_Level', 0)
            )

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
                    return 'background-color: #f8d7da; color: #721c24'
                elif val < 30:
                    return 'background-color: #d4edda; color: #155724'
                return ''

            def style_pnl(val):
                if isinstance(val, str) and '+' in val or (isinstance(val, (int, float)) and val > 0):
                    return 'color: #28a745; font-weight: bold'
                elif isinstance(val, str) and '-' in val or (isinstance(val, (int, float)) and val < 0):
                    return 'color: #dc3545; font-weight: bold'
                return ''

            table_data = portfolio_display.copy()
            table_data['Buy_Price'] = table_data['Buy_Price'].round(2)
            table_data['Current_Price'] = table_data['Current_Price'].round(2)
            table_data['Invested_Amount'] = table_data['Invested_Amount'].round(0)
            table_data['Current_Value'] = table_data['Current_Value'].round(0)
            table_data['PnL'] = table_data['PnL'].round(0)
            table_data['PnL_Percentage'] = table_data['PnL_Percentage'].round(1)
            table_data['RSI_1Y'] = table_data['RSI_1Y'].round(1)
            table_data['MACD_1Y'] = table_data['MACD_1Y'].round(3)
            table_data['SMA_44'] = table_data['SMA_44'].round(2)
            table_data['PE_Ratio'] = table_data['PE_Ratio'].round(2)
            table_data['PB_Ratio'] = table_data['PB_Ratio'].round(2)
            table_data['Support_Level'] = table_data['Support_Level'].round(2)
            table_data['Resistance_Level'] = table_data['Resistance_Level'].round(2)
            table_sorted = table_data.sort_values(by='PnL_Percentage', ascending=False)
            column_config = {
                "Symbol": st.column_config.TextColumn("🏢 Symbol", width="small"),
                "Quantity": st.column_config.NumberColumn("📦 Qty", width="small"),
                "Buy_Price": st.column_config.NumberColumn("💰 Buy Price", format="₹%.2f", width="small"),
                "Current_Price": st.column_config.NumberColumn("💎 Current Price", format="₹%.2f", width="small"),
                "Invested_Amount": st.column_config.TextColumn("💸 Invested", width="small"),
                "Current_Value": st.column_config.TextColumn("💰 Current Value", width="small"),
                "PnL": st.column_config.TextColumn("📈 P&L", width="small"),
                "PnL_Percentage": st.column_config.NumberColumn("📊 P&L %", format="%.1f%%", width="small"),
                "RSI_1Y": st.column_config.NumberColumn("🎯 RSI (1Y)", format="%.1f", width="small"),
                "MACD_1Y": st.column_config.NumberColumn("📉 MACD (1Y)", format="%.3f", width="small"),
                "SMA_44": st.column_config.NumberColumn("📈 44 SMA", format="₹%.2f", width="small"),
                "PE_Ratio": st.column_config.NumberColumn("📊 P/E Ratio", format="%.2f", width="small"),
                "PB_Ratio": st.column_config.NumberColumn("📊 P/B Ratio", format="%.2f", width="small"),
                "Support_Level": st.column_config.NumberColumn("📉 Support", format="₹%.2f", width="small"),
                "Resistance_Level": st.column_config.NumberColumn("📈 Resistance", format="₹%.2f", width="small"),
                "Recommendation": st.column_config.TextColumn("🤖 Signal", width="small"),
                "Reason": st.column_config.TextColumn("💡 Analysis", width="large")
            }
            display_data = table_sorted.copy()
            display_data['Invested_Amount'] = display_data['Invested_Amount'].apply(format_indian_currency)
            display_data['Current_Value'] = display_data['Current_Value'].apply(format_indian_currency)
            display_data['PnL'] = display_data['PnL'].apply(format_indian_currency)
            st.markdown("**💡 Click on any column header to sort the data**")
            st.data_editor(
                display_data[['Symbol', 'Quantity', 'Buy_Price', 'Current_Price',
                              'Invested_Amount', 'Current_Value', 'PnL', 'PnL_Percentage',
                              'RSI_1Y', 'MACD_1Y', 'SMA_44', 'PE_Ratio', 'PB_Ratio',
                              'Support_Level', 'Resistance_Level', 'Recommendation', 'Reason']],
                column_config=column_config,
                use_container_width=True,
                height=400,
                disabled=True,
                hide_index=True
            )
            st.markdown("---")
            st.markdown("### 🎯 AI-Powered Investment Recommendations")
            if model_trained:
                st.success(
                    "🧠 **Advanced AI Analysis Active** - Using ensemble machine learning models trained on 5-year data with RSI, MACD, and moving averages")
            else:
                st.info(
                    "📊 **Technical Analysis Mode** - Using traditional indicators with 5-year and 1-year RSI, MACD, SMA, P/E, P/B, and support/resistance levels")
            buy_stocks = [symbol for symbol in symbols if recommendations.get(symbol, {}).get('action') == 'BUY']
            sell_stocks = [symbol for symbol in symbols if recommendations.get(symbol, {}).get('action') == 'SELL']
            hold_stocks = [symbol for symbol in symbols if recommendations.get(symbol, {}).get('action') == 'HOLD']
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
        except Exception as e:
            st.error(f"❌ Error processing portfolio: {str(e)}")
            st.info("Please check your CSV format and try again.")


if __name__ == "__main__":
    main()