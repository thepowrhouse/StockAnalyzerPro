import os, sys
from subprocess import run, CalledProcessError

# List of core dependencies
dependencies = [
    "numpy==2.2.6",
    "pandas==2.2.3",
    "scikit-learn==1.3.2",
    "streamlit==1.45.1"
]


def install_missing_packages():
    missing = []
    for dep in dependencies:
        try:
            __import__(dep.split('==')[0])
        except ImportError:
            missing.append(dep)

    if missing:
        print(f"⛔ Missing packages: {missing}", file=sys.stderr)
        try:
            run([sys.executable, "-m", "pip", "install"] + missing, check=True)
            print("✅ Missing packages installed", file=sys.stderr)
        except CalledProcessError:
            print("💥 FAILED to install packages", file=sys.stderr)
            sys.exit(1)


#install_missing_packages() //Commented as Streamlit cloud doesnt support this

import urllib.error
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Portfolio Manager",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Check for LightGBM availability after set_page_config
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from portfolio_analyzer import PortfolioAnalyzer

warnings.filterwarnings('ignore')

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
        self.model = None
        self.support_model = None
        self.resistance_model = None
        self.model_fitted = False
        self.support_model_fitted = False
        self.resistance_model_fitted = False

        # Define numerical features for training (EXCLUDING portfolio-specific fields)
        self.numerical_features = [
            'RSI_1Y', 'RSI_5Y', 'MACD_1Y', 'MACD_5Y', 'SMA_44',
            'Momentum', 'Volatility', 'MA20_Cross_SMA44', 'Volume_Ratio'
        ]

        # Initialize LightGBM model if available
        if LIGHTGBM_AVAILABLE:
            self.model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('classifier', LGBMClassifier(
                    random_state=42,
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    num_leaves=31,
                    n_jobs=-1
                ))
            ])
            self.support_model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(
                    random_state=42,
                    n_estimators=50,
                    max_depth=5,
                    n_jobs=-1
                ))
            ])
            self.resistance_model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(
                    random_state=42,
                    n_estimators=50,
                    max_depth=5,
                    n_jobs=-1
                ))
            ])
        else:
            st.info("LightGBM unavailable. Using traditional analysis for recommendations. Install LightGBM with `pip install lightgbm`.")

        # Define numerical features for training
        self.numerical_features = [
            'RSI_1Y', 'RSI_5Y', 'MACD_1Y', 'MACD_5Y', 'SMA_44',
            'Momentum', 'Volatility', 'MA20_Cross_SMA44', 'Volume_Ratio'
        ]

    def initialize_models(self, stock_data, symbol):
        """Initialize and train the models with the provided stock data"""
        try:
            X_train, y_train = self.create_training_data(stock_data, symbol)
            if len(X_train) == 0 or len(y_train) == 0:
                st.warning(f"Not enough training data for {symbol}")
                return False
            if LIGHTGBM_AVAILABLE and self.model:
                self.model.fit(X_train, y_train)
                self.model_fitted = True
            if LIGHTGBM_AVAILABLE and self.support_model and self.resistance_model:
                X_support_resistance, y_support, y_resistance = self.create_support_resistance_data(stock_data, symbol)
                if len(X_support_resistance) > 0:
                    self.support_model.fit(X_support_resistance, y_support)
                    self.resistance_model.fit(X_support_resistance, y_resistance)
                    self.support_model_fitted = True
                    self.resistance_model_fitted = True
                    print(f"Support and resistance models trained successfully for {symbol}")
                else:
                    st.warning(f"Insufficient data to train support/resistance models for {symbol}")
                    self.support_model_fitted = False
                    self.resistance_model_fitted = False
            return True
        except Exception as e:
            st.warning(f"Error initializing models for {symbol}: {str(e)}")
            return False

    def _calculate_rsi(self, prices, timeframe):
        """
        Calculate RSI with optimized periods based on timeframe
        """
        try:
            period = 14 if timeframe == "1y" else 50 if timeframe == "5y" else 14
            delta = prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gain = gains.rolling(window=period, min_periods=1).mean()
            avg_loss = losses.rolling(window=period, min_periods=1).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
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
        Calculate MACD with optimized periods
        """
        try:
            fast_period = 12 if timeframe == "1y" else 24 if timeframe == "5y" else 12
            slow_period = 26 if timeframe == "1y" else 52 if timeframe == "5y" else 26
            signal_period = 9 if timeframe == "1y" else 18 if timeframe == "5y" else 9
            exp1 = prices.ewm(span=fast_period, adjust=False).mean()
            exp2 = prices.ewm(span=slow_period, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            macd_histogram = macd_line - signal_line
            current_macd = macd_histogram.iloc[-1]
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

    def _calculate_sma(self, prices, period=44):
        """
        Calculate Simple Moving Average
        """
        try:
            sma = prices.rolling(window=period, min_periods=1).mean()
            return float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else 0.0
        except Exception as e:
            print(f"Error in SMA calculation: {str(e)}")
            return 0.0

    def calculate_advanced_features(self, data, mode="REAL_TIME"):
        """Calculate optimized technical indicators for ML models"""
        try:
            if data.empty or len(data) < 50:
                return None

            features = {}
            numerical_features = self.numerical_features
            for feature in numerical_features:
                features[feature] = np.float64(0.0)

            close_prices = data['Close'].astype(np.float64)
            volume = data['Volume'].astype(np.float64) if 'Volume' in data.columns else pd.Series(np.zeros(len(data)), index=data.index)

            # RSI Calculations
            rsi_1y_value, rsi_1y_signals = self._calculate_rsi(close_prices, timeframe="1y")
            rsi_5y_value, rsi_5y_signals = self._calculate_rsi(close_prices, timeframe="5y")
            features['RSI_1Y'] = np.float64(rsi_1y_value)
            features['RSI_5Y'] = np.float64(rsi_5y_value)

            # MACD Calculations
            macd_1y_value, macd_1y_signals = self._calculate_macd(close_prices, timeframe="1y")
            macd_5y_value, macd_5y_signals = self._calculate_macd(close_prices, timeframe="5y")
            features['MACD_1Y'] = np.float64(macd_1y_value)
            features['MACD_5Y'] = np.float64(macd_5y_value)

            # SMA 44
            sma_44 = self._calculate_sma(close_prices, period=44)
            ma_20 = self._calculate_sma(close_prices, period=20)
            features['SMA_44'] = np.float64(sma_44)

            # Momentum
            current_price = close_prices.iloc[-1]
            price_5_days_ago = close_prices.iloc[-6] if len(close_prices) > 5 else close_prices.iloc[0]
            momentum = ((current_price - price_5_days_ago) / price_5_days_ago) * 100
            features['Momentum'] = np.float64(momentum if not pd.isna(momentum) else 0.0)

            # Volatility
            returns = close_prices.pct_change()
            volatility = returns.std() * np.sqrt(252)
            features['Volatility'] = np.float64(volatility if not pd.isna(volatility) else 0.0)

            # Moving Average Crossover
            features['MA20_Cross_SMA44'] = np.float64(1.0 if ma_20 > sma_44 else 0.0)

            # Volume Ratio
            avg_volume = volume.rolling(window=20, min_periods=1).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            features['Volume_Ratio'] = np.float64(current_volume / avg_volume if avg_volume != 0 else 1.0)

            # Collect signals
            signals = rsi_1y_signals + rsi_5y_signals + macd_1y_signals + macd_5y_signals
            features['Signals'] = signals

            return features
        except Exception as e:
            print(f"Error calculating features: {str(e)}")
            return None

    def create_training_data(self, stock_data, symbol):
        """Create optimized training data for ML models (EXCLUDES portfolio-specific fields)"""
        try:
            features_list = []
            labels_list = []
            for i in range(44, len(stock_data) - 5):
                current_data = stock_data.iloc[:i + 1]
                features = self.calculate_advanced_features(current_data)
                if not features:
                    continue

                # Only include technical features (EXCLUDE portfolio-specific fields)
                feature_values = [features.get(key, 0.0) for key in self.numerical_features]
                feature_values = [np.float64(val) if not pd.isna(val) and not np.isinf(val) else 0.0 for val in
                                  feature_values]
                if all(val == 0.0 for val in feature_values):
                    continue

                # Determine label based on price movement only
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

            X = np.array(features_list, dtype=np.float64)
            y = np.array(labels_list, dtype=np.int32)
            print(f"Training data shape for {symbol}: {X.shape}")
            return X, y
        except Exception as e:
            st.warning(f"Error creating training data for {symbol}: {str(e)}")
            return np.array([]), np.array([])

    def create_support_resistance_data(self, stock_data, symbol):
        """Create training data for support and resistance prediction"""
        try:
            features_list = []
            support_list = []
            resistance_list = []
            lookback = 20

            for i in range(lookback, len(stock_data)):
                current_data = stock_data.iloc[:i + 1]
                features = self.calculate_advanced_features(current_data)
                if not features:
                    continue

                feature_values = [features.get(key, 0.0) for key in self.numerical_features]
                feature_values = [np.float64(val) if not pd.isna(val) and not np.isinf(val) else 0.0 for val in feature_values]
                if all(val == 0.0 for val in feature_values):
                    continue

                recent_data = stock_data.iloc[i - lookback:i]
                pivot_high = recent_data['High'].max()
                pivot_low = recent_data['Low'].min()
                support_level = pivot_low
                resistance_level = pivot_high

                if pd.isna(support_level) or pd.isna(resistance_level):
                    continue

                current_price = stock_data['Close'].iloc[i]
                if support_level > current_price * 0.7 and resistance_level < current_price * 1.3:
                    features_list.append(feature_values)
                    support_list.append(support_level)
                    resistance_list.append(resistance_level)

            if not features_list:
                print(f"No valid support/resistance data for {symbol}")
                return np.array([]), np.array([]), np.array([])

            X = np.array(features_list, dtype=np.float64)
            y_support = np.array(support_list, dtype=np.float64)
            y_resistance = np.array(resistance_list, dtype=np.float64)
            print(f"Support/resistance training data shape for {symbol}: {X.shape}")
            return X, y_support, y_resistance
        except Exception as e:
            print(f"Error creating support/resistance data for {symbol}: {str(e)}")
            return np.array([]), np.array([]), np.array([])

    def train_ensemble_models(self, features_data, labels_data):
        """Train the LightGBM model"""
        if len(features_data) == 0 or len(labels_data) == 0:
            return False
        try:
            if LIGHTGBM_AVAILABLE and self.model:
                self.model.fit(features_data, labels_data)
                self.model_fitted = True
                return True
            return False
        except Exception as e:
            st.warning(f"Error training model: {str(e)}")
            return False

    def get_ai_recommendation(self, features):
        """Get AI-powered recommendation (EXCLUDES portfolio-specific fields)"""
        try:
            if not features or not all(key in features for key in self.numerical_features):
                return "HOLD", "Insufficient data for AI analysis", 0.33

            # Use only technical features for prediction
            feature_array = np.array([[features.get(key, 0.0) for key in self.numerical_features]], dtype=np.float64)

            if LIGHTGBM_AVAILABLE and self.model and self.model_fitted:
                pred = self.model.predict(feature_array)[0]
                prob = self.model.predict_proba(feature_array)[0]
                confidence = np.max(prob)
                recommendation_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
                recommendation = recommendation_map[pred]
                reason = self._generate_ai_reason(features, recommendation, confidence, prob)
                return recommendation, reason, confidence
            else:
                return self._traditional_recommendation(features)
        except Exception as e:
            return "HOLD", f"AI analysis error: {str(e)}", 0.33

    def _traditional_recommendation(self, features):
        """Fallback to traditional analysis (EXCLUDES portfolio-specific fields)"""
        reasons = []
        rsi_1y = features.get('RSI_1Y', 50)
        rsi_5y = features.get('RSI_5Y', 50)
        macd_1y = features.get('MACD_1Y', 0)
        sma_44 = features.get('SMA_44', 0)
        current_price = features.get('Market_Price', 0)  # Renamed for clarity

        if rsi_1y > 70 or rsi_5y > 70:
            reasons.append("Overbought RSI")
        elif rsi_1y < 30 or rsi_5y < 30:
            reasons.append("Oversold RSI")
        if macd_1y > 0:
            reasons.append("Bullish MACD")
        elif macd_1y < 0:
            reasons.append("Bearish MACD")
        if current_price > sma_44 * 1.05:
            reasons.append("Price above SMA 44 (bullish)")
        elif current_price < sma_44 * 0.95:
            reasons.append("Price below SMA 44 (bearish)")

        bullish_count = sum(1 for r in reasons if "bullish" in r.lower() or "oversold" in r.lower())
        bearish_count = sum(1 for r in reasons if "bearish" in r.lower() or "overbought" in r.lower())
        action = "BUY" if bullish_count >= 2 else "SELL" if bearish_count >= 2 else "HOLD"
        reason = ", ".join(reasons) if reasons else "Neutral signals"
        confidence = 0.6 if action != "HOLD" else 0.33
        return action, reason, confidence

    def _generate_ai_reason(self, features, recommendation, confidence, probabilities):
        """Generate detailed AI reasoning"""
        reasons = []
        rsi_1y = features.get('RSI_1Y', 50)
        rsi_5y = features.get('RSI_5Y', 50)
        macd_1y = features.get('MACD_1Y', 0)
        sma_44 = features.get('SMA_44', 0)
        current_price = features.get('Current_Price', 0)

        if rsi_1y > 70:
            reasons.append("Overbought conditions (RSI 1Y > 70)")
        elif rsi_1y < 30:
            reasons.append("Oversold conditions (RSI 1Y < 30)")
        if rsi_5y > 70:
            reasons.append("Overbought conditions (RSI 5Y > 70)")
        elif rsi_5y < 30:
            reasons.append("Oversold conditions (RSI 5Y < 30)")
        if macd_1y > 0:
            reasons.append("Bullish MACD momentum (1Y)")
        elif macd_1y < 0:
            reasons.append("Bearish MACD momentum (1Y)")
        if current_price > sma_44:
            reasons.append("Price above SMA 44 (bullish)")
        elif current_price < sma_44:
            reasons.append("Price below SMA 44 (bearish)")

        buy_prob, hold_prob, sell_prob = probabilities[2], probabilities[1], probabilities[0]
        reason_text = f"AI Confidence: {confidence:.1%} | "
        reason_text += ", ".join(reasons) if reasons else "Neutral signals detected"
        reason_text += f" (Buy:{buy_prob:.1%}, Hold:{hold_prob:.1%}, Sell:{sell_prob:.1%})"
        return reason_text

    def _calculate_support_resistance(self, data, lookback=20):
        """Calculate support and resistance levels using Random Forest and pivot points"""
        try:
            if data.empty or len(data) < lookback:
                return np.float64(0.0), np.float64(0.0)

            current_price = data['Close'].iloc[-1]
            features = self.calculate_advanced_features(data)
            if not features:
                print("No valid features for support/resistance prediction")
                return np.float64(current_price * 0.95), np.float64(current_price * 1.05)

            if (LIGHTGBM_AVAILABLE and self.support_model and self.resistance_model and
                    self.support_model_fitted and self.resistance_model_fitted):
                feature_array = np.array([[features.get(key, 0.0) for key in self.numerical_features]], dtype=np.float64)
                try:
                    support_level = self.support_model.predict(feature_array)[0]
                    resistance_level = self.resistance_model.predict(feature_array)[0]
                except Exception as e:
                    print(f"Error in model prediction: {str(e)}")
                    recent_data = data.tail(lookback)
                    pivot_high = recent_data['High'].max()
                    pivot_low = recent_data['Low'].min()
                    support_level = pivot_low
                    resistance_level = pivot_high
                    st.warning("AI-based support/resistance prediction failed, using pivot point method")
            else:
                recent_data = data.tail(lookback)
                pivot_high = recent_data['High'].max()
                pivot_low = recent_data['Low'].min()
                support_level = pivot_low
                resistance_level = pivot_high
                print("Using pivot point method for support/resistance")

            support_level = np.float64(support_level) if not pd.isna(support_level) else np.float64(current_price * 0.95)
            resistance_level = np.float64(resistance_level) if not pd.isna(resistance_level) else np.float64(current_price * 1.05)

            if not (support_level < current_price < resistance_level):
                support_level = np.float64(current_price * 0.95)
                resistance_level = np.float64(current_price * 1.05)

            return support_level, resistance_level
        except Exception as e:
            print(f"Error calculating support/resistance: {str(e)}")
            return np.float64(current_price * 0.95), np.float64(current_price * 1.05)

def main():
    if not LIGHTGBM_AVAILABLE:
        st.info("LightGBM unavailable. Using traditional analysis for recommendations. Install LightGBM with `pip install lightgbm`.")

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
            help="CSV should contain columns: Symbol, Quantity, Buy_Price, Buy_Date, and optionally Exchange",
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("### 📋 Required Format")
        st.markdown("""
        **Columns needed:**
        - **Symbol**: Stock symbol (e.g., RELIANCE, AAPL)
        - **Quantity**: Number of shares
        - **Buy_Price**: Purchase price per share
        - **Buy_Date**: Date of purchase (YYYY-MM-DD)
        - **Exchange** (optional): Exchange code (e.g., NSE, BSE, NASDAQ). Defaults to NSE if not provided
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("💡 See Sample CSV Format", expanded=False):
        sample_data = {
            'Symbol': ['RELIANCE', 'TCS', 'INFY', 'AAPL'],
            'Quantity': [10, 25, 50, 15],
            'Buy_Price': [2150.00, 3200.00, 1450.00, 150.00],
            'Buy_Date': ['2023-01-15', '2023-03-20', '2023-06-10', '2023-08-05'],
            'Exchange': ['NSE', 'NSE', 'BSE', 'NASDAQ']
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        st.info("💡 Tip: Save your data in this exact format as a CSV file for best results. Exchange column is optional.")
    if uploaded_file is not None:
        try:
            # Read CSV with UTF-8-SIG to handle BOM
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            # Normalize column names: strip whitespace, convert to title case
            df.columns = [col.strip().title() for col in df.columns]
            st.success("✅ CSV file uploaded successfully!")
            print("Detected columns:", df.columns.tolist())  # Debug output
            # Validate CSV format with required columns
            required_columns = ['Symbol', 'Quantity', 'Buy_Price', 'Buy_Date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.stop()
            # Add Exchange column with default 'NSE' if not present
            if 'Exchange' not in df.columns:
                df['Exchange'] = 'NSE'
                st.info("ℹ️ Exchange column not found in CSV. Defaulting to NSE for all stocks.")
            else:
                # Validate Exchange values
                valid_exchanges = ['NSE', 'BSE', 'NASDAQ']
                invalid_exchanges = df['Exchange'].dropna().apply(
                    lambda x: x if isinstance(x, str) and x.strip().upper() in valid_exchanges else None
                ).isna()
                if invalid_exchanges.any():
                    st.error(f"❌ Invalid exchange values found: {df[invalid_exchanges]['Exchange'].unique()}. Supported exchanges: {', '.join(valid_exchanges)}")
                    st.stop()
                df['Exchange'] = df['Exchange'].fillna('NSE')  # Default for NaN values
            # Pass the full DataFrame to validate_csv_format
            is_valid, result = portfolio_analyzer.validate_csv_format(df)
            if not is_valid:
                st.error(f"❌ {result}")
                st.stop()
            df = result
            with st.spinner("🔄 Fetching live prices and calculating metrics..."):
                # Group by Symbol and Exchange to handle unique combinations
                symbol_exchange_pairs = df[['Symbol', 'Exchange']].drop_duplicates().to_dict('records')
                current_prices = {}
                for pair in symbol_exchange_pairs:
                    symbol = pair['Symbol']
                    exchange = pair['Exchange']
                    try:
                        price = portfolio_analyzer.get_current_prices([symbol], exchange=exchange)
                        current_prices[symbol] = price.get(symbol, np.nan)
                    except Exception as e:
                        st.warning(f"Error fetching price for {symbol} on {exchange}: {str(e)}")
                        current_prices[symbol] = np.nan
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
                    st.warning("""
                    ⚠️ XIRR calculation failed. Common causes:
                    - Invalid date formats (use YYYY-MM-DD)
                    - Missing buy price/quantity values
                    - Portfolio with only one transaction
                    """)
            st.markdown("---")
            st.markdown("### 📋 Detailed Holdings & Investment Analysis")
            recommendations = {}
            technical_data = {}
            ai_analyzer = AIPortfolioAnalyzer()
            model_trained = False
            with st.spinner("🤖 Training AI models and analyzing technical indicators..."):
                all_training_features = []
                all_training_labels = []
                valid_symbols = []

                for _, pair in enumerate(symbol_exchange_pairs):
                    symbol = pair['Symbol']
                    exchange = pair['Exchange']
                    try:
                        stock_data = portfolio_analyzer.data_fetcher.get_stock_data(symbol, exchange, "5y")
                        stock_data_1y = portfolio_analyzer.data_fetcher.get_stock_data(symbol, exchange, "1y")
                        if stock_data is not None and not stock_data.empty:
                            try:
                                rsi_5y = portfolio_analyzer.tech_indicators.calculate_rsi(stock_data['Close'])
                                macd_data_5y = portfolio_analyzer.tech_indicators.calculate_macd(stock_data['Close'])
                                rsi_1y = portfolio_analyzer.tech_indicators.calculate_rsi(stock_data_1y['Close'])
                                macd_data_1y = portfolio_analyzer.tech_indicators.calculate_macd(stock_data_1y['Close'])
                                sma_44 = stock_data['Close'].rolling(44).mean()
                                info = portfolio_analyzer.data_fetcher.get_stock_info(symbol, exchange)
                                current_price = stock_data['Close'].iloc[-1]
                                ai_analyzer.initialize_models(stock_data, symbol)
                                support_level, resistance_level = ai_analyzer._calculate_support_resistance(stock_data)
                            except Exception as e:
                                st.warning(f"Error in fetching Stock INFO for {symbol} on {exchange}: {str(e)}")
                                support_level, resistance_level = current_price * 0.95, current_price * 1.05

                            if model_trained:
                                features = ai_analyzer.calculate_advanced_features(stock_data)
                                if features:
                                    # Use Market_Price instead of Current_Price for clarity
                                    features['Market_Price'] = current_price

                                    # Get recommendation using ONLY technical features
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
                                    features = {
                                        'RSI_1Y': rsi_1y.iloc[-1] if not rsi_1y.empty else 50,
                                        'RSI_5Y': rsi_5y.iloc[-1] if not rsi_5y.empty else 50,
                                        'MACD_1Y': macd_data_1y['MACD'].iloc[-1] if not macd_data_1y['MACD'].empty else 0,
                                        'SMA_44': current_sma44,
                                        'Current_Price': current_price
                                    }
                                    action, reason, confidence = ai_analyzer._traditional_recommendation(features)
                                    recommendations[symbol] = {'action': action, 'reason': reason}
                                except Exception as e:
                                    st.warning(f"Error in traditional analysis for {symbol} on {exchange}: {str(e)}")
                                    recommendations[symbol] = {'action': 'HOLD', 'reason': 'Analysis failed due to data issues'}
                            try:
                                technical_data[symbol] = {
                                    'RSI_5Y': rsi_5y.iloc[-1] if not rsi_5y.empty else 0,
                                    'MACD_5Y': macd_data_5y['MACD'].iloc[-1] if not macd_data_5y['MACD'].empty else 0,
                                    'RSI_1Y': rsi_1y.iloc[-1] if not rsi_1y.empty else 0,
                                    'MACD_1Y': macd_data_1y['MACD'].iloc[-1] if not macd_data_1y['MACD'].empty else 0,
                                    'SMA_44': sma_44.iloc[-1] if not sma_44.empty else 0,
                                    'PE_Ratio': info.get('trailingPE', 0),
                                    'PB_Ratio': info.get('priceToBook', 0),
                                    'Support_Level': support_level,
                                    'Resistance_Level': resistance_level
                                }
                            except Exception as e:
                                st.warning(f"Error calculating indicators for {symbol} on {exchange}: {str(e)}")
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
                        st.warning(f"Error processing data for {symbol} on {exchange}: {str(e)}")
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

                for _, pair in enumerate(symbol_exchange_pairs):
                    symbol = pair['Symbol']
                    exchange = pair['Exchange']
                    try:
                        # Format symbol with exchange suffix
                        formatted_symbol = portfolio_analyzer.format_symbol(symbol, exchange)
                        stock_data = portfolio_analyzer.data_fetcher.get_stock_data(formatted_symbol, exchange, "5y")
                        if stock_data is not None and not stock_data.empty and len(stock_data) > 50:
                            features, labels = ai_analyzer.create_training_data(stock_data, symbol)
                            if len(features) > 0:
                                all_training_features.extend(features)
                                all_training_labels.extend(labels)
                                valid_symbols.append(symbol)
                    except (ValueError, KeyError, urllib.error.URLError) as e:
                        st.warning(f"Skipping training for {symbol} on {exchange}: {str(e)}")

                if len(all_training_features) > 100 and valid_symbols:
                    model_trained = ai_analyzer.train_ensemble_models(
                        np.array(all_training_features),
                        np.array(all_training_labels)
                    )
                    if model_trained:
                        st.success("✅ AI models trained successfully with LightGBM!")
                        for symbol in valid_symbols:
                            try:
                                stock_data = portfolio_analyzer.data_fetcher.get_stock_data(symbol, df[df['Symbol'] == symbol]['Exchange'].iloc[0], "5y")
                                if stock_data is not None and not stock_data.empty:
                                    features = ai_analyzer.calculate_advanced_features(stock_data)
                                    if features:
                                        features['Current_Price'] = stock_data['Close'].iloc[-1]
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
            portfolio_display['Exchange'] = portfolio_display['Symbol'].map(
                lambda x: df[df['Symbol'] == x]['Exchange'].iloc[0]
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
                "Exchange": st.column_config.TextColumn("🌐 Exchange", width="small"),
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
            display_data['Invested_Amount'] = table_data['Invested_Amount'].apply(format_indian_currency)
            display_data['Current_Value'] = table_data['Current_Value'].apply(format_indian_currency)
            display_data['PnL'] = table_data['PnL'].apply(format_indian_currency)
            st.markdown("**💡 Click on any column header to sort the data**")
            st.data_editor(
                display_data[['Symbol', 'Exchange', 'Quantity', 'Buy_Price', 'Current_Price',
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
            if model_trained and LIGHTGBM_AVAILABLE and ai_analyzer.support_model_fitted:
                st.success("🧠 **Advanced AI Analysis Active** - Using LightGBM with RSI, MACD, and SMA 44, with AI-predicted support/resistance levels")
            else:
                st.info("📊 **Technical Analysis Mode** - Using traditional indicators with RSI, MACD, SMA 44, and pivot point support/resistance levels")
            buy_stocks = [pair for pair in symbol_exchange_pairs if recommendations.get(pair['Symbol'], {}).get('action') == 'BUY']
            sell_stocks = [pair for pair in symbol_exchange_pairs if recommendations.get(pair['Symbol'], {}).get('action') == 'SELL']
            hold_stocks = [pair for pair in symbol_exchange_pairs if recommendations.get(pair['Symbol'], {}).get('action') == 'HOLD']
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
                        symbol = stock['Symbol']
                        exchange = stock['Exchange']
                        reason = recommendations[symbol]['reason']
                        st.success(f"**{symbol} ({exchange})** - {reason}")
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
                        symbol = stock['Symbol']
                        exchange = stock['Exchange']
                        reason = recommendations[symbol]['reason']
                        st.error(f"**{symbol} ({exchange})** - {reason}")
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
                        symbol = stock['Symbol']
                        exchange = stock['Exchange']
                        reason = recommendations[symbol]['reason']
                        st.warning(f"**{symbol} ({exchange})** - {reason}")
                else:
                    st.info("No hold recommendations at this time")
        except Exception as e:
            st.error(f"❌ Error processing portfolio: {str(e)}")
            st.info("Please check your CSV format and try again.")

if __name__ == "__main__":
    main()