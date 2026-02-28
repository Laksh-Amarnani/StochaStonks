import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats

# Setting the page configuration
st.set_page_config(
    page_title = "StochaStonks - Stock Analysis and Prediction",
    page_icon = "📈",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# Custom CSS styling for Nice UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
            
    .main-header1 {
        text=align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-top: -15px;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 20px;
    }
    
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>    
""", unsafe_allow_html=True)

# Title

st.markdown('<p class="main-header">📈 StochaStonks</p>', unsafe_allow_html=True)
st.markdown('<p class="main-header1">Stochastic Process Analyzer for Financial Markets</p>', unsafe_allow_html=True)
st.markdown('### MPSTME - NMIMS | STPA Project')
st.markdown('---')

# Sidebar for user input
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Line_chart.svg/1200px-Line_chart.svg.png", width=100)
    st.markdown("## 🎯 Project Settings")

    # Stock Selection - Using yfinance for dynamic stock data
    stock_symbol = st.text_input(
        "Stock Symbol",
        value="RELIANCE.NS",
        help="Enter stock ticker (e.g., RELIANCE.NS, TCS.NS, ^NSEI for Nifty50)"
    )

    # Date Range Selection
    st.markdown("### 📅 Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Default to last 1 year

    date_range = st.date_input(
        "Select Period",
        value=(start_date, end_date),
        max_value=end_date
    )

    # Simulation Params
    st.markdown("### ⚙️ Simulation Parameters")
    num_simulations = st.slider("Number of Simulatiosns", 10, 1000, 100, 10)
    forecast_days = st.slider("Forecast Days", 1, 365, 90, 30)

    # Analysis Type
    st.markdown("### 🔍 Analysis Type")
    analysis_type = st.selectbox(
        "Choose Analysis",
        ["Random Walks", "Geometric Brownian Motion", "Monte Carlo Simulation", "All"]
    )

    # Run Analysis button
    run_analysis = st.button("🚀 Run Analysis", type="primary")

# Helper Functions
@st.cache_data
def fetch_stock_data(symbol, start, end):
    """Fetch stock data from yahoo finance."""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start, end=end)
        return df, stock.info
    except Exception as e:
        st.error(f"Error Fetching Data: {str(e)}")
        return None, None
    
def calculate_returns(prices):
    """Calculate daily returns."""
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()

def random_walk_simulation(initial_price, returns, days):
    """Simple Random Walk Simulation."""
    daily_returns = np.random.choice(returns, size=days)
    price_path = [initial_price]

    for ret in daily_returns:
        price_path.append(price_path[-1] * np.exp(ret))

    return np.array(price_path)

def geometric_brownian_motion(S0, mu, sigma, T, dt, N):
    """Geometric Brownian Motion Simulation."""
    np.random.seed(None)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N) 
    W = np.cumsum(W)*np.sqrt(dt)  # Brownian motion
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)  # Geometric Brownian Motion
    return S

def monte_carlo_simulation(S0, mu, sigma, T, dt, num_sims):
    """Monte Carlo Simulation Using Geometric Brownian Motion."""
    N = int(T / dt)
    simulations = np.zeros((num_sims, N))

    for i in range(num_sims):
        simulations[i] = geometric_brownian_motion(S0, mu, sigma, T, dt, N)

    return simulations
