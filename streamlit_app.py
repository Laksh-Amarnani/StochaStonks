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

# Main Analysis Section
if run_analysis:
    with st.spinner("Fetching stock data..."):
        # Fetching Stock Data As per user input from yfinance
        if len(date_range) == 2:
            start, end = date_range
            df, info = fetch_stock_data(stock_symbol, start, end)

            if df is not None and not df.empty:
                # Displaying the information of stock
                col1, col2, col3, col4, = st.columns(4)

                with col1:
                    st.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")

                with col2:
                    change = ((df['Close'].iloc[-1] - df['Close'].iloc[0])/df['Close'].iloc[0]) * 100
                    st.metric("Period Change", f"{change:.2f}%")
                
                with col3:
                    st.metric("Highest", f"₹{df['High'].max():.2f}")

                with col4:
                    st.metric("Lowest", f"₹{df['Low'].min():.2f}")

                st.mardown("---")

                # Calculating Returns and statistics
                prices = df['Close']
                returns = calculate_returns(prices)

                mu = returns.mean()
                sigma = returns.std()

                # Display Statistics of the data that is fetched.
                st.markdown('<p class="sub-header">📊 Stock Statistics</p>', unsafe_allow_html=True)

                col1, cols2 = st.columns(2)

                with col1:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**Returns Statistics**")
                    st.write(f"- Mean Daily Return (μ): {mu:.6f}")
                    st.write(f"- Daily Volatility (σ): {sigma:.6f}")
                    st.markdown(f"- Annual Return: {(mu*252):.2f}")
                    st.markdown(f'- Annual Volatility: {(sigma*np.sqrt(252)):.2f}%')
                    st.markdown('</div>', unsafe_allow_html=True)

                with cols2:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**Distribution Analysis**")
                    st.markdown(f"- Skewness: {stats.skew(returns):.4f}")
                    st.markdown(f"- Kurtosis: {stats.kurtosis(returns):.4f}")
                    st.markdown(f"- Min Return: {returns.min():.4f}")
                    st.markdown(f"- Max Return: {returns.max():.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Displaying the Price Plot of Historical Data
                st.markdown('<p class="sub-header">📈 Historical Price Movement</p>', unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df.index, df['Close'], linewidth=2, color='#1f77b4', label='Actual Price')
                ax.fill_between(df.index, df['Close'], alpha=0.3, color='#1f77b4')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Price (₹)', fontsize=12)
                ax.set_title(f"{stock_symbol} - Historical Closing Price", fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Returns Distribution
                st.markdown('<p class="sub-header">📊 Returns Distribution</p>', unsafe_allow_html=True)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Histogram of Returns
                ax1.hist(returns, bins=50, density=True, alpha=0.7, color="#2ecc71", edgecolor='black')
                mu_fit, sigma_fit = returns.mean(), returns.std()
                x = np.linspace(returns.min(), returns.max(), 100)
                ax1.plot(x, stats.norm.pdf(x, mu_fit, sigma_fit), 'r-', linewidth=2, label="Normal Distribution")
                ax1.set_xlabel('Daily Return', fontsize=11)
                ax1.set_ylabel('Frequency', fontsize=11)
                ax1.set_title('Returns Distribution', fontsize=12, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Q-Q Plot
                stats.probplot(returns, dist="norm", plot=ax2)
                ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig)

                # Stochastic Process Analysis
                st.markdown("---")
                st.markdown('<p class="sub-header">🎲 Stochastic Process Simulations</p>', unsafe_allow_html=True)

                S0 = df['Close'].iloc[-1]
                T = forecast_days / 252  # Convert days to years
                dt = 1/252  # Daily time steps
                N = forecast_days

                # Random walks Shows it's magic here
                if analysis_type in ["Random Walks", "All"]:
                    st.markdown("### 🚶 Random Walk Simulation")

                    with st.spinner("Running Random Walk Simulation..."):
                        fig, ax = plt.subplots(figsize=(12, 6))

                        for i in range(min(num_simulations, 100)):
                            path = random_walk_simulation(S0, returns, forecast_days)
                            ax.plot(path, alpha=0.3, linewidth=0.8, color='blue')

                        # Plot actual historical data
                        historical_days = len(prices)
                        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Forecast Start')
                        ax.set_xlabel('Days', fontsize=12)
                        ax.set_ylabel('Price (₹)', fontsize=12)
                        ax.set_title(f'Random Walk Simulation ({num_simulations} paths)', fontsize=14, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        st.pyplot(fig)

                        st.info("📝 **Random Walk:** Assumes future price movements are independent and equally likely to go up or down based on historical returns.")
                
                # Geometric Brownian Motion Simulation
                if analysis_type in ["Geometric Brownian Motion", "All"]:
                    st.markdown("### 📈 Geometric Brownian Motion (GBM)")

                    with st.spinner("Running Geometric Brownian Motion Simulation..."):
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        for i in range(min(num_simulations, 100)):
                            path = geometric_brownian_motion(S0, mu, sigma, T, dt, N)
                            ax.plot(path, alpha=0.3, linewidth=0.8, color='green')

                        ax.axhline(y=S0, color='red', linestyle='--', linewidth=2, label='Current Price')

                        ax.set_xlabel('Days', fontsize=12)
                        ax.set_ylabel('Price (₹)', fontsize=12)
                        ax.set_title(f'Geometric Brownian Motion ({num_simulations} paths)', fontsize=14, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                        st.info("📝 **GBM:** Assumes returns are log-normally distributed. Commonly used in Black-Scholes option pricing model.")

                # Monte Carlo Simulation
                if analysis_type in ["Monte Carlo Simulation", "All"]:
                    st.markdown("### 🎰 Monte Carlo Simulation")

                    with st.spinner("Running Monte Carlo Simulation..."):
                        simulations = monte_carlo_simulation(S0, mu, sigma, T, dt, num_simulations)

                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                        for i in range(min(num_simulations, 100)):
                            ax1.plot(simulations[i], alpha=0.3, linewidth=0.8, color='purple')
                            
                        mean_path = simulations.mean(axis=0)
                        ax1.plot(mean_path, color='red', linewidth=3, label='Mean Path')
                        ax1.axhline(y=S0, color='red', linestyle='--', linewidth=2, label='Current Price')
                        
                        ax1.set_xlabel('Days', fontsize=12)
                        ax1.set_ylabel('Price (₹)', fontsize=12)
                        ax1.set_title(f'Monte Carlo Simulation ({num_simulations} paths)', fontsize=14, fontweight='bold')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Distribution of final prices
                        final_prices = simulations[:, -1]
                        ax2.hist(final_prices, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
                        ax2.axvline(x=S0, color='orange', linestyle='--', linewidth=2, label='Current Price')
                        ax2.axvline(x=mean_path[-1], color='red', linewidth=2, label='Mean Final Price')
                        ax2.set_xlabel(f'Price after {forecast_days} days (₹)', fontsize=11)
                        ax2.set_ylabel('Probability Density', fontsize=11)
                        ax2.set_title('Distribution of Final Prices', fontsize=13, fontweight='bold')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)

                        st.pyplot(fig)

                        # Statistics 
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Mean Final Price", f"₹{final_prices.mean():.2f}")
                        with col2:
                            percentile_5 = np.percentile(final_prices, 5)
                            st.metric("5% Percentile (VaR)", f"₹{percentile_5:.2f}")
                        with col3:
                            percentile_95 = np.percentile(final_prices, 95)
                            st.metric("95% Percentile", f"₹{percentile_95:.2f}")

                        st.info("📝 **Monte Carlo:** Uses repeated random sampling to obtain numerical results. Provides probability distribution of possible outcomes.")

                    # key Insights
                    st.markdown("---")
                    st.markdown('<p class="sub-header">💡 Key Insights</p>', unsafe_allow_html=True)              

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**✅ Model Assumptions:**")
                        st.write("- Markets are continuous and frictionless.")
                        st.write("- Returns are normally distributed.")
                        st.write("- No arbitrage opportunities.")
                        st.write("- Constant volatility (simplified).")

                    with col2:
                        st.write("**⚠️ Limitations:**")
                        st.write("- Real markets have jumps and gaps.")
                        st.write("- Volatility changes over time.")
                        st.write("- Ignores market microstructure.")
                        st.write("- Past performance ≠ future results")

            else:
                st.error("❌ Unable to fetch data. Please check the stock symbol and try again.")
        else:
            st.warning("⚠️ Please select both start and end dates.")
    
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p><strong>StochaStonks - Stochastic Processes and Applications</strong></p>
    <p>MPSTME, NMIMS | Mukesh Patel School of Technology Management & Engineering</p>
    <p><em>Disclaimer: This is an academic project. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)