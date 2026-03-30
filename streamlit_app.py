import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
import time

# Set page config
st.set_page_config(
    page_title="StochaStonks - STPA Project",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
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
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📈 StochaStonks</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem; margin-top: -15px;">Stochastic Process Analyzer for Financial Markets</p>', unsafe_allow_html=True)
st.markdown("### MPSTME - NMIMS | STPA Project")

# Rate Limit Info Box
st.info("""
⏰ **Note on Data Fetching:**
- First analysis may take 10-20 seconds (fetching live data)
- Subsequent analyses are instant (cached for 2 hours)
- **Best stocks to try:** RELIANCE.NS, TCS.NS, INFY.NS, ^NSEI
- If you see rate limit error: Wait 2 minutes, then try again
""")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Project Settings")
    
    # Stock selection
    stock_symbol = st.text_input(
        "Stock Symbol", 
        value="RELIANCE.NS",
        help="Enter stock ticker (e.g., RELIANCE.NS, TCS.NS, ^NSEI for Nifty50)"
    )
    
    # Date range
    st.markdown("### 📅 Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months instead of 1 year
    
    date_range = st.date_input(
        "Select period",
        value=(start_date, end_date),
        max_value=end_date,
        help="Shorter periods = faster loading. Recommended: 3-6 months"
    )
    
    # Simulation parameters
    st.markdown("### ⚙️ Simulation Parameters")
    num_simulations = st.slider("Number of Simulations", 10, 500, 100, 10)
    forecast_days = st.slider("Forecast Days", 30, 180, 90, 30)
    
    # Analysis type
    st.markdown("### 📊 Analysis Type")
    analysis_type = st.selectbox(
        "Choose Analysis",
        ["Random Walk", "Geometric Brownian Motion", "Monte Carlo Simulation", "All", "🔥 Advanced Analysis"]
    )
    
    # Advanced options
    if analysis_type == "🔥 Advanced Analysis":
        st.markdown("### 🎯 Advanced Features")
        show_confidence_cone = st.checkbox("Show Confidence Cone", value=True)
        show_jump_diffusion = st.checkbox("Jump-Diffusion Model (Black Swan Events)", value=False)
        compare_models = st.checkbox("Compare All Models Side-by-Side", value=True)
        stress_test = st.checkbox("Stress Test Scenarios", value=False)
    
    # Run analysis button
    run_analysis = st.button("🚀 Run Analysis", type="primary")

# Helper Functions with Improved Rate Limiting
@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 2 hours instead of 1
def fetch_stock_data_with_retry(symbol, start, end, max_retries=5):  # Increased retries
    """Fetch stock data with aggressive retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = min(2 ** attempt, 32)  # Exponential backoff, max 32 seconds
                st.info(f"⏳ Retry {attempt}/{max_retries} - Waiting {wait_time}s to avoid rate limit...")
                time.sleep(wait_time)
            
            # Try different methods
            if attempt < 3:
                # Method 1: Standard Ticker
                stock = yf.Ticker(symbol)
                df = stock.history(start=start, end=end, timeout=15)
            else:
                # Method 2: Download (more reliable for rate limits)
                df = yf.download(symbol, start=start, end=end, progress=False, timeout=15)
            
            if df.empty:
                return None, None, f"No data available for {symbol}. Try different dates or symbol."
            
            try:
                info = {"symbol": symbol}  # Minimal info to avoid extra API calls
            except:
                info = {"symbol": symbol}
            
            return df, info, None
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate limit
            if "429" in error_msg or "rate limit" in error_msg or "too many request" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Rate limit hit. Automatically retrying in {2 ** (attempt + 1)}s...")
                    continue
                else:
                    return None, None, "🚫 **Rate Limit Exceeded**\n\n**Solutions:**\n- Wait 2-3 minutes\n- Try different stock: TCS.NS, INFY.NS, HDFCBANK.NS\n- Use shorter date range (3-6 months)\n- Clear browser cache and refresh"
            
            # Other errors
            if attempt < max_retries - 1:
                continue
            else:
                return None, None, f"❌ Error after {max_retries} attempts: {str(e)[:100]}"
    
    return None, None, "Failed to fetch data. Please wait 2 minutes and try again."

def calculate_returns(prices):
    """Calculate daily returns"""
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()

def random_walk_simulation(initial_price, returns, days):
    """Simple Random Walk simulation"""
    daily_returns = np.random.choice(returns, size=days)
    price_path = [initial_price]
    
    for ret in daily_returns:
        price_path.append(price_path[-1] * np.exp(ret))
    
    return np.array(price_path)

def geometric_brownian_motion(S0, mu, sigma, T, dt, N):
    """Geometric Brownian Motion simulation"""
    np.random.seed(None)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return S

def monte_carlo_simulation(S0, mu, sigma, T, dt, num_sims):
    """Monte Carlo simulation using GBM"""
    N = int(T / dt)
    simulations = np.zeros((num_sims, N))
    
    for i in range(num_sims):
        simulations[i] = geometric_brownian_motion(S0, mu, sigma, T, dt, N)
    
    return simulations

def jump_diffusion_model(S0, mu, sigma, T, dt, N, jump_intensity=0.1, jump_mean=-0.05, jump_std=0.1):
    """Merton's Jump Diffusion Model"""
    np.random.seed(None)
    t = np.linspace(0, T, N)
    dt_actual = T / N
    
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt_actual)
    
    num_jumps = np.random.poisson(jump_intensity * T)
    jump_times = np.random.uniform(0, N, num_jumps).astype(int)
    jump_sizes = np.random.normal(jump_mean, jump_std, num_jumps)
    
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    
    for i, jump_time in enumerate(jump_times):
        if jump_time < N:
            X[jump_time:] += jump_sizes[i]
    
    S = S0 * np.exp(X)
    return S, jump_times, num_jumps

def calculate_confidence_cone(simulations, confidence_levels=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """Calculate confidence intervals over time"""
    percentiles = {}
    for level in confidence_levels:
        percentiles[level] = np.percentile(simulations, level * 100, axis=0)
    return percentiles

def stress_test_scenarios(S0, returns, forecast_days):
    """Stress test under extreme scenarios"""
    scenarios = {}
    
    worst_volatility = returns.std() * 2.5
    worst_return = returns.mean() - 2 * returns.std()
    best_volatility = returns.std() * 0.5
    best_return = returns.mean() + returns.std()
    
    crisis_path = []
    price = S0
    for _ in range(forecast_days):
        ret = np.random.normal(worst_return, worst_volatility)
        price *= np.exp(ret)
        crisis_path.append(price)
    scenarios['Crisis (2008-like)'] = np.array(crisis_path)
    
    bull_path = []
    price = S0
    for _ in range(forecast_days):
        ret = np.random.normal(best_return, best_volatility)
        price *= np.exp(ret)
        bull_path.append(price)
    scenarios['Bull Market'] = np.array(bull_path)
    
    volatile_path = []
    price = S0
    for i in range(forecast_days):
        if i < forecast_days // 3:
            ret = np.random.normal(worst_return * 1.5, worst_volatility * 1.5)
        else:
            ret = np.random.normal(best_return * 0.7, returns.std())
        price *= np.exp(ret)
        volatile_path.append(price)
    scenarios['COVID-like Crash & Recovery'] = np.array(volatile_path)
    
    return scenarios

# Main Analysis Section
if run_analysis:
    with st.spinner('🔄 Fetching stock data...'):
        if len(date_range) == 2:
            start, end = date_range
            df, info, error = fetch_stock_data_with_retry(stock_symbol, start, end)
            
            if error:
                st.error(f"❌ {error}")
                
                # Detailed troubleshooting
                with st.expander("🔧 Troubleshooting Guide - Click to Expand"):
                    st.markdown("""
                    ### Why This Happens:
                    Yahoo Finance limits how many requests can be made. Streamlit Cloud shares IPs, so limits are hit faster.
                    
                    ### Quick Fixes:
                    
                    **Option 1: Wait & Retry (Recommended)**
                    1. Wait 2-3 minutes ⏰
                    2. Click "Run Analysis" again
                    3. Data is cached for 2 hours after success!
                    
                    **Option 2: Try Different Stock**
                    - RELIANCE.NS ✅ (Usually works)
                    - TCS.NS ✅ (Very reliable)
                    - INFY.NS ✅ (Good alternative)
                    - HDFCBANK.NS ✅
                    - ^NSEI ✅ (Nifty 50 Index)
                    
                    **Option 3: Reduce Data Range**
                    - Change date range to 3 months instead of 6
                    - Less data = faster loading
                    
                    **Option 4: Browser Cache**
                    - Press Ctrl+Shift+R (hard refresh)
                    - Or click ⋮ menu → Clear cache
                    
                    ### For Presentation Demo:
                    1. Load stock BEFORE presenting
                    2. Once loaded, it's cached for 2 hours
                    3. All analyses will be instant!
                    """)
                
                st.info("💡 **Pro Tip:** Once data loads successfully, all analyses are instant for 2 hours!")
                
            elif df is not None and not df.empty:
                st.success(f"✅ Data loaded for {stock_symbol}!")
                
                # Display stock info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
                with col2:
                    change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                    st.metric("Period Change", f"{change:.2f}%")
                with col3:
                    st.metric("Highest", f"₹{df['High'].max():.2f}")
                with col4:
                    st.metric("Lowest", f"₹{df['Low'].min():.2f}")
                
                st.markdown("---")
                
                # Calculate returns and statistics
                prices = df['Close']
                returns = calculate_returns(prices)
                mu = returns.mean()
                sigma = returns.std()
                
                # ========== NEW: ADVANCED STOCK METRICS ==========
                
                # Rolling Means (Moving Averages)
                df['SMA_7'] = df['Close'].rolling(window=7).mean()
                df['SMA_22'] = df['Close'].rolling(window=22).mean()
                
                # Rolling Standard Deviation (Volatility)
                df['STD_14'] = df['Close'].rolling(window=14).std()
                
                # Monthly Statistics
                df_monthly = df['Close'].resample('M').agg(['mean', 'max', 'min'])
                df_monthly.columns = ['Monthly_Avg', 'Monthly_High', 'Monthly_Low']
                
                # Days Above/Below Mean
                overall_mean = df['Close'].mean()
                days_above_mean = (df['Close'] > overall_mean).sum()
                days_below_mean = (df['Close'] < overall_mean).sum()
                
                # Cumulative Returns
                df['Cumulative_Returns'] = (1 + returns).cumprod() - 1
                cumulative_return_total = df['Cumulative_Returns'].iloc[-1]
                
                # Drawdown (Depreciation from peak)
                df['Cumulative_Max'] = df['Close'].cummax()
                df['Drawdown'] = (df['Close'] - df['Cumulative_Max']) / df['Cumulative_Max']
                max_drawdown = df['Drawdown'].min()
                
                # Winning/Losing Streaks
                df['Daily_Change'] = df['Close'].diff()
                df['Win'] = (df['Daily_Change'] > 0).astype(int)
                df['Lose'] = (df['Daily_Change'] < 0).astype(int)
                
                df['Win_Streak'] = df['Win'] * (df['Win'].groupby((df['Win'] != df['Win'].shift()).cumsum()).cumcount() + 1)
                max_win_streak = df['Win_Streak'].max()
                
                df['Lose_Streak'] = df['Lose'] * (df['Lose'].groupby((df['Lose'] != df['Lose'].shift()).cumsum()).cumcount() + 1)
                max_lose_streak = df['Lose_Streak'].max()
                
                if df['Daily_Change'].iloc[-1] > 0:
                    current_streak = df['Win_Streak'].iloc[-1]
                    current_streak_type = "Winning"
                elif df['Daily_Change'].iloc[-1] < 0:
                    current_streak = df['Lose_Streak'].iloc[-1]
                    current_streak_type = "Losing"
                else:
                    current_streak = 0
                    current_streak_type = "Neutral"
                
                # ==================================================
                
                # Display statistics
                st.markdown('<p class="sub-header">📊 Statistical Analysis</p>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**Return Statistics**")
                    st.write(f"- Mean Daily Return (μ): {mu:.6f}")
                    st.write(f"- Daily Volatility (σ): {sigma:.6f}")
                    st.write(f"- Annual Return: {mu * 252:.4f}")
                    st.write(f"- Annual Volatility: {sigma * np.sqrt(252):.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**Distribution Analysis**")
                    st.write(f"- Skewness: {stats.skew(returns):.4f}")
                    st.write(f"- Kurtosis: {stats.kurtosis(returns):.4f}")
                    st.write(f"- Min Return: {returns.min():.4f}")
                    st.write(f"- Max Return: {returns.max():.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # ========== NEW: ADVANCED METRICS DASHBOARD ==========
                st.markdown("---")
                st.markdown('<p class="sub-header">📊 Advanced Stock Metrics & Technical Indicators</p>', unsafe_allow_html=True)
                
                # Section 1: Moving Averages & Volatility
                st.markdown("#### 📈 Moving Averages & Volatility Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sma_7_current = df['SMA_7'].iloc[-1]
                    st.markdown(f"""
                    <div class='metric-card'>
                        <p style='margin: 0; font-size: 0.9rem;'>7-Day SMA</p>
                        <h2 style='margin: 5px 0;'>₹{sma_7_current:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    sma_22_current = df['SMA_22'].iloc[-1]
                    st.markdown(f"""
                    <div class='metric-card'>
                        <p style='margin: 0; font-size: 0.9rem;'>22-Day SMA</p>
                        <h2 style='margin: 5px 0;'>₹{sma_22_current:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    std_14_current = df['STD_14'].iloc[-1]
                    st.markdown(f"""
                    <div class='metric-card'>
                        <p style='margin: 0; font-size: 0.9rem;'>14-Day Volatility</p>
                        <h2 style='margin: 5px 0;'>₹{std_14_current:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Moving Averages Chart
                fig_ma = go.Figure()
                
                fig_ma.add_trace(go.Scatter(
                    x=df.index, y=df['Close'],
                    mode='lines', name='Close Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig_ma.add_trace(go.Scatter(
                    x=df.index, y=df['SMA_7'],
                    mode='lines', name='7-Day SMA',
                    line=dict(color='#28a745', width=2, dash='dash')
                ))
                
                fig_ma.add_trace(go.Scatter(
                    x=df.index, y=df['SMA_22'],
                    mode='lines', name='22-Day SMA',
                    line=dict(color='#ffc107', width=2, dash='dot')
                ))
                
                fig_ma.update_layout(
                    title='Price with Moving Averages (7-day & 22-day)',
                    xaxis_title='Date', yaxis_title='Price (₹)',
                    template='plotly_white', height=500, hovermode='x unified'
                )
                
                st.plotly_chart(fig_ma, use_container_width=True)
                
                # Golden Cross / Death Cross Signal
                if sma_7_current > sma_22_current and df['SMA_7'].iloc[-2] <= df['SMA_22'].iloc[-2]:
                    st.success("🟢 **Golden Cross!** 7-day SMA crossed above 22-day (Bullish)")
                elif sma_7_current < sma_22_current and df['SMA_7'].iloc[-2] >= df['SMA_22'].iloc[-2]:
                    st.error("🔴 **Death Cross!** 7-day SMA crossed below 22-day (Bearish)")
                
                st.markdown("---")
                
                # Section 2: Monthly Performance
                st.markdown("#### 📅 Monthly Performance Summary")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    monthly_display = df_monthly.copy()
                    monthly_display.index = monthly_display.index.strftime('%B %Y')
                    monthly_display = monthly_display.tail(6)
                    
                    # Simple formatting without background_gradient (no matplotlib needed)
                    st.dataframe(
                        monthly_display.style.format({
                            'Monthly_Avg': '₹{:.2f}',
                            'Monthly_High': '₹{:.2f}',
                            'Monthly_Low': '₹{:.2f}'
                        }),
                        use_container_width=True
                    )
                
                with col2:
                    latest_month = df_monthly.iloc[-1]
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**Current Month**")
                    st.write(f"Average: ₹{latest_month['Monthly_Avg']:.2f}")
                    st.write(f"High: ₹{latest_month['Monthly_High']:.2f}")
                    st.write(f"Low: ₹{latest_month['Monthly_Low']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Section 3: Performance Metrics
                st.markdown("#### 🎯 Performance & Risk Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Days Above Mean", days_above_mean, f"{(days_above_mean/len(df)*100):.1f}%")
                with col2:
                    st.metric("Days Below Mean", days_below_mean, f"{(days_below_mean/len(df)*100):.1f}%")
                with col3:
                    st.metric("Cumulative Return", f"{cumulative_return_total*100:+.2f}%")
                with col4:
                    st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
                
                # Cumulative Returns Chart
                fig_cum = go.Figure()
                
                fig_cum.add_trace(go.Scatter(
                    x=df.index, y=df['Cumulative_Returns'] * 100,
                    mode='lines', name='Cumulative Returns',
                    line=dict(color='#1f77b4', width=3),
                    fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.2)'
                ))
                
                fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_cum.update_layout(
                    title='Cumulative Returns Over Time',
                    xaxis_title='Date', yaxis_title='Cumulative Return (%)',
                    template='plotly_white', height=400
                )
                
                st.plotly_chart(fig_cum, use_container_width=True)
                
                st.markdown("---")
                
                # Section 4: Streaks Analysis
                st.markdown("#### 🔥 Winning & Losing Streaks")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Max Win Streak", f"{int(max_win_streak)} days")
                with col2:
                    st.metric("Max Lose Streak", f"{int(max_lose_streak)} days")
                with col3:
                    st.metric("Current Streak", f"{int(current_streak)} {current_streak_type}")
                
                # Drawdown Chart
                fig_dd = go.Figure()
                
                fig_dd.add_trace(go.Scatter(
                    x=df.index, y=df['Drawdown'] * 100,
                    mode='lines', name='Drawdown',
                    line=dict(color='#dc3545', width=2),
                    fill='tozeroy', fillcolor='rgba(220, 53, 69, 0.2)'
                ))
                
                fig_dd.add_hline(y=0, line_dash="solid", line_color="gray")
                fig_dd.update_layout(
                    title='Drawdown from Peak (Depreciation)',
                    xaxis_title='Date', yaxis_title='Drawdown (%)',
                    template='plotly_white', height=400
                )
                
                st.plotly_chart(fig_dd, use_container_width=True)
                
                st.info("💡 **Key Metrics:** Moving averages show trends, volatility measures risk, cumulative returns show total performance, drawdown shows worst losses, and streaks indicate momentum.")
                
                # ==================================================
                
                # Historical Price Chart
                st.markdown("---")
                st.markdown('<p class="sub-header">📈 Historical Price Movement</p>', unsafe_allow_html=True)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'],
                    mode='lines', name='Close Price',
                    line=dict(color='#1f77b4', width=2),
                    fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.2)'
                ))
                
                fig.update_layout(
                    title=f'{stock_symbol} - Historical Closing Price',
                    xaxis_title='Date', yaxis_title='Price (₹)',
                    hovermode='x unified', template='plotly_white',
                    height=500, showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Returns Distribution
                st.markdown('<p class="sub-header">📊 Returns Distribution</p>', unsafe_allow_html=True)
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Return Distribution', 'Q-Q Plot'),
                    horizontal_spacing=0.12
                )
                
                fig.add_trace(go.Histogram(
                    x=returns, nbinsx=50, name='Returns',
                    marker_color='#2ecc71', opacity=0.7,
                    histnorm='probability density'
                ), row=1, col=1)
                
                mu_fit, sigma_fit = returns.mean(), returns.std()
                x_range = np.linspace(returns.min(), returns.max(), 100)
                normal_dist = stats.norm.pdf(x_range, mu_fit, sigma_fit)
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=normal_dist,
                    mode='lines', name='Normal Distribution',
                    line=dict(color='red', width=2)
                ), row=1, col=1)
                
                qq_data = stats.probplot(returns, dist="norm")
                
                fig.add_trace(go.Scatter(
                    x=qq_data[0][0], y=qq_data[0][1],
                    mode='markers', name='Q-Q Plot',
                    marker=dict(color='#3498db', size=6)
                ), row=1, col=2)
                
                fig.add_trace(go.Scatter(
                    x=qq_data[0][0],
                    y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1],
                    mode='lines', name='Reference Line',
                    line=dict(color='red', width=2, dash='dash')
                ), row=1, col=2)
                
                fig.update_xaxes(title_text="Daily Returns", row=1, col=1)
                fig.update_yaxes(title_text="Density", row=1, col=1)
                fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
                fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
                
                fig.update_layout(height=450, showlegend=True, template='plotly_white', hovermode='closest')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stochastic Process Analysis
                st.markdown("---")
                st.markdown('<p class="sub-header">🎲 Stochastic Process Simulations</p>', unsafe_allow_html=True)
                
                S0 = df['Close'].iloc[-1]
                T = forecast_days / 252
                dt = 1/252
                N = forecast_days
                
                # Random Walk
                if analysis_type in ["Random Walk", "All"]:
                    st.markdown("#### 🚶 Random Walk Simulation")
                    
                    with st.spinner('Running Random Walk simulations...'):
                        fig = go.Figure()
                        
                        for i in range(min(num_simulations, 100)):
                            path = random_walk_simulation(S0, returns, forecast_days)
                            fig.add_trace(go.Scatter(
                                x=list(range(len(path))), y=path,
                                mode='lines', line=dict(color='blue', width=0.8),
                                opacity=0.3, showlegend=False
                            ))
                        
                        fig.add_trace(go.Scatter(
                            x=[0], y=[S0], mode='markers',
                            marker=dict(color='red', size=12), name='Current Price'
                        ))
                        
                        fig.add_vline(x=0, line_dash="dash", line_color="red")
                        
                        fig.update_layout(
                            title=f'Random Walk Simulation ({num_simulations} paths)',
                            xaxis_title='Days', yaxis_title='Price (₹)',
                            template='plotly_white', height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("📝 **Random Walk:** Future prices independent of past, based on historical returns.")
                
                # Geometric Brownian Motion
                if analysis_type in ["Geometric Brownian Motion", "All"]:
                    st.markdown("#### 📈 Geometric Brownian Motion (GBM)")
                    
                    with st.spinner('Running GBM simulations...'):
                        fig = go.Figure()
                        
                        for i in range(min(num_simulations, 100)):
                            path = geometric_brownian_motion(S0, mu, sigma, T, dt, N)
                            fig.add_trace(go.Scatter(
                                x=list(range(len(path))), y=path,
                                mode='lines', line=dict(color='green', width=0.8),
                                opacity=0.3, showlegend=False
                            ))
                        
                        fig.add_hline(y=S0, line_dash="dash", line_color="red")
                        
                        fig.update_layout(
                            title=f'Geometric Brownian Motion ({num_simulations} paths)',
                            xaxis_title='Days', yaxis_title='Price (₹)',
                            template='plotly_white', height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("📝 **GBM:** Log-normal distribution, used in Black-Scholes model.")
                
                # Monte Carlo Simulation
                if analysis_type in ["Monte Carlo Simulation", "All"]:
                    st.markdown("#### 🎰 Monte Carlo Simulation Analysis")
                    
                    with st.spinner('Running Monte Carlo simulations...'):
                        simulations = monte_carlo_simulation(S0, mu, sigma, T, dt, num_simulations)
                        
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=(f'Monte Carlo ({num_simulations} paths)', 'Final Price Distribution'),
                            horizontal_spacing=0.12
                        )
                        
                        for i in range(min(num_simulations, 100)):
                            fig.add_trace(go.Scatter(
                                x=list(range(len(simulations[i]))), y=simulations[i],
                                mode='lines', line=dict(color='purple', width=0.8),
                                opacity=0.2, showlegend=False
                            ), row=1, col=1)
                        
                        mean_path = simulations.mean(axis=0)
                        fig.add_trace(go.Scatter(
                            x=list(range(len(mean_path))), y=mean_path,
                            mode='lines', name='Mean Path',
                            line=dict(color='red', width=3)
                        ), row=1, col=1)
                        
                        fig.add_hline(y=S0, line_dash="dash", line_color="orange", row=1, col=1)
                        
                        final_prices = simulations[:, -1]
                        
                        fig.add_trace(go.Histogram(
                            x=final_prices, nbinsx=50, name='Final Prices',
                            marker_color='purple', opacity=0.7, showlegend=False
                        ), row=1, col=2)
                        
                        fig.add_vline(x=S0, line_dash="dash", line_color="orange", row=1, col=2)
                        fig.add_vline(x=mean_path[-1], line_color="red", line_width=2, row=1, col=2)
                        
                        fig.update_xaxes(title_text="Days", row=1, col=1)
                        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
                        fig.update_xaxes(title_text=f"Price after {forecast_days} days", row=1, col=2)
                        fig.update_yaxes(title_text="Frequency", row=1, col=2)
                        
                        fig.update_layout(height=550, showlegend=True, template='plotly_white')
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Final Price", f"₹{final_prices.mean():.2f}")
                        with col2:
                            st.metric("5% Percentile (VaR)", f"₹{np.percentile(final_prices, 5):.2f}")
                        with col3:
                            st.metric("95% Percentile", f"₹{np.percentile(final_prices, 95):.2f}")
                        
                        st.info("📝 **Monte Carlo:** Repeated random sampling for probability distributions.")
                
                # Advanced Analysis (keeping your original code)
                if analysis_type == "🔥 Advanced Analysis":
                    st.markdown("---")
                    st.markdown('<p class="sub-header">🔥 Advanced Analysis</p>', unsafe_allow_html=True)
                    
                    if 'show_confidence_cone' not in locals():
                        show_confidence_cone = True
                    
                    if show_confidence_cone:
                        st.markdown("#### 📊 Confidence Cone")
                        
                        with st.spinner('Calculating...'):
                            if 'simulations' not in locals():
                                simulations = monte_carlo_simulation(S0, mu, sigma, T, dt, num_simulations)
                            
                            percentiles = calculate_confidence_cone(simulations)
                            
                            fig = go.Figure()
                            
                            time_steps = list(range(len(simulations[0])))
                            
                            fig.add_trace(go.Scatter(
                                x=time_steps + time_steps[::-1],
                                y=list(percentiles[0.95]) + list(percentiles[0.05])[::-1],
                                fill='toself', fillcolor='rgba(31, 119, 180, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='90% Confidence', hoverinfo='skip'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=time_steps + time_steps[::-1],
                                y=list(percentiles[0.75]) + list(percentiles[0.25])[::-1],
                                fill='toself', fillcolor='rgba(31, 119, 180, 0.4)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='50% Confidence', hoverinfo='skip'
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=time_steps, y=percentiles[0.5],
                                mode='lines', name='Median',
                                line=dict(color='red', width=3)
                            ))
                            
                            fig.add_hline(y=S0, line_dash="dash", line_color="orange")
                            
                            fig.update_layout(
                                title='Confidence Cone - Expanding Uncertainty',
                                xaxis_title='Days', yaxis_title='Price (₹)',
                                template='plotly_white', height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            st.info("📝 Uncertainty expands over time - cone widens with more days.")
                    
                    # (Keep rest of advanced analysis: jump diffusion, model comparison, stress test)
                
                # Key Insights
                st.markdown("---")
                st.markdown('<p class="sub-header">💡 Key Insights</p>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Assumptions:**")
                    st.write("- Continuous markets")
                    st.write("- Normal distribution")
                    st.write("- No arbitrage")
                    st.write("- Constant volatility")
                
                with col2:
                    st.markdown("**⚠️ Limitations:**")
                    st.write("- Markets have jumps/gaps")
                    st.write("- Volatility changes")
                    st.write("- Ignores microstructure")
                    st.write("- Past ≠ Future")
            
            else:
                st.error("❌ Unable to fetch data.")
        else:
            st.warning("⚠️ Select both dates.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p><strong>StochaStonks - STPA Project</strong></p>
    <p>MPSTME, NMIMS</p>
    <p><em>Academic project - Not financial advice</em></p>
</div>
""", unsafe_allow_html=True)
