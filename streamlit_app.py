import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats

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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📈 StochaStonks</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.2rem; margin-top: -15px;">Stochastic Process Analyzer for Financial Markets</p>', unsafe_allow_html=True)
st.markdown("### MPSTME - NMIMS | STPA Project")
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
    start_date = end_date - timedelta(days=365)
    
    date_range = st.date_input(
        "Select period",
        value=(start_date, end_date),
        max_value=end_date
    )
    
    # Simulation parameters
    st.markdown("### ⚙️ Simulation Parameters")
    num_simulations = st.slider("Number of Simulations", 10, 1000, 100, 10)
    forecast_days = st.slider("Forecast Days", 30, 365, 90, 30)
    
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

# Helper Functions
@st.cache_data
def fetch_stock_data(symbol, start, end):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start, end=end)
        return df, stock.info
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None

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
    """
    Merton's Jump Diffusion Model - captures sudden market crashes/rallies
    Models both continuous movements AND discrete jumps (Black Swan events)
    
    dS = μS dt + σS dW + S dJ
    where dJ represents jump process (Poisson)
    """
    np.random.seed(None)
    t = np.linspace(0, T, N)
    dt_actual = T / N
    
    # Regular GBM component
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt_actual)
    
    # Jump component (Poisson process)
    num_jumps = np.random.poisson(jump_intensity * T)
    jump_times = np.random.uniform(0, N, num_jumps).astype(int)
    jump_sizes = np.random.normal(jump_mean, jump_std, num_jumps)
    
    # Construct price path
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    
    # Add jumps
    for i, jump_time in enumerate(jump_times):
        if jump_time < N:
            X[jump_time:] += jump_sizes[i]
    
    S = S0 * np.exp(X)
    return S, jump_times, num_jumps

def calculate_confidence_cone(simulations, confidence_levels=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """
    Calculate confidence intervals over time
    Shows the 'cone of uncertainty' expanding into the future
    """
    percentiles = {}
    for level in confidence_levels:
        percentiles[level] = np.percentile(simulations, level * 100, axis=0)
    return percentiles

def stress_test_scenarios(S0, returns, forecast_days):
    """
    Stress test under extreme scenarios
    - 2008 Financial Crisis scenario
    - COVID-19 Crash scenario
    - Bull Market scenario
    """
    scenarios = {}
    
    # Historical worst period volatility
    worst_volatility = returns.std() * 2.5
    worst_return = returns.mean() - 2 * returns.std()
    
    # Best period
    best_volatility = returns.std() * 0.5
    best_return = returns.mean() + returns.std()
    
    # Crisis scenario (like 2008)
    crisis_path = []
    price = S0
    for _ in range(forecast_days):
        ret = np.random.normal(worst_return, worst_volatility)
        price *= np.exp(ret)
        crisis_path.append(price)
    scenarios['Crisis (2008-like)'] = np.array(crisis_path)
    
    # Bull market scenario
    bull_path = []
    price = S0
    for _ in range(forecast_days):
        ret = np.random.normal(best_return, best_volatility)
        price *= np.exp(ret)
        bull_path.append(price)
    scenarios['Bull Market'] = np.array(bull_path)
    
    # Volatile scenario (like COVID)
    volatile_path = []
    price = S0
    for i in range(forecast_days):
        if i < forecast_days // 3:  # Crash phase
            ret = np.random.normal(worst_return * 1.5, worst_volatility * 1.5)
        else:  # Recovery phase
            ret = np.random.normal(best_return * 0.7, returns.std())
        price *= np.exp(ret)
        volatile_path.append(price)
    scenarios['COVID-like Crash & Recovery'] = np.array(volatile_path)
    
    return scenarios

# Main Analysis Section
if run_analysis:
    with st.spinner('Fetching stock data...'):
        # Fetch data
        if len(date_range) == 2:
            start, end = date_range
            df, info = fetch_stock_data(stock_symbol, start, end)
            
            if df is not None and not df.empty:
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
                
                # Historical Price Chart
                st.markdown('<p class="sub-header">📈 Historical Price Movement</p>', unsafe_allow_html=True)
                
                fig = go.Figure()
                
                # Add price line
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#1f77b4', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.2)'
                ))
                
                fig.update_layout(
                    title=f'{stock_symbol} - Historical Closing Price',
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    showlegend=True,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Returns Distribution
                st.markdown('<p class="sub-header">📊 Returns Distribution</p>', unsafe_allow_html=True)
                
                # Create subplots
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Return Distribution', 'Q-Q Plot'),
                    horizontal_spacing=0.12
                )
                
                # Histogram with normal distribution overlay
                fig.add_trace(
                    go.Histogram(
                        x=returns,
                        nbinsx=50,
                        name='Returns',
                        marker_color='#2ecc71',
                        opacity=0.7,
                        histnorm='probability density'
                    ),
                    row=1, col=1
                )
                
                # Normal distribution curve
                mu_fit, sigma_fit = returns.mean(), returns.std()
                x_range = np.linspace(returns.min(), returns.max(), 100)
                normal_dist = stats.norm.pdf(x_range, mu_fit, sigma_fit)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=normal_dist,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=1
                )
                
                # Q-Q Plot
                qq_data = stats.probplot(returns, dist="norm")
                
                fig.add_trace(
                    go.Scatter(
                        x=qq_data[0][0],
                        y=qq_data[0][1],
                        mode='markers',
                        name='Q-Q Plot',
                        marker=dict(color='#3498db', size=6)
                    ),
                    row=1, col=2
                )
                
                # Add reference line for Q-Q plot
                fig.add_trace(
                    go.Scatter(
                        x=qq_data[0][0],
                        y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1],
                        mode='lines',
                        name='Reference Line',
                        line=dict(color='red', width=2, dash='dash')
                    ),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Daily Returns", row=1, col=1)
                fig.update_yaxes(title_text="Density", row=1, col=1)
                fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
                fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
                
                fig.update_layout(
                    height=450,
                    showlegend=True,
                    template='plotly_white',
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stochastic Process Analysis
                st.markdown("---")
                st.markdown('<p class="sub-header">🎲 Stochastic Process Simulations</p>', unsafe_allow_html=True)
                
                S0 = df['Close'].iloc[-1]
                T = forecast_days / 252  # Convert to years
                dt = 1/252  # Daily time step
                N = forecast_days
                
                # Random Walk
                if analysis_type in ["Random Walk", "All"]:
                    st.markdown("#### 🚶 Random Walk Simulation")
                    
                    with st.spinner('Running Random Walk simulations...'):
                        fig = go.Figure()
                        
                        # Run simulations
                        for i in range(min(num_simulations, 100)):
                            path = random_walk_simulation(S0, returns, forecast_days)
                            fig.add_trace(go.Scatter(
                                x=list(range(len(path))),
                                y=path,
                                mode='lines',
                                line=dict(color='blue', width=0.8),
                                opacity=0.3,
                                showlegend=False,
                                hovertemplate='Day: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
                            ))
                        
                        # Add current price marker
                        fig.add_trace(go.Scatter(
                            x=[0],
                            y=[S0],
                            mode='markers',
                            marker=dict(color='red', size=12),
                            name='Current Price',
                            hovertemplate='Current Price: ₹%{y:.2f}<extra></extra>'
                        ))
                        
                        # Add vertical line at forecast start
                        fig.add_vline(
                            x=0,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Forecast Start"
                        )
                        
                        fig.update_layout(
                            title=f'Random Walk Simulation ({num_simulations} paths)',
                            xaxis_title='Days',
                            yaxis_title='Price (₹)',
                            hovermode='closest',
                            template='plotly_white',
                            height=600,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("📝 **Random Walk:** Assumes future price movements are independent and equally likely to go up or down based on historical returns.")
                
                # Geometric Brownian Motion
                if analysis_type in ["Geometric Brownian Motion", "All"]:
                    st.markdown("#### 📈 Geometric Brownian Motion (GBM)")
                    
                    with st.spinner('Running GBM simulations...'):
                        fig = go.Figure()
                        
                        for i in range(min(num_simulations, 100)):
                            path = geometric_brownian_motion(S0, mu, sigma, T, dt, N)
                            fig.add_trace(go.Scatter(
                                x=list(range(len(path))),
                                y=path,
                                mode='lines',
                                line=dict(color='green', width=0.8),
                                opacity=0.3,
                                showlegend=False,
                                hovertemplate='Day: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
                            ))
                        
                        # Add current price line
                        fig.add_hline(
                            y=S0,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Current Price: ₹{S0:.2f}",
                            annotation_position="right"
                        )
                        
                        fig.update_layout(
                            title=f'Geometric Brownian Motion ({num_simulations} paths)',
                            xaxis_title='Days',
                            yaxis_title='Price (₹)',
                            hovermode='closest',
                            template='plotly_white',
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("📝 **GBM:** Assumes returns are log-normally distributed. Commonly used in Black-Scholes option pricing model.")
                
                # Monte Carlo Simulation
                if analysis_type in ["Monte Carlo Simulation", "All"]:
                    st.markdown("#### 🎰 Monte Carlo Simulation Analysis")
                    
                    with st.spinner('Running Monte Carlo simulations...'):
                        simulations = monte_carlo_simulation(S0, mu, sigma, T, dt, num_simulations)
                        
                        # Create subplots
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=(f'Monte Carlo Simulation ({num_simulations} paths)', 'Distribution of Final Prices'),
                            horizontal_spacing=0.12
                        )
                        
                        # Plot all simulation paths
                        for i in range(min(num_simulations, 100)):
                            fig.add_trace(
                                go.Scatter(
                                    x=list(range(len(simulations[i]))),
                                    y=simulations[i],
                                    mode='lines',
                                    line=dict(color='purple', width=0.8),
                                    opacity=0.2,
                                    showlegend=False,
                                    hovertemplate='Day: %{x}<br>Price: ₹%{y:.2f}<extra></extra>'
                                ),
                                row=1, col=1
                            )
                        
                        # Mean path
                        mean_path = simulations.mean(axis=0)
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(len(mean_path))),
                                y=mean_path,
                                mode='lines',
                                name='Mean Path',
                                line=dict(color='red', width=3),
                                hovertemplate='Day: %{x}<br>Mean Price: ₹%{y:.2f}<extra></extra>'
                            ),
                            row=1, col=1
                        )
                        
                        # Current price line
                        fig.add_hline(
                            y=S0,
                            line_dash="dash",
                            line_color="orange",
                            row=1, col=1
                        )
                        
                        # Distribution of final prices
                        final_prices = simulations[:, -1]
                        
                        fig.add_trace(
                            go.Histogram(
                                x=final_prices,
                                nbinsx=50,
                                name='Final Prices',
                                marker_color='purple',
                                opacity=0.7,
                                showlegend=False,
                                hovertemplate='Price Range: %{x:.2f}<br>Count: %{y}<extra></extra>'
                            ),
                            row=1, col=2
                        )
                        
                        # Add vertical lines for current price and mean
                        fig.add_vline(
                            x=S0,
                            line_dash="dash",
                            line_color="orange",
                            annotation_text=f"Current: ₹{S0:.2f}",
                            annotation_position="top",
                            row=1, col=2
                        )
                        
                        fig.add_vline(
                            x=mean_path[-1],
                            line_color="red",
                            line_width=2,
                            annotation_text=f"Mean: ₹{mean_path[-1]:.2f}",
                            annotation_position="top",
                            row=1, col=2
                        )
                        
                        # Update axes
                        fig.update_xaxes(title_text="Days", row=1, col=1)
                        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
                        fig.update_xaxes(title_text=f"Price after {forecast_days} days (₹)", row=1, col=2)
                        fig.update_yaxes(title_text="Frequency", row=1, col=2)
                        
                        fig.update_layout(
                            height=550,
                            showlegend=True,
                            template='plotly_white',
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
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
                
                # 🔥 ADVANCED ANALYSIS SECTION
                if analysis_type == "🔥 Advanced Analysis":
                    st.markdown("---")
                    st.markdown('<p class="sub-header">🔥 Advanced Stochastic Analysis</p>', unsafe_allow_html=True)
                    st.markdown("*Going beyond basic models - capturing real market complexities*")
                    
                    # Confidence Cone Analysis
                    if 'show_confidence_cone' not in locals():
                        show_confidence_cone = True
                    
                    if show_confidence_cone:
                        st.markdown("#### 📊 Confidence Cone - The Expanding Uncertainty")
                        
                        with st.spinner('Calculating confidence intervals...'):
                            # Run simulations if not already done
                            if 'simulations' not in locals():
                                simulations = monte_carlo_simulation(S0, mu, sigma, T, dt, num_simulations)
                            
                            # Calculate percentiles
                            percentiles = calculate_confidence_cone(simulations)
                            
                            fig = go.Figure()
                            
                            # Add confidence bands
                            time_steps = list(range(len(simulations[0])))
                            
                            # 90% confidence interval (shaded)
                            fig.add_trace(go.Scatter(
                                x=time_steps + time_steps[::-1],
                                y=list(percentiles[0.95]) + list(percentiles[0.05])[::-1],
                                fill='toself',
                                fillcolor='rgba(31, 119, 180, 0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='90% Confidence',
                                hoverinfo='skip'
                            ))
                            
                            # 50% confidence interval (darker)
                            fig.add_trace(go.Scatter(
                                x=time_steps + time_steps[::-1],
                                y=list(percentiles[0.75]) + list(percentiles[0.25])[::-1],
                                fill='toself',
                                fillcolor='rgba(31, 119, 180, 0.4)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='50% Confidence',
                                hoverinfo='skip'
                            ))
                            
                            # Median path
                            fig.add_trace(go.Scatter(
                                x=time_steps,
                                y=percentiles[0.5],
                                mode='lines',
                                name='Median Forecast',
                                line=dict(color='red', width=3)
                            ))
                            
                            # Current price
                            fig.add_hline(y=S0, line_dash="dash", line_color="orange",
                                        annotation_text=f"Current: ₹{S0:.2f}")
                            
                            fig.update_layout(
                                title='Confidence Cone - How Uncertainty Grows Over Time',
                                xaxis_title='Days into Future',
                                yaxis_title='Price (₹)',
                                template='plotly_white',
                                height=600,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Explanation with metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                uncertainty_today = percentiles[0.95][0] - percentiles[0.05][0]
                                st.metric("Uncertainty Today", f"₹{uncertainty_today:.2f}")
                            with col2:
                                uncertainty_future = percentiles[0.95][-1] - percentiles[0.05][-1]
                                st.metric(f"Uncertainty Day {forecast_days}", f"₹{uncertainty_future:.2f}")
                            with col3:
                                uncertainty_growth = ((uncertainty_future - uncertainty_today) / uncertainty_today) * 100
                                st.metric("Uncertainty Growth", f"{uncertainty_growth:.1f}%")
                            
                            st.info("📝 **Confidence Cone:** Shows how prediction uncertainty expands over time. The 'cone' widens because more time = more possible outcomes. This is why long-term predictions are less reliable!")
                    
                    # Jump-Diffusion Model
                    if 'show_jump_diffusion' not in locals():
                        show_jump_diffusion = False
                        
                    if show_jump_diffusion:
                        st.markdown("#### ⚡ Jump-Diffusion Model - Black Swan Events")
                        
                        with st.spinner('Simulating jump processes...'):
                            fig = go.Figure()
                            
                            # Regular GBM for comparison
                            regular_path = geometric_brownian_motion(S0, mu, sigma, T, dt, N)
                            fig.add_trace(go.Scatter(
                                x=list(range(len(regular_path))),
                                y=regular_path,
                                mode='lines',
                                name='Regular GBM (No Jumps)',
                                line=dict(color='green', width=2, dash='dash'),
                                opacity=0.6
                            ))
                            
                            # Multiple jump-diffusion paths
                            for i in range(5):
                                path, jump_times, num_jumps = jump_diffusion_model(
                                    S0, mu, sigma, T, dt, N,
                                    jump_intensity=3,  # ~3 jumps per year on average
                                    jump_mean=-0.05,   # Average 5% drop per jump
                                    jump_std=0.08      # But can vary ±8%
                                )
                                
                                fig.add_trace(go.Scatter(
                                    x=list(range(len(path))),
                                    y=path,
                                    mode='lines',
                                    name=f'Path {i+1} ({num_jumps} jumps)',
                                    line=dict(width=2),
                                    opacity=0.7
                                ))
                                
                                # Mark jump locations
                                if len(jump_times) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=jump_times,
                                        y=[path[min(jt, len(path)-1)] for jt in jump_times],
                                        mode='markers',
                                        name=f'Jumps (Path {i+1})',
                                        marker=dict(
                                            size=12,
                                            symbol='x',
                                            color='red',
                                            line=dict(width=2)
                                        ),
                                        showlegend=(i == 0)
                                    ))
                            
                            fig.update_layout(
                                title='Jump-Diffusion: Modeling Market Crashes & Sudden Events',
                                xaxis_title='Days',
                                yaxis_title='Price (₹)',
                                template='plotly_white',
                                height=600,
                                hovermode='closest'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.warning("⚡ **Jump-Diffusion Model:** Unlike regular GBM, this model includes sudden jumps to capture market crashes, flash crashes, and unexpected news events. The red X marks show when 'Black Swan' events occurred. This is Merton's model - more realistic than basic GBM!")
                    
                    # Model Comparison
                    if 'compare_models' not in locals():
                        compare_models = True
                        
                    if compare_models:
                        st.markdown("#### 🔬 Model Comparison - Which Performs Best?")
                        
                        with st.spinner('Comparing all models...'):
                            fig = go.Figure()
                            
                            # Generate one path from each model
                            rw_path = random_walk_simulation(S0, returns, forecast_days)
                            gbm_path = geometric_brownian_motion(S0, mu, sigma, T, dt, N)
                            jd_path, _, _ = jump_diffusion_model(S0, mu, sigma, T, dt, N)
                            mc_mean = simulations.mean(axis=0)
                            
                            # Plot all on same chart
                            fig.add_trace(go.Scatter(
                                x=list(range(len(rw_path))),
                                y=rw_path,
                                mode='lines',
                                name='Random Walk',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=list(range(len(gbm_path))),
                                y=gbm_path,
                                mode='lines',
                                name='Geometric Brownian Motion',
                                line=dict(color='green', width=2)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=list(range(len(jd_path))),
                                y=jd_path,
                                mode='lines',
                                name='Jump-Diffusion',
                                line=dict(color='red', width=2)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=list(range(len(mc_mean))),
                                y=mc_mean,
                                mode='lines',
                                name='Monte Carlo Mean',
                                line=dict(color='purple', width=3, dash='dot')
                            ))
                            
                            fig.add_hline(y=S0, line_dash="dash", line_color="orange",
                                        annotation_text=f"Starting: ₹{S0:.2f}")
                            
                            fig.update_layout(
                                title='Model Comparison - Different Approaches, Different Results',
                                xaxis_title='Days',
                                yaxis_title='Price (₹)',
                                template='plotly_white',
                                height=600,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Comparison table
                            comparison_data = {
                                'Model': ['Random Walk', 'GBM', 'Jump-Diffusion', 'Monte Carlo'],
                                'Final Price': [
                                    f"₹{rw_path[-1]:.2f}",
                                    f"₹{gbm_path[-1]:.2f}",
                                    f"₹{jd_path[-1]:.2f}",
                                    f"₹{mc_mean[-1]:.2f}"
                                ],
                                'Change %': [
                                    f"{((rw_path[-1] - S0) / S0 * 100):.2f}%",
                                    f"{((gbm_path[-1] - S0) / S0 * 100):.2f}%",
                                    f"{((jd_path[-1] - S0) / S0 * 100):.2f}%",
                                    f"{((mc_mean[-1] - S0) / S0 * 100):.2f}%"
                                ],
                                'Best For': [
                                    'Simple analysis',
                                    'Option pricing',
                                    'Risk management',
                                    'Uncertainty quantification'
                                ]
                            }
                            
                            st.dataframe(comparison_data, use_container_width=True, hide_index=True)
                            
                            st.info("📝 **Key Insight:** Different models give different results because they make different assumptions. In reality, markets combine all these effects - random walks, trends, AND sudden jumps!")
                    
                    # Stress Testing
                    if 'stress_test' not in locals():
                        stress_test = False
                        
                    if stress_test:
                        st.markdown("#### 🚨 Stress Test - Extreme Scenarios")
                        
                        with st.spinner('Running stress tests...'):
                            scenarios = stress_test_scenarios(S0, returns, forecast_days)
                            
                            fig = go.Figure()
                            
                            # Plot each scenario
                            colors = {'Crisis (2008-like)': 'red', 
                                    'Bull Market': 'green',
                                    'COVID-like Crash & Recovery': 'orange'}
                            
                            for scenario_name, path in scenarios.items():
                                fig.add_trace(go.Scatter(
                                    x=list(range(len(path))),
                                    y=path,
                                    mode='lines',
                                    name=scenario_name,
                                    line=dict(color=colors[scenario_name], width=3)
                                ))
                            
                            # Add normal forecast for reference
                            normal_path = geometric_brownian_motion(S0, mu, sigma, T, dt, N)
                            fig.add_trace(go.Scatter(
                                x=list(range(len(normal_path))),
                                y=normal_path,
                                mode='lines',
                                name='Normal Forecast',
                                line=dict(color='blue', width=2, dash='dash')
                            ))
                            
                            fig.add_hline(y=S0, line_dash="dash", line_color="gray",
                                        annotation_text=f"Today: ₹{S0:.2f}")
                            
                            fig.update_layout(
                                title='Stress Test - How Would This Stock Perform in Extreme Scenarios?',
                                xaxis_title='Days',
                                yaxis_title='Price (₹)',
                                template='plotly_white',
                                height=600,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Scenario metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                crisis_final = scenarios['Crisis (2008-like)'][-1]
                                crisis_loss = ((crisis_final - S0) / S0) * 100
                                st.metric("Crisis Scenario", 
                                        f"₹{crisis_final:.2f}",
                                        f"{crisis_loss:.1f}%",
                                        delta_color="inverse")
                            
                            with col2:
                                bull_final = scenarios['Bull Market'][-1]
                                bull_gain = ((bull_final - S0) / S0) * 100
                                st.metric("Bull Market", 
                                        f"₹{bull_final:.2f}",
                                        f"+{bull_gain:.1f}%")
                            
                            with col3:
                                covid_final = scenarios['COVID-like Crash & Recovery'][-1]
                                covid_change = ((covid_final - S0) / S0) * 100
                                st.metric("COVID-like", 
                                        f"₹{covid_final:.2f}",
                                        f"{covid_change:+.1f}%")
                            
                            st.warning("🚨 **Stress Testing:** Banks and hedge funds run these tests to see how portfolios would perform in extreme scenarios. This is required by regulators after 2008! It shows your maximum potential loss in a crisis.")
                
                # Key Insights
                st.markdown("---")
                st.markdown('<p class="sub-header">💡 Key Insights</p>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Model Assumptions:**")
                    st.write("- Markets are continuous and frictionless")
                    st.write("- Returns follow normal distribution")
                    st.write("- No arbitrage opportunities")
                    st.write("- Constant volatility (simplified)")
                
                with col2:
                    st.markdown("**⚠️ Limitations:**")
                    st.write("- Real markets have jumps and gaps")
                    st.write("- Volatility changes over time")
                    st.write("- Ignores market microstructure")
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