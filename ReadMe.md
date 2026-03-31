# 📈 StochaStonks: Stochastic Process Analyzer for Financial Markets

**StochaStonks** is an advanced, fintech-grade web application developed for the **Stochastic Processes and Applications (STPA)** course at **NMIMS Mukesh Patel School of Technology Management & Engineering (MPSTME)**. It bridges theoretical probability concepts from the classroom with real-world financial data analysis. [cite: 145, 146]

## 🎓 Academic Context
- **Course**: STPA (702BS0C033) [cite: 145]
- **Instructor**: Prof. Rashmi Patel [cite: 82, 145]
- **Institution**: NMIMS MPSTME, Mumbai [cite: 1]
- **Syllabus Alignment**: Directly maps to Units 1 through 5, covering Random Walks, Stationary Processes, Markov Chains, and Poisson Processes. [cite: 77, 79]

## ✨ Key Features
### 📊 Syllabus-Aligned Stochastic Models
* **Random Walk (Unit 2)**: Modeling unpredictable price movements. [cite: 81, 147]
* **Moving Average Process (Unit 3)**: Demonstrating weak stationarity and mean reversion. [cite: 81, 147]
* **Discrete-time Markov Chains (Unit 4)**: Transition matrices and Chapman-Kolmogorov equations for regime switching. [cite: 81, 147, 153]
* **Poisson Process (Unit 5)**: Continuous-time event counting for market news/shocks. [cite: 81, 147, 154]
* **Gambler’s Ruin (Unit 4)**: Assessing the probability of capital ruin vs. profit targets. [cite: 81, 147]

### 🚀 Advanced Technical Indicators
* **Trend Analysis**: 7-day and 22-day Moving Averages with **Golden Cross/Death Cross** detection. [cite: 209, 210, 212]
* **Volatility Metrics**: 14-day rolling standard deviation for risk assessment. [cite: 209, 211]
* **Performance Tracking**: Cumulative returns, monthly high/low/average tables, and winning/losing streak analysis. 
* **Risk Management**: Maximum Drawdown (depreciation) tracking from peak values. [cite: 210, 213]

### 🎨 UI/UX & Technology
* **Glassmorphism Design**: A modern, animated interface with purple gradient aesthetics. [cite: 62, 63]
* **Interactive Visualizations**: Powered by **Plotly** for hover-data, zooming, and panning. [cite: 34, 35]
* **Production-Ready**: Implements exponential backoff retry logic and 2-hour caching to handle API rate limits. [cite: 254, 257]

## 🛠️ Installation
```bash
pip install -r requirements.txt
streamlit run stock_stochastic_app_complete.py