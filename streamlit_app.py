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
