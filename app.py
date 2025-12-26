import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta, datetime
import time

# ==========================================
# PAGE CONFIGURATION (Must be first)
# ==========================================
st.set_page_config(
    page_title="Macro-Optimized Stock Picker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Financial Terminal" Look
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #41424b;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        font-weight: 600;
    }
    .stDataFrame {
        border: 1px solid #41424b;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. FRED MACRO LOGIC
# ==========================================
class FredAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_series(self, series_id):
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "asc"
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json().get("observations", [])
            df = pd.DataFrame(data)
            if df.empty: return pd.DataFrame()
            df = df[df['value'] != '.']
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'])
            return df[['date', 'value']].sort_values('date')
        except Exception as e:
            st.error(f"Error fetching {series_id}: {e}")
            return pd.DataFrame()

    def analyze_regime(self):
        # Fetch Data
        df_rate = self.fetch_series("FEDFUNDS")
        df_bs = self.fetch_series("WALCL")

        if df_rate.empty or df_bs.empty:
            return None, None, None

        # Rename
        df_rate.rename(columns={'value': 'interest_rate'}, inplace=True)
        df_bs.rename(columns={'value': 'balance_sheet'}, inplace=True)

        # Merge
        df_merged = pd.merge_asof(df_bs, df_rate, on='date', direction='backward')
        
        # Calculate BS Change
        df_merged['bs_change'] = df_merged['balance_sheet'].diff()
        
        # Filter Last 10 Years for Average calculation
        latest_date = df_merged['date'].max()
        cutoff_date = latest_date - timedelta(days=365 * 10)
        df_10y = df_merged[df_merged['date'] >= cutoff_date].dropna().copy()
        
        # Determine Thresholds
        avg_rate_10y = df_10y['interest_rate'].mean()
        
        # Get Current State (Last row)
        current = df_10y.iloc[-1]
        is_high_rate = current['interest_rate'] > avg_rate_10y
        is_growing_bs = current['bs_change'] >= 0
        
        # Determine Category
        if is_high_rate and is_growing_bs:
            category = "High Rate / Growing BS"
            strategy = "growth_pe" # Growth at a reasonable price
        elif is_high_rate and not is_growing_bs:
            category = "High Rate / Shrinking BS"
            strategy = "value" # Pure Value (Low P/E)
        elif not is_high_rate and is_growing_bs:
            category = "Low Rate / Growing BS"
            strategy = "growth" # Pure Growth
        else: # Low Rate / Shrinking BS
            category = "Low Rate / Shrinking BS"
            strategy = "growth_pe"

        stats = {
            "current_rate": current['interest_rate'],
            "avg_rate_10y": avg_rate_10y,
            "current_bs": current['balance_sheet'],
            "bs_change": current['bs_change']
        }
        
        return category, strategy, stats

# ==========================================
# 2. STOCK FETCHING LOGIC
# ==========================================
@st.cache_data(ttl=3600) # Cache for 1 hour to prevent constant re-fetching
def get_sp500_data():
    """Fetches S&P 500 tickers and data using yfinance."""
    
    # 1. Get Tickers
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        tickers = tables[0]['Symbol'].tolist()
        # Clean tickers (BF.B -> BF-B)
        tickers = [t.replace('.', '-') for t in tickers]
    except Exception as e:
        st.error(f"Error scraping tickers: {e}")
        return pd.DataFrame()

    # 2. Fetch Data in Bulk (Much faster than loops)
    # We fetch a subset of tickers for demo speed if needed, but here we try all.
    # For user experience, we'll limit to first 100 or full 500 if requested.
    # Let's do a smart fetch using Ticker object which is slower but more detailed for metrics
    # OR use download for price history. We need Metrics (PE, Growth).
    # Bulk fetching metrics via yfinance is tricky. We iterate.
    
    data = []
    
    # Progress Bar
    progress_text = "Scanning S&P 500 Financials... (This takes a moment)"
    my_bar = st.progress(0, text=progress_text)
    
    # Optimization: We cannot fetch 500 stocks metrics instantly in free yfinance.
    # We will fetch a representative sample or the user must wait. 
    # For this "Production" script, we will fetch the first 50 to ensure UI responsiveness,
    # or you can uncomment the full list line.
    
    target_tickers = tickers[:60] # LIMITING TO 60 FOR DEMO SPEED. 
    # target_tickers = tickers # Uncomment for full production run (takes ~5 mins)

    for i, ticker in enumerate(target_tickers):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            
            # Safe extraction
            pe = info.get('trailingPE')
            fwd_pe = info.get('forwardPE')
            growth = info.get('earningsGrowth') # This is a decimal (0.15 = 15%)
            name = info.get('shortName')
            sector = info.get('sector')
            price = info.get('currentPrice')
            
            if pe is not None:
                data.append({
                    'Ticker': ticker,
                    'Name': name,
                    'Sector': sector,
                    'Price': price,
                    'PE': pe,
                    'Growth': growth if growth else 0.0,
                    'PEG_Proxy': (growth / pe) if (growth and pe and pe > 0) else -1
                })
        except:
            pass
            
        percent_complete = int((i / len(target_tickers)) * 100)
        my_bar.progress(percent_complete, text=f"Scanning {ticker}...")
        
    my_bar.empty()
    return pd.DataFrame(data)

def optimize_portfolio(df, strategy):
    """
    Strategies:
    - growth: Sort by Growth (Desc)
    - value: Sort by PE (Asc)
    - growth_pe: Sort by Growth/PE (Desc)
    """
    df = df.copy()
    
    if strategy == "growth":
        st.info("🎯 Optimization Strategy: **Pure Growth** (Maximizing Earnings Growth)")
        df_sorted = df.sort_values(by='Growth', ascending=False)
        best = df_sorted.head(5)
        worst = df_sorted.tail(5) # Lowest growth
        
    elif strategy == "value":
        st.info("🎯 Optimization Strategy: **Deep Value** (Minimizing P/E Ratio)")
        # Filter out negative PE or zero if any leaked through
        clean_df = df[df['PE'] > 0]
        df_sorted = clean_df.sort_values(by='PE', ascending=True)
        best = df_sorted.head(5)
        worst = df_sorted.tail(5) # Highest PE
        
    elif strategy == "growth_pe":
        st.info("🎯 Optimization Strategy: **GARP** (Growth at a Reasonable Price)")
        # Metric: Growth / PE. Higher is better.
        # We need positive PE for this to make sense.
        clean_df = df[(df['PE'] > 0) & (df['Growth'] > 0)]
        clean_df['Score'] = clean_df['Growth'] / clean_df['PE']
        df_sorted = clean_df.sort_values(by='Score', ascending=False)
        best = df_sorted.head(5)
        worst = df_sorted.tail(5) # Lowest score
        
    return best, worst

# ==========================================
# 3. MAIN APP UI
# ==========================================
def main():
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    api_key = st.sidebar.text_input("FRED API Key", type="password", help="Get one at https://fred.stlouisfed.org/docs/api/api_key.html")
    
    st.title("🦅 Macro-Driven Portfolio Optimizer")
    st.markdown("This tool analyzes Federal Reserve data to determine the current economic cycle, then selects the optimal S&P 500 stocks based on that regime.")
    
    if not api_key:
        st.warning("Please enter your FRED API Key in the sidebar to begin analysis.")
        st.stop()
        
    # --- PHASE 1: MACRO ANALYSIS ---
    analyzer = FredAnalyzer(api_key)
    
    with st.spinner("Analyzing Federal Reserve Data..."):
        category, strategy, stats = analyzer.analyze_regime()
        
    if not category:
        st.error("Failed to fetch FRED data. Check your API Key.")
        st.stop()

    # Display Macro Dashboard
    st.markdown("### 1. Macroeconomic Regime Detection")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Fed Funds Rate", f"{stats['current_rate']:.2f}%", 
                delta=f"{stats['current_rate'] - stats['avg_rate_10y']:.2f}% vs 10y Avg")
    
    col2.metric("Fed Balance Sheet", f"${stats['current_bs']/1000000:.2f}T", 
                delta=f"{stats['bs_change']/1000:.2f}B Weekly Change")
    
    col3.metric("Regime Quadrant", category)
    
    col4.markdown(f"**Strategy:** `{strategy.upper()}`")

    st.divider()

    # --- PHASE 2: STOCK OPTIMIZATION ---
    st.markdown("### 2. Stock Selection & Optimization")
    
    # Fetch Data
    df_stocks = get_sp500_data()
    
    if df_stocks.empty:
        st.error("No stock data found.")
        st.stop()
        
    # Optimize
    best_stocks, worst_stocks = optimize_portfolio(df_stocks, strategy)
    
    # Display Results
    tab1, tab2, tab3 = st.tabs(["🏆 Top 5 Selections", "📉 Bottom 5 Avoids", "📊 Detailed Performance"])
    
    with tab1:
        st.success(f"Top 5 Stocks for {category} Environment")
        # Format for display
        display_cols = ['Ticker', 'Name', 'Sector', 'Price', 'PE', 'Growth']
        
        # Enhanced Table
        st.dataframe(
            best_stocks[display_cols].style.format({
                'Price': '${:.2f}',
                'PE': '{:.1f}',
                'Growth': '{:.1%}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Mini charts for top 5
        st.subheader("Price Action (Last 30 Days)")
        cols = st.columns(5)
        for i, row in enumerate(best_stocks.itertuples()):
            with cols[i]:
                st.caption(f"**{row.Ticker}**")
                # Quick fetch of history for sparkline
                hist = yf.Ticker(row.Ticker).history(period="1mo")
                st.line_chart(hist['Close'], height=100)

    with tab2:
        st.error(f"Bottom 5 Stocks (Least Optimal for {category})")
        st.dataframe(
            worst_stocks[display_cols].style.format({
                'Price': '${:.2f}',
                'PE': '{:.1f}',
                'Growth': '{:.1%}'
            }),
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        st.subheader("Comparative Analysis")
        
        # Scatter Plot: Growth vs PE
        fig = px.scatter(
            df_stocks, 
            x="PE", 
            y="Growth", 
            hover_data=['Ticker', 'Name'],
            color="Sector",
            title="S&P 500: P/E Ratio vs Earnings Growth",
            log_x=True # Log scale because PE can vary wildly
        )
        # Highlight selections
        fig.add_trace(go.Scatter(
            x=best_stocks['PE'], 
            y=best_stocks['Growth'], 
            mode='markers', 
            marker=dict(color='green', size=12, symbol='star'),
            name='Top Picks'
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Full Dataset"):
            st.dataframe(df_stocks)

if __name__ == "__main__":
    main()