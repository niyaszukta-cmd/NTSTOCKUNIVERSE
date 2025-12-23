import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from functools import wraps
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

st.set_page_config(
    page_title="NYZTrade Pro Valuation", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSV-BASED STOCK UNIVERSE LOADER - OPTIMIZED FOR 10,000+ STOCKS
# ============================================================================

@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_stocks_from_csv(csv_path='stocks_universe_full.csv'):
    """
    Load stocks from CSV file - Optimized for large datasets
    Expected columns: Ticker, Name, Category Name
    """
    try:
        # Use chunksize for large files
        df = pd.read_csv(csv_path, dtype=str)
        
        # Normalize column names
        df.columns = df.columns.str.strip()
        column_mapping = {
            'Ticker': 'ticker', 'ticker': 'ticker', 'TICKER': 'ticker',
            'Name': 'name', 'name': 'name', 'NAME': 'name',
            'Category Name': 'category', 'category': 'category', 'Category': 'category'
        }
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns
        if 'ticker' not in df.columns or 'name' not in df.columns:
            raise ValueError("CSV must contain 'Ticker' and 'Name' columns")
        
        # Add category if missing
        if 'category' not in df.columns:
            df['category'] = '📊 All Stocks'
        
        # Clean data
        df = df.dropna(subset=['ticker', 'name'])
        df['ticker'] = df['ticker'].str.strip().str.upper()
        df['name'] = df['name'].str.strip()
        df['category'] = df['category'].str.strip()
        df['category'] = df['category'].fillna('📊 Uncategorized')
        df.loc[df['category'] == '', 'category'] = '📊 Uncategorized'
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['ticker'])
        
        # Filter only Indian stocks (.NS or .BO)
        df = df[df['ticker'].str.contains('.NS|.BO', na=False, regex=True)]
        
        st.success(f"✅ Loaded {len(df):,} Indian stocks from CSV")
        return df
        
    except FileNotFoundError:
        st.warning(f"⚠️ CSV file '{csv_path}' not found. Using minimal sample.")
        sample_data = {
            'ticker': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
            'name': ['Reliance Industries', 'Tata Consultancy Services', 'Infosys', 'HDFC Bank', 'ICICI Bank'],
            'category': ['Energy', 'Technology', 'Technology', 'Financial Services', 'Financial Services']
        }
        return pd.DataFrame(sample_data)
    
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        return pd.DataFrame(columns=['ticker', 'name', 'category'])

def load_stocks_from_uploaded_file(uploaded_file):
    """Load stocks from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file, dtype=str)
        
        # Normalize columns
        df.columns = df.columns.str.strip()
        column_mapping = {
            'Ticker': 'ticker', 'ticker': 'ticker', 'TICKER': 'ticker',
            'Name': 'name', 'name': 'name', 'NAME': 'name',
            'Category Name': 'category', 'category': 'category'
        }
        df = df.rename(columns=column_mapping)
        
        if 'ticker' not in df.columns or 'name' not in df.columns:
            st.error("❌ CSV must contain 'Ticker' and 'Name' columns")
            return None
        
        if 'category' not in df.columns:
            df['category'] = '📊 All Stocks'
        
        # Clean
        df = df.dropna(subset=['ticker', 'name'])
        df['ticker'] = df['ticker'].str.strip().str.upper()
        df['name'] = df['name'].str.strip()
        df['category'] = df['category'].str.strip().fillna('📊 Uncategorized')
        df = df.drop_duplicates(subset=['ticker'])
        
        # Filter Indian stocks
        df = df[df['ticker'].str.contains('.NS|.BO', na=False, regex=True)]
        
        return df
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

@st.cache_data
def get_stocks_by_category_from_df(df):
    """Convert DataFrame to dictionary grouped by category - with limit"""
    stocks_dict = {}
    for category in sorted(df['category'].unique())[:50]:  # Limit categories shown
        category_df = df[df['category'] == category]
        stocks_dict[category] = dict(zip(category_df['ticker'], category_df['name']))
    return stocks_dict

@st.cache_data
def get_all_stocks_from_df(df):
    """Get all stocks as a dictionary"""
    return dict(zip(df['ticker'], df['name']))

@st.cache_data
def get_categories_from_df(df):
    """Get list of top categories"""
    # Return top 50 categories by stock count
    top_cats = df['category'].value_counts().head(50).index.tolist()
    return sorted(top_cats)

@st.cache_data
def search_stock_in_df(df, query):
    """Search for stocks - optimized"""
    if df.empty or not query:
        return {}
    
    query_upper = query.upper()
    mask = (df['ticker'].str.contains(query_upper, na=False, case=False) | 
            df['name'].str.upper().str.contains(query_upper, na=False))
    
    results = {}
    for _, row in df[mask].head(100).iterrows():  # Limit to 100 results
        results[row['ticker']] = {
            'name': row['name'],
            'category': row['category']
        }
    return results

def get_category_for_ticker(df, ticker):
    """Get category for ticker"""
    result = df[df['ticker'] == ticker]
    return result.iloc[0]['category'] if not result.empty else 'Default'

# Category to Sector Mapping
CATEGORY_TO_SECTOR = {
    'Information Technology Services': 'Technology', 'IT Services': 'Technology',
    'Software': 'Technology', 'Application Software': 'Technology',
    'Technical & System Software': 'Technology', 'Computer Software': 'Technology',
    'Banks': 'Financial Services', 'Finance': 'Financial Services',
    'Financial Services': 'Financial Services', 'Money Center Banks': 'Financial Services',
    'NBFC': 'Financial Services', 'Insurance': 'Financial Services',
    'Asset Management': 'Financial Services', 'Investment Brokerage - National': 'Financial Services',
    'Industrial Metals & Minerals': 'Industrials', 'General Contractors': 'Industrials',
    'Diversified Machinery': 'Industrials', 'Farm & Construction Machinery': 'Industrials',
    'General Building Materials': 'Industrials', 'Steel & Iron': 'Industrials',
    'Auto Parts': 'Industrials', 'Auto Manufacturers - Major': 'Industrials',
    'Oil & Gas Drilling & Exploration': 'Energy', 'Oil & Gas': 'Energy',
    'Electric Utilities': 'Energy', 'Power': 'Energy', 'Diversified Utilities': 'Energy',
    'Chemicals': 'Basic Materials', 'Specialty Chemicals': 'Basic Materials',
    'Drug Manufacturers - Major': 'Healthcare', 'Drugs - Generic': 'Healthcare',
    'Pharmaceuticals': 'Healthcare', 'Healthcare': 'Healthcare', 'Medical': 'Healthcare',
    'Textile Industrial': 'Consumer Cyclical', 'Textile - Apparel Clothing': 'Consumer Cyclical',
    'Food - Major Diversified': 'Consumer Defensive', 'Entertainment - Diversified': 'Consumer Cyclical',
    'Telecommunications': 'Communication Services', 'Telecom': 'Communication Services',
    'Wireless Communications': 'Communication Services',
    'Real Estate': 'Real Estate', 'REIT': 'Real Estate', 'Real Estate Development': 'Real Estate',
    'Construction': 'Real Estate', 'Conglomerates': 'Industrials',
    'Aerospace/Defense - Major Diversified': 'Industrials', 'Shipping': 'Industrials',
    'Beverages - Soft Drinks': 'Consumer Defensive', 'Agricultural Chemicals': 'Basic Materials',
    'Rubber & Plastics': 'Basic Materials', 'Farm Products': 'Consumer Defensive',
}

def map_category_to_sector(category):
    """Map category to sector"""
    if not category or pd.isna(category):
        return 'Default'
    category = str(category).strip()
    if category in CATEGORY_TO_SECTOR:
        return CATEGORY_TO_SECTOR[category]
    category_lower = category.lower()
    for key, value in CATEGORY_TO_SECTOR.items():
        if key.lower() in category_lower or category_lower in key.lower():
            return value
    return 'Default'

# ============================================================================
# INDUSTRY BENCHMARKS
# ============================================================================

INDUSTRY_BENCHMARKS = {
    'Technology': {'pe': 25, 'ev_ebitda': 15},
    'Financial Services': {'pe': 18, 'ev_ebitda': 12},
    'Consumer Cyclical': {'pe': 30, 'ev_ebitda': 14},
    'Consumer Defensive': {'pe': 35, 'ev_ebitda': 16},
    'Healthcare': {'pe': 28, 'ev_ebitda': 14},
    'Industrials': {'pe': 22, 'ev_ebitda': 12},
    'Energy': {'pe': 15, 'ev_ebitda': 8},
    'Basic Materials': {'pe': 18, 'ev_ebitda': 10},
    'Communication Services': {'pe': 20, 'ev_ebitda': 12},
    'Real Estate': {'pe': 25, 'ev_ebitda': 18},
    'Utilities': {'pe': 16, 'ev_ebitda': 10},
    'Default': {'pe': 20, 'ev_ebitda': 12}
}

def get_benchmark(sector):
    return INDUSTRY_BENCHMARKS.get(sector, INDUSTRY_BENCHMARKS['Default'])

# ============================================================================
# [KEEP ALL THE CSS STYLING FROM BEFORE - EXACT SAME]
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 3rem; border-radius: 20px; text-align: center; color: white;
    margin-bottom: 2rem; box-shadow: 0 20px 60px rgba(0,0,0,0.3); position: relative; overflow: hidden;
}
.main-header::before {
    content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 0.5; } 50% { transform: scale(1.1); opacity: 0.8; } }
.main-header h1 {
    font-size: 2.8rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    position: relative; z-index: 1;
}
.main-header p { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; position: relative; z-index: 1; color: #e2e8f0; }
.company-header {
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 50%, #8b5cf6 100%);
    border: none; border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(124, 58, 237, 0.3);
}
.company-name { font-size: 2rem; font-weight: 700; color: #ffffff !important; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }
.company-meta { display: flex; gap: 1rem; margin-top: 0.8rem; flex-wrap: wrap; }
.meta-badge {
    background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(10px);
    padding: 0.5rem 1rem; border-radius: 25px; font-size: 0.9rem;
    color: #ffffff !important; font-weight: 500; border: 1px solid rgba(255,255,255,0.3);
}
.fair-value-card {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%);
    padding: 2rem; border-radius: 20px; text-align: center; color: white; margin: 1.5rem 0;
    box-shadow: 0 20px 40px rgba(124, 58, 237, 0.3); position: relative; overflow: hidden;
}
.fair-value-label { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.9; margin-bottom: 0.5rem; color: #ffffff; }
.fair-value-amount { font-size: 3rem; font-weight: 700; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace; color: #ffffff; }
.current-price { font-size: 1rem; opacity: 0.85; color: #ffffff; }
.upside-badge {
    display: inline-block; background: rgba(255,255,255,0.2); padding: 0.5rem 1.5rem;
    border-radius: 30px; margin-top: 1rem; font-weight: 600; font-size: 1.2rem;
    backdrop-filter: blur(10px); color: #ffffff;
}
.rec-strong-buy { background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%); color: white !important; padding: 1.5rem 2rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; box-shadow: 0 15px 35px rgba(16, 185, 129, 0.35); }
.rec-buy { background: linear-gradient(135deg, #0d9488 0%, #14b8a6 50%, #2dd4bf 100%); color: white !important; padding: 1.5rem 2rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; box-shadow: 0 15px 35px rgba(20, 184, 166, 0.35); }
.rec-hold { background: linear-gradient(135deg, #d97706 0%, #f59e0b 50%, #fbbf24 100%); color: white !important; padding: 1.5rem 2rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; box-shadow: 0 15px 35px rgba(245, 158, 11, 0.35); }
.rec-avoid { background: linear-gradient(135deg, #dc2626 0%, #ef4444 50%, #f87171 100%); color: white !important; padding: 1.5rem 2rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; box-shadow: 0 15px 35px rgba(239, 68, 68, 0.35); }
.rec-subtitle { font-size: 1rem; font-weight: 500; opacity: 0.9; margin-top: 0.3rem; color: white !important; }
.metric-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none; border-radius: 16px; padding: 1.5rem; text-align: center;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3); transition: all 0.3s ease;
}
.metric-card:hover { transform: translateY(-5px); box-shadow: 0 15px 40px rgba(99, 102, 241, 0.4); }
.metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: #ffffff !important; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.8rem; color: rgba(255,255,255,0.85) !important; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }
.section-header { font-size: 1.4rem; font-weight: 600; color: #a78bfa; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 3px solid #7c3aed; display: inline-block; }
.stock-count { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white !important; padding: 0.8rem 1.2rem; border-radius: 12px; text-align: center; margin: 1rem 0; font-weight: 600; }
.valuation-method { background: linear-gradient(135deg, #312e81 0%, #3730a3 100%); border-radius: 16px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid #a78bfa; box-shadow: 0 8px 25px rgba(55, 48, 163, 0.3); }
.method-title { font-weight: 600; color: #ffffff !important; font-size: 1.1rem; margin-bottom: 1rem; }
.method-row { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); }
.method-label { color: rgba(255,255,255,0.7) !important; }
.method-value { font-weight: 600; color: #ffffff !important; font-family: 'JetBrains Mono', monospace; }
.info-box { background: linear-gradient(135deg, #312e81 0%, #3730a3 100%); border: 1px solid #6366f1; border-radius: 12px; padding: 1.5rem 2rem; color: #ffffff !important; margin: 1rem 0; }
.info-box h3 { color: #a78bfa !important; margin-bottom: 0.5rem; }
.info-box p, .info-box li { color: rgba(255,255,255,0.9) !important; }
.warning-box { background: linear-gradient(135deg, #78350f 0%, #92400e 100%); border: 1px solid #f59e0b; border-radius: 12px; padding: 1rem 1.5rem; color: #fef3c7 !important; margin: 1rem 0; }
.range-card { background: linear-gradient(135deg, #312e81 0%, #3730a3 100%); border-radius: 16px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 8px 25px rgba(55, 48, 163, 0.3); }
.range-header { display: flex; justify-content: space-between; margin-bottom: 1rem; }
.range-low { color: #34d399; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.range-high { color: #f87171; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.range-bar-container { background: rgba(255,255,255,0.1); border-radius: 10px; height: 20px; position: relative; overflow: hidden; }
.range-bar-fill { height: 100%; border-radius: 10px; background: linear-gradient(90deg, #34d399, #fbbf24, #f87171); }
.range-current { text-align: center; margin-top: 1rem; color: #ffffff; font-size: 1.2rem; font-weight: 600; }
.range-current span { color: #a78bfa; font-family: 'JetBrains Mono', monospace; }
.footer { text-align: center; padding: 2rem; color: #a78bfa; font-size: 0.9rem; margin-top: 3rem; border-top: 1px solid rgba(167, 139, 250, 0.3); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PASSWORD AUTHENTICATION
# ============================================================================
def check_password():
    def password_entered():
        username = st.session_state["username"].strip().lower()
        password = st.session_state["password"]
        users = {"demo": "demo123", "premium": "1nV3st!ng", "niyas": "buffet"}
        if username in users and password == users[username]:
            st.session_state["password_correct"] = True
            st.session_state["authenticated_user"] = username
            del st.session_state["password"]
            return
        st.session_state["password_correct"] = False
    
    if "password_correct" not in st.session_state:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
                    padding: 4rem; border-radius: 24px; text-align: center; margin: 2rem auto; max-width: 500px;
                    box-shadow: 0 25px 50px rgba(0,0,0,0.3);'>
            <h1 style='color: white; font-size: 2.5rem; margin-bottom: 0.5rem;'>
                <span style='background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                    NYZTrade Pro
                </span>
            </h1>
            <p style='color: rgba(255,255,255,0.7); margin-bottom: 2rem;'>Professional Stock Valuation Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("👤 Username", key="username", placeholder="Enter username")
            st.text_input("🔒 Password", type="password", key="password", placeholder="Enter password")
            st.button("🚀 Login", on_click=password_entered, use_container_width=True, type="primary")
            st.info("💡 Demo: demo/demo123")
        return False
    elif not st.session_state["password_correct"]:
        st.error("❌ Incorrect credentials. Please try again.")
        return False
    return True

if not check_password():
    st.stop()

# ============================================================================
# LOAD STOCK DATA
# ============================================================================
if 'stocks_df' not in st.session_state:
    st.session_state.stocks_df = load_stocks_from_csv('stocks_universe_full.csv')

STOCKS_DF = st.session_state.stocks_df

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def retry_with_backoff(retries=5, backoff_in_seconds=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise
                    time.sleep(backoff_in_seconds * 2 ** x)
                    x += 1
        return wrapper
    return decorator

@st.cache_data(ttl=7200)
@retry_with_backoff(retries=5, backoff_in_seconds=3)
def fetch_stock_data(ticker):
    try:
        time.sleep(0.5)
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or len(info) < 5:
            return None, "Unable to fetch data"
        return info, None
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate" in error_msg.lower():
            return None, "Rate limit reached"
        return None, str(e)[:100]

def calculate_valuations(info, ticker_symbol, stocks_df):
    try:
        price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
        trailing_pe = info.get('trailingPE', 0)
        forward_pe = info.get('forwardPE', 0)
        trailing_eps = info.get('trailingEps', 0)
        enterprise_value = info.get('enterpriseValue', 0)
        ebitda = info.get('ebitda', 0)
        market_cap = info.get('marketCap', 0)
        shares = info.get('sharesOutstanding', 1)
        
        yf_sector = info.get('sector', None)
        csv_category = get_category_for_ticker(stocks_df, ticker_symbol)
        mapped_sector = map_category_to_sector(csv_category)
        sector = yf_sector if yf_sector else mapped_sector
        
        book_value = info.get('bookValue', 0)
        revenue = info.get('totalRevenue', 0)
        
        benchmark = get_benchmark(sector)
        industry_pe = benchmark['pe']
        industry_ev_ebitda = benchmark['ev_ebitda']
        
        historical_pe = trailing_pe * 0.9 if trailing_pe and trailing_pe > 0 else industry_pe
        blended_pe = (industry_pe + historical_pe) / 2
        fair_value_pe = trailing_eps * blended_pe if trailing_eps else None
        upside_pe = ((fair_value_pe - price) / price * 100) if fair_value_pe and price else None
        
        current_ev_ebitda = enterprise_value / ebitda if ebitda and ebitda > 0 else None
        target_ev_ebitda = (industry_ev_ebitda + current_ev_ebitda * 0.9) / 2 if current_ev_ebitda and 0 < current_ev_ebitda < 50 else industry_ev_ebitda
        
        if ebitda and ebitda > 0:
            fair_ev = ebitda * target_ev_ebitda
            net_debt = (info.get('totalDebt', 0) or 0) - (info.get('totalCash', 0) or 0)
            fair_mcap = fair_ev - net_debt
            fair_value_ev = fair_mcap / shares if shares else None
            upside_ev = ((fair_value_ev - price) / price * 100) if fair_value_ev and price else None
        else:
            fair_value_ev = None
            upside_ev = None
        
        pb_ratio = price / book_value if book_value and book_value > 0 else None
        ps_ratio = market_cap / revenue if revenue and revenue > 0 else None
        
        return {
            'price': price, 'trailing_pe': trailing_pe, 'forward_pe': forward_pe,
            'trailing_eps': trailing_eps, 'industry_pe': industry_pe,
            'fair_value_pe': fair_value_pe, 'upside_pe': upside_pe,
            'enterprise_value': enterprise_value, 'ebitda': ebitda,
            'market_cap': market_cap, 'current_ev_ebitda': current_ev_ebitda,
            'industry_ev_ebitda': industry_ev_ebitda,
            'fair_value_ev': fair_value_ev, 'upside_ev': upside_ev,
            'pb_ratio': pb_ratio, 'ps_ratio': ps_ratio,
            'book_value': book_value, 'revenue': revenue,
            'net_debt': (info.get('totalDebt', 0) or 0) - (info.get('totalCash', 0) or 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'roe': info.get('returnOnEquity', 0),
            'profit_margin': info.get('profitMargins', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0),
            '52w_low': info.get('fiftyTwoWeekLow', 0),
            'sector': sector,
            'csv_category': csv_category
        }
    except:
        return None

# ============================================================================
# CHART FUNCTIONS - [KEEP ALL FROM BEFORE]
# ============================================================================

def create_gauge_chart(upside_pe, upside_ev):
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'indicator'}, {'type': 'indicator'}]], horizontal_spacing=0.15)
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta", value=upside_pe if upside_pe else 0,
        number={'suffix': "%", 'font': {'size': 36, 'color': '#e2e8f0', 'family': 'Inter'}},
        delta={'reference': 0, 'increasing': {'color': "#34d399"}, 'decreasing': {'color': "#f87171"}},
        title={'text': "PE Multiple", 'font': {'size': 16, 'color': '#a78bfa', 'family': 'Inter'}},
        gauge={'axis': {'range': [-50, 50], 'tickwidth': 2, 'tickcolor': "#64748b", 'tickfont': {'color': '#94a3b8'}},
            'bar': {'color': "#7c3aed", 'thickness': 0.75}, 'bgcolor': "#1e1b4b", 'borderwidth': 2, 'bordercolor': "#4c1d95",
            'steps': [{'range': [-50, -20], 'color': '#7f1d1d'}, {'range': [-20, 0], 'color': '#78350f'},
                {'range': [0, 20], 'color': '#14532d'}, {'range': [20, 50], 'color': '#065f46'}],
            'threshold': {'line': {'color': "#f472b6", 'width': 4}, 'thickness': 0.8, 'value': 0}}
    ), row=1, col=1)
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta", value=upside_ev if upside_ev else 0,
        number={'suffix': "%", 'font': {'size': 36, 'color': '#e2e8f0', 'family': 'Inter'}},
        delta={'reference': 0, 'increasing': {'color': "#34d399"}, 'decreasing': {'color': "#f87171"}},
        title={'text': "EV/EBITDA", 'font': {'size': 16, 'color': '#a78bfa', 'family': 'Inter'}},
        gauge={'axis': {'range': [-50, 50], 'tickwidth': 2, 'tickcolor': "#64748b", 'tickfont': {'color': '#94a3b8'}},
            'bar': {'color': "#ec4899", 'thickness': 0.75}, 'bgcolor': "#1e1b4b", 'borderwidth': 2, 'bordercolor': "#4c1d95",
            'steps': [{'range': [-50, -20], 'color': '#7f1d1d'}, {'range': [-20, 0], 'color': '#78350f'},
                {'range': [0, 20], 'color': '#14532d'}, {'range': [20, 50], 'color': '#065f46'}],
            'threshold': {'line': {'color': "#f472b6", 'width': 4}, 'thickness': 0.8, 'value': 0}}
    ), row=1, col=2)
    fig.update_layout(height=350, margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family': 'Inter', 'color': '#e2e8f0'})
    return fig

def create_valuation_comparison_chart(vals):
    categories, current_vals, fair_vals = [], [], []
    if vals['fair_value_pe']:
        categories.append('PE Multiple')
        current_vals.append(vals['price'])
        fair_vals.append(vals['fair_value_pe'])
    if vals['fair_value_ev']:
        categories.append('EV/EBITDA')
        current_vals.append(vals['price'])
        fair_vals.append(vals['fair_value_ev'])
    if not categories:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Current Price', x=categories, y=current_vals,
        marker=dict(color='#6366f1', line=dict(color='#818cf8', width=2)),
        text=[f'₹{v:,.2f}' for v in current_vals], textposition='outside',
        textfont=dict(size=14, color='#e2e8f0', family='JetBrains Mono')))
    colors = ['#34d399' if fv > cv else '#f87171' for fv, cv in zip(fair_vals, current_vals)]
    fig.add_trace(go.Bar(name='Fair Value', x=categories, y=fair_vals,
        marker=dict(color=colors, line=dict(color=['#6ee7b7' if c == '#34d399' else '#fca5a5' for c in colors], width=2)),
        text=[f'₹{v:,.2f}' for v in fair_vals], textposition='outside',
        textfont=dict(size=14, color='#e2e8f0', family='JetBrains Mono')))
    fig.update_layout(barmode='group', height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12, color='#e2e8f0'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=14, color='#e2e8f0')),
        xaxis=dict(showgrid=False, showline=True, linecolor='#4c1d95', tickfont=dict(size=14, color='#e2e8f0')),
        yaxis=dict(showgrid=True, gridcolor='rgba(167, 139, 250, 0.2)', showline=False, tickprefix='₹', tickfont=dict(size=12, color='#a78bfa')),
        margin=dict(l=60, r=40, t=60, b=40))
    return fig

def create_52week_range_display(vals):
    low, high, current = vals.get('52w_low', 0), vals.get('52w_high', 0), vals.get('price', 0)
    if not all([low, high, current]) or high <= low:
        return None
    position = ((current - low) / (high - low)) * 100
    position = max(0, min(100, position))
    return f'''<div class="range-card"><div class="range-header">
        <div class="range-low">52W Low: ₹{low:,.2f}</div><div class="range-high">52W High: ₹{high:,.2f}</div></div>
        <div class="range-bar-container"><div class="range-bar-fill" style="width: {position}%;"></div></div>
        <div class="range-current">Current Price: <span>₹{current:,.2f}</span> ({position:.1f}% of range)</div></div>'''

def create_radar_chart(vals):
    categories = ['PE Ratio', 'EV/EBITDA', 'P/B Ratio', 'Profit Margin', 'ROE']
    pe_score = max(0, min(100, 100 - (vals['trailing_pe'] / vals['industry_pe'] * 50))) if vals['trailing_pe'] and vals['industry_pe'] else 50
    ev_score = max(0, min(100, 100 - (vals['current_ev_ebitda'] / vals['industry_ev_ebitda'] * 50))) if vals['current_ev_ebitda'] and vals['industry_ev_ebitda'] else 50
    pb_score = max(0, min(100, 100 - (vals['pb_ratio'] * 20))) if vals['pb_ratio'] else 50
    margin_score = vals['profit_margin'] * 500 if vals['profit_margin'] else 50
    roe_score = vals['roe'] * 300 if vals['roe'] else 50
    values = [max(0, min(100, v)) for v in [pe_score, ev_score, pb_score, margin_score, roe_score]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself',
        fillcolor='rgba(124, 58, 237, 0.3)', line=dict(color='#a78bfa', width=2), marker=dict(size=8, color='#c4b5fd')))
    fig.add_trace(go.Scatterpolar(r=[50]*6, theta=categories + [categories[0]], fill='none',
        line=dict(color='#6366f1', width=2, dash='dash'), name='Benchmark'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False,
        gridcolor='rgba(167, 139, 250, 0.2)', linecolor='rgba(167, 139, 250, 0.3)'),
        angularaxis=dict(tickfont=dict(size=12, color='#a78bfa'), linecolor='rgba(167, 139, 250, 0.3)',
        gridcolor='rgba(167, 139, 250, 0.2)'), bgcolor='rgba(0,0,0,0)'),
        showlegend=False, height=350, margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
    return fig

# ============================================================================
# PDF REPORT - [KEEP FROM BEFORE]
# ============================================================================
def create_pdf_report(company, ticker, sector, vals):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=28, 
        textColor=colors.HexColor('#7c3aed'), alignment=TA_CENTER, spaceAfter=20)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=12,
        textColor=colors.HexColor('#64748b'), alignment=TA_CENTER, spaceAfter=30)
    story = []
    story.append(Paragraph("NYZTrade Pro", title_style))
    story.append(Paragraph("Professional Valuation Report", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>{company}</b>", styles['Heading2']))
    story.append(Paragraph(f"Ticker: {ticker} | Sector: {sector}", styles['Normal']))
    story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 30))
    ups = [v for v in [vals['upside_pe'], vals['upside_ev']] if v is not None]
    avg_up = np.mean(ups) if ups else 0
    fairs = [v for v in [vals['fair_value_pe'], vals['fair_value_ev']] if v is not None]
    avg_fair = np.mean(fairs) if fairs else vals['price']
    fair_data = [['Metric', 'Value'], ['Fair Value', f"₹ {avg_fair:,.2f}"],
        ['Current Price', f"₹ {vals['price']:,.2f}"], ['Potential Upside', f"{avg_up:+.2f}%"]]
    fair_table = Table(fair_data, colWidths=[3*inch, 2.5*inch])
    fair_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7c3aed')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')), ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('TOPPADDING', (0, 1), (-1, -1), 8), ('BOTTOMPADDING', (0, 1), (-1, -1), 8)]))
    story.append(fair_table)
    story.append(Spacer(1, 25))
    story.append(Paragraph("<b>Valuation Metrics</b>", styles['Heading3']))
    metrics_data = [['Metric', 'Current', 'Industry Benchmark'],
        ['PE Ratio', f"{vals['trailing_pe']:.2f}x" if vals['trailing_pe'] else 'N/A', f"{vals['industry_pe']:.2f}x"],
        ['EV/EBITDA', f"{vals['current_ev_ebitda']:.2f}x" if vals['current_ev_ebitda'] else 'N/A', f"{vals['industry_ev_ebitda']:.2f}x"],
        ['P/B Ratio', f"{vals['pb_ratio']:.2f}x" if vals['pb_ratio'] else 'N/A', '-'],
        ['EPS', f"₹ {vals['trailing_eps']:.2f}" if vals['trailing_eps'] else 'N/A', '-'],
        ['Market Cap', f"₹ {vals['market_cap']/10000000:,.0f} Cr", '-']]
    metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch, 2*inch])
    metrics_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('FONTSIZE', (0, 1), (-1, -1), 10), ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])]))
    story.append(metrics_table)
    story.append(Spacer(1, 30))
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8,
        textColor=colors.HexColor('#94a3b8'), spaceBefore=20)
    story.append(Paragraph("<b>DISCLAIMER:</b> This report is for educational purposes only and does not constitute financial advice. "
        "Always consult a qualified financial advisor before making investment decisions. Past performance is not indicative of future results.",
        disclaimer_style))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================================
# CSV TEMPLATE GENERATOR
# ============================================================================
def create_csv_template():
    template_data = {
        'Ticker': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
        'Name': ['Reliance Industries', 'Tata Consultancy Services', 'Infosys', 'HDFC Bank', 'ICICI Bank'],
        'Category Name': ['Energy', 'Technology', 'Technology', 'Financial Services', 'Financial Services']
    }
    return pd.DataFrame(template_data).to_csv(index=False)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.markdown('''
<div class="main-header">
    <h1>STOCK VALUATION PRO</h1>
    <p>📊 10,500+ Indian Stocks | Professional Multi-Factor Analysis | Real-Time Data</p>
</div>
''', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔐 Account")
    st.markdown(f"**User:** {st.session_state.get('authenticated_user', 'Guest').title()}")
    
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📤 Upload Stock Universe")
    
    template_csv = create_csv_template()
    st.download_button("📋 Download CSV Template", data=template_csv,
        file_name="stocks_template.csv", mime="text/csv", use_container_width=True, help="Download sample template")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Your Stock CSV", type=['csv'],
        help="Upload CSV with: Ticker, Name, Category Name")
    
    if uploaded_file is not None:
        new_df = load_stocks_from_uploaded_file(uploaded_file)
        if new_df is not None and not new_df.empty:
            st.session_state.stocks_df = new_df
            STOCKS_DF = new_df
            st.success(f"✅ Loaded {len(new_df):,} stocks!")
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📈 Stock Selection")
    
    all_stocks = get_all_stocks_from_df(STOCKS_DF)
    
    if not all_stocks:
        st.error("❌ No stocks available. Upload CSV.")
        st.stop()
    
    st.markdown(f'<div class="stock-count">📊 {len(all_stocks):,} Stocks Available</div>', unsafe_allow_html=True)
    
    categories = get_categories_from_df(STOCKS_DF)
    category = st.selectbox("🏷️ Category", ["📋 All Stocks"] + categories, help="Filter by category")
    
    search = st.text_input("🔍 Search", placeholder="Company name or ticker...", help="Search stocks")
    
    if search:
        search_results = search_stock_in_df(STOCKS_DF, search)
        filtered = {t: info['name'] for t, info in search_results.items()}
    elif category == "📋 All Stocks":
        filtered = dict(list(all_stocks.items())[:1000])  # Limit display
    else:
        category_stocks = get_stocks_by_category_from_df(STOCKS_DF)
        filtered = category_stocks.get(category, {})
    
    if filtered:
        options = sorted([f"{n} ({t})" for t, n in filtered.items()])
        selected = st.selectbox("🎯 Select Stock", options, help="Choose stock")
        ticker = selected.split("(")[1].strip(")")
    else:
        ticker = None
        st.warning("⚠️ No stocks found")
    
    st.markdown("---")
    custom = st.text_input("✏️ Custom Ticker", placeholder="e.g., TATAMOTORS.NS", help="Enter ticker manually")
    
    st.markdown("---")
    analyze_clicked = st.button("🚀 ANALYZE STOCK", use_container_width=True, type="primary")

# Main content
if analyze_clicked:
    st.session_state.analyze = custom.upper() if custom else ticker

if 'analyze' in st.session_state and st.session_state.analyze:
    t = st.session_state.analyze
    
    with st.spinner(f"🔄 Fetching data for {t}..."):
        info, error = fetch_stock_data(t)
    
    if error or not info:
        st.error(f"❌ Error: {error if error else 'Failed to fetch data'}")
        st.markdown('<div class="warning-box"><strong>Tips:</strong><br>• Verify ticker (e.g., RELIANCE.NS)<br>• Check internet<br>• Try again later</div>', unsafe_allow_html=True)
        st.stop()
    
    vals = calculate_valuations(info, t, STOCKS_DF)
    if not vals:
        st.error("❌ Unable to calculate valuations")
        st.stop()
    
    company = info.get('longName', t)
    sector = vals.get('sector', 'N/A')
    csv_category = vals.get('csv_category', 'N/A')
    industry = info.get('industry', 'N/A')
    
    st.markdown(f'''
    <div class="company-header">
        <h2 class="company-name">{company}</h2>
        <div class="company-meta">
            <span class="meta-badge">🏷️ {t}</span>
            <span class="meta-badge">🏢 {sector}</span>
            <span class="meta-badge">📊 {csv_category}</span>
            <span class="meta-badge">🏭 {industry}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    ups = [v for v in [vals['upside_pe'], vals['upside_ev']] if v is not None]
    avg_up = np.mean(ups) if ups else 0
    fairs = [v for v in [vals['fair_value_pe'], vals['fair_value_ev']] if v is not None]
    avg_fair = np.mean(fairs) if fairs else vals['price']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f'''
        <div class="fair-value-card">
            <div class="fair-value-label">📊 Calculated Fair Value</div>
            <div class="fair-value-amount">₹{avg_fair:,.2f}</div>
            <div class="current-price">Current Price: ₹{vals["price"]:,.2f}</div>
            <div class="upside-badge">{"📈" if avg_up > 0 else "📉"} {avg_up:+.2f}% Potential</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        if avg_up > 25:
            rec_class, rec_text, rec_icon = "rec-strong-buy", "Highly Undervalued", "🚀"
        elif avg_up > 15:
            rec_class, rec_text, rec_icon = "rec-buy", "Undervalued", "✅"
        elif avg_up > 0:
            rec_class, rec_text, rec_icon = "rec-buy", "Fairly Valued", "📥"
        elif avg_up > -10:
            rec_class, rec_text, rec_icon = "rec-hold", "HOLD", "⏸️"
        else:
            rec_class, rec_text, rec_icon = "rec-avoid", "Overvalued", "⚠️"
        
        st.markdown(f'''
        <div class="rec-container">
            <div class="{rec_class}">
                {rec_icon} {rec_text}
                <div class="rec-subtitle">Expected Return: {avg_up:+.2f}%</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        pdf = create_pdf_report(company, t, sector, vals)
        st.download_button("📥 Download PDF Report", data=pdf,
            file_name=f"NYZTrade_{t}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf", use_container_width=True)
    
    st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-icon">💰</div><div class="metric-value">₹{vals["price"]:,.2f}</div><div class="metric-label">Current Price</div></div>', unsafe_allow_html=True)
    with m2:
        pe_val = f"{vals['trailing_pe']:.2f}x" if vals['trailing_pe'] else "N/A"
        st.markdown(f'<div class="metric-card"><div class="metric-icon">📈</div><div class="metric-value">{pe_val}</div><div class="metric-label">PE Ratio</div></div>', unsafe_allow_html=True)
    with m3:
        eps_val = f"₹{vals['trailing_eps']:.2f}" if vals['trailing_eps'] else "N/A"
        st.markdown(f'<div class="metric-card"><div class="metric-icon">💵</div><div class="metric-value">{eps_val}</div><div class="metric-label">EPS (TTM)</div></div>', unsafe_allow_html=True)
    with m4:
        mcap_val = f"₹{vals['market_cap']/10000000:,.0f}Cr"
        st.markdown(f'<div class="metric-card"><div class="metric-icon">🏦</div><div class="metric-value">{mcap_val}</div><div class="metric-label">Market Cap</div></div>', unsafe_allow_html=True)
    with m5:
        ev_ebitda = f"{vals['current_ev_ebitda']:.2f}x" if vals['current_ev_ebitda'] else "N/A"
        st.markdown(f'<div class="metric-card"><div class="metric-icon">📊</div><div class="metric-value">{ev_ebitda}</div><div class="metric-label">EV/EBITDA</div></div>', unsafe_allow_html=True)
    with m6:
        pb_val = f"{vals['pb_ratio']:.2f}x" if vals['pb_ratio'] else "N/A"
        st.markdown(f'<div class="metric-card"><div class="metric-icon">📚</div><div class="metric-value">{pb_val}</div><div class="metric-label">P/B Ratio</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown('<div class="section-header">🎯 Valuation Gauges</div>', unsafe_allow_html=True)
        if vals['upside_pe'] is not None or vals['upside_ev'] is not None:
            fig_gauge = create_gauge_chart(vals['upside_pe'] if vals['upside_pe'] else 0,
                vals['upside_ev'] if vals['upside_ev'] else 0)
            st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("Insufficient data")
    
    with chart_col2:
        st.markdown('<div class="section-header">📊 Price vs Fair Value</div>', unsafe_allow_html=True)
        fig_bar = create_valuation_comparison_chart(vals)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Insufficient data")
    
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        st.markdown('<div class="section-header">📍 52-Week Range</div>', unsafe_allow_html=True)
        range_html = create_52week_range_display(vals)
        if range_html:
            st.markdown(range_html, unsafe_allow_html=True)
        else:
            st.info("52-week data not available")
    
    with chart_col4:
        st.markdown('<div class="section-header">🎯 Metrics Radar</div>', unsafe_allow_html=True)
        fig_radar = create_radar_chart(vals)
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Valuation Breakdown</div>', unsafe_allow_html=True)
    
    val_col1, val_col2 = st.columns(2)
    
    with val_col1:
        if vals['fair_value_pe'] and vals['trailing_pe']:
            upside_color = '#34d399' if vals['upside_pe'] and vals['upside_pe'] > 0 else '#f87171'
            fair_color = '#34d399' if vals['fair_value_pe'] > vals['price'] else '#f87171'
            st.markdown(f'''
            <div class="valuation-method">
                <div class="method-title">📈 PE Multiple Method</div>
                <div class="method-row"><span class="method-label">Current PE</span><span class="method-value">{vals['trailing_pe']:.2f}x</span></div>
                <div class="method-row"><span class="method-label">Industry PE</span><span class="method-value">{vals['industry_pe']:.2f}x</span></div>
                <div class="method-row"><span class="method-label">EPS (TTM)</span><span class="method-value">₹{vals['trailing_eps']:.2f}</span></div>
                <div class="method-row"><span class="method-label">Fair Value (PE)</span><span class="method-value" style="color: {fair_color}">₹{vals['fair_value_pe']:,.2f}</span></div>
                <div class="method-row"><span class="method-label">Upside (PE)</span><span class="method-value" style="color: {upside_color}">{vals['upside_pe']:+.2f}%</span></div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.info("PE valuation not available")
    
    with val_col2:
        if vals['fair_value_ev'] and vals['current_ev_ebitda']:
            upside_color_ev = '#34d399' if vals['upside_ev'] and vals['upside_ev'] > 0 else '#f87171'
            fair_color_ev = '#34d399' if vals['fair_value_ev'] > vals['price'] else '#f87171'
            st.markdown(f'''
            <div class="valuation-method">
                <div class="method-title">💼 EV/EBITDA Method</div>
                <div class="method-row"><span class="method-label">Current EV/EBITDA</span><span class="method-value">{vals['current_ev_ebitda']:.2f}x</span></div>
                <div class="method-row"><span class="method-label">Industry EV/EBITDA</span><span class="method-value">{vals['industry_ev_ebitda']:.2f}x</span></div>
                <div class="method-row"><span class="method-label">EBITDA</span><span class="method-value">₹{vals['ebitda']/10000000:,.0f} Cr</span></div>
                <div class="method-row"><span class="method-label">Fair Value (EV)</span><span class="method-value" style="color: {fair_color_ev}">₹{vals['fair_value_ev']:,.2f}</span></div>
                <div class="method-row"><span class="method-label">Upside (EV)</span><span class="method-value" style="color: {upside_color_ev}">{vals['upside_ev']:+.2f}%</span></div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.info("EV/EBITDA valuation not available")
    
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Complete Financial Summary</div>', unsafe_allow_html=True)
    
    financial_data = pd.DataFrame({
        'Metric': ['Current Price', 'Market Cap', 'Enterprise Value', 'PE Ratio (TTM)', 'Forward PE', 'EV/EBITDA',
            'P/B Ratio', 'P/S Ratio', 'EPS (TTM)', 'EBITDA', 'Book Value', 'Net Debt',
            '52W High', '52W Low', 'Beta', 'Dividend Yield', 'ROE', 'Profit Margin'],
        'Value': [
            f"₹{vals['price']:,.2f}", f"₹{vals['market_cap']/10000000:,.0f} Cr",
            f"₹{vals['enterprise_value']/10000000:,.0f} Cr" if vals['enterprise_value'] else 'N/A',
            f"{vals['trailing_pe']:.2f}x" if vals['trailing_pe'] else 'N/A',
            f"{vals['forward_pe']:.2f}x" if vals['forward_pe'] else 'N/A',
            f"{vals['current_ev_ebitda']:.2f}x" if vals['current_ev_ebitda'] else 'N/A',
            f"{vals['pb_ratio']:.2f}x" if vals['pb_ratio'] else 'N/A',
            f"{vals['ps_ratio']:.2f}x" if vals['ps_ratio'] else 'N/A',
            f"₹{vals['trailing_eps']:.2f}" if vals['trailing_eps'] else 'N/A',
            f"₹{vals['ebitda']/10000000:,.0f} Cr" if vals['ebitda'] else 'N/A',
            f"₹{vals['book_value']:.2f}" if vals['book_value'] else 'N/A',
            f"₹{vals['net_debt']/10000000:,.0f} Cr",
            f"₹{vals['52w_high']:,.2f}" if vals['52w_high'] else 'N/A',
            f"₹{vals['52w_low']:,.2f}" if vals['52w_low'] else 'N/A',
            f"{vals['beta']:.2f}" if vals['beta'] else 'N/A',
            f"{vals['dividend_yield']*100:.2f}%" if vals['dividend_yield'] else 'N/A',
            f"{vals['roe']*100:.2f}%" if vals['roe'] else 'N/A',
            f"{vals['profit_margin']*100:.2f}%" if vals['profit_margin'] else 'N/A'
        ]
    })
    
    st.dataframe(financial_data, use_container_width=True, hide_index=True,
        column_config={"Metric": st.column_config.TextColumn("📊 Metric", width="medium"),
            "Value": st.column_config.TextColumn("📈 Value", width="medium")})

else:
    st.markdown('''
    <div class="info-box">
        <h3>👋 Welcome to NYZTrade Pro Valuation</h3>
        <p>Select a stock from <strong>10,500+ Indian stocks</strong> and click <strong>ANALYZE STOCK</strong>!</p>
        <br><strong>Features:</strong><ul>
            <li>📊 10,500+ Indian stocks (NSE & BSE)</li>
            <li>📈 Multi-factor valuation (PE, EV/EBITDA, P/B)</li>
            <li>📉 Professional charts</li>
            <li>📥 PDF reports</li>
            <li>🎯 Buy/Sell recommendations</li>
            <li>📋 Complete financial metrics</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('''
<div class="footer">
    <p><strong>NYZTrade Pro</strong> | 10,500+ Indian Stocks Database</p>
    <p style="font-size: 0.8rem; color: #94a3b8;">
        ⚠️ Disclaimer: Educational purposes only. Consult a qualified advisor.
    </p>
</div>
''', unsafe_allow_html=True)
