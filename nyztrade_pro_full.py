import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
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
# RATE LIMITING PROTECTION
# ============================================================================

def init_rate_limiter():
    """Initialize rate limiter in session state"""
    if 'api_calls' not in st.session_state:
        st.session_state.api_calls = []
    if 'rate_limit_cooldown' not in st.session_state:
        st.session_state.rate_limit_cooldown = None

def check_rate_limit():
    """Check if we're within rate limits"""
    current_time = time.time()
    
    # Clean old calls (older than 1 minute)
    st.session_state.api_calls = [
        call_time for call_time in st.session_state.api_calls 
        if current_time - call_time < 60
    ]
    
    # Check cooldown period
    if st.session_state.rate_limit_cooldown:
        cooldown_end = st.session_state.rate_limit_cooldown
        if current_time < cooldown_end:
            remaining = int(cooldown_end - current_time)
            return False, remaining
        else:
            st.session_state.rate_limit_cooldown = None
    
    # Check if too many calls in last minute
    if len(st.session_state.api_calls) >= 5:  # Max 5 calls per minute
        return False, 60
    
    return True, 0

def record_api_call():
    """Record an API call"""
    st.session_state.api_calls.append(time.time())

def set_rate_limit_cooldown(seconds=60):
    """Set a cooldown period after rate limit"""
    st.session_state.rate_limit_cooldown = time.time() + seconds

# ============================================================================
# CSV-BASED STOCK UNIVERSE LOADER
# ============================================================================

@st.cache_data(ttl=86400)
def load_stocks_from_csv(csv_path='stocks_universe_full.csv'):
    """Load stocks from CSV file"""
    try:
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = df.columns.str.strip()
        column_mapping = {
            'Ticker': 'ticker', 'ticker': 'ticker', 'TICKER': 'ticker',
            'Name': 'name', 'name': 'name', 'NAME': 'name',
            'Category Name': 'category', 'category': 'category', 'Category': 'category'
        }
        df = df.rename(columns=column_mapping)
        
        if 'ticker' not in df.columns or 'name' not in df.columns:
            raise ValueError("CSV must contain 'Ticker' and 'Name' columns")
        
        if 'category' not in df.columns:
            df['category'] = '📊 All Stocks'
        
        df = df.dropna(subset=['ticker', 'name'])
        df['ticker'] = df['ticker'].str.strip().str.upper()
        df['name'] = df['name'].str.strip()
        df['category'] = df['category'].str.strip()
        df['category'] = df['category'].fillna('📊 Uncategorized')
        df.loc[df['category'] == '', 'category'] = '📊 Uncategorized'
        df = df.drop_duplicates(subset=['ticker'])
        df = df[df['ticker'].str.contains('.NS|.BO', na=False, regex=True)]
        
        return df
        
    except FileNotFoundError:
        return pd.DataFrame(columns=['ticker', 'name', 'category'])
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        return pd.DataFrame(columns=['ticker', 'name', 'category'])

def load_stocks_from_uploaded_file(uploaded_file):
    """Load stocks from uploaded file"""
    try:
        df = pd.read_csv(uploaded_file, dtype=str)
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
        
        df = df.dropna(subset=['ticker', 'name'])
        df['ticker'] = df['ticker'].str.strip().str.upper()
        df['name'] = df['name'].str.strip()
        df['category'] = df['category'].str.strip().fillna('📊 Uncategorized')
        df = df.drop_duplicates(subset=['ticker'])
        df = df[df['ticker'].str.contains('.NS|.BO', na=False, regex=True)]
        
        return df
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def get_stocks_by_category_from_df(df):
    """Convert DataFrame to dictionary grouped by category"""
    if df is None or df.empty:
        return {}
    stocks_dict = {}
    for category in sorted(df['category'].unique())[:50]:
        category_df = df[df['category'] == category]
        stocks_dict[category] = dict(zip(category_df['ticker'], category_df['name']))
    return stocks_dict

def get_all_stocks_from_df(df):
    """Get all stocks as a dictionary"""
    if df is None or df.empty:
        return {}
    return dict(zip(df['ticker'], df['name']))

def get_categories_from_df(df):
    """Get list of top categories"""
    if df is None or df.empty:
        return []
    top_cats = df['category'].value_counts().head(50).index.tolist()
    return sorted(top_cats)

def search_stock_in_df(df, query):
    """Search for stocks"""
    if df is None or df.empty or not query:
        return {}
    
    query_upper = query.upper()
    mask = (df['ticker'].str.contains(query_upper, na=False, case=False) | 
            df['name'].str.upper().str.contains(query_upper, na=False))
    
    results = {}
    for _, row in df[mask].head(100).iterrows():
        results[row['ticker']] = {
            'name': row['name'],
            'category': row['category']
        }
    return results

def get_category_for_ticker(df, ticker):
    """Get category for ticker"""
    if df is None or df.empty:
        return 'Default'
    result = df[df['ticker'] == ticker]
    return result.iloc[0]['category'] if not result.empty else 'Default'

# Category Mappings
CATEGORY_TO_SECTOR = {
    'Information Technology Services': 'Technology', 'IT Services': 'Technology',
    'Software': 'Technology', 'Application Software': 'Technology',
    'Banks': 'Financial Services', 'Finance': 'Financial Services',
    'Money Center Banks': 'Financial Services', 'NBFC': 'Financial Services',
    'Industrial Metals & Minerals': 'Industrials', 'General Contractors': 'Industrials',
    'Diversified Machinery': 'Industrials', 'Steel & Iron': 'Industrials',
    'Auto Parts': 'Industrials', 'Auto Manufacturers - Major': 'Industrials',
    'Oil & Gas Drilling & Exploration': 'Energy', 'Oil & Gas': 'Energy',
    'Chemicals': 'Basic Materials', 'Specialty Chemicals': 'Basic Materials',
    'Drug Manufacturers - Major': 'Healthcare', 'Drugs - Generic': 'Healthcare',
    'Textile Industrial': 'Consumer Cyclical', 'Food - Major Diversified': 'Consumer Defensive',
    'Telecommunications': 'Communication Services', 'Wireless Communications': 'Communication Services',
    'Real Estate': 'Real Estate', 'REIT': 'Real Estate',
}

def map_category_to_sector(category):
    """Map category to sector"""
    if not category or pd.isna(category):
        return 'Default'
    category = str(category).strip()
    if category in CATEGORY_TO_SECTOR:
        return CATEGORY_TO_SECTOR[category]
    return 'Default'

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
    'Default': {'pe': 20, 'ev_ebitda': 12}
}

def get_benchmark(sector):
    return INDUSTRY_BENCHMARKS.get(sector, INDUSTRY_BENCHMARKS['Default'])

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; background-color: #0a0a0a; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 3rem; border-radius: 20px; text-align: center; color: white;
    margin-bottom: 2rem; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}
.main-header h1 {
    font-size: 2.8rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.company-header {
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 50%, #8b5cf6 100%);
    border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
}
.company-name { font-size: 2rem; font-weight: 700; color: #ffffff !important; }
.meta-badge { background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 25px; color: #ffffff !important; margin-right: 0.5rem; display: inline-block; }
.fair-value-card {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
    padding: 2rem; border-radius: 20px; text-align: center; color: white; margin: 1.5rem 0;
}
.fair-value-amount { font-size: 3rem; font-weight: 700; color: #ffffff !important; }
.metric-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 16px; padding: 1.5rem; text-align: center;
}
.metric-value { font-size: 1.5rem; font-weight: 700; color: #ffffff !important; }
.rec-strong-buy { background: linear-gradient(135deg, #059669 0%, #10b981 100%); color: white !important; padding: 1.5rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; }
.rec-buy { background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%); color: white !important; padding: 1.5rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; }
.rec-hold { background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); color: white !important; padding: 1.5rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; }
.rec-avoid { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); color: white !important; padding: 1.5rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; }
.stock-count { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white !important; padding: 0.8rem 1.2rem; border-radius: 12px; text-align: center; margin: 1rem 0; font-weight: 600; }
.rate-limit-warning {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    padding: 1.5rem 2rem; border-radius: 16px; color: white !important;
    margin: 1rem 0; text-align: center; font-weight: 600;
    box-shadow: 0 10px 30px rgba(220, 38, 38, 0.3);
}
.cooldown-timer {
    font-size: 2rem; font-weight: 700; margin-top: 0.5rem;
    font-family: monospace;
}
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
                    padding: 4rem; border-radius: 24px; text-align: center; margin: 2rem auto; max-width: 500px;'>
            <h1 style='color: white; font-size: 2.5rem;'>
                <span style='background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                    NYZTrade Pro
                </span>
            </h1>
            <p style='color: rgba(255,255,255,0.7);'>Professional Stock Valuation</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("👤 Username", key="username")
            st.text_input("🔒 Password", type="password", key="password")
            st.button("🚀 Login", on_click=password_entered, use_container_width=True, type="primary")
            st.info("💡 Demo: demo/demo123")
        return False
    elif not st.session_state["password_correct"]:
        st.error("❌ Incorrect credentials")
        return False
    return True

if not check_password():
    st.stop()

# Initialize rate limiter
init_rate_limiter()

# ============================================================================
# LOAD STOCK DATA
# ============================================================================
if 'stocks_df' not in st.session_state:
    initial_df = load_stocks_from_csv('stocks_universe_full.csv')
    if initial_df.empty:
        st.session_state.stocks_df = pd.DataFrame(columns=['ticker', 'name', 'category'])
    else:
        st.session_state.stocks_df = initial_df

STOCKS_DF = st.session_state.stocks_df

# ============================================================================
# DATA FETCHING WITH RATE LIMITING
# ============================================================================

@st.cache_data(ttl=7200, show_spinner=False)
def fetch_stock_data_cached(ticker):
    """Cached version of stock data fetch"""
    try:
        time.sleep(1)  # Mandatory delay between requests
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or len(info) < 5:
            return None, "Unable to fetch data"
        return info, None
    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "too many" in error_str or "rate" in error_str:
            return None, "RATE_LIMIT"
        return None, str(e)[:100]

def fetch_stock_data(ticker):
    """Fetch stock data with rate limiting protection"""
    # Check rate limit
    allowed, cooldown = check_rate_limit()
    if not allowed:
        return None, f"RATE_LIMIT:{cooldown}"
    
    # Record this API call
    record_api_call()
    
    # Fetch data
    info, error = fetch_stock_data_cached(ticker)
    
    # Handle rate limit error
    if error == "RATE_LIMIT":
        set_rate_limit_cooldown(120)  # 2 minute cooldown
        return None, "RATE_LIMIT:120"
    
    return info, error

def calculate_valuations(info, ticker_symbol, stocks_df):
    """Calculate valuations"""
    try:
        price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
        trailing_pe = info.get('trailingPE', 0)
        trailing_eps = info.get('trailingEps', 0)
        enterprise_value = info.get('enterpriseValue', 0)
        ebitda = info.get('ebitda', 0)
        market_cap = info.get('marketCap', 0)
        shares = info.get('sharesOutstanding', 1)
        
        yf_sector = info.get('sector', None)
        csv_category = get_category_for_ticker(stocks_df, ticker_symbol)
        mapped_sector = map_category_to_sector(csv_category)
        sector = yf_sector if yf_sector else mapped_sector
        
        benchmark = get_benchmark(sector)
        industry_pe = benchmark['pe']
        industry_ev_ebitda = benchmark['ev_ebitda']
        
        blended_pe = industry_pe
        fair_value_pe = trailing_eps * blended_pe if trailing_eps else None
        upside_pe = ((fair_value_pe - price) / price * 100) if fair_value_pe and price else None
        
        if ebitda and ebitda > 0:
            fair_ev = ebitda * industry_ev_ebitda
            net_debt = (info.get('totalDebt', 0) or 0) - (info.get('totalCash', 0) or 0)
            fair_mcap = fair_ev - net_debt
            fair_value_ev = fair_mcap / shares if shares else None
            upside_ev = ((fair_value_ev - price) / price * 100) if fair_value_ev and price else None
        else:
            fair_value_ev = None
            upside_ev = None
        
        return {
            'price': price, 'trailing_pe': trailing_pe, 'trailing_eps': trailing_eps,
            'industry_pe': industry_pe, 'fair_value_pe': fair_value_pe, 'upside_pe': upside_pe,
            'enterprise_value': enterprise_value, 'ebitda': ebitda, 'market_cap': market_cap,
            'industry_ev_ebitda': industry_ev_ebitda, 'fair_value_ev': fair_value_ev,
            'upside_ev': upside_ev, 'sector': sector, 'csv_category': csv_category,
            '52w_high': info.get('fiftyTwoWeekHigh', 0), '52w_low': info.get('fiftyTwoWeekLow', 0)
        }
    except:
        return None

def create_csv_template():
    return pd.DataFrame({
        'Ticker': ['RELIANCE.NS', 'TCS.NS'],
        'Name': ['Reliance', 'TCS'],
        'Category Name': ['Energy', 'Technology']
    }).to_csv(index=False)

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown('<div class="main-header"><h1>STOCK VALUATION PRO</h1><p>📊 Professional Multi-Factor Analysis | Real-Time Data</p></div>', unsafe_allow_html=True)

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
    st.download_button("📋 Download Template", data=template_csv,
        file_name="template.csv", mime="text/csv", use_container_width=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != file_id:
            new_df = load_stocks_from_uploaded_file(uploaded_file)
            if new_df is not None and not new_df.empty:
                st.session_state.stocks_df = new_df
                st.session_state.last_uploaded_file = file_id
                st.cache_data.clear()
                st.success(f"✅ Loaded {len(new_df):,} stocks!")
                st.rerun()
    
    st.markdown("---")
    st.markdown("### 📈 Stock Selection")
    
    STOCKS_DF = st.session_state.stocks_df if 'stocks_df' in st.session_state else pd.DataFrame(columns=['ticker', 'name', 'category'])
    all_stocks = get_all_stocks_from_df(STOCKS_DF)
    
    if not all_stocks:
        st.error("❌ No stocks. Upload CSV.")
        ticker = None
    else:
        st.markdown(f'<div class="stock-count">📊 {len(all_stocks):,} Stocks Available</div>', unsafe_allow_html=True)
        
        categories = get_categories_from_df(STOCKS_DF)
        category = st.selectbox("🏷️ Category", ["📋 All Stocks"] + categories, key="cat")
        
        search = st.text_input("🔍 Search", placeholder="Name or ticker...", key="search")
        
        if search:
            results = search_stock_in_df(STOCKS_DF, search)
            filtered = {t: info['name'] for t, info in results.items()}
        elif category == "📋 All Stocks":
            filtered = dict(list(all_stocks.items())[:1000])
        else:
            cat_stocks = get_stocks_by_category_from_df(STOCKS_DF)
            filtered = cat_stocks.get(category, {})
        
        if filtered:
            options = sorted([f"{n} ({t})" for t, n in filtered.items()])
            selected = st.selectbox("🎯 Select Stock", options, key="stock")
            ticker = selected.split("(")[1].strip(")")
        else:
            ticker = None
    
    st.markdown("---")
    custom = st.text_input("✏️ Custom Ticker", placeholder="e.g., TATAMOTORS.NS", key="custom")
    
    st.markdown("---")
    
    # Rate limit warning
    allowed, cooldown = check_rate_limit()
    if not allowed:
        st.markdown(f'''
        <div class="rate-limit-warning">
            ⚠️ RATE LIMIT ACTIVE
            <div class="cooldown-timer">Wait {cooldown}s</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">Too many requests. Please wait.</div>
        </div>
        ''', unsafe_allow_html=True)
        analyze_clicked = False
    else:
        analyze_clicked = st.button("🚀 ANALYZE STOCK", use_container_width=True, type="primary")

# Main content
if analyze_clicked:
    st.session_state.analyze = custom.upper() if custom else ticker

if 'analyze' in st.session_state and st.session_state.analyze:
    t = st.session_state.analyze
    
    # Check rate limit again before fetching
    allowed, cooldown = check_rate_limit()
    if not allowed:
        st.markdown(f'''
        <div class="rate-limit-warning">
            ⚠️ RATE LIMIT PROTECTION
            <div class="cooldown-timer">{cooldown} seconds</div>
            <div style="font-size: 1rem; margin-top: 1rem;">
                You're making requests too quickly. Yahoo Finance has rate limits to prevent abuse.
                <br><br>
                <strong>Please wait {cooldown} seconds before trying again.</strong>
                <br><br>
                💡 Tip: Data is cached for 2 hours. Previously analyzed stocks load instantly!
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.stop()
    
    with st.spinner(f"🔄 Fetching data for {t}..."):
        info, error = fetch_stock_data(t)
    
    if error:
        if error.startswith("RATE_LIMIT"):
            cooldown_time = int(error.split(":")[1])
            st.markdown(f'''
            <div class="rate-limit-warning">
                ⚠️ YAHOO FINANCE RATE LIMIT
                <div class="cooldown-timer">{cooldown_time} seconds</div>
                <div style="font-size: 1rem; margin-top: 1rem;">
                    Yahoo Finance is temporarily blocking requests due to high traffic.
                    <br><br>
                    <strong>Solutions:</strong><br>
                    • Wait {cooldown_time} seconds and try again<br>
                    • Previously analyzed stocks are cached (try another stock first)<br>
                    • Try during off-peak hours for better reliability<br>
                    <br>
                    💡 This is a Yahoo Finance limitation, not an app issue.
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.error(f"❌ Error: {error}")
            st.info("💡 Tips: Check ticker format (STOCK.NS or STOCK.BO), internet connection, or try another stock.")
        st.stop()
    
    if not info:
        st.error(f"❌ Unable to fetch data for {t}")
        st.stop()
    
    STOCKS_DF = st.session_state.stocks_df
    vals = calculate_valuations(info, t, STOCKS_DF)
    if not vals:
        st.error("❌ Unable to calculate valuations")
        st.stop()
    
    company = info.get('longName', t)
    sector = vals.get('sector', 'N/A')
    
    st.markdown(f'''
    <div class="company-header">
        <h2 class="company-name">{company}</h2>
        <div><span class="meta-badge">🏷️ {t}</span><span class="meta-badge">🏢 {sector}</span></div>
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
            <div style="font-size: 0.9rem; opacity: 0.9;">📊 Fair Value</div>
            <div class="fair-value-amount">₹{avg_fair:,.2f}</div>
            <div style="opacity: 0.85;">Current: ₹{vals["price"]:,.2f}</div>
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1.5rem; border-radius: 30px; margin-top: 1rem; display: inline-block;">
                {"📈" if avg_up > 0 else "📉"} {avg_up:+.2f}%
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        if avg_up > 25:
            rec_class, rec_text = "rec-strong-buy", "🚀 Strong Buy"
        elif avg_up > 15:
            rec_class, rec_text = "rec-buy", "✅ Buy"
        elif avg_up > 0:
            rec_class, rec_text = "rec-buy", "📥 Buy"
        elif avg_up > -10:
            rec_class, rec_text = "rec-hold", "⏸️ Hold"
        else:
            rec_class, rec_text = "rec-avoid", "⚠️ Avoid"
        
        st.markdown(f'<div class="{rec_class}">{rec_text}<div style="font-size: 1rem; margin-top: 0.3rem;">Return: {avg_up:+.2f}%</div></div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2rem;">💰</div><div class="metric-value">₹{vals["price"]:,.2f}</div><div style="font-size: 0.8rem; color: #e0e0e0;">Price</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2rem;">📈</div><div class="metric-value">{vals["trailing_pe"]:.2f}x" if vals["trailing_pe"] else "N/A</div><div style="font-size: 0.8rem; color: #e0e0e0;">PE Ratio</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2rem;">🏦</div><div class="metric-value">₹{vals["market_cap"]/10000000:,.0f}Cr</div><div style="font-size: 0.8rem; color: #e0e0e0;">Market Cap</div></div>', unsafe_allow_html=True)
    with m4:
        if vals['52w_high'] and vals['52w_low']:
            range_text = f"{vals['52w_low']:.0f}-{vals['52w_high']:.0f}"
        else:
            range_text = "N/A"
        st.markdown(f'<div class="metric-card"><div style="font-size: 2rem;">📊</div><div class="metric-value" style="font-size: 1.2rem;">{range_text}</div><div style="font-size: 0.8rem; color: #e0e0e0;">52W Range</div></div>', unsafe_allow_html=True)

else:
    st.markdown('''
    <div style="background: linear-gradient(135deg, #312e81 0%, #3730a3 100%); border-radius: 16px; padding: 2rem; color: white; margin: 2rem 0;">
        <h3 style="color: #a78bfa; margin-bottom: 1rem;">👋 Welcome to NYZTrade Pro</h3>
        <p>Select a stock and click <strong>ANALYZE STOCK</strong>!</p>
        <br><strong>Features:</strong>
        <ul style="margin-top: 1rem;">
            <li>📊 Professional valuation analysis</li>
            <li>📈 Real-time market data</li>
            <li>🎯 Buy/Sell recommendations</li>
            <li>⚡ Smart rate limiting protection</li>
        </ul>
        <br>
        <div style="background: rgba(255,193,7,0.2); padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107;">
            <strong>💡 Tip:</strong> Data is cached for 2 hours. Previously analyzed stocks load instantly without API calls!
        </div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('<div style="text-align: center; padding: 2rem; color: #a78bfa;"><strong>NYZTrade Pro</strong> | Protected by Smart Rate Limiting</div>', unsafe_allow_html=True)
