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
# CSV-BASED STOCK UNIVERSE LOADER - FIXED STATE MANAGEMENT
# ============================================================================

@st.cache_data(ttl=86400)
def load_stocks_from_csv(csv_path='stocks_universe_full.csv'):
    """Load stocks from CSV file - Optimized for large datasets"""
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
    """Convert DataFrame to dictionary grouped by category - with limit"""
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
    """Search for stocks - optimized"""
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

# Industry Benchmarks
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
# [KEEP ALL CSS STYLING - SAME AS BEFORE]
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
.main-header h1 {
    font-size: 2.8rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #00d4ff, #7c3aed, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    position: relative; z-index: 1;
}
.main-header p { font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; position: relative; z-index: 1; color: #e2e8f0; }
.company-header {
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 50%, #8b5cf6 100%);
    border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(124, 58, 237, 0.3);
}
.company-name { font-size: 2rem; font-weight: 700; color: #ffffff !important; margin: 0; }
.company-meta { display: flex; gap: 1rem; margin-top: 0.8rem; flex-wrap: wrap; }
.meta-badge {
    background: rgba(255, 255, 255, 0.2); padding: 0.5rem 1rem; border-radius: 25px;
    font-size: 0.9rem; color: #ffffff !important; font-weight: 500;
}
.fair-value-card {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%);
    padding: 2rem; border-radius: 20px; text-align: center; color: white; margin: 1.5rem 0;
    box-shadow: 0 20px 40px rgba(124, 58, 237, 0.3);
}
.fair-value-amount { font-size: 3rem; font-weight: 700; margin: 0.5rem 0; font-family: 'JetBrains Mono', monospace; color: #ffffff; }
.rec-strong-buy { background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%); color: white !important; padding: 1.5rem 2rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; }
.rec-buy { background: linear-gradient(135deg, #0d9488 0%, #14b8a6 50%, #2dd4bf 100%); color: white !important; padding: 1.5rem 2rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; }
.rec-hold { background: linear-gradient(135deg, #d97706 0%, #f59e0b 50%, #fbbf24 100%); color: white !important; padding: 1.5rem 2rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; }
.rec-avoid { background: linear-gradient(135deg, #dc2626 0%, #ef4444 50%, #f87171 100%); color: white !important; padding: 1.5rem 2rem; border-radius: 16px; text-align: center; font-size: 1.5rem; font-weight: 700; }
.metric-card {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 16px; padding: 1.5rem; text-align: center;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3); transition: all 0.3s ease;
}
.metric-value { font-size: 1.5rem; font-weight: 700; color: #ffffff !important; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.8rem; color: rgba(255,255,255,0.85) !important; text-transform: uppercase; }
.section-header { font-size: 1.4rem; font-weight: 600; color: #a78bfa; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 3px solid #7c3aed; }
.stock-count { background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); color: white !important; padding: 0.8rem 1.2rem; border-radius: 12px; text-align: center; margin: 1rem 0; font-weight: 600; }
.valuation-method { background: linear-gradient(135deg, #312e81 0%, #3730a3 100%); border-radius: 16px; padding: 1.5rem; margin: 1rem 0; border-left: 4px solid #a78bfa; }
.method-row { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); }
.method-value { font-weight: 600; color: #ffffff !important; font-family: 'JetBrains Mono', monospace; }
.info-box { background: linear-gradient(135deg, #312e81 0%, #3730a3 100%); border: 1px solid #6366f1; border-radius: 12px; padding: 1.5rem 2rem; color: #ffffff !important; margin: 1rem 0; }
.footer { text-align: center; padding: 2rem; color: #a78bfa; font-size: 0.9rem; margin-top: 3rem; }
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
            <p style='color: rgba(255,255,255,0.7);'>Professional Stock Valuation Platform</p>
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
        st.error("❌ Incorrect credentials")
        return False
    return True

if not check_password():
    st.stop()

# ============================================================================
# LOAD STOCK DATA - FIXED STATE MANAGEMENT
# ============================================================================
if 'stocks_df' not in st.session_state:
    # Try to load from default CSV file
    initial_df = load_stocks_from_csv('stocks_universe_full.csv')
    if initial_df.empty:
        # If no default file, create empty dataframe
        st.session_state.stocks_df = pd.DataFrame(columns=['ticker', 'name', 'category'])
    else:
        st.session_state.stocks_df = initial_df
        st.sidebar.success(f"✅ Loaded {len(initial_df):,} stocks from default database")

# Always use the latest DataFrame from session state
STOCKS_DF = st.session_state.stocks_df

# ============================================================================
# UTILITY FUNCTIONS - [SAME AS BEFORE, SHORTENED FOR SPACE]
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
        
        return {
            'price': price, 'trailing_pe': trailing_pe, 'forward_pe': forward_pe,
            'trailing_eps': trailing_eps, 'industry_pe': industry_pe,
            'fair_value_pe': fair_value_pe, 'upside_pe': upside_pe,
            'enterprise_value': enterprise_value, 'ebitda': ebitda,
            'market_cap': market_cap, 'current_ev_ebitda': current_ev_ebitda,
            'industry_ev_ebitda': industry_ev_ebitda,
            'fair_value_ev': fair_value_ev, 'upside_ev': upside_ev,
            'pb_ratio': price / book_value if book_value and book_value > 0 else None,
            'ps_ratio': market_cap / revenue if revenue and revenue > 0 else None,
            'book_value': book_value, 'revenue': revenue,
            'net_debt': (info.get('totalDebt', 0) or 0) - (info.get('totalCash', 0) or 0),
            'dividend_yield': info.get('dividendYield', 0), 'beta': info.get('beta', 0),
            'roe': info.get('returnOnEquity', 0), 'profit_margin': info.get('profitMargins', 0),
            '52w_high': info.get('fiftyTwoWeekHigh', 0), '52w_low': info.get('fiftyTwoWeekLow', 0),
            'sector': sector, 'csv_category': csv_category
        }
    except:
        return None

# [CHART FUNCTIONS - KEEP ALL FROM PREVIOUS VERSION]
def create_gauge_chart(upside_pe, upside_ev):
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'indicator'}, {'type': 'indicator'}]])
    fig.add_trace(go.Indicator(mode="gauge+number+delta", value=upside_pe if upside_pe else 0,
        number={'suffix': "%", 'font': {'size': 36, 'color': '#e2e8f0'}},
        title={'text': "PE Multiple", 'font': {'size': 16, 'color': '#a78bfa'}},
        gauge={'axis': {'range': [-50, 50]}, 'bar': {'color': "#7c3aed"},
            'steps': [{'range': [-50, -20], 'color': '#7f1d1d'}, {'range': [-20, 0], 'color': '#78350f'},
                {'range': [0, 20], 'color': '#14532d'}, {'range': [20, 50], 'color': '#065f46'}]}), row=1, col=1)
    fig.add_trace(go.Indicator(mode="gauge+number+delta", value=upside_ev if upside_ev else 0,
        number={'suffix': "%", 'font': {'size': 36, 'color': '#e2e8f0'}},
        title={'text': "EV/EBITDA", 'font': {'size': 16, 'color': '#a78bfa'}},
        gauge={'axis': {'range': [-50, 50]}, 'bar': {'color': "#ec4899"},
            'steps': [{'range': [-50, -20], 'color': '#7f1d1d'}, {'range': [-20, 0], 'color': '#78350f'},
                {'range': [0, 20], 'color': '#14532d'}, {'range': [20, 50], 'color': '#065f46'}]}), row=1, col=2)
    fig.update_layout(height=350, margin=dict(l=30, r=30, t=60, b=30), paper_bgcolor='rgba(0,0,0,0)')
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
        marker=dict(color='#6366f1'), text=[f'₹{v:,.2f}' for v in current_vals], textposition='outside'))
    colors = ['#34d399' if fv > cv else '#f87171' for fv, cv in zip(fair_vals, current_vals)]
    fig.add_trace(go.Bar(name='Fair Value', x=categories, y=fair_vals,
        marker=dict(color=colors), text=[f'₹{v:,.2f}' for v in fair_vals], textposition='outside'))
    fig.update_layout(barmode='group', height=400, paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_52week_range_display(vals):
    low, high, current = vals.get('52w_low', 0), vals.get('52w_high', 0), vals.get('price', 0)
    if not all([low, high, current]) or high <= low:
        return None
    position = ((current - low) / (high - low)) * 100
    return f'<div style="background: #312e81; padding: 1.5rem; border-radius: 16px;"><div style="display: flex; justify-content: space-between;"><span style="color: #34d399;">52W Low: ₹{low:,.2f}</span><span style="color: #f87171;">52W High: ₹{high:,.2f}</span></div><div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 20px; margin: 1rem 0;"><div style="width: {position}%; height: 100%; background: linear-gradient(90deg, #34d399, #fbbf24, #f87171); border-radius: 10px;"></div></div><div style="text-align: center; color: white;">Current: ₹{current:,.2f} ({position:.1f}%)</div></div>'

def create_radar_chart(vals):
    categories = ['PE', 'EV/EBITDA', 'P/B', 'Margin', 'ROE']
    values = [
        max(0, min(100, 100 - (vals['trailing_pe'] / vals['industry_pe'] * 50))) if vals['trailing_pe'] and vals['industry_pe'] else 50,
        max(0, min(100, 100 - (vals['current_ev_ebitda'] / vals['industry_ev_ebitda'] * 50))) if vals['current_ev_ebitda'] and vals['industry_ev_ebitda'] else 50,
        max(0, min(100, 100 - (vals['pb_ratio'] * 20))) if vals['pb_ratio'] else 50,
        vals['profit_margin'] * 500 if vals['profit_margin'] else 50,
        vals['roe'] * 300 if vals['roe'] else 50
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself'))
    fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_pdf_report(company, ticker, sector, vals):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("NYZTrade Pro", styles['Title']), Paragraph(f"{company} - {ticker}", styles['Heading2'])]
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_csv_template():
    return pd.DataFrame({
        'Ticker': ['RELIANCE.NS', 'TCS.NS'],
        'Name': ['Reliance', 'TCS'],
        'Category Name': ['Energy', 'Technology']
    }).to_csv(index=False)

# ============================================================================
# MAIN APPLICATION
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
    st.download_button("📋 Download CSV Template", data=template_csv,
        file_name="stocks_template.csv", mime="text/csv", use_container_width=True)
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Your Stock CSV", type=['csv'])
    
    # FIXED: Proper state update after upload with immediate rerun
    if uploaded_file is not None:
        # Check if this is a new file (not previously processed)
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != file_id:
            new_df = load_stocks_from_uploaded_file(uploaded_file)
            if new_df is not None and not new_df.empty:
                # Update session state
                st.session_state.stocks_df = new_df
                st.session_state.last_uploaded_file = file_id
                # Clear cache to ensure fresh data
                st.cache_data.clear()
                st.success(f"✅ Loaded {len(new_df):,} stocks!")
                # Force immediate rerun to update UI
                st.rerun()
    
    st.markdown("---")
    st.markdown("### 📈 Stock Selection")
    
    # CRITICAL FIX: Always get fresh data from session state
    STOCKS_DF = st.session_state.stocks_df if 'stocks_df' in st.session_state else pd.DataFrame(columns=['ticker', 'name', 'category'])
    all_stocks = get_all_stocks_from_df(STOCKS_DF)
    
    if not all_stocks or len(all_stocks) == 0:
        st.error("❌ No stocks available. Upload CSV above.")
        ticker = None
    else:
        st.markdown(f'<div class="stock-count">📊 {len(all_stocks):,} Stocks Available</div>', unsafe_allow_html=True)
        
        categories = get_categories_from_df(STOCKS_DF)
        category = st.selectbox("🏷️ Category", ["📋 All Stocks"] + categories, key="category_select")
        
        search = st.text_input("🔍 Search", placeholder="Company name or ticker...", key="stock_search")
        
        if search:
            search_results = search_stock_in_df(STOCKS_DF, search)
            filtered = {t: info['name'] for t, info in search_results.items()}
        elif category == "📋 All Stocks":
            filtered = dict(list(all_stocks.items())[:1000])
        else:
            category_stocks = get_stocks_by_category_from_df(STOCKS_DF)
            filtered = category_stocks.get(category, {})
        
        if filtered and len(filtered) > 0:
            options = sorted([f"{n} ({t})" for t, n in filtered.items()])
            selected = st.selectbox("🎯 Select Stock", options, key="stock_select")
            ticker = selected.split("(")[1].strip(")")
        else:
            ticker = None
            st.warning("⚠️ No stocks in this category. Try 'All Stocks' or search.")
    
    st.markdown("---")
    custom = st.text_input("✏️ Custom Ticker", placeholder="e.g., TATAMOTORS.NS")
    
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
        st.stop()
    
    # Use the latest STOCKS_DF from session state
    STOCKS_DF = st.session_state.stocks_df
    vals = calculate_valuations(info, t, STOCKS_DF)
    if not vals:
        st.error("❌ Unable to calculate valuations")
        st.stop()
    
    company = info.get('longName', t)
    sector = vals.get('sector', 'N/A')
    csv_category = vals.get('csv_category', 'N/A')
    
    st.markdown(f'''
    <div class="company-header">
        <h2 class="company-name">{company}</h2>
        <div class="company-meta">
            <span class="meta-badge">🏷️ {t}</span>
            <span class="meta-badge">🏢 {sector}</span>
            <span class="meta-badge">📊 {csv_category}</span>
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
            <div style="font-size: 0.9rem; opacity: 0.9;">📊 Calculated Fair Value</div>
            <div class="fair-value-amount">₹{avg_fair:,.2f}</div>
            <div style="font-size: 1rem; opacity: 0.85;">Current Price: ₹{vals["price"]:,.2f}</div>
            <div style="display: inline-block; background: rgba(255,255,255,0.2); padding: 0.5rem 1.5rem; border-radius: 30px; margin-top: 1rem; font-weight: 600; font-size: 1.2rem;">
                {"📈" if avg_up > 0 else "📉"} {avg_up:+.2f}% Potential
            </div>
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
        
        st.markdown(f'<div class="{rec_class}">{rec_icon} {rec_text}<div style="font-size: 1rem; margin-top: 0.3rem;">Expected Return: {avg_up:+.2f}%</div></div>', unsafe_allow_html=True)
        
        pdf = create_pdf_report(company, t, sector, vals)
        st.download_button("📥 Download PDF", data=pdf,
            file_name=f"NYZTrade_{t}.pdf", mime="application/pdf", use_container_width=True)
    
    st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2rem;">💰</div><div class="metric-value">₹{vals["price"]:,.2f}</div><div class="metric-label">Price</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2rem;">📈</div><div class="metric-value">{vals["trailing_pe"]:.2f}x" if vals["trailing_pe"] else "N/A</div><div class="metric-label">PE Ratio</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2rem;">💵</div><div class="metric-value">₹{vals["trailing_eps"]:.2f}" if vals["trailing_eps"] else "N/A</div><div class="metric-label">EPS</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2rem;">🏦</div><div class="metric-value">₹{vals["market_cap"]/10000000:,.0f}Cr</div><div class="metric-label">Market Cap</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2rem;">📊</div><div class="metric-value">{vals["current_ev_ebitda"]:.2f}x" if vals["current_ev_ebitda"] else "N/A</div><div class="metric-label">EV/EBITDA</div></div>', unsafe_allow_html=True)
    with m6:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2rem;">📚</div><div class="metric-value">{vals["pb_ratio"]:.2f}x" if vals["pb_ratio"] else "N/A</div><div class="metric-label">P/B</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown('<div class="section-header">🎯 Valuation Gauges</div>', unsafe_allow_html=True)
        if vals['upside_pe'] or vals['upside_ev']:
            st.plotly_chart(create_gauge_chart(vals['upside_pe'] or 0, vals['upside_ev'] or 0), use_container_width=True)
    with chart_col2:
        st.markdown('<div class="section-header">📊 Price vs Fair Value</div>', unsafe_allow_html=True)
        fig_bar = create_valuation_comparison_chart(vals)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
    
    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.markdown('<div class="section-header">📍 52-Week Range</div>', unsafe_allow_html=True)
        range_html = create_52week_range_display(vals)
        if range_html:
            st.markdown(range_html, unsafe_allow_html=True)
    with chart_col4:
        st.markdown('<div class="section-header">🎯 Metrics Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(create_radar_chart(vals), use_container_width=True)

else:
    st.markdown('''
    <div class="info-box">
        <h3>👋 Welcome to NYZTrade Pro</h3>
        <p>Select a stock and click <strong>ANALYZE STOCK</strong>!</p>
        <strong>Features:</strong><ul>
            <li>📊 Professional valuation analysis</li>
            <li>📈 Real-time market data</li>
            <li>📥 PDF reports</li>
            <li>🎯 Buy/Sell recommendations</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('<div class="footer"><p><strong>NYZTrade Pro</strong> | Professional Stock Valuation</p></div>', unsafe_allow_html=True)
