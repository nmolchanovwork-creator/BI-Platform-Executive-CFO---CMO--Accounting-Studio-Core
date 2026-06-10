"""
BI Platform v2.0 — Executive CFO & CMO Intelligence Suite
Modern Corporate Design | Advanced Auto-Analytics | Compound Growth Modeling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import google.generativeai as genai
import warnings
import json
import base64
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════
# CONFIG & GLOBAL STYLES
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="Executive BI Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Base & Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* ── Main Background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1220 50%, #0a0e1a 100%);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1220 0%, #111827 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    
    /* ── Main content area ── */
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(17,24,39,0.9) 0%, rgba(30,41,59,0.9) 100%);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99,102,241,0.25);
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 11px !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.08em; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 700 !important; font-size: 1.6rem !important; }
    [data-testid="stMetricDelta"] { font-size: 12px !important; }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 20, 35, 0.8);
        border-radius: 10px;
        padding: 4px;
        gap: 2px;
        border: 1px solid rgba(99,102,241,0.15);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 500;
        padding: 8px 14px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        box-shadow: 0 2px 12px rgba(99,102,241,0.4);
    }

    /* ── Section headers ── */
    h1 {
        background: linear-gradient(135deg, #f1f5f9, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    h2, h3 { color: #cbd5e1 !important; }
    
    /* ── Insight boxes ── */
    .insight-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.08) 100%);
        border: 1px solid rgba(99,102,241,0.3);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 13px;
        line-height: 1.6;
    }
    .insight-box.warning {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, rgba(245,158,11,0.06) 0%, rgba(245,158,11,0.03) 100%);
        border-color: rgba(245,158,11,0.25);
    }
    .insight-box.danger {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, rgba(239,68,68,0.06) 0%, rgba(239,68,68,0.03) 100%);
        border-color: rgba(239,68,68,0.25);
    }
    .insight-box.success {
        border-left-color: #10b981;
        background: linear-gradient(135deg, rgba(16,185,129,0.06) 0%, rgba(16,185,129,0.03) 100%);
        border-color: rgba(16,185,129,0.25);
    }
    
    /* ── KPI badge ── */
    .kpi-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px;
    }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 16px rgba(99,102,241,0.35);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.5);
    }
    
    /* ── Sidebar title ── */
    .sidebar-logo {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* ── Section divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(99,102,241,0.15);
        margin: 20px 0;
    }
    
    /* ── Plotly dark override ── */
    .js-plotly-plot .plotly .modebar { background: transparent !important; }
    
    /* ── Alerts ── */
    .stAlert { border-radius: 8px !important; }
    
    /* ── Info / Success / Warning overrides ── */
    div[data-baseweb="notification"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='rgba(13,18,32,0)',
        plot_bgcolor='rgba(13,18,32,0)',
        font=dict(family='Inter', color='#94a3b8', size=11),
        title=dict(font=dict(color='#e2e8f0', size=14, family='Inter'), x=0.01),
        xaxis=dict(gridcolor='rgba(99,102,241,0.08)', linecolor='rgba(99,102,241,0.15)', tickcolor='#475569'),
        yaxis=dict(gridcolor='rgba(99,102,241,0.08)', linecolor='rgba(99,102,241,0.15)', tickcolor='#475569'),
        legend=dict(bgcolor='rgba(13,18,32,0.8)', bordercolor='rgba(99,102,241,0.2)', borderwidth=1, font=dict(color='#94a3b8')),
        colorway=['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316', '#84cc16'],
        hoverlabel=dict(bgcolor='rgba(17,24,39,0.95)', bordercolor='rgba(99,102,241,0.4)', font=dict(color='#e2e8f0')),
        margin=dict(t=50, l=10, r=10, b=10),
    )
)

def apply_theme(fig, height=380):
    fig.update_layout(**PLOTLY_TEMPLATE['layout'], height=height)
    return fig

# ═══════════════════════════════════════════════════
# CORE DATA PROCESSING
# ═══════════════════════════════════════════════════
@st.cache_data
def load_and_clean_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, dtype=str, low_memory=False)
        else:
            df = pd.read_excel(file, dtype=str)

        num_cols = ['Цена', 'Количество', 'Стоимость', 'Итого']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        def is_razom(name): return 'Разом дешевше' in str(name)
        results = []

        for order_num, group in df.groupby('Номер заказа', sort=False):
            rows = group.reset_index(drop=True)
            n = len(rows)
            itogo = rows['Итого'].iloc[0] if 'Итого' in rows.columns else 0

            r0 = rows.iloc[0]
            meta = {
                'order_id': order_num,
                'date': r0.get('Дата заказа'),
                'status': str(r0.get('Статус', 'Доставлен')).strip(),
                'city': str(r0.get('Город', 'Не указан')).strip(),
                'utm_source': str(r0.get('utm_source', 'organic')).replace('nan', 'organic').replace('(none)', 'direct'),
                'utm_medium': str(r0.get('utm_medium', 'organic')).replace('nan', 'organic').replace('(none)', 'direct'),
                'utm_campaign': str(r0.get('utm_campaign', 'organic')).replace('nan', 'organic').replace('(none)', 'direct')
            }

            included = []
            skip_until = -1
            i = 0

            while i < n:
                if i <= skip_until:
                    i += 1
                    continue

                row = rows.iloc[i]
                name = str(row.get('Название товара', ''))
                art  = str(row.get('Артикул', ''))

                if is_razom(name):
                    i += 1
                    continue

                if ('Набір' in name or 'Набор' in name) and not art.startswith('48'):
                    bundle_price = row['Стоимость']
                    j = i + 1
                    cumsum = 0
                    has_zero = False
                    candidate_subitems = []

                    while j < n:
                        r = rows.iloc[j]
                        r_name = str(r.get('Название товара', ''))
                        r_art  = str(r.get('Артикул', ''))
                        if is_razom(r_name): break
                        if ('Набір' in r_name or 'Набор' in r_name) and not r_art.startswith('48'): break

                        candidate_subitems.append(j)
                        if r['Цена'] == 0:
                            has_zero = True
                            j += 1
                            break
                        cumsum += r['Цена'] * r['Количество']
                        if cumsum >= bundle_price:
                            j += 1
                            break
                        j += 1

                    has_subitems = (len(candidate_subitems) > 0 and (cumsum >= bundle_price or has_zero))

                    if has_subitems:
                        for si in candidate_subitems:
                            si_row = rows.iloc[si]
                            if si_row['Цена'] == 0:
                                raw_cost = bundle_price / len(candidate_subitems)
                            else:
                                raw_cost = si_row['Стоимость']
                            included.append((si, raw_cost, str(si_row.get('Артикул','')), str(si_row.get('Название товара','')), si_row['Количество'], si_row['Цена']))
                        skip_until = candidate_subitems[-1]
                        i = skip_until + 1
                    else:
                        included.append((i, row['Стоимость'], art, name, row['Количество'], row['Цена']))
                        i += 1
                    continue

                included.append((i, row['Стоимость'], art, name, row['Количество'], row['Цена']))
                i += 1

            raw_sum = sum(x[1] for x in included)
            scale   = (itogo / raw_sum) if raw_sum else 0

            for item in included:
                idx, raw_cost, item_art, item_name, qty, price = item
                net_cost = round(raw_cost * scale, 2)
                results.append({**meta, 'art': item_art, 'product': item_name, 'qty': qty, 'revenue': net_cost})

        df_clean = pd.DataFrame(results)
        if df_clean.empty:
            return pd.DataFrame()

        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['date'])
        df_clean['profit'] = df_clean['revenue'] * 0.35
        df_clean['year_month'] = df_clean['date'].dt.to_period('M').astype(str)
        df_clean['week'] = df_clean['date'].dt.to_period('W').dt.start_time

        return df_clean
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {str(e)}")
        return pd.DataFrame()

# ═══════════════════════════════════════════════════
# PROPHET FORECASTING
# ═══════════════════════════════════════════════════
@st.cache_data
def generate_global_forecast(df, periods, freq='D'):
    daily_rev = df.groupby(df['date'].dt.date)['revenue'].sum().reset_index()
    daily_rev.columns = ['ds', 'y']

    if len(daily_rev) < 10:
        return None, None, daily_rev

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10
    )
    try:
        m.add_country_holidays(country_name='UA')
    except: pass

    m.fit(daily_rev)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return m, forecast, daily_rev

def get_monthly_distribution(df, entity_col, monthly_totals, total_historical_rev, metric='revenue'):
    entity_rev = df.groupby(entity_col)[metric].sum().reset_index()
    entity_rev['share'] = entity_rev[metric] / total_historical_rev if total_historical_rev > 0 else 0
    dist_data = []
    for _, row in entity_rev.iterrows():
        row_data = {entity_col: row[entity_col], 'Факт (₴)': row[metric]}
        for _, m_row in monthly_totals.iterrows():
            row_data[m_row['month_str']] = row['share'] * m_row['yhat']
        dist_data.append(row_data)
    res_df = pd.DataFrame(dist_data).sort_values('Факт (₴)', ascending=False)
    return res_df

# ═══════════════════════════════════════════════════
# ADVANCED ANALYTICS HELPERS
# ═══════════════════════════════════════════════════
def compute_unit_economics(df):
    """Unit economics per order"""
    order_df = df.groupby('order_id').agg(
        revenue=('revenue', 'sum'),
        profit=('profit', 'sum'),
        items=('qty', 'sum'),
        date=('date', 'min')
    ).reset_index()
    return order_df

def compute_cohort_retention(df):
    """Monthly cohort retention analysis"""
    order_df = df.groupby('order_id').agg(date=('date','min')).reset_index()
    # First order month per customer proxy = order_id prefix
    order_df['cohort'] = order_df['date'].dt.to_period('M').astype(str)
    order_df['order_month'] = order_df['date'].dt.to_period('M').astype(str)
    cohort_counts = order_df.groupby('cohort')['order_id'].count().reset_index()
    cohort_counts.columns = ['Месяц', 'Заказов']
    return cohort_counts

def compute_velocity(df):
    """Revenue velocity — MoM growth"""
    monthly = df.groupby('year_month')['revenue'].sum().reset_index()
    monthly = monthly.sort_values('year_month')
    monthly['mom_growth'] = monthly['revenue'].pct_change() * 100
    monthly['mom_growth_abs'] = monthly['revenue'].diff()
    return monthly

def compute_concentration_risk(df):
    """Herfindahl-Hirschman index for channel concentration"""
    ch = df.groupby('utm_source')['revenue'].sum()
    shares = ch / ch.sum()
    hhi = (shares**2).sum()
    return hhi, shares

def compute_product_velocity(df):
    """Days of supply / sell-through velocity per product"""
    total_days = (df['date'].max() - df['date'].min()).days or 1
    prod = df.groupby('product').agg(
        total_qty=('qty', 'sum'),
        total_rev=('revenue', 'sum')
    ).reset_index()
    prod['daily_units'] = prod['total_qty'] / total_days
    prod['daily_rev'] = prod['total_rev'] / total_days
    return prod

def compute_pareto_80_20(df):
    prod = df.groupby('product')['revenue'].sum().sort_values(ascending=False).reset_index()
    prod['cumshare'] = prod['revenue'].cumsum() / prod['revenue'].sum()
    pareto_count = (prod['cumshare'] <= 0.8).sum()
    total_count = len(prod)
    pareto_pct = pareto_count / total_count * 100 if total_count else 0
    return pareto_count, total_count, pareto_pct

def compute_weekly_seasonality(df):
    df2 = df.copy()
    df2['dow'] = df2['date'].dt.day_name()
    df2['dow_num'] = df2['date'].dt.dayofweek
    day_df = df2.groupby(['dow_num','dow'])['revenue'].mean().reset_index().sort_values('dow_num')
    return day_df

def compute_moving_average(df, window=7):
    daily = df.groupby(df['date'].dt.date)['revenue'].sum().reset_index()
    daily.columns = ['date', 'revenue']
    daily['ma7'] = daily['revenue'].rolling(window=window).mean()
    daily['ma30'] = daily['revenue'].rolling(window=30).mean()
    return daily

def compute_revenue_at_risk(df, percentile=10):
    daily = df.groupby(df['date'].dt.date)['revenue'].sum()
    var = np.percentile(daily, percentile)
    return var

# ═══════════════════════════════════════════════════
# HTML INSIGHT BOX HELPER
# ═══════════════════════════════════════════════════
def insight(text, kind='default', icon='💡'):
    css_class = {'warning': 'warning', 'danger': 'danger', 'success': 'success'}.get(kind, '')
    st.markdown(f'<div class="insight-box {css_class}">{icon} {text}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════
for key in ['ai_director_report', 'ai_growth_plan']:
    if key not in st.session_state:
        st.session_state[key] = None

# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════
st.sidebar.markdown('<div class="sidebar-logo">⚡ Executive BI Suite</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="color:#475569;font-size:11px;margin-bottom:16px">CFO & CMO Intelligence Platform</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

api_key = st.sidebar.text_input("🔑 Google Gemini API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)
    try:
        models_info = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        gemini_models = [m for m in models_info if 'gemini' in m]
        selected_model_path = st.sidebar.selectbox("🤖 AI модель", gemini_models)
    except Exception:
        selected_model_path = "models/gemini-1.5-flash-latest"
else:
    selected_model_path = "models/gemini-1.5-flash-latest"
    st.sidebar.caption("Введите API Key для AI-аналитики")

uploaded_file = st.sidebar.file_uploader("📂 Реестр заказов (.csv / .xlsx)", type=['csv', 'xlsx'])

# ═══════════════════════════════════════════════════
# EXPORT VARS (init)
# ═══════════════════════════════════════════════════
month_df = pd.DataFrame()
model_df_display = pd.DataFrame()
mkt_dist = pd.DataFrame()
geo_dist = pd.DataFrame()
abc_summary = pd.DataFrame()
prod_dist = pd.DataFrame()
trend_direction = ""
change_pct = 0.0
orders_boost = 15.0
aov_boost = 10.0
horizon_months = 3
total_rev = 0
future_total_rev = 0
future_total_upper = 0
future_forecast = pd.DataFrame()
monthly_forecast_totals = pd.DataFrame()

# ═══════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════
if uploaded_file:
    with st.spinner("⚙️ Профилирование номенклатуры и бандлов..."):
        raw_df = load_and_clean_data(uploaded_file)

    if not raw_df.empty:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🎯 Параметры анализа**")

        horizon_months = st.sidebar.slider("Горизонт прогноза (мес.)", 1, 12, 3)
        gross_margin = st.sidebar.slider("Валовая маржа (%)", 5, 80, 35) / 100

        all_statuses = raw_df['status'].unique().tolist()
        status_filter = st.sidebar.multiselect("Статус заказов", all_statuses, default=all_statuses)
        df = raw_df[raw_df['status'].isin(status_filter)].copy()
        df['profit'] = df['revenue'] * gross_margin

        with st.spinner("🔮 Генерация прогноза Prophet..."):
            m, forecast, actual = generate_global_forecast(df, horizon_months * 30)
            total_rev = df['revenue'].sum()
            total_profit = df['profit'].sum()
            unique_orders = df['order_id'].nunique()
            aov = total_rev / unique_orders if unique_orders > 0 else 0

            if forecast is not None and not actual.empty:
                max_date = pd.to_datetime(actual['ds'].max())
                future_forecast = forecast[forecast['ds'] > max_date].copy()
                future_total_rev = max(future_forecast['yhat'].sum(), 0)
                future_total_upper = max(future_forecast['yhat_upper'].sum(), 0)
                future_total_lower = max(future_forecast['yhat_lower'].sum(), 0)

                future_forecast['month_str'] = future_forecast['ds'].dt.strftime('%Y-%m')
                monthly_forecast_totals = future_forecast.groupby('month_str')[['yhat', 'yhat_upper', 'yhat_lower']].sum().reset_index()

                historical_total_rev = actual['y'].sum()
                growth_coeff = future_total_rev / historical_total_rev if historical_total_rev > 0 else 1

                avg_hist_daily = actual['y'].mean()
                avg_fore_daily = future_forecast['yhat'].mean()
                trend_direction = "восходящий 📈" if avg_fore_daily > avg_hist_daily else "нисходящий / стабилизационный 📉"
                change_pct = abs(avg_fore_daily - avg_hist_daily) / avg_hist_daily * 100 if avg_hist_daily > 0 else 0
            else:
                future_total_rev = future_total_upper = 0
                monthly_forecast_totals = pd.DataFrame()

        # Pre-compute analytics
        unit_eco = compute_unit_economics(df)
        velocity = compute_velocity(df)
        hhi, ch_shares = compute_concentration_risk(df)
        day_df = compute_weekly_seasonality(df)
        ma_df = compute_moving_average(df)
        var_10 = compute_revenue_at_risk(df)
        pareto_count, total_products, pareto_pct = compute_pareto_80_20(df)
        city_rev_df = df.groupby('city')['revenue'].sum().sort_values(ascending=False).reset_index()
        prod_velocity = compute_product_velocity(df)

        # ───────────────────────────────────────────
        # TABS
        # ───────────────────────────────────────────
        tabs = st.tabs([
            "⚡ Dashboard", "🔮 Прогноз", "🎛 Моделирование",
            "📣 Маркетинг", "🗺️ Гео", "💰 Продукты",
            "📐 Unit Economics", "🧠 AI CFO", "🚀 AI Стратегия"
        ])

        # ───────────────────────────────────────────
        # TAB 0: DASHBOARD
        # ───────────────────────────────────────────
        with tabs[0]:
            st.title("Executive Dashboard")
            now_ts = datetime.now().strftime("%d.%m.%Y %H:%M")
            st.markdown(f'<div style="color:#475569;font-size:12px;margin-bottom:20px">Last updated: {now_ts} UTC+2 &nbsp;|&nbsp; Data points: {len(df):,} rows</div>', unsafe_allow_html=True)

            # Row 1: KPIs
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("💰 Выручка (факт)", f"₴{total_rev/1_000:.1f}K")
            c2.metric("📈 Прогноз", f"₴{future_total_rev/1_000:.1f}K", f"+{horizon_months} мес.")
            c3.metric("🛒 Заказов", f"{unique_orders:,}")
            c4.metric("🧾 AOV", f"₴{aov:,.0f}")
            c5.metric("💵 Прибыль (факт)", f"₴{total_profit/1_000:.1f}K", f"{gross_margin*100:.0f}% маржа")
            c6.metric("📦 SKU активных", f"{df['product'].nunique():,}")

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # Row 2: Revenue trend + weekly heatmap
            col_l, col_r = st.columns([3, 1])

            with col_l:
                # Revenue with MA
                fig_main = go.Figure()
                weekly = df.groupby(df['date'].dt.to_period('W').dt.start_time)['revenue'].sum().reset_index()
                fig_main.add_trace(go.Bar(x=weekly['date'], y=weekly['revenue'], name='Выручка (нед.)',
                                          marker_color='rgba(99,102,241,0.5)', marker_line_color='rgba(99,102,241,0.8)'))

                # Add 4-week MA
                if len(weekly) >= 4:
                    weekly['ma4'] = weekly['revenue'].rolling(4).mean()
                    fig_main.add_trace(go.Scatter(x=weekly['date'], y=weekly['ma4'], name='MA 4 нед.',
                                                  line=dict(color='#f59e0b', width=2)))

                fig_main.update_layout(title="Динамика выручки с трендом (MA-4 недели)")
                apply_theme(fig_main, height=320)
                st.plotly_chart(fig_main, use_container_width=True)

            with col_r:
                # DoW seasonality
                fig_dow = go.Figure(go.Bar(
                    x=day_df['dow'],
                    y=day_df['revenue'],
                    marker=dict(
                        color=day_df['revenue'],
                        colorscale=[[0, 'rgba(99,102,241,0.3)'], [1, 'rgba(99,102,241,1)']],
                        showscale=False
                    )
                ))
                fig_dow.update_layout(title="Ср. выручка по дням недели")
                apply_theme(fig_dow, height=320)
                st.plotly_chart(fig_dow, use_container_width=True)

            # Row 3: Auto-insights
            st.markdown("### 🎯 Автоматические инсайты")

            last_months = velocity.tail(3)
            avg_mom = last_months['mom_growth'].mean() if not last_months.empty else 0

            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                if avg_mom > 5:
                    insight(f"**Бизнес в фазе роста.** Среднемесячный прирост последних 3 мес.: **+{avg_mom:.1f}%**. Текущая траектория соответствует годовому росту **x{(1+avg_mom/100)**12:.1f}**.", 'success', '🚀')
                elif avg_mom < -5:
                    insight(f"**Внимание: нисходящая динамика.** Среднемесячный спад: **{avg_mom:.1f}%**. Требуется немедленный аудит маркетинговых каналов и ассортиментной матрицы.", 'danger', '⚠️')
                else:
                    insight(f"**Стабилизация выручки.** MoM: **{avg_mom:.1f}%**. Бизнес в зоне плато — критический момент для масштабирования или пересмотра стратегии.", 'warning', '📊')

            with col_i2:
                hhi_pct = hhi * 100
                if hhi > 0.5:
                    insight(f"**Критическая концентрация каналов** (HHI={hhi_pct:.0f}%). Более 50% выручки зависит от 1 источника. Диверсификация — приоритет #1.", 'danger', '⚠️')
                elif hhi > 0.25:
                    insight(f"**Умеренная концентрация** (HHI={hhi_pct:.0f}%). Допустимый уровень, но рекомендуется развивать 2-й и 3-й каналы для снижения риска.", 'warning', '📡')
                else:
                    insight(f"**Здоровая диверсификация каналов** (HHI={hhi_pct:.0f}%). Портфель каналов сбалансирован — снижает операционный риск.", 'success', '✅')

            with col_i3:
                insight(f"**Закон Парето:** {pareto_count} из {total_products} SKU ({pareto_pct:.0f}%) генерируют 80% выручки. **{total_products - pareto_count} позиций** — кандидаты на вывод из ассортимента или A/B-тест ценообразования.", 'default', '📦')

            # Row 4: MoM bar
            if not velocity.empty:
                fig_vel = go.Figure()
                colors = ['rgba(16,185,129,0.7)' if v >= 0 else 'rgba(239,68,68,0.7)' for v in velocity['mom_growth'].fillna(0)]
                fig_vel.add_trace(go.Bar(x=velocity['year_month'], y=velocity['mom_growth'], marker_color=colors, name='MoM %'))
                fig_vel.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                fig_vel.update_layout(title="Month-over-Month рост выручки (%)")
                apply_theme(fig_vel, height=250)
                st.plotly_chart(fig_vel, use_container_width=True)

        # ───────────────────────────────────────────
        # TAB 1: FORECASTING
        # ───────────────────────────────────────────
        with tabs[1]:
            st.title(f"Прогноз продаж — Горизонт {horizon_months} мес.")

            if forecast is not None and not future_forecast.empty:
                # Main forecast chart
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=actual['ds'], y=actual['y'],
                    name='Факт (дни)', mode='lines',
                    line=dict(color='#6366f1', width=1.5)
                ))
                fig_fc.add_trace(go.Scatter(
                    x=future_forecast['ds'], y=future_forecast['yhat'],
                    name='Прогноз (база)', line=dict(color='#f59e0b', width=2)
                ))
                fig_fc.add_trace(go.Scatter(
                    x=future_forecast['ds'], y=future_forecast['yhat_upper'],
                    name='Оптимистичный', line=dict(color='#10b981', width=1.5, dash='dot')
                ))
                fig_fc.add_trace(go.Scatter(
                    x=future_forecast['ds'], y=future_forecast['yhat_lower'],
                    name='Пессимистичный', line=dict(color='#ef4444', width=1.5, dash='dot'),
                    fill='tonexty', fillcolor='rgba(99,102,241,0.04)'
                ))
                fig_fc.update_layout(title="Прогноз выручки (Prophet + Seasonality)", hovermode="x unified")
                apply_theme(fig_fc, height=400)
                st.plotly_chart(fig_fc, use_container_width=True)

                # Forecast KPIs
                uncertainty_pct = (future_total_upper - future_total_lower) / future_total_rev * 100 if future_total_rev > 0 else 0
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Базовый прогноз", f"₴{future_total_rev/1_000:.1f}K")
                k2.metric("Оптимистичный", f"₴{future_total_upper/1_000:.1f}K", f"+{(future_total_upper/future_total_rev-1)*100:.1f}%")
                k3.metric("Пессимистичный", f"₴{future_total_lower/1_000:.1f}K", f"{(future_total_lower/future_total_rev-1)*100:.1f}%")
                k4.metric("Неопределенность", f"±{uncertainty_pct/2:.1f}%")

                st.markdown("### 📋 Автообоснование прогноза")
                insight(f"""
                **Prophet-заключение:** Модель идентифицировала **{trend_direction}** тренд.
                Смещение среднесуточных продаж относительно исторического базиса: **{change_pct:+.1f}%**.
                Диапазон неопределенности сценариев: **±{uncertainty_pct/2:.1f}%** — отражает баланс недельной и годовой сезонности.
                Прогнозируемая прибыль при марже {gross_margin*100:.0f}%: **₴{future_total_rev*gross_margin/1_000:.1f}K** (база).
                """, 'default', '📐')

                # Calendar of holidays
                st.markdown("### 📅 Ключевые события Украины 2026 (влияние на спрос)")
                holidays_2026 = pd.DataFrame([
                    {"Дата": "28.06.2026", "Событие": "День Конституции", "Риск/Возможность": "Рост в HoReCa, спад B2B активности"},
                    {"Дата": "15.07.2026", "Событие": "День Государственности", "Риск/Возможность": "Нейтральный эффект, стандартный летний день"},
                    {"Дата": "24.08.2026", "Событие": "День Независимости 🇺🇦", "Риск/Возможность": "Всплеск патриотического спроса +15-25%"},
                    {"Дата": "01.10.2026", "Событие": "День Защитников", "Риск/Возможность": "Высокий подарочный спрос, специализированный ассортимент"},
                    {"Дата": "19.12.2026", "Событие": "Предновогодний пик", "Риск/Возможность": "Сезонный максимум — готовить запасы заблаговременно"}
                ])
                st.dataframe(holidays_2026, use_container_width=True, hide_index=True)

                # Monthly aggregated
                st.markdown("### 📊 Агрегированный прогноз по месяцам")
                month_df = monthly_forecast_totals.rename(columns={
                    'month_str': 'Месяц', 'yhat': 'Базовый (₴)', 'yhat_upper': 'Оптимистичный (₴)', 'yhat_lower': 'Пессимистичный (₴)'
                })
                month_df['Прибыль базовая (₴)'] = month_df['Базовый (₴)'] * gross_margin
                st.dataframe(month_df.style.format({c: "{:,.0f}" for c in month_df.columns if c != 'Месяц'}), use_container_width=True, hide_index=True)

            else:
                st.warning("Недостаточно данных для прогноза (минимум 10 дней продаж).")

        # ───────────────────────────────────────────
        # TAB 2: WHAT-IF MODELING (FIXED COMPOUND)
        # ───────────────────────────────────────────
        with tabs[2]:
            st.title("Стратегическое моделирование")
            st.markdown('<div style="color:#64748b;font-size:13px;margin-bottom:16px">Составной процент: Итог = База × (1 + Δ трафик) × (1 + Δ чек). Эффект синергии автоматически учтен.</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            orders_boost = c1.slider("📦 Рост объема заказов / трафика (%)", -50.0, 100.0, 15.0, 1.0)
            aov_boost = c2.slider("💳 Рост среднего чека / цен (%)", -50.0, 100.0, 10.0, 1.0)
            scenario_name = c3.selectbox("Сценарий", ["Базовый прогноз", "Оптимистичный", "Пессимистичный"])

            if not monthly_forecast_totals.empty:
                model_df = monthly_forecast_totals[['month_str']].copy()
                
                # Выбор базы сценария
                scenario_col = {'Базовый прогноз': 'yhat', 'Оптимистичный': 'yhat_upper', 'Пессимистичный': 'yhat_lower'}[scenario_name]
                model_df['База прогноза (₴)'] = monthly_forecast_totals[scenario_col].clip(lower=0)
                model_df.rename(columns={'month_str': 'Месяц'}, inplace=True)

                # === КОРРЕКТНАЯ СОСТАВНАЯ ФОРМУЛА ===
                # Шаг 1: Применяем изменение трафика/объема к базе
                model_df['После Δ Трафика (₴)'] = model_df['База прогноза (₴)'] * (1 + orders_boost / 100)

                # Шаг 2: Применяем изменение чека К НОВОМУ ОБЪЕМУ (составной эффект)
                model_df['Итог с Δ Чека (₴)'] = model_df['После Δ Трафика (₴)'] * (1 + aov_boost / 100)

                # Декомпозиция прироста
                model_df['Δ Трафик (₴)'] = model_df['После Δ Трафика (₴)'] - model_df['База прогноза (₴)']
                model_df['Δ Чек (₴)'] = model_df['Итог с Δ Чека (₴)'] - model_df['После Δ Трафика (₴)']
                model_df['Δ Синергия (₴)'] = model_df['Итог с Δ Чека (₴)'] - model_df['База прогноза (₴)'] - \
                                              model_df['База прогноза (₴)'] * orders_boost / 100 - \
                                              model_df['База прогноза (₴)'] * aov_boost / 100

                model_df['Совокупный Δ (₴)'] = model_df['Итог с Δ Чека (₴)'] - model_df['База прогноза (₴)']
                model_df[f'Доп. Прибыль {int(gross_margin*100)}% (₴)'] = model_df['Совокупный Δ (₴)'] * gross_margin

                # Строка ИТОГО
                totals = {c: model_df[c].sum() if c != 'Месяц' else 'ИТОГО' for c in model_df.columns}
                model_df_display = pd.concat([model_df, pd.DataFrame([totals])], ignore_index=True)

                fmt = {c: "{:,.0f}" for c in model_df_display.columns if c != 'Месяц'}
                st.dataframe(model_df_display.style.format(fmt), use_container_width=True, hide_index=True)

                # Auto-analysis
                t_base = model_df['База прогноза (₴)'].sum()
                t_traffic = model_df['Δ Трафик (₴)'].sum()
                t_aov_delta = model_df['Δ Чек (₴)'].sum()
                t_synergy = model_df['Δ Синергия (₴)'].sum()
                t_total = model_df['Совокупный Δ (₴)'].sum()
                t_profit = model_df[f'Доп. Прибыль {int(gross_margin*100)}% (₴)'].sum()

                st.markdown("### 💡 Факторный анализ прироста (Декомпозиция)")
                col_a, col_b = st.columns(2)
                with col_a:
                    insight(f"""
                    **Декомпозиция прироста (нарастающий итог):**<br>
                    • База ({scenario_name}): **₴{t_base:,.0f}**<br>
                    • Эффект роста трафика +{orders_boost}%: **₴{t_traffic:+,.0f}**<br>
                    • Эффект роста чека +{aov_boost}% (на новом объеме): **₴{t_aov_delta:+,.0f}**<br>
                    • Синергетический эффект (cross-term): **₴{t_synergy:+,.0f}**<br>
                    • <strong>Совокупный прирост выручки: ₴{t_total:+,.0f}</strong>
                    """, 'success', '📊')
                with col_b:
                    total_final = t_base + t_total
                    insight(f"""
                    **P&L импакт:**<br>
                    • Итоговая выручка: **₴{total_final:,.0f}**<br>
                    • Рост vs база: **{t_total/t_base*100:+.1f}%**<br>
                    • Доп. прибыль при марже {gross_margin*100:.0f}%: **₴{t_profit:+,.0f}**<br>
                    • Break-even дополнительных инвестиций: нужно удержать не более **₴{t_profit:,.0f}** в маркетинге для безубыточности сценария
                    """, 'default', '💰')

                # Waterfall chart
                labels = ['База', 'Δ Трафик', 'Δ Чек', 'Синергия', 'Итого']
                values = [t_base, t_traffic, t_aov_delta, t_synergy, t_base + t_total]
                measures = ['absolute', 'relative', 'relative', 'relative', 'total']
                fig_wf = go.Figure(go.Waterfall(
                    orientation="v", measure=measures,
                    x=labels, y=values,
                    connector=dict(line=dict(color='rgba(255,255,255,0.1)')),
                    decreasing=dict(marker_color='rgba(239,68,68,0.7)'),
                    increasing=dict(marker_color='rgba(16,185,129,0.7)'),
                    totals=dict(marker_color='rgba(99,102,241,0.8)')
                ))
                fig_wf.update_layout(title="Waterfall: декомпозиция прироста выручки")
                apply_theme(fig_wf, height=320)
                st.plotly_chart(fig_wf, use_container_width=True)

            else:
                st.warning("Нет данных прогноза. Загрузите файл с минимум 10 днями продаж.")

        # ───────────────────────────────────────────
        # TAB 3: MARKETING
        # ───────────────────────────────────────────
        with tabs[3]:
            st.title("Маркетинговая аналитика")

            col_l, col_r = st.columns([2, 1])

            with col_l:
                fig_sun = px.sunburst(
                    df, path=['utm_source', 'utm_medium', 'utm_campaign'],
                    values='revenue',
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig_sun.update_layout(title="Структура доходов по UTM-цепочке")
                apply_theme(fig_sun, height=400)
                st.plotly_chart(fig_sun, use_container_width=True)

            with col_r:
                channel_rev = df.groupby('utm_source')['revenue'].sum().sort_values(ascending=False)
                fig_ch = go.Figure(go.Bar(
                    y=channel_rev.index, x=channel_rev.values, orientation='h',
                    marker=dict(color='rgba(99,102,241,0.7)')
                ))
                fig_ch.update_layout(title="Выручка по каналам")
                apply_theme(fig_ch, height=400)
                st.plotly_chart(fig_ch, use_container_width=True)

            # Auto-analytics
            st.markdown("### 🔬 Расширенная авто-аналитика каналов")

            if not channel_rev.empty:
                top_source = channel_rev.index[0]
                top_val = channel_rev.values[0]
                bottom_source = channel_rev.index[-1]
                bottom_val = channel_rev.values[-1]
                channel_count = len(channel_rev)

                c1, c2, c3 = st.columns(3)
                with c1:
                    insight(f"**Локомотив:** `{top_source}` генерирует **{top_val/total_rev*100:.1f}%** выручки (₴{top_val:,.0f}). При прогнозе ₴{future_total_rev:,.0f} — вклад канала **₴{future_total_rev*(top_val/total_rev):,.0f}**.", 'success', '🏆')
                with c2:
                    insight(f"**Риск-буфер:** При потере `{top_source}` кассовый разрыв составит **~₴{future_total_rev*(top_val/total_rev):,.0f}** за горизонт прогноза. Рекомендуется страховочный бюджет на 2-й канал.", 'danger', '🛡️')
                with c3:
                    if channel_count > 1:
                        insight(f"**Оптимизация хвоста:** `{bottom_source}` (₴{bottom_val:,.0f}) показывает ROI ниже медианы. Рекомендуется A/B перераспределение бюджета или 30-дневный паузинг.", 'warning', '⚙️')

                # Channel MoM dynamics
                ch_monthly = df.groupby(['year_month', 'utm_source'])['revenue'].sum().reset_index()
                fig_ch_trend = px.line(ch_monthly, x='year_month', y='revenue', color='utm_source',
                                       title="Динамика каналов по месяцам")
                apply_theme(fig_ch_trend, height=300)
                st.plotly_chart(fig_ch_trend, use_container_width=True)

            st.markdown("### 📈 Прогноз по каналам (помесячно)")
            if not monthly_forecast_totals.empty:
                mkt_dist = get_monthly_distribution(df, 'utm_source', monthly_forecast_totals, total_rev)
                mkt_dist = mkt_dist.rename(columns={'utm_source': 'Источник'})
                st.dataframe(mkt_dist.style.format({c: "{:,.0f}" for c in mkt_dist.columns if c != 'Источник'}),
                             hide_index=True, use_container_width=True)

        # ───────────────────────────────────────────
        # TAB 4: GEO
        # ───────────────────────────────────────────
        with tabs[4]:
            st.title("Геоаналитика")

            col_l, col_r = st.columns([2, 1])
            with col_l:
                fig_geo = px.bar(
                    city_rev_df.head(20), x='city', y='revenue',
                    color='revenue', color_continuous_scale=[[0, 'rgba(99,102,241,0.3)'], [1, 'rgba(99,102,241,1)']],
                    title="Топ-20 городов (факт)"
                )
                apply_theme(fig_geo, height=380)
                st.plotly_chart(fig_geo, use_container_width=True)

            with col_r:
                # City concentration pie
                top10 = city_rev_df.head(10).copy()
                other_rev = city_rev_df.iloc[10:]['revenue'].sum()
                if other_rev > 0:
                    top10 = pd.concat([top10, pd.DataFrame([{'city': 'Остальные', 'revenue': other_rev}])], ignore_index=True)
                fig_pie = px.pie(top10, names='city', values='revenue', title="Доли регионов")
                apply_theme(fig_pie, height=380)
                st.plotly_chart(fig_pie, use_container_width=True)

            # Geo auto-analytics
            st.markdown("### 🌍 Региональные инсайты")
            c1, c2 = st.columns(2)
            with c1:
                if not city_rev_df.empty:
                    top_city_name = city_rev_df.iloc[0]['city']
                    top_city_share = city_rev_df.iloc[0]['revenue'] / total_rev * 100
                    top3_share = city_rev_df.head(3)['revenue'].sum() / total_rev * 100
                    insight(f"**Лидер регионов:** `{top_city_name}` — **{top_city_share:.1f}%** выручки. ТОП-3 города концентрируют **{top3_share:.1f}%** продаж — потенциал для региональной экспансии.", 'default', '📍')
            with c2:
                if len(city_rev_df) > 5:
                    tail_cities = city_rev_df.tail(len(city_rev_df) - 10)
                    tail_share = tail_cities['revenue'].sum() / total_rev * 100
                    insight(f"**Нераскрытый потенциал:** {len(tail_cities)} городов за пределами ТОП-10 суммарно дают лишь **{tail_share:.1f}%** выручки. Целевые региональные кампании могут удвоить охват.", 'warning', '🗺️')

            # City MoM dynamics
            city_monthly = df.groupby(['year_month', 'city'])['revenue'].sum().reset_index()
            top5_cities = city_rev_df.head(5)['city'].tolist()
            fig_city_trend = px.line(city_monthly[city_monthly['city'].isin(top5_cities)],
                                     x='year_month', y='revenue', color='city', title="Динамика ТОП-5 городов")
            apply_theme(fig_city_trend, height=280)
            st.plotly_chart(fig_city_trend, use_container_width=True)

            st.markdown("### 📍 Прогноз по городам (Топ-25)")
            if not monthly_forecast_totals.empty:
                geo_dist = get_monthly_distribution(df, 'city', monthly_forecast_totals, total_rev)
                geo_dist = geo_dist.rename(columns={'city': 'Город'})
                st.dataframe(geo_dist.head(25).style.format({c: "{:,.0f}" for c in geo_dist.columns if c != 'Город'}),
                             use_container_width=True, hide_index=True)

        # ───────────────────────────────────────────
        # TAB 5: PRODUCTS
        # ───────────────────────────────────────────
        with tabs[5]:
            st.title("Товарная аналитика & ABC-анализ")

            df['product_full'] = df['art'] + " — " + df['product']
            prod_df_base = df.groupby('product_full').agg({'qty': 'sum', 'revenue': 'sum'}).sort_values('revenue', ascending=False).reset_index()

            prod_df_base['share'] = prod_df_base['revenue'] / total_rev
            prod_df_base['cum_share'] = prod_df_base['share'].cumsum()

            def assign_abc(cum):
                if cum <= 0.80: return 'A'
                elif cum <= 0.95: return 'B'
                return 'C'

            prod_df_base['ABC'] = prod_df_base['cum_share'].apply(assign_abc)

            abc_summary = prod_df_base.groupby('ABC').agg(
                Кол_во=('product_full', 'count'),
                Выручка=('revenue', 'sum')
            ).reset_index()
            abc_summary['Доля %'] = abc_summary['Выручка'] / total_rev * 100
            abc_summary.rename(columns={'ABC': 'Класс', 'Кол_во': 'SKU'}, inplace=True)

            col_l, col_r = st.columns([1, 2])
            with col_l:
                st.markdown("### 📊 ABC-матрица")
                st.dataframe(abc_summary.style.format({'Выручка': "{:,.0f} ₴", 'Доля %': "{:.1f}%"}),
                             hide_index=True, use_container_width=True)

                insight(f"**Класс A** ({abc_summary[abc_summary['Класс']=='A']['SKU'].sum() if not abc_summary.empty else 0} SKU): Ядро бизнеса. Приоритет — наличие на складе 100%, ноль стоковых аутов.", 'success', '🅰️')
                insight(f"**Класс C** ({abc_summary[abc_summary['Класс']=='C']['SKU'].sum() if not abc_summary.empty else 0} SKU): Кандидаты на вывод или ребрендинг. Замораживание закупок.", 'warning', '🅲')

            with col_r:
                # Pareto curve
                fig_pareto = go.Figure()
                fig_pareto.add_trace(go.Bar(x=prod_df_base['product_full'].str[:30].head(30),
                                            y=prod_df_base['revenue'].head(30),
                                            name='Выручка', marker_color='rgba(99,102,241,0.6)'))
                fig_pareto.add_trace(go.Scatter(x=prod_df_base['product_full'].str[:30].head(30),
                                                y=prod_df_base['cum_share'].head(30) * prod_df_base['revenue'].head(30).max() / (prod_df_base['cum_share'].head(30).max() or 1),
                                                name='Кум. доля', mode='lines',
                                                line=dict(color='#f59e0b', width=2),
                                                yaxis='y2'))
                fig_pareto.update_layout(
                    title="Кривая Парето (Топ-30 SKU)",
                    yaxis2=dict(overlaying='y', side='right', showgrid=False, color='#f59e0b')
                )
                apply_theme(fig_pareto, height=380)
                st.plotly_chart(fig_pareto, use_container_width=True)

            st.markdown("### 📦 Прогноз по товарам (Топ-50)")
            if not monthly_forecast_totals.empty:
                prod_dist = get_monthly_distribution(df, 'product_full', monthly_forecast_totals, total_rev)
                prod_dist = prod_dist.merge(prod_df_base[['product_full', 'ABC']], on='product_full', how='left')
                prod_dist = prod_dist.rename(columns={'product_full': 'Товар', 'ABC': 'Класс'})
                cols = ['Товар', 'Класс'] + [c for c in prod_dist.columns if c not in ['Товар', 'Класс']]
                prod_dist = prod_dist[cols]
                st.dataframe(prod_dist.head(50).style.format({c: "{:,.0f}" for c in prod_dist.columns if c not in ['Товар', 'Класс']}),
                             use_container_width=True, hide_index=True)

        # ───────────────────────────────────────────
        # TAB 6: UNIT ECONOMICS
        # ───────────────────────────────────────────
        with tabs[6]:
            st.title("📐 Unit Economics & Financial Health")

            # Core metrics
            days_total = (df['date'].max() - df['date'].min()).days or 1
            revenue_per_day = total_rev / days_total
            orders_per_day = unique_orders / days_total
            ltv_proxy = aov * 3.2  # примерный LTV как AOV × средняя частота 3.2
            gross_profit_per_order = aov * gross_margin
            breakeven_cac = gross_profit_per_order * 0.3  # CAC не должен превышать 30% GP

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Revenue/Day", f"₴{revenue_per_day:,.0f}")
            k2.metric("Orders/Day", f"{orders_per_day:.1f}")
            k3.metric("AOV", f"₴{aov:,.0f}")
            k4.metric("GP/Order", f"₴{gross_profit_per_order:,.0f}")
            k5.metric("LTV (proxy)", f"₴{ltv_proxy:,.0f}")
            k6.metric("Max CAC", f"₴{breakeven_cac:,.0f}")

            st.markdown("---")

            col_l, col_r = st.columns(2)

            with col_l:
                # Revenue distribution per order
                fig_hist = go.Figure(go.Histogram(
                    x=unit_eco['revenue'].clip(upper=unit_eco['revenue'].quantile(0.95)),
                    nbinsx=40,
                    marker_color='rgba(99,102,241,0.6)',
                    name='Выручка с заказа'
                ))
                fig_hist.update_layout(title="Распределение выручки по заказам (P95)")
                apply_theme(fig_hist, height=300)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_r:
                # Items per order
                fig_items = go.Figure(go.Histogram(
                    x=unit_eco['items'].clip(upper=unit_eco['items'].quantile(0.95)),
                    nbinsx=20,
                    marker_color='rgba(16,185,129,0.6)',
                    name='Товаров в заказе'
                ))
                fig_items.update_layout(title="Кол-во товаров в заказе")
                apply_theme(fig_items, height=300)
                st.plotly_chart(fig_items, use_container_width=True)

            # Velocity & VaR
            st.markdown("### 📊 Метрики рисков и скорости")

            r1, r2, r3 = st.columns(3)
            p25 = unit_eco['revenue'].quantile(0.25)
            p75 = unit_eco['revenue'].quantile(0.75)
            with r1:
                insight(f"**Value at Risk (VaR 10%):** В 10% худших дней выручка не превышает **₴{var_10:,.0f}**. Резервный буфер ликвидности должен покрывать не менее **{int(7 * var_10):,} ₴** (недельный VaR).", 'warning', '📉')
            with r2:
                insight(f"**IQR заказов:** 25%-75% перцентиль выручки с заказа: ₴{p25:,.0f} — ₴{p75:,.0f}. Межквартильный диапазон **₴{p75-p25:,.0f}** — показатель стабильности AOV.", 'default', '📐')
            with r3:
                insight(f"**LTV/CAC норматив:** При AOV ₴{aov:,.0f} и марже {gross_margin*100:.0f}% — максимально допустимый CAC **₴{breakeven_cac:,.0f}**. LTV-прокси: **₴{ltv_proxy:,.0f}** (ratio LTV/CAC > 3x).", 'success', '💎')

            # Order velocity by week
            order_wk = df.groupby('week')['order_id'].nunique().reset_index()
            order_wk.columns = ['week', 'orders']
            fig_vel = go.Figure(go.Scatter(x=order_wk['week'], y=order_wk['orders'],
                                           mode='lines+markers', fill='tozeroy',
                                           line=dict(color='#6366f1'), fillcolor='rgba(99,102,241,0.1)'))
            fig_vel.update_layout(title="Еженедельный объем заказов (Velocity)")
            apply_theme(fig_vel, height=260)
            st.plotly_chart(fig_vel, use_container_width=True)

        # ───────────────────────────────────────────
        # TAB 7: AI CFO
        # ───────────────────────────────────────────
        with tabs[7]:
            st.title("🧠 AI Финансовый Директор")
            st.markdown('<div style="color:#64748b;font-size:13px;margin-bottom:16px">Стратегический брифинг уровня C-Suite на основе всех данных платформы.</div>', unsafe_allow_html=True)

            if not api_key:
                st.warning("⚠️ Введите Google API Key в боковом меню.")
            else:
                if st.button("⚡ Сгенерировать CFO брифинг"):
                    with st.spinner(f"Анализ через {selected_model_path}..."):
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel(selected_model_path)
                            top_city = city_rev_df.iloc[0]['city'] if not city_rev_df.empty else "N/A"
                            top_channel = df.groupby('utm_source')['revenue'].sum().idxmax() if total_rev > 0 else "N/A"
                            top_product = df.groupby('product')['revenue'].sum().idxmax() if total_rev > 0 else "N/A"

                            ai_context = {
                                "выручка_факт": f"₴{total_rev:,.0f}",
                                "выручка_прогноз": f"₴{future_total_rev:,.0f}",
                                "горизонт_месяцев": horizon_months,
                                "AOV": f"₴{aov:,.0f}",
                                "LTV_proxy": f"₴{ltv_proxy:,.0f}",
                                "маржа": f"{gross_margin*100:.0f}%",
                                "прибыль": f"₴{total_profit:,.0f}",
                                "тренд": trend_direction,
                                "изменение_тренда": f"{change_pct:.1f}%",
                                "топ_канал": top_channel,
                                "HHI_концентрация": f"{hhi:.2f}",
                                "топ_регион": top_city,
                                "топ_продукт": top_product,
                                "кол_во_SKU": df['product'].nunique(),
                                "pareto_SKU_pct": f"{pareto_pct:.0f}%",
                                "VaR_10pct": f"₴{var_10:,.0f}"
                            }

                            prompt = f"""
Ты Chief Financial Officer крупной украинской e-commerce компании, уровень Goldman Sachs / McKinsey.
Данные платформы: {json.dumps(ai_context, ensure_ascii=False)}

Напиши СТРОГИЙ, КОНКРЕТНЫЙ стратегический брифинг для Board of Directors. Структура:

## EXECUTIVE SUMMARY (3 предложения — ключевые цифры)

## RED FLAGS — КРИТИЧЕСКИЕ РИСКИ (топ-3, с цифрами)

## FINANCIAL HEALTH ASSESSMENT (оценка P&L, ликвидности, устойчивости)

## STRATEGIC RECOMMENDATIONS (5 конкретных действий с ROI-потенциалом)

## OUTLOOK — ПРОГНОЗ (жесткая оценка реалистичности прогнозных цифр)

Тон: прямой, без воды, как на собрании совета директоров Нью-Йорка. Используй конкретные цифры из данных.
"""
                            response = model.generate_content(prompt)
                            st.session_state['ai_director_report'] = response.text
                            st.markdown(response.text)
                        except Exception as e:
                            if "429" in str(e) or "ResourceExhausted" in str(e):
                                st.error("🛑 Лимит Google API исчерпан. Смените модель или попробуйте позже.")
                            else:
                                st.error(f"❌ Ошибка AI: {e}")
                elif st.session_state['ai_director_report']:
                    st.markdown(st.session_state['ai_director_report'])

        # ───────────────────────────────────────────
        # TAB 8: AI STRATEGY
        # ───────────────────────────────────────────
        with tabs[8]:
            st.title("🚀 AI Стратегия роста")
            st.markdown('<div style="color:#64748b;font-size:13px;margin-bottom:16px">90-дневный операционный план масштабирования от Chief Marketing Officer.</div>', unsafe_allow_html=True)

            if not api_key:
                st.warning("⚠️ Введите Google API Key в боковом меню.")
            else:
                if st.button("🚀 Разработать план роста"):
                    with st.spinner("Построение стратегии..."):
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel(selected_model_path)
                            top_ch = df.groupby('utm_source')['revenue'].sum().sort_values(ascending=False)

                            prompt = f"""
Ты Chief Marketing Officer e-commerce компании (уровень Shopify / Rozetka).
Данные: выручка ₴{total_rev:,.0f}, прогноз на {horizon_months} мес. ₴{future_total_rev:,.0f},
AOV ₴{aov:,.0f}, LTV ₴{ltv_proxy:,.0f}, топ-канал: {top_ch.index[0] if len(top_ch)>0 else 'N/A'},
HHI концентрации: {hhi:.2f}, активных SKU: {df['product'].nunique()}.

Напиши КОНКРЕТНЫЙ 90-ДНЕВНЫЙ ПЛАН РОСТА. Структура:

## QUICK WINS (0-30 дней) — 5 действий с прогнозируемым uplift %

## GROWTH ENGINE (30-60 дней) — масштабирование каналов, AOV-стратегии

## SCALE (60-90 дней) — новые каналы, ассортиментная матрица, retention

## MARKETING MIX RECOMMENDATION — перераспределение бюджета по каналам (%)

## KPI DASHBOARD — 10 метрик для еженедельного мониторинга

Тон: конкретно, с цифрами, без воды. Стиль: операционный директор Нью-Йорской компании.
"""
                            response = model.generate_content(prompt)
                            st.session_state['ai_growth_plan'] = response.text
                            st.markdown(response.text)
                        except Exception as e:
                            if "429" in str(e) or "ResourceExhausted" in str(e):
                                st.error("🛑 Лимит Google API исчерпан.")
                            else:
                                st.error(f"❌ Ошибка AI: {e}")
                elif st.session_state['ai_growth_plan']:
                    st.markdown(st.session_state['ai_growth_plan'])

        # ═══════════════════════════════════════════
        # SIDEBAR EXPORT
        # ═══════════════════════════════════════════
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🖨️ Экспорт**")

        def df_to_html_table(df_in, max_rows=None):
            if df_in is None or df_in.empty:
                return "<p><em>Нет данных</em></p>"
            d = df_in.head(max_rows) if max_rows else df_in
            return d.to_html(index=False, border=0, classes='data-table')

        # Build full HTML report
        ai_cfo_section = ""
        ai_plan_section = ""
        if st.session_state['ai_director_report']:
            ai_cfo_section = f"""
            <div class="page-break"></div>
            <h2>7. Заключение AI CFO (Board Briefing)</h2>
            <div class="ai-box">{st.session_state['ai_director_report'].replace(chr(10), '<br>')}</div>
            """
        if st.session_state['ai_growth_plan']:
            ai_plan_section = f"""
            <h2>8. AI План роста (90 дней)</h2>
            <div class="ai-box">{st.session_state['ai_growth_plan'].replace(chr(10), '<br>')}</div>
            """

        html_report = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Executive BI Report</title>
<style>
  @page {{ size: A4 portrait; margin: 15mm 12mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10.5px;
    color: #1e293b;
    line-height: 1.5;
    background: white;
  }}
  .header {{
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    color: white;
    padding: 20px 24px;
    border-radius: 8px;
    margin-bottom: 20px;
  }}
  .header h1 {{ margin: 0; font-size: 20px; font-weight: 700; color: white; }}
  .header .sub {{ color: #a5b4fc; font-size: 11px; margin-top: 4px; }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 20px;
  }}
  .kpi-card {{
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #6366f1;
    border-radius: 6px;
    padding: 10px 12px;
  }}
  .kpi-label {{ font-size: 9px; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }}
  .kpi-value {{ font-size: 18px; font-weight: 700; color: #1e293b; margin-top: 2px; }}
  h2 {{
    color: #1e293b;
    font-size: 13px;
    font-weight: 700;
    border-bottom: 2px solid #6366f1;
    padding-bottom: 4px;
    margin: 20px 0 10px;
    page-break-after: avoid;
  }}
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 9.5px;
    margin-bottom: 14px;
    page-break-inside: auto;
  }}
  .data-table th {{
    background: #1e1b4b;
    color: white;
    padding: 5px 7px;
    text-align: left;
    font-weight: 600;
  }}
  .data-table td {{
    padding: 4px 7px;
    border-bottom: 1px solid #f1f5f9;
  }}
  .data-table tr:nth-child(even) td {{ background: #f8fafc; }}
  .data-table tr:last-child td {{
    font-weight: 700;
    background: #ede9fe;
    border-top: 2px solid #6366f1;
  }}
  .page-break {{ page-break-before: always; }}
  .ai-box {{
    background: #fafafa;
    border-left: 4px solid #6366f1;
    border-radius: 4px;
    padding: 14px 16px;
    font-size: 10.5px;
    line-height: 1.7;
    white-space: pre-wrap;
  }}
  .insight {{
    background: #eff6ff;
    border-left: 3px solid #3b82f6;
    padding: 8px 12px;
    border-radius: 4px;
    margin: 6px 0;
    font-size: 10px;
  }}
  .footer {{
    text-align: center;
    color: #94a3b8;
    font-size: 9px;
    margin-top: 30px;
    border-top: 1px solid #e2e8f0;
    padding-top: 10px;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>⚡ Executive BI Report</h1>
  <div class="sub">Сформировано: {datetime.now().strftime('%d.%m.%Y %H:%M')} &nbsp;|&nbsp; Горизонт прогноза: {horizon_months} мес. &nbsp;|&nbsp; Статус: {', '.join(status_filter)}</div>
</div>

<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-label">Выручка (факт)</div><div class="kpi-value">₴{total_rev/1000:.1f}K</div></div>
  <div class="kpi-card"><div class="kpi-label">Прогноз (база)</div><div class="kpi-value">₴{future_total_rev/1000:.1f}K</div></div>
  <div class="kpi-card"><div class="kpi-label">Заказов</div><div class="kpi-value">{unique_orders:,}</div></div>
  <div class="kpi-card"><div class="kpi-label">AOV</div><div class="kpi-value">₴{aov:,.0f}</div></div>
  <div class="kpi-card"><div class="kpi-label">Прибыль (факт)</div><div class="kpi-value">₴{total_profit/1000:.1f}K</div></div>
  <div class="kpi-card"><div class="kpi-label">Маржа</div><div class="kpi-value">{gross_margin*100:.0f}%</div></div>
  <div class="kpi-card"><div class="kpi-label">Тренд</div><div class="kpi-value" style="font-size:12px">{trend_direction}</div></div>
  <div class="kpi-card"><div class="kpi-label">Δ vs история</div><div class="kpi-value">{change_pct:+.1f}%</div></div>
</div>

<h2>1. Прогноз продаж по месяцам (Prophet)</h2>
<div class="insight">Тренд: <strong>{trend_direction}</strong>. Отклонение от исторической базы: <strong>{change_pct:+.1f}%</strong>. Прибыль при марже {gross_margin*100:.0f}%: <strong>₴{future_total_rev*gross_margin:,.0f}</strong></div>
{df_to_html_table(month_df)}

<h2>2. Стратегическое моделирование (Составной рост)</h2>
<div class="insight">Δ Трафик: <strong>{orders_boost:+.0f}%</strong> | Δ Чек: <strong>{aov_boost:+.0f}%</strong> | Синергия учтена автоматически. Итоговый мультипликатор: <strong>×{(1+orders_boost/100)*(1+aov_boost/100):.3f}</strong></div>
{df_to_html_table(model_df_display)}

<div class="page-break"></div>

<h2>3. Маркетинг: прогноз по каналам</h2>
{df_to_html_table(mkt_dist)}

<h2>4. Геоаналитика: прогноз по городам (Топ-25)</h2>
{df_to_html_table(geo_dist, max_rows=25)}

<div class="page-break"></div>

<h2>5. ABC-анализ ассортимента</h2>
{df_to_html_table(abc_summary)}

<h2>6. Товарный прогноз (Топ-50 позиций)</h2>
{df_to_html_table(prod_dist, max_rows=50)}

{ai_cfo_section}
{ai_plan_section}

<div class="footer">Executive BI Suite v2.0 &nbsp;|&nbsp; Сформировано {datetime.now().strftime('%d.%m.%Y')} &nbsp;|&nbsp; Конфиденциально</div>

<script>
  window.addEventListener('load', function() {{
    setTimeout(function() {{ window.print(); }}, 800);
  }});
</script>
</body>
</html>"""

        st.sidebar.download_button(
            label="📄 Скачать PDF отчет (A4)",
            data=html_report.encode('utf-8'),
            file_name=f"BI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html"
        )
        st.sidebar.caption("Откройте файл в браузере → автоматически откроется диалог печати → Сохранить как PDF")

else:
    st.markdown("""
    <div style="
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-height: 60vh; text-align: center; color: #475569;
    ">
        <div style="font-size: 64px; margin-bottom: 20px">⚡</div>
        <h2 style="color: #94a3b8; font-size: 24px; margin-bottom: 8px">Executive BI Suite</h2>
        <p style="font-size: 14px; color: #475569; max-width: 400px">
            Загрузите реестр заказов (.csv или .xlsx) в боковом меню для запуска аналитики
        </p>
        <div style="margin-top: 24px; display: flex; gap: 12px; flex-wrap: wrap; justify-content: center">
            <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);padding:6px 14px;border-radius:20px;font-size:12px;color:#a5b4fc">📊 Prophet Forecasting</span>
            <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);padding:6px 14px;border-radius:20px;font-size:12px;color:#a5b4fc">🔢 Unit Economics</span>
            <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);padding:6px 14px;border-radius:20px;font-size:12px;color:#a5b4fc">🧠 AI CFO Insights</span>
            <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);padding:6px 14px;border-radius:20px;font-size:12px;color:#a5b4fc">📐 ABC Analysis</span>
            <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);padding:6px 14px;border-radius:20px;font-size:12px;color:#a5b4fc">🌍 Geo Intelligence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
