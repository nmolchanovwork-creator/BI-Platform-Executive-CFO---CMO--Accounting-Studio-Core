"""
BI Platform v3.0 — Executive CFO & CMO Intelligence Suite
Читабельный UI | Расшифровка всех терминов | Вертикальный layout
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import google.generativeai as genai
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════
# CONFIG & STYLES
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="Executive BI Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1220 50%, #0a0e1a 100%);
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1220 0%, #111827 100%);
        border-right: 1px solid rgba(99,102,241,0.2);
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }

    .main .block-container { padding: 2rem 2.5rem; max-width: 1300px; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(17,24,39,0.95) 0%, rgba(30,41,59,0.95) 100%);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricDelta"] { font-size: 13px !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,20,35,0.9);
        border-radius: 12px;
        padding: 5px;
        gap: 3px;
        border: 1px solid rgba(99,102,241,0.15);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        padding: 10px 18px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        box-shadow: 0 2px 12px rgba(99,102,241,0.4);
    }

    /* Headers */
    h1 {
        background: linear-gradient(135deg, #f1f5f9, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.3rem !important;
    }
    h2 { color: #e2e8f0 !important; font-size: 1.2rem !important; }
    h3 { color: #cbd5e1 !important; font-size: 1rem !important; }

    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08));
        border: 1px solid rgba(99,102,241,0.3);
        border-left: 5px solid #6366f1;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0 18px 0;
        font-size: 14px;
        line-height: 1.7;
        color: #cbd5e1;
    }
    .insight-box.warning {
        border-left-color: #f59e0b;
        background: rgba(245,158,11,0.06);
        border-color: rgba(245,158,11,0.25);
    }
    .insight-box.danger {
        border-left-color: #ef4444;
        background: rgba(239,68,68,0.06);
        border-color: rgba(239,68,68,0.25);
    }
    .insight-box.success {
        border-left-color: #10b981;
        background: rgba(16,185,129,0.06);
        border-color: rgba(16,185,129,0.25);
    }

    /* Chart caption */
    .chart-caption {
        background: rgba(30,41,59,0.6);
        border-left: 3px solid rgba(99,102,241,0.5);
        border-radius: 0 6px 6px 0;
        padding: 10px 16px;
        font-size: 13px;
        color: #94a3b8;
        margin-bottom: 24px;
        line-height: 1.6;
    }

    /* Section title */
    .section-title {
        color: #a5b4fc;
        font-size: 15px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 28px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(99,102,241,0.2);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 14px;
        box-shadow: 0 4px 16px rgba(99,102,241,0.35);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99,102,241,0.5);
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 10px;
        overflow: hidden;
    }

    .sidebar-logo {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .stAlert { border-radius: 10px !important; }

    /* Term tooltip style */
    .term-label {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a5b4fc;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# PLOTLY THEME
# ═══════════════════════════════════════════════════
PTPL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    title=dict(font=dict(color='#e2e8f0', size=15, family='Inter'), x=0.01),
    xaxis=dict(gridcolor='rgba(99,102,241,0.08)', linecolor='rgba(99,102,241,0.15)'),
    yaxis=dict(gridcolor='rgba(99,102,241,0.08)', linecolor='rgba(99,102,241,0.15)'),
    legend=dict(bgcolor='rgba(13,18,32,0.8)', bordercolor='rgba(99,102,241,0.2)', borderwidth=1, font=dict(color='#94a3b8', size=12)),
    colorway=['#6366f1','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#f97316','#84cc16'],
    hoverlabel=dict(bgcolor='rgba(17,24,39,0.95)', bordercolor='rgba(99,102,241,0.4)', font=dict(color='#e2e8f0', size=12)),
    margin=dict(t=55, l=10, r=10, b=15),
)

def T(fig, h=420):
    fig.update_layout(**PTPL, height=h)
    return fig

# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════
def insight(text, kind='default', icon='💡'):
    cls = {'warning':'warning','danger':'danger','success':'success'}.get(kind,'')
    st.markdown(f'<div class="insight-box {cls}">{icon}&nbsp;&nbsp;{text}</div>', unsafe_allow_html=True)

def caption(text):
    st.markdown(f'<div class="chart-caption">📌 {text}</div>', unsafe_allow_html=True)

def section(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

DOW_RU = {
    'Monday':    'Понедельник',
    'Tuesday':   'Вторник',
    'Wednesday': 'Среда',
    'Thursday':  'Четверг',
    'Friday':    'Пятница',
    'Saturday':  'Суббота',
    'Sunday':    'Воскресенье',
}

# ═══════════════════════════════════════════════════
# DATA PROCESSING
# ═══════════════════════════════════════════════════
@st.cache_data
def load_and_clean_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, dtype=str, low_memory=False)
        else:
            df = pd.read_excel(file, dtype=str)

        num_cols = ['Цена','Количество','Стоимость','Итого']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        def is_razom(name): return 'Разом дешевше' in str(name)
        results = []

        for order_num, group in df.groupby('Номер заказа', sort=False):
            rows = group.reset_index(drop=True)
            n    = len(rows)
            itogo = rows['Итого'].iloc[0] if 'Итого' in rows.columns else 0
            r0 = rows.iloc[0]
            meta = {
                'order_id': order_num,
                'date': r0.get('Дата заказа'),
                'status': str(r0.get('Статус','Доставлен')).strip(),
                'city': str(r0.get('Город','Не указан')).strip(),
                'utm_source':   str(r0.get('utm_source','organic')).replace('nan','organic').replace('(none)','direct'),
                'utm_medium':   str(r0.get('utm_medium','organic')).replace('nan','organic').replace('(none)','direct'),
                'utm_campaign': str(r0.get('utm_campaign','organic')).replace('nan','organic').replace('(none)','direct'),
            }
            included = []; skip_until = -1; i = 0
            while i < n:
                if i <= skip_until: i += 1; continue
                row  = rows.iloc[i]
                name = str(row.get('Название товара',''))
                art  = str(row.get('Артикул',''))
                if is_razom(name): i += 1; continue
                if ('Набір' in name or 'Набор' in name) and not art.startswith('48'):
                    bundle_price = row['Стоимость']; j = i+1; cumsum = 0
                    has_zero = False; candidate_subitems = []
                    while j < n:
                        r = rows.iloc[j]; r_name = str(r.get('Название товара','')); r_art = str(r.get('Артикул',''))
                        if is_razom(r_name): break
                        if ('Набір' in r_name or 'Набор' in r_name) and not r_art.startswith('48'): break
                        candidate_subitems.append(j)
                        if r['Цена'] == 0: has_zero = True; j += 1; break
                        cumsum += r['Цена'] * r['Количество']
                        if cumsum >= bundle_price: j += 1; break
                        j += 1
                    has_sub = len(candidate_subitems)>0 and (cumsum>=bundle_price or has_zero)
                    if has_sub:
                        for si in candidate_subitems:
                            si_row = rows.iloc[si]
                            raw_cost = bundle_price/len(candidate_subitems) if si_row['Цена']==0 else si_row['Стоимость']
                            included.append((si, raw_cost, str(si_row.get('Артикул','')), str(si_row.get('Название товара','')), si_row['Количество'], si_row['Цена']))
                        skip_until = candidate_subitems[-1]; i = skip_until+1
                    else:
                        included.append((i, row['Стоимость'], art, name, row['Количество'], row['Цена'])); i += 1
                    continue
                included.append((i, row['Стоимость'], art, name, row['Количество'], row['Цена'])); i += 1
            raw_sum = sum(x[1] for x in included)
            scale   = (itogo/raw_sum) if raw_sum else 0
            for item in included:
                _, raw_cost, item_art, item_name, qty, price = item
                results.append({**meta, 'art': item_art, 'product': item_name, 'qty': qty, 'revenue': round(raw_cost*scale,2)})

        df_clean = pd.DataFrame(results)
        if df_clean.empty: return pd.DataFrame()
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['date'])
        df_clean['year_month'] = df_clean['date'].dt.to_period('M').astype(str)
        df_clean['week']       = df_clean['date'].dt.to_period('W').dt.start_time
        return df_clean
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")
        return pd.DataFrame()

@st.cache_data
def generate_global_forecast(df, periods):
    daily = df.groupby(df['date'].dt.date)['revenue'].sum().reset_index()
    daily.columns = ['ds','y']
    if len(daily) < 10: return None, None, daily
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                changepoint_prior_scale=0.15, seasonality_prior_scale=10)
    try: m.add_country_holidays(country_name='UA')
    except: pass
    m.fit(daily)
    future   = m.make_future_dataframe(periods=periods, freq='D')
    forecast = m.predict(future)
    return m, forecast, daily

def get_monthly_distribution(df, entity_col, monthly_totals, total_rev):
    entity_rev = df.groupby(entity_col)['revenue'].sum().reset_index()
    entity_rev['share'] = entity_rev['revenue'] / total_rev if total_rev > 0 else 0
    dist = []
    for _, row in entity_rev.iterrows():
        rd = {entity_col: row[entity_col], 'Факт (₴)': row['revenue']}
        for _, mr in monthly_totals.iterrows():
            rd[mr['month_str']] = row['share'] * mr['yhat']
        dist.append(rd)
    return pd.DataFrame(dist).sort_values('Факт (₴)', ascending=False)

def compute_velocity(df):
    m = df.groupby('year_month')['revenue'].sum().reset_index().sort_values('year_month')
    m['mom_growth']     = m['revenue'].pct_change() * 100
    m['mom_growth_abs'] = m['revenue'].diff()
    return m

def compute_hhi(df):
    ch = df.groupby('utm_source')['revenue'].sum()
    shares = ch / ch.sum()
    return (shares**2).sum(), shares

def compute_weekly_seasonality(df):
    df2 = df.copy()
    df2['dow']     = df2['date'].dt.day_name()
    df2['dow_num'] = df2['date'].dt.dayofweek
    d = df2.groupby(['dow_num','dow'])['revenue'].mean().reset_index().sort_values('dow_num')
    d['dow_ru'] = d['dow'].map(DOW_RU)
    return d

def compute_pareto(df):
    p = df.groupby('product')['revenue'].sum().sort_values(ascending=False).reset_index()
    p['cum'] = p['revenue'].cumsum() / p['revenue'].sum()
    cnt = (p['cum'] <= 0.8).sum(); total = len(p)
    return cnt, total, cnt/total*100 if total else 0

def compute_unit_eco(df):
    return df.groupby('order_id').agg(revenue=('revenue','sum'), items=('qty','sum'), date=('date','min')).reset_index()

def compute_var(df, pct=10):
    daily = df.groupby(df['date'].dt.date)['revenue'].sum()
    return np.percentile(daily, pct)

# ═══════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════
for k in ['ai_cfo','ai_plan']:
    if k not in st.session_state: st.session_state[k] = None

# ═══════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════
st.sidebar.markdown('<div class="sidebar-logo">⚡ Executive BI Suite</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="color:#475569;font-size:11px;margin-bottom:14px">Аналитическая платформа CFO & CMO</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

api_key = st.sidebar.text_input("🔑 Google Gemini API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)
    try:
        mlist = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        sel_model = st.sidebar.selectbox("🤖 AI модель", [m for m in mlist if 'gemini' in m])
    except:
        sel_model = "models/gemini-1.5-flash-latest"
else:
    sel_model = "models/gemini-1.5-flash-latest"
    st.sidebar.caption("Введите API Key для AI-аналитики")

uploaded_file = st.sidebar.file_uploader("📂 Реестр заказов (.csv / .xlsx)", type=['csv','xlsx'])

# Init export vars
month_df = model_df_display = mkt_dist = geo_dist = abc_summary = prod_dist = pd.DataFrame()
trend_direction = ""; change_pct = 0.0; orders_boost = 15.0; aov_boost = 10.0
horizon_months = 3; total_rev = 0; future_total_rev = future_total_upper = 0
future_forecast = monthly_forecast_totals = pd.DataFrame()
unique_orders = 0; aov = 0

# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
if uploaded_file:
    with st.spinner("⚙️ Профилирование номенклатуры..."):
        raw_df = load_and_clean_data(uploaded_file)

    if not raw_df.empty:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🎯 Параметры анализа**")
        horizon_months = st.sidebar.slider("Горизонт прогноза (мес.)", 1, 12, 3)
        gross_margin   = st.sidebar.slider("Валовая маржа (%)", 5, 80, 35) / 100
        all_statuses   = raw_df['status'].unique().tolist()
        status_filter  = st.sidebar.multiselect("Статус заказов", all_statuses, default=all_statuses)
        df = raw_df[raw_df['status'].isin(status_filter)].copy()

        with st.spinner("🔮 Расчёт прогноза..."):
            m_prophet, forecast, actual = generate_global_forecast(df, horizon_months * 30)
            total_rev     = df['revenue'].sum()
            unique_orders = df['order_id'].nunique()
            aov           = total_rev / unique_orders if unique_orders > 0 else 0

            if forecast is not None and not actual.empty:
                max_date      = pd.to_datetime(actual['ds'].max())
                future_forecast = forecast[forecast['ds'] > max_date].copy()
                future_total_rev   = max(future_forecast['yhat'].sum(), 0)
                future_total_upper = max(future_forecast['yhat_upper'].sum(), 0)
                future_total_lower = max(future_forecast['yhat_lower'].sum(), 0)
                future_forecast['month_str'] = future_forecast['ds'].dt.strftime('%Y-%m')
                monthly_forecast_totals = future_forecast.groupby('month_str')[['yhat','yhat_upper','yhat_lower']].sum().reset_index()
                avg_hist_daily = actual['y'].mean()
                avg_fore_daily = future_forecast['yhat'].mean()
                trend_direction = "восходящий 📈" if avg_fore_daily > avg_hist_daily else "нисходящий / стабилизационный 📉"
                change_pct = abs(avg_fore_daily - avg_hist_daily) / avg_hist_daily * 100 if avg_hist_daily > 0 else 0
            else:
                future_total_rev = future_total_upper = 0
                monthly_forecast_totals = pd.DataFrame()

        velocity    = compute_velocity(df)
        hhi, ch_sh  = compute_hhi(df)
        day_df      = compute_weekly_seasonality(df)
        var_10      = compute_var(df)
        p_cnt, p_tot, p_pct = compute_pareto(df)
        unit_eco    = compute_unit_eco(df)
        city_rev_df = df.groupby('city')['revenue'].sum().sort_values(ascending=False).reset_index()
        days_total  = max((df['date'].max() - df['date'].min()).days, 1)
        ltv_proxy   = aov * 3.2
        gp_per_order = aov * gross_margin
        max_cac     = gp_per_order * 0.3

        # ───────────────────────────────────────────
        # TABS
        # ───────────────────────────────────────────
        tabs = st.tabs([
            "⚡ Дашборд", "🔮 Прогноз", "🎛 Моделирование",
            "📣 Маркетинг", "🗺️ Гео", "💰 Продукты",
            "📐 Юнит-экономика", "🧠 AI CFO", "🚀 AI Стратегия"
        ])

        # ═══════════════════════════
        # TAB 0: DASHBOARD
        # ═══════════════════════════
        with tabs[0]:
            st.title("Главный дашборд")
            st.markdown(f'<div style="color:#475569;font-size:13px;margin-bottom:20px">Обновлено: {datetime.now().strftime("%d.%m.%Y %H:%M")} &nbsp;|&nbsp; Строк данных: {len(df):,}</div>', unsafe_allow_html=True)

            # KPIs — без прибыли, полные цифры
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("💰 Выручка (факт)", f"₴ {total_rev:,.0f}")
            c2.metric("🔮 Прогноз выручки", f"₴ {future_total_rev:,.0f}", f"на {horizon_months} мес. вперёд")
            c3.metric("🛒 Заказов (факт)", f"{unique_orders:,}")
            c4.metric("🧾 Средний чек", f"₴ {aov:,.0f}")
            c5.metric("📦 Активных SKU", f"{df['product'].nunique():,}")

            st.markdown("---")

            # ── Динамика выручки (недельная)
            section("📊 Динамика выручки по неделям с линией тренда")
            weekly = df.groupby(df['date'].dt.to_period('W').dt.start_time)['revenue'].sum().reset_index()
            weekly.columns = ['date','revenue']
            fig_main = go.Figure()
            fig_main.add_trace(go.Bar(x=weekly['date'], y=weekly['revenue'], name='Выручка (неделя)',
                                      marker_color='rgba(99,102,241,0.55)', marker_line_color='rgba(99,102,241,0.9)'))
            if len(weekly) >= 4:
                weekly['ma4'] = weekly['revenue'].rolling(4).mean()
                fig_main.add_trace(go.Scatter(x=weekly['date'], y=weekly['ma4'], name='Скользящая средняя (4 нед.)',
                                              line=dict(color='#f59e0b', width=2.5)))
            fig_main.update_layout(title="Выручка по неделям")
            T(fig_main, 400)
            st.plotly_chart(fig_main, use_container_width=True)
            caption("Столбцы — фактическая выручка за каждую неделю. Жёлтая линия — скользящая средняя за 4 недели (сглаживает колебания и показывает реальный тренд). Если линия идёт вверх — бизнес растёт.")

            # ── MoM рост
            section("📈 Прирост выручки месяц к месяцу (MoM)")
            if not velocity.empty:
                colors_mom = ['rgba(16,185,129,0.75)' if v >= 0 else 'rgba(239,68,68,0.75)' for v in velocity['mom_growth'].fillna(0)]
                fig_mom = go.Figure()
                fig_mom.add_trace(go.Bar(x=velocity['year_month'], y=velocity['mom_growth'],
                                         marker_color=colors_mom, name='Прирост, %'))
                fig_mom.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
                fig_mom.update_layout(title="Прирост выручки, % (каждый месяц к предыдущему)")
                T(fig_mom, 320)
                st.plotly_chart(fig_mom, use_container_width=True)
            caption("MoM (Month-over-Month) — показывает на сколько % выросла или упала выручка по сравнению с предыдущим месяцем. Зелёный = рост, красный = спад. Используется для оценки момента: растёт ли бизнес прямо сейчас.")

            # ── Средняя выручка по дням недели
            section("📅 Средняя выручка по дням недели — сезонность")
            fig_dow = go.Figure(go.Bar(
                x=day_df['dow_ru'], y=day_df['revenue'],
                marker=dict(color=day_df['revenue'],
                            colorscale=[[0,'rgba(99,102,241,0.3)'],[1,'rgba(99,102,241,1)']],
                            showscale=False)
            ))
            fig_dow.update_layout(title="Средняя дневная выручка по дням недели")
            T(fig_dow, 340)
            st.plotly_chart(fig_dow, use_container_width=True)
            caption("Показывает в какие дни недели в среднем приходит больше выручки. Помогает планировать запуск акций, рассылок и рекламных кампаний на пиковые дни.")

            # ── Автоинсайты
            section("🎯 Автоматические инсайты")
            last3 = velocity.tail(3)
            avg_mom_3 = last3['mom_growth'].mean() if not last3.empty else 0
            hhi_pct = hhi * 100

            c1, c2, c3 = st.columns(3)
            with c1:
                if avg_mom_3 > 5:
                    insight(f"<strong>Бизнес в фазе роста.</strong> Средний прирост за последние 3 месяца: <strong>+{avg_mom_3:.1f}%</strong> в месяц. При такой скорости годовой рост составит <strong>×{(1+avg_mom_3/100)**12:.1f}</strong> к текущей базе.", 'success', '🚀')
                elif avg_mom_3 < -5:
                    insight(f"<strong>⚠️ Нисходящая динамика.</strong> Средний спад за 3 мес.: <strong>{avg_mom_3:.1f}%</strong>. Требуется срочный аудит маркетинговых каналов и ассортиментной матрицы.", 'danger', '⚠️')
                else:
                    insight(f"<strong>Стабилизация.</strong> Прирост за 3 мес.: <strong>{avg_mom_3:.1f}%</strong>. Бизнес на плато — критический момент для выбора стратегии: масштабировать или оптимизировать.", 'warning', '📊')
            with c2:
                if hhi_pct > 50:
                    insight(f"<strong>Критическая зависимость от 1 канала.</strong> Индекс концентрации HHI = {hhi_pct:.0f}% (норма &lt;25%). Потеря основного источника трафика обнулит бизнес. Диверсификация — приоритет #1.", 'danger', '⚠️')
                elif hhi_pct > 25:
                    insight(f"<strong>Умеренная концентрация каналов.</strong> Индекс HHI = {hhi_pct:.0f}%. Допустимый уровень, но рекомендуется развивать дополнительные источники продаж.", 'warning', '📡')
                else:
                    insight(f"<strong>Здоровая диверсификация каналов.</strong> HHI = {hhi_pct:.0f}% — портфель источников сбалансирован, риск концентрации низкий.", 'success', '✅')
            with c3:
                insight(f"<strong>Закон Парето (80/20):</strong> {p_cnt} из {p_tot} товаров ({p_pct:.0f}%) генерируют 80% всей выручки. Остальные <strong>{p_tot-p_cnt} позиций</strong> — кандидаты на пересмотр ценообразования или вывод из ассортимента.", 'default', '📦')

        # ═══════════════════════════
        # TAB 1: FORECAST
        # ═══════════════════════════
        with tabs[1]:
            st.title(f"Прогноз продаж — {horizon_months} мес.")
            st.markdown('<div style="color:#64748b;font-size:14px;margin-bottom:20px">Математическая модель Prophet анализирует исторические данные, выявляет недельную и годовую сезонность, строит три сценария развития.</div>', unsafe_allow_html=True)

            if forecast is not None and not future_forecast.empty:

                # Прогнозный график
                section("📉 График прогноза (факт + 3 сценария)")
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=actual['ds'], y=actual['y'], name='Факт (история)',
                                            mode='lines', line=dict(color='#6366f1', width=2)))
                fig_fc.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'],
                                            name='Базовый прогноз', line=dict(color='#f59e0b', width=2.5)))
                fig_fc.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper'],
                                            name='Оптимистичный сценарий', line=dict(color='#10b981', width=1.5, dash='dot')))
                fig_fc.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower'],
                                            name='Пессимистичный сценарий', line=dict(color='#ef4444', width=1.5, dash='dot'),
                                            fill='tonexty', fillcolor='rgba(99,102,241,0.04)'))
                fig_fc.update_layout(title="Прогноз выручки (дни)", hovermode="x unified")
                T(fig_fc, 430)
                st.plotly_chart(fig_fc, use_container_width=True)
                caption("Синяя линия — реальные продажи в прошлом. Жёлтая — математически рассчитанный базовый прогноз с учётом трендов и сезонности. Зелёная — оптимистичный сценарий (верхняя граница). Красная — пессимистичный (нижняя граница). Закрашенная область — зона неопределённости.")

                # KPI прогноза
                uncertainty_pct = (future_total_upper - future_total_lower) / future_total_rev * 100 if future_total_rev > 0 else 0
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Базовый прогноз", f"₴ {future_total_rev:,.0f}")
                k2.metric("Оптимистичный", f"₴ {future_total_upper:,.0f}", f"+{(future_total_upper/future_total_rev-1)*100:.1f}%")
                k3.metric("Пессимистичный", f"₴ {future_total_lower:,.0f}", f"{(future_total_lower/future_total_rev-1)*100:.1f}%")
                k4.metric("Коридор неопределённости", f"±{uncertainty_pct/2:.1f}%")

                section("📋 Обоснование прогноза")
                insight(f"""
                <strong>Как работает прогноз (Prophet — алгоритм от Meta/Facebook):</strong> Модель находит в исторических данных два типа закономерностей — <em>недельную сезонность</em> (в какие дни продаётся лучше) и <em>годовую сезонность</em> (в какие месяцы пик/спад). Затем экстраполирует их вперёд.<br><br>
                Выявленный тренд: <strong>{trend_direction}</strong>. Среднесуточные продажи прогнозного периода отличаются от исторического базиса на <strong>{change_pct:+.1f}%</strong>.<br>
                Ширина коридора неопределённости: <strong>±{uncertainty_pct/2:.1f}%</strong> — чем шире, тем менее предсказуема ваша выручка (высокая волатильность в прошлых данных).
                """, 'default', '🧮')

                # Календарь событий
                section("📅 Ключевые события Украины 2026 — влияние на спрос")
                holidays_df = pd.DataFrame([
                    {"Дата": "28.06.2026", "Событие": "День Конституции Украины",    "Влияние на продажи": "Кратковременный спад B2B-активности, рост в HoReCa и досуге"},
                    {"Дата": "15.07.2026", "Событие": "День Государственности",      "Влияние на продажи": "Нейтральный эффект, стандартный летний торговый день"},
                    {"Дата": "24.08.2026", "Событие": "День Независимости 🇺🇦",       "Влияние на продажи": "Всплеск патриотического спроса +15-25%, мерч, подарки"},
                    {"Дата": "01.10.2026", "Событие": "День защитников Украины",     "Влияние на продажи": "Высокий спрос на подарки, специализированный ассортимент"},
                    {"Дата": "19.12.2026", "Событие": "Предновогодний пик",          "Влияние на продажи": "Сезонный максимум — готовить запасы заблаговременно"},
                ])
                st.dataframe(holidays_df, use_container_width=True, hide_index=True)

                # Прогноз по месяцам
                section("📊 Агрегированный прогноз по месяцам")
                month_df = monthly_forecast_totals.rename(columns={
                    'month_str':'Месяц','yhat':'Базовый (₴)','yhat_upper':'Оптимистичный (₴)','yhat_lower':'Пессимистичный (₴)'
                })
                st.dataframe(month_df.style.format({c:"{:,.0f}" for c in month_df.columns if c!='Месяц'}),
                             use_container_width=True, hide_index=True)
                caption("Суммарная выручка по каждому прогнозному месяцу в трёх сценариях. Базовый — наиболее вероятный. Используйте для бюджетирования и планирования закупок.")

                # Прогноз по дням
                section("📋 Детализация прогноза по дням")
                future_daily = future_forecast[['ds','yhat','yhat_upper','yhat_lower']].copy()
                future_daily['Дата']                = future_daily['ds'].dt.strftime('%d.%m.%Y')
                future_daily['Месяц']               = future_daily['ds'].dt.strftime('%Y-%m')
                future_daily['Базовый прогноз (₴)'] = future_daily['yhat']
                future_daily['Оптимистичный (₴)']   = future_daily['yhat_upper']
                future_daily['Пессимистичный (₴)']  = future_daily['yhat_lower']
                show_daily = future_daily.set_index(['Месяц','Дата'])[['Базовый прогноз (₴)','Оптимистичный (₴)','Пессимистичный (₴)']]
                st.dataframe(show_daily.style.format("{:,.0f}"), use_container_width=True, height=320)
                caption("Разбивка прогноза на каждый день горизонта. Полезно для еженедельного план-факт контроля, закупок и логистического планирования.")

            else:
                st.warning("Недостаточно данных для прогноза (минимум 10 дней продаж).")

        # ═══════════════════════════
        # TAB 2: MODELING
        # ═══════════════════════════
        with tabs[2]:
            st.title("Стратегическое моделирование")
            st.markdown('<div style="color:#94a3b8;font-size:14px;margin-bottom:16px">Симулятор «что если»: задайте желаемый рост числа заказов и среднего чека — платформа рассчитает совокупный финансовый эффект с учётом синергии.</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            orders_boost  = c1.slider("📦 Рост кол-ва заказов / трафика (%)", -50.0, 100.0, 15.0, 1.0)
            aov_boost     = c2.slider("💳 Рост среднего чека / цен (%)", -50.0, 100.0, 10.0, 1.0)
            scenario_name = c3.selectbox("Сценарий-база", ["Базовый прогноз","Оптимистичный","Пессимистичный"])

            insight(f"""
            <strong>Как считается (составной процент):</strong><br>
            Шаг 1: База × (1 + {orders_boost:.0f}%) = выручка после роста количества заказов<br>
            Шаг 2: Результат шага 1 × (1 + {aov_boost:.0f}%) = итоговая выручка после роста среднего чека<br>
            Мультипликатор: <strong>×{(1+orders_boost/100)*(1+aov_boost/100):.3f}</strong> к базовому прогнозу.<br>
            <em>Важно:</em> рост чека применяется к уже увеличенному объёму заказов — поэтому итог выше, чем простая сумма двух процентов. Это и есть эффект синергии.
            """, 'default', '📐')

            if not monthly_forecast_totals.empty:
                sc_col = {'Базовый прогноз':'yhat','Оптимистичный':'yhat_upper','Пессимистичный':'yhat_lower'}[scenario_name]
                model_df = monthly_forecast_totals[['month_str']].copy()
                model_df.rename(columns={'month_str':'Месяц'}, inplace=True)
                model_df['База прогноза (₴)']    = monthly_forecast_totals[sc_col].clip(lower=0)
                model_df['После роста заказов (₴)'] = model_df['База прогноза (₴)'] * (1 + orders_boost/100)
                model_df['Итог с ростом чека (₴)']  = model_df['После роста заказов (₴)'] * (1 + aov_boost/100)
                model_df['Δ от роста заказов (₴)']  = model_df['После роста заказов (₴)'] - model_df['База прогноза (₴)']
                model_df['Δ от роста чека (₴)']     = model_df['Итог с ростом чека (₴)']  - model_df['После роста заказов (₴)']
                model_df['Δ синергия (₴)']           = (model_df['Итог с ростом чека (₴)'] - model_df['База прогноза (₴)']
                                                         - model_df['База прогноза (₴)'] * orders_boost/100
                                                         - model_df['База прогноза (₴)'] * aov_boost/100)
                model_df['Совокупный прирост (₴)']  = model_df['Итог с ростом чека (₴)'] - model_df['База прогноза (₴)']

                totals = {c: model_df[c].sum() if c!='Месяц' else 'ИТОГО' for c in model_df.columns}
                model_df_display = pd.concat([model_df, pd.DataFrame([totals])], ignore_index=True)
                st.dataframe(model_df_display.style.format({c:"{:,.0f}" for c in model_df_display.columns if c!='Месяц'}),
                             use_container_width=True, hide_index=True)
                caption("Таблица декомпозиции: каждый столбец показывает сколько выручки добавляет каждый рычаг управления. Строка ИТОГО — суммарный эффект за весь прогнозный период.")

                t_base     = model_df['База прогноза (₴)'].sum()
                t_orders   = model_df['Δ от роста заказов (₴)'].sum()
                t_aov_d    = model_df['Δ от роста чека (₴)'].sum()
                t_syn      = model_df['Δ синергия (₴)'].sum()
                t_total    = model_df['Совокупный прирост (₴)'].sum()

                section("💡 Факторный анализ прироста")
                col_a, col_b = st.columns(2)
                with col_a:
                    insight(f"""
                    <strong>Декомпозиция прироста выручки:</strong><br>
                    • База ({scenario_name}): <strong>₴ {t_base:,.0f}</strong><br>
                    • Эффект роста кол-ва заказов (+{orders_boost:.0f}%): <strong>₴ {t_orders:+,.0f}</strong><br>
                    • Эффект роста среднего чека (+{aov_boost:.0f}%, применён к новому объёму): <strong>₴ {t_aov_d:+,.0f}</strong><br>
                    • Синергетический «перемножающий» эффект: <strong>₴ {t_syn:+,.0f}</strong><br>
                    ━━━━━━━━━━━━━━━━━━━━━━━━━<br>
                    Совокупный прирост выручки: <strong>₴ {t_total:+,.0f}</strong>
                    """, 'success', '📊')
                with col_b:
                    insight(f"""
                    <strong>Итоговый P&L эффект (прибыль и убытки):</strong><br>
                    • Итоговая выручка сценария: <strong>₴ {t_base+t_total:,.0f}</strong><br>
                    • Рост выручки vs база: <strong>{t_total/t_base*100:+.1f}%</strong><br>
                    • Дополнительная валовая прибыль (маржа {gross_margin*100:.0f}%): <strong>₴ {t_total*gross_margin:+,.0f}</strong><br>
                    • Максимальный бюджет на маркетинг для безубыточности сценария: <strong>₴ {t_total*gross_margin:,.0f}</strong>
                    """, 'default', '💰')

                # Waterfall
                section("📊 Водопадная диаграмма декомпозиции прироста")
                fig_wf = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=['absolute','relative','relative','relative','total'],
                    x=['База','Δ Заказы','Δ Чек','Δ Синергия','Итого'],
                    y=[t_base, t_orders, t_aov_d, t_syn, t_base+t_total],
                    connector=dict(line=dict(color='rgba(255,255,255,0.1)')),
                    decreasing=dict(marker_color='rgba(239,68,68,0.7)'),
                    increasing=dict(marker_color='rgba(16,185,129,0.7)'),
                    totals=dict(marker_color='rgba(99,102,241,0.8)')
                ))
                fig_wf.update_layout(title="Декомпозиция прироста выручки (₴)")
                T(fig_wf, 360)
                st.plotly_chart(fig_wf, use_container_width=True)
                caption("Водопадная диаграмма (Waterfall chart) наглядно показывает: сколько выручки даёт каждый рычаг. «База» — стартовая точка, «Δ Заказы» — вклад роста числа заказов, «Δ Чек» — вклад роста суммы заказа, «Δ Синергия» — дополнительный эффект от одновременного роста обоих показателей, «Итого» — финальная сумма.")
            else:
                st.warning("Нет данных прогноза. Нужно минимум 10 дней продаж.")

        # ═══════════════════════════
        # TAB 3: MARKETING
        # ═══════════════════════════
        with tabs[3]:
            st.title("Маркетинговая аналитика")

            channel_rev = df.groupby('utm_source')['revenue'].sum().sort_values(ascending=False)

            section("🌐 Структура доходов по UTM-цепочке (источник → канал → кампания)")
            fig_sun = px.sunburst(df, path=['utm_source','utm_medium','utm_campaign'], values='revenue',
                                  color_discrete_sequence=px.colors.qualitative.Vivid)
            fig_sun.update_layout(title="Доходы по маркетинговым каналам (иерархия)")
            T(fig_sun, 460)
            st.plotly_chart(fig_sun, use_container_width=True)
            caption("Солнечная диаграмма (Sunburst) показывает иерархию: внешнее кольцо — UTM Source (откуда пришёл клиент: google, facebook, email и т.д.), среднее — UTM Medium (тип трафика: cpc, organic, email), внутреннее — конкретная кампания. Размер сектора = доля выручки.")

            section("📊 Выручка по источникам трафика")
            fig_ch = go.Figure(go.Bar(y=channel_rev.index, x=channel_rev.values, orientation='h',
                                      marker=dict(color='rgba(99,102,241,0.75)')))
            fig_ch.update_layout(title="Выручка по каналам (₴)")
            T(fig_ch, 320)
            st.plotly_chart(fig_ch, use_container_width=True)
            caption("Горизонтальный барчарт: каждая строка — один источник трафика (UTM Source). Длина полосы = сумма выручки, которую принёс этот канал за весь период.")

            if not channel_rev.empty:
                top_s = channel_rev.index[0]; top_v = channel_rev.values[0]
                bot_s = channel_rev.index[-1]; bot_v = channel_rev.values[-1]
                c1, c2, c3 = st.columns(3)
                with c1:
                    insight(f"<strong>Главный канал роста:</strong> <code>{top_s}</code> — генерирует <strong>{top_v/total_rev*100:.1f}%</strong> всей выручки (₴ {top_v:,.0f}). В прогнозном периоде его вклад составит ~₴ {future_total_rev*(top_v/total_rev):,.0f}.", 'success', '🏆')
                with c2:
                    insight(f"<strong>Риск зависимости:</strong> если канал <code>{top_s}</code> перестанет работать — потеря за прогнозный период составит ~₴ {future_total_rev*(top_v/total_rev):,.0f}. Рекомендуется развивать страховочный канал.", 'danger', '🛡️')
                with c3:
                    if len(channel_rev) > 1:
                        insight(f"<strong>Слабый канал:</strong> <code>{bot_s}</code> принёс лишь ₴ {bot_v:,.0f}. Рекомендуется: пересмотр бюджета, смена креативов или 30-дневная пауза с последующей оценкой.", 'warning', '⚙️')

            section("📈 Динамика каналов по месяцам")
            ch_monthly = df.groupby(['year_month','utm_source'])['revenue'].sum().reset_index()
            fig_ch_t = px.line(ch_monthly, x='year_month', y='revenue', color='utm_source',
                               title="Выручка по каналам (каждый месяц)")
            T(fig_ch_t, 340)
            st.plotly_chart(fig_ch_t, use_container_width=True)
            caption("Линейный график: каждый цвет — отдельный канал. Позволяет видеть сезонность и динамику каждого источника трафика. Падающая линия = канал теряет эффективность.")

            section("📈 Прогноз выручки по каналам (помесячно)")
            if not monthly_forecast_totals.empty:
                mkt_dist = get_monthly_distribution(df, 'utm_source', monthly_forecast_totals, total_rev)
                mkt_dist = mkt_dist.rename(columns={'utm_source':'Источник'})
                st.dataframe(mkt_dist.style.format({c:"{:,.0f}" for c in mkt_dist.columns if c!='Источник'}),
                             hide_index=True, use_container_width=True)
            caption("Прогноз распределяется пропорционально исторической доле каждого канала. Используйте для планирования маркетинговых бюджетов и KPI по каналам.")

        # ═══════════════════════════
        # TAB 4: GEO
        # ═══════════════════════════
        with tabs[4]:
            st.title("Геоаналитика — продажи по регионам")

            section("🏙️ Топ-20 городов по выручке (исторический факт)")
            fig_geo = px.bar(city_rev_df.head(20), x='city', y='revenue',
                             color='revenue', color_continuous_scale=[[0,'rgba(99,102,241,0.3)'],[1,'rgba(99,102,241,1)']],
                             title="Топ-20 городов — факт")
            T(fig_geo, 400)
            st.plotly_chart(fig_geo, use_container_width=True)
            caption("Столбчатая диаграмма: каждый столбец — один город. Высота = суммарная выручка за весь период. Цветовая интенсивность дублирует высоту для визуальной наглядности.")

            section("🥧 Распределение доли выручки по городам")
            top10 = city_rev_df.head(10).copy()
            oth = city_rev_df.iloc[10:]['revenue'].sum()
            if oth > 0:
                top10 = pd.concat([top10, pd.DataFrame([{'city':'Остальные','revenue':oth}])], ignore_index=True)
            fig_pie = px.pie(top10, names='city', values='revenue', title="Доли городов в выручке")
            T(fig_pie, 380)
            st.plotly_chart(fig_pie, use_container_width=True)
            caption("Круговая диаграмма: наглядно показывает насколько выручка сконцентрирована в нескольких городах vs распределена по всей стране. Большой «Остальные» = потенциал для региональной экспансии.")

            if not city_rev_df.empty:
                top_city = city_rev_df.iloc[0]['city']
                top_share = city_rev_df.iloc[0]['revenue'] / total_rev * 100
                top3_share = city_rev_df.head(3)['revenue'].sum() / total_rev * 100
                c1, c2 = st.columns(2)
                with c1:
                    insight(f"<strong>Доминирующий регион:</strong> <code>{top_city}</code> формирует <strong>{top_share:.1f}%</strong> выручки. ТОП-3 города — <strong>{top3_share:.1f}%</strong>. Рекомендуется усиливать логистику и складской сток под эти регионы.", 'default', '📍')
                with c2:
                    if len(city_rev_df) > 10:
                        tail_sh = city_rev_df.iloc[10:]['revenue'].sum() / total_rev * 100
                        insight(f"<strong>Потенциал экспансии:</strong> города за пределами ТОП-10 дают лишь <strong>{tail_sh:.1f}%</strong> выручки. Целевые региональные кампании могут значительно увеличить охват.", 'warning', '🗺️')

            section("📈 Динамика ТОП-5 городов по месяцам")
            top5 = city_rev_df.head(5)['city'].tolist()
            cmon = df.groupby(['year_month','city'])['revenue'].sum().reset_index()
            fig_ct = px.line(cmon[cmon['city'].isin(top5)], x='year_month', y='revenue', color='city',
                             title="Выручка по месяцам (ТОП-5 городов)")
            T(fig_ct, 340)
            st.plotly_chart(fig_ct, use_container_width=True)
            caption("Помесячная динамика продаж в разрезе городов. Помогает выявить региональную сезонность и отслеживать рост/спад конкретных рынков.")

            section("📍 Прогноз продаж по городам (Топ-25, помесячно)")
            if not monthly_forecast_totals.empty:
                geo_dist = get_monthly_distribution(df, 'city', monthly_forecast_totals, total_rev)
                geo_dist = geo_dist.rename(columns={'city':'Город'})
                st.dataframe(geo_dist.head(25).style.format({c:"{:,.0f}" for c in geo_dist.columns if c!='Город'}),
                             use_container_width=True, hide_index=True)
            caption("Прогнозная выручка распределена пропорционально исторической доле каждого города. Используйте для регионального бюджетирования и планирования доставки.")

        # ═══════════════════════════
        # TAB 5: PRODUCTS
        # ═══════════════════════════
        with tabs[5]:
            st.title("Товарная аналитика & ABC-анализ")

            df['product_full'] = df['art'] + " — " + df['product']
            prod_base = df.groupby('product_full').agg({'qty':'sum','revenue':'sum'}).sort_values('revenue',ascending=False).reset_index()
            prod_base['share'] = prod_base['revenue'] / total_rev
            prod_base['cum_share'] = prod_base['share'].cumsum()
            prod_base['ABC'] = prod_base['cum_share'].apply(lambda c: 'A' if c<=0.8 else ('B' if c<=0.95 else 'C'))
            abc_summary = prod_base.groupby('ABC').agg(SKU=('product_full','count'), Выручка=('revenue','sum')).reset_index()
            abc_summary['Доля %'] = abc_summary['Выручка'] / total_rev * 100
            abc_summary.rename(columns={'ABC':'Класс'}, inplace=True)

            section("📊 ABC-матрица ассортимента (анализ Парето)")
            st.dataframe(abc_summary.style.format({'Выручка':"{:,.0f} ₴",'Доля %':"{:.1f}%"}),
                         hide_index=True, use_container_width=True)
            caption("ABC-анализ делит все товары на 3 группы по вкладу в выручку: Класс A — товары формирующие первые 80% выручки (ядро бизнеса, нельзя допускать стокаутов); Класс B — следующие 15% (поддерживающий ассортимент); Класс C — последние 5% (длинный хвост, кандидаты на вывод или оптимизацию).")

            a_sku = abc_summary[abc_summary['Класс']=='A']['SKU'].sum() if not abc_summary.empty else 0
            c_sku = abc_summary[abc_summary['Класс']=='C']['SKU'].sum() if not abc_summary.empty else 0
            c1, c2 = st.columns(2)
            with c1:
                insight(f"<strong>Класс A ({a_sku} SKU):</strong> Ядро бизнеса. Максимальный приоритет при закупках — ноль стоковых аутов (отсутствие товара на складе). Контролировать наличие ежедневно.", 'success', '🅰️')
            with c2:
                insight(f"<strong>Класс C ({c_sku} SKU):</strong> Минимальный вклад в выручку. Рекомендуется: заморозить закупки, провести распродажу остатков, освободить складской капитал.", 'warning', '🅲')

            section("📈 Кривая Парето — топ-30 товаров по выручке")
            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(x=prod_base['product_full'].str[:35].head(30), y=prod_base['revenue'].head(30),
                                   name='Выручка', marker_color='rgba(99,102,241,0.65)'))
            max_rev_top30 = prod_base['revenue'].head(30).max() or 1
            cum_max = prod_base['cum_share'].head(30).max() or 1
            fig_p.add_trace(go.Scatter(x=prod_base['product_full'].str[:35].head(30),
                                       y=prod_base['cum_share'].head(30) * max_rev_top30 / cum_max,
                                       name='Накопленная доля', mode='lines',
                                       line=dict(color='#f59e0b', width=2.5), yaxis='y2'))
            fig_p.update_layout(title="Кривая Парето (Топ-30 SKU)",
                                yaxis2=dict(overlaying='y', side='right', showgrid=False, color='#f59e0b'))
            T(fig_p, 420)
            st.plotly_chart(fig_p, use_container_width=True)
            caption("Синие столбцы — выручка каждого товара (от большего к меньшему). Жёлтая линия — накопленная доля: точка где она достигает 80% — это граница ABC-класса A. Всё правее — классы B и C.")

            section("📦 Прогноз по товарам — Топ-50 (помесячно)")
            if not monthly_forecast_totals.empty:
                prod_dist = get_monthly_distribution(df, 'product_full', monthly_forecast_totals, total_rev)
                prod_dist = prod_dist.merge(prod_base[['product_full','ABC']], on='product_full', how='left')
                prod_dist = prod_dist.rename(columns={'product_full':'Товар','ABC':'Класс'})
                cols = ['Товар','Класс'] + [c for c in prod_dist.columns if c not in ['Товар','Класс']]
                prod_dist = prod_dist[cols]
                st.dataframe(prod_dist.head(50).style.format({c:"{:,.0f}" for c in prod_dist.columns if c not in ['Товар','Класс']}),
                             use_container_width=True, hide_index=True)
            caption("Для каждого товара рассчитана ожидаемая выручка в каждом прогнозном месяце — пропорционально его исторической доле. Колонка «Класс» показывает ABC-категорию для приоритизации закупок.")

        # ═══════════════════════════
        # TAB 6: UNIT ECONOMICS
        # ═══════════════════════════
        with tabs[6]:
            st.title("Юнит-экономика — финансовое здоровье бизнеса")
            st.markdown("""
            <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);border-radius:10px;padding:16px 20px;margin-bottom:20px;font-size:14px;line-height:1.8;color:#cbd5e1">
            <strong>Что такое юнит-экономика?</strong><br>
            Юнит-экономика — это финансовый анализ одной единицы бизнеса (одного заказа, одного клиента). 
            Отвечает на вопрос: <em>сколько денег зарабатывает или теряет бизнес с каждого заказа?</em>
            Позволяет понять — масштабирование будет прибыльным или убыточным.
            </div>
            """, unsafe_allow_html=True)

            rev_per_day   = total_rev / days_total
            orders_per_day = unique_orders / days_total

            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Выручка в день", f"₴ {rev_per_day:,.0f}")
            k2.metric("Заказов в день", f"{orders_per_day:.1f}")
            k3.metric("Средний чек (AOV)", f"₴ {aov:,.0f}")
            k4.metric("Валовая прибыль / заказ", f"₴ {gp_per_order:,.0f}")
            k5.metric("LTV-прокси", f"₴ {ltv_proxy:,.0f}")
            k6.metric("Макс. допустимый CAC", f"₴ {max_cac:,.0f}")

            insight("""
            <strong>Расшифровка ключевых показателей:</strong><br>
            • <strong>AOV (Average Order Value)</strong> — средний чек, средняя сумма одного заказа. Формула: Выручка ÷ Кол-во заказов.<br>
            • <strong>Валовая прибыль / заказ</strong> — сколько зарабатывает бизнес с одного заказа после вычета себестоимости товара (задаётся в настройках как % маржи).<br>
            • <strong>LTV (Lifetime Value, пожизненная ценность клиента)</strong> — сколько денег приносит один клиент за всё время сотрудничества. Прокси-формула: AOV × 3.2 (среднее кол-во повторных заказов).<br>
            • <strong>CAC (Customer Acquisition Cost)</strong> — стоимость привлечения одного нового клиента. Максимально допустимый CAC = 30% от валовой прибыли с заказа. Если реальный CAC выше — бизнес работает в убыток на привлечении.
            """, 'default', '📚')

            section("📊 Распределение заказов по сумме (гистограмма)")
            p95 = unit_eco['revenue'].quantile(0.95)
            fig_hist = go.Figure(go.Histogram(x=unit_eco['revenue'].clip(upper=p95), nbinsx=40,
                                              marker_color='rgba(99,102,241,0.65)', name='Заказы'))
            fig_hist.update_layout(title=f"Сумма заказов (до 95-го перцентиля, P95 = ₴{p95:,.0f})")
            T(fig_hist, 340)
            st.plotly_chart(fig_hist, use_container_width=True)
            caption("Гистограмма показывает сколько заказов попадает в каждый ценовой диапазон. Пик гистограммы — это самый популярный диапазон сумм заказа. Срезано на 95-м перцентиле чтобы убрать выбросы (очень крупные нетипичные заказы) и сделать график читабельным.")

            section("📦 Количество товаров в одном заказе")
            items_p95 = unit_eco['items'].quantile(0.95)
            fig_items = go.Figure(go.Histogram(x=unit_eco['items'].clip(upper=items_p95), nbinsx=20,
                                               marker_color='rgba(16,185,129,0.65)', name='Товаров в заказе'))
            fig_items.update_layout(title="Кол-во товаров в заказе (до P95)")
            T(fig_items, 300)
            st.plotly_chart(fig_items, use_container_width=True)
            caption("Показывает типичную «глубину» заказа. Если большинство заказов содержат 1 товар — есть потенциал для up-sell и cross-sell (предложения сопутствующих товаров). Рост этого показателя = рост среднего чека без привлечения новых клиентов.")

            section("📈 Еженедельный объём заказов (скорость бизнеса)")
            ord_wk = df.groupby('week')['order_id'].nunique().reset_index()
            ord_wk.columns = ['week','orders']
            fig_vel = go.Figure(go.Scatter(x=ord_wk['week'], y=ord_wk['orders'], mode='lines+markers',
                                           fill='tozeroy', line=dict(color='#6366f1'),
                                           fillcolor='rgba(99,102,241,0.1)'))
            fig_vel.update_layout(title="Заказов в неделю (Velocity)")
            T(fig_vel, 300)
            st.plotly_chart(fig_vel, use_container_width=True)
            caption("Velocity (скорость) — еженедельное кол-во заказов. Нарастающий тренд = бизнес ускоряется. Плато = стагнация. Резкие провалы = аномалии (технические проблемы, сезонный спад, отключение рекламы).")

            section("⚠️ Метрики финансового риска")
            p25 = unit_eco['revenue'].quantile(0.25)
            p75 = unit_eco['revenue'].quantile(0.75)
            r1, r2, r3 = st.columns(3)
            with r1:
                insight(f"""
                <strong>VaR 10% — «Выручка под риском»:</strong><br>
                Value at Risk (VaR) — в 10% худших дней выручка не превысит <strong>₴ {var_10:,.0f}</strong>.<br>
                Это минимальный резерв ликвидности на 1 неделю: <strong>₴ {int(7*var_10):,}</strong>.<br>
                <em>Практический смысл:</em> сколько денег нужно держать «на чёрный день» чтобы покрывать операционные расходы в худшие периоды.
                """, 'warning', '📉')
            with r2:
                insight(f"""
                <strong>IQR — межквартильный диапазон чеков:</strong><br>
                Межквартильный размах (IQR) = разница между 75-м и 25-м перцентилями.<br>
                25% заказов: до ₴ {p25:,.0f} | 75% заказов: до ₴ {p75:,.0f}<br>
                IQR = <strong>₴ {p75-p25:,.0f}</strong>.<br>
                <em>Практический смысл:</em> чем меньше IQR — тем стабильнее средний чек. Большой IQR = высокая вариативность заказов.
                """, 'default', '📐')
            with r3:
                insight(f"""
                <strong>LTV / CAC — золотое правило юнит-экономики:</strong><br>
                LTV (прокси) = ₴ {ltv_proxy:,.0f} | Макс. CAC = ₴ {max_cac:,.0f}<br>
                Отношение LTV/CAC должно быть <strong>≥ 3</strong>.<br>
                При CAC ≤ ₴ {max_cac:,.0f} — бизнес масштабируется прибыльно.<br>
                <em>Практический смысл:</em> если вы тратите больше на привлечение клиента чем он приносит — масштабирование убыточно.
                """, 'success', '💎')

        # ═══════════════════════════
        # TAB 7: AI CFO
        # ═══════════════════════════
        with tabs[7]:
            st.title("🧠 AI Финансовый Директор — CFO Брифинг")
            st.markdown('<div style="color:#64748b;font-size:14px;margin-bottom:20px">AI анализирует все показатели платформы и генерирует стратегический брифинг уровня совета директоров — с конкретными выводами, рисками и рекомендациями.</div>', unsafe_allow_html=True)
            if not api_key:
                st.warning("⚠️ Введите Google API Key в боковом меню.")
            else:
                if st.button("⚡ Сгенерировать CFO брифинг"):
                    with st.spinner("AI анализирует данные..."):
                        try:
                            genai.configure(api_key=api_key)
                            mdl = genai.GenerativeModel(sel_model)
                            top_city    = city_rev_df.iloc[0]['city'] if not city_rev_df.empty else "N/A"
                            top_channel = df.groupby('utm_source')['revenue'].sum().idxmax() if total_rev > 0 else "N/A"
                            ctx = {
                                "выручка_факт": f"₴{total_rev:,.0f}", "выручка_прогноз": f"₴{future_total_rev:,.0f}",
                                "горизонт_мес": horizon_months, "AOV": f"₴{aov:,.0f}",
                                "LTV_прокси": f"₴{ltv_proxy:,.0f}", "маржа": f"{gross_margin*100:.0f}%",
                                "тренд": trend_direction, "изм_тренда_%": f"{change_pct:.1f}%",
                                "топ_канал": top_channel, "HHI_концентрация": f"{hhi:.2f}",
                                "топ_город": top_city, "SKU": df['product'].nunique(),
                                "pareto_pct": f"{p_pct:.0f}%", "VaR_10%": f"₴{var_10:,.0f}",
                            }
                            prompt = f"""Ты Chief Financial Officer крупной украинской e-commerce компании, уровень Goldman Sachs / McKinsey.
Данные: {json.dumps(ctx, ensure_ascii=False)}

Напиши СТРОГИЙ, КОНКРЕТНЫЙ стратегический брифинг для Board of Directors. Структура:

## EXECUTIVE SUMMARY (3 предложения — ключевые цифры)
## RED FLAGS — КРИТИЧЕСКИЕ РИСКИ (топ-3, с цифрами)
## FINANCIAL HEALTH ASSESSMENT (P&L, ликвидность, устойчивость)
## STRATEGIC RECOMMENDATIONS (5 конкретных действий с ROI-потенциалом)
## OUTLOOK (реалистичность прогнозных цифр)

Тон: прямой, без воды. Стиль: совет директоров Нью-Йорка. Конкретные цифры из данных."""
                            resp = mdl.generate_content(prompt)
                            st.session_state['ai_cfo'] = resp.text
                            st.markdown(resp.text)
                        except Exception as e:
                            if "429" in str(e) or "ResourceExhausted" in str(e):
                                st.error("🛑 Лимит Google API исчерпан. Смените модель или попробуйте позже.")
                            else:
                                st.error(f"❌ Ошибка AI: {e}")
                elif st.session_state['ai_cfo']:
                    st.markdown(st.session_state['ai_cfo'])

        # ═══════════════════════════
        # TAB 8: AI STRATEGY
        # ═══════════════════════════
        with tabs[8]:
            st.title("🚀 AI Стратегия роста — 90-дневный план")
            st.markdown('<div style="color:#64748b;font-size:14px;margin-bottom:20px">AI разрабатывает операционный план масштабирования на 90 дней с конкретными действиями, KPI и рекомендацией по маркетинговому миксу.</div>', unsafe_allow_html=True)
            if not api_key:
                st.warning("⚠️ Введите Google API Key в боковом меню.")
            else:
                if st.button("🚀 Разработать план роста"):
                    with st.spinner("AI строит стратегию..."):
                        try:
                            genai.configure(api_key=api_key)
                            mdl = genai.GenerativeModel(sel_model)
                            top_ch = df.groupby('utm_source')['revenue'].sum().sort_values(ascending=False)
                            prompt = f"""Ты Chief Marketing Officer e-commerce компании (уровень Shopify / Rozetka).
Данные: выручка ₴{total_rev:,.0f}, прогноз {horizon_months} мес. ₴{future_total_rev:,.0f}, AOV ₴{aov:,.0f}, LTV ₴{ltv_proxy:,.0f}, топ-канал: {top_ch.index[0] if len(top_ch)>0 else 'N/A'}, HHI: {hhi:.2f}, SKU: {df['product'].nunique()}.

Напиши КОНКРЕТНЫЙ 90-ДНЕВНЫЙ ПЛАН РОСТА:

## QUICK WINS (0-30 дней) — 5 действий с прогнозируемым uplift %
## GROWTH ENGINE (30-60 дней) — масштабирование каналов, рост AOV
## SCALE (60-90 дней) — новые каналы, retention, ассортимент
## MARKETING MIX — перераспределение бюджета по каналам (%)
## KPI DASHBOARD — 10 метрик для еженедельного мониторинга

Конкретно, с цифрами, без воды. Стиль: операционный директор Нью-Йорской компании."""
                            resp = mdl.generate_content(prompt)
                            st.session_state['ai_plan'] = resp.text
                            st.markdown(resp.text)
                        except Exception as e:
                            if "429" in str(e) or "ResourceExhausted" in str(e):
                                st.error("🛑 Лимит Google API исчерпан.")
                            else:
                                st.error(f"❌ Ошибка AI: {e}")
                elif st.session_state['ai_plan']:
                    st.markdown(st.session_state['ai_plan'])

        # ═══════════════════════════════════════════
        # SIDEBAR EXPORT
        # ═══════════════════════════════════════════
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🖨️ Экспорт отчёта**")

        def df2html(dff, maxr=None):
            if dff is None or dff.empty: return "<p><em>Нет данных</em></p>"
            d = dff.head(maxr) if maxr else dff
            return d.to_html(index=False, border=0, classes='dt')

        ai_cfo_html = ""
        ai_plan_html = ""
        if st.session_state['ai_cfo']:
            ai_cfo_html = f"""<div class="pb"></div>
            <h2>7. Заключение AI CFO (Board Briefing)</h2>
            <div class="ai-box">{st.session_state['ai_cfo'].replace(chr(10),'<br>')}</div>"""
        if st.session_state['ai_plan']:
            ai_plan_html = f"""<h2>8. AI План роста (90 дней)</h2>
            <div class="ai-box">{st.session_state['ai_plan'].replace(chr(10),'<br>')}</div>"""

        html_report = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Executive BI Report — {datetime.now().strftime('%d.%m.%Y')}</title>
<style>
  @page {{ size: A4 portrait; margin: 14mm 12mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5px; color: #1e293b; line-height: 1.5; background: #fff; }}
  .header {{ background: linear-gradient(135deg, #1e1b4b, #312e81); color: #fff; padding: 18px 22px; border-radius: 8px; margin-bottom: 18px; }}
  .header h1 {{ margin: 0; font-size: 20px; font-weight: 700; color: #fff; }}
  .header .sub {{ color: #a5b4fc; font-size: 10.5px; margin-top: 3px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(5,1fr); gap: 8px; margin-bottom: 18px; }}
  .kpi {{ background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #6366f1; border-radius: 6px; padding: 9px 11px; }}
  .kpi-l {{ font-size: 9px; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }}
  .kpi-v {{ font-size: 16px; font-weight: 700; color: #1e293b; margin-top: 2px; }}
  h2 {{ color: #1e293b; font-size: 13px; font-weight: 700; border-bottom: 2px solid #6366f1; padding-bottom: 4px; margin: 18px 0 9px; page-break-after: avoid; }}
  .note {{ background: #eff6ff; border-left: 3px solid #3b82f6; padding: 7px 11px; border-radius: 0 5px 5px 0; font-size: 9.5px; margin: 6px 0 10px; }}
  table.dt {{ width: 100%; border-collapse: collapse; font-size: 9px; margin-bottom: 12px; page-break-inside: auto; }}
  table.dt th {{ background: #1e1b4b; color: #fff; padding: 5px 6px; text-align: left; font-weight: 600; }}
  table.dt td {{ padding: 4px 6px; border-bottom: 1px solid #f1f5f9; }}
  table.dt tr:nth-child(even) td {{ background: #f8fafc; }}
  table.dt tr:last-child td {{ font-weight: 700; background: #ede9fe; border-top: 2px solid #6366f1; }}
  .pb {{ page-break-before: always; }}
  .ai-box {{ background: #fafafa; border-left: 4px solid #6366f1; border-radius: 4px; padding: 12px 15px; font-size: 10px; line-height: 1.7; white-space: pre-wrap; }}
  .footer {{ text-align: center; color: #94a3b8; font-size: 9px; margin-top: 28px; border-top: 1px solid #e2e8f0; padding-top: 8px; }}
</style>
</head>
<body>
<div class="header">
  <h1>⚡ Executive BI Report</h1>
  <div class="sub">Сформировано: {datetime.now().strftime('%d.%m.%Y %H:%M')} &nbsp;|&nbsp; Горизонт: {horizon_months} мес. &nbsp;|&nbsp; Статусы: {', '.join(status_filter)}</div>
</div>
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-l">Выручка (факт)</div><div class="kpi-v">₴ {total_rev:,.0f}</div></div>
  <div class="kpi"><div class="kpi-l">Прогноз (база)</div><div class="kpi-v">₴ {future_total_rev:,.0f}</div></div>
  <div class="kpi"><div class="kpi-l">Заказов</div><div class="kpi-v">{unique_orders:,}</div></div>
  <div class="kpi"><div class="kpi-l">Средний чек</div><div class="kpi-v">₴ {aov:,.0f}</div></div>
  <div class="kpi"><div class="kpi-l">Тренд / Δ</div><div class="kpi-v" style="font-size:12px">{trend_direction} {change_pct:+.1f}%</div></div>
</div>

<h2>1. Прогноз продаж по месяцам (алгоритм Prophet)</h2>
<div class="note">Prophet — алгоритм прогнозирования временных рядов от Meta. Учитывает сезонность (недельную и годовую) и праздники Украины.</div>
{df2html(month_df)}

<h2>2. Стратегическое моделирование «что если»</h2>
<div class="note">Составной прирост: Итог = База × (1 + Δ заказы) × (1 + Δ чек). Мультипликатор: ×{(1+orders_boost/100)*(1+aov_boost/100):.3f}. Рост заказов: {orders_boost:+.0f}% | Рост чека: {aov_boost:+.0f}%</div>
{df2html(model_df_display)}

<div class="pb"></div>

<h2>3. Маркетинг — прогноз по каналам трафика</h2>
<div class="note">Распределение прогнозной выручки по UTM-источникам пропорционально исторической доле каждого канала.</div>
{df2html(mkt_dist)}

<h2>4. Геоаналитика — прогноз по городам (Топ-25)</h2>
<div class="note">Распределение прогнозной выручки по городам пропорционально историческим продажам.</div>
{df2html(geo_dist, 25)}

<div class="pb"></div>

<h2>5. ABC-анализ ассортимента</h2>
<div class="note">A — первые 80% выручки (ядро). B — следующие 15%. C — последние 5% (кандидаты на вывод).</div>
{df2html(abc_summary)}

<h2>6. Товарный прогноз — Топ-50 позиций</h2>
{df2html(prod_dist, 50)}

{ai_cfo_html}
{ai_plan_html}

<div class="footer">Executive BI Suite v3.0 &nbsp;|&nbsp; {datetime.now().strftime('%d.%m.%Y')} &nbsp;|&nbsp; Конфиденциально</div>
<script>window.addEventListener('load', function(){{ setTimeout(function(){{ window.print(); }}, 800); }});</script>
</body>
</html>"""

        st.sidebar.download_button(
            label="📄 Скачать PDF-отчёт (A4)",
            data=html_report.encode('utf-8'),
            file_name=f"BI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html"
        )
        st.sidebar.caption("Откройте файл в браузере → автоматически откроется диалог печати → выберите «Сохранить как PDF»")

else:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:65vh;text-align:center;color:#475569">
        <div style="font-size:72px;margin-bottom:22px">⚡</div>
        <h2 style="color:#94a3b8;font-size:26px;margin-bottom:8px">Executive BI Suite v3.0</h2>
        <p style="font-size:15px;color:#475569;max-width:420px;line-height:1.7">
            Загрузите реестр заказов (.csv или .xlsx) в боковом меню для запуска полного аналитического цикла
        </p>
        <div style="margin-top:28px;display:flex;gap:10px;flex-wrap:wrap;justify-content:center">
            <span style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);padding:7px 16px;border-radius:20px;font-size:13px;color:#a5b4fc">📊 Прогноз Prophet</span>
            <span style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);padding:7px 16px;border-radius:20px;font-size:13px;color:#a5b4fc">🔢 Юнит-экономика</span>
            <span style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);padding:7px 16px;border-radius:20px;font-size:13px;color:#a5b4fc">🧠 AI CFO брифинг</span>
            <span style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);padding:7px 16px;border-radius:20px;font-size:13px;color:#a5b4fc">📐 ABC-анализ</span>
            <span style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.3);padding:7px 16px;border-radius:20px;font-size:13px;color:#a5b4fc">🌍 Геоаналитика</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
