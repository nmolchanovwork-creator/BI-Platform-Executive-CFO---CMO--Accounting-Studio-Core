"""
BI Platform: Executive CFO & CMO / Accounting Studio Core
(Integrated with advanced Nomenclature Parsing, Global Forecasting & Export)
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

warnings.filterwarnings('ignore')

# --- CONFIG ---
st.set_page_config(page_title="Executive BI Platform", page_icon="📊", layout="wide")

# --- СТИЛИЗАЦИЯ ПОД КОРПОРАТИВНЫЙ СТАНДАРТ ---
st.markdown("""
<style>
    .report-text { font-size: 14px; line-height: 1.6; }
    .metric-box { padding: 15px; background-color: #f8f9fa; border-left: 5px solid #d9534f; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# --- CORE DATA PROCESSING WITH BUNDLE LOGIC ---
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
        if df_clean.empty: return pd.DataFrame()
            
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['date'])
        df_clean['profit'] = df_clean['revenue'] * 0.35

        return df_clean
    except Exception as e:
        st.error(f"❌ Ошибка при профилировании: {str(e)}")
        return pd.DataFrame()

# --- PROPHET FORECASTING ---
@st.cache_data
def generate_global_forecast(df, periods, freq='D'):
    daily_rev = df.groupby(df['date'].dt.date)['revenue'].sum().reset_index()
    daily_rev.columns = ['ds', 'y']
    
    if len(daily_rev) < 10:
        return None, None, daily_rev
        
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
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

# --- Инициализация переменных сессии для хранения ИИ вывода ---
if 'ai_director_report' not in st.session_state:
    st.session_state['ai_director_report'] = None
if 'ai_growth_plan' not in st.session_state:
    st.session_state['ai_growth_plan'] = None

# --- UI & SIDEBAR ---
st.sidebar.title("Настройки платформы")

api_key = st.sidebar.text_input("🔑 Google API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)
    try:
        models_info = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        gemini_models = [m for m in models_info if 'gemini' in m]
        selected_model_path = st.sidebar.selectbox("🤖 Выберите ИИ-модель", gemini_models)
    except Exception as e:
        st.sidebar.error("Не удалось загрузить список моделей. Проверьте API Key.")
        selected_model_path = "models/gemini-1.5-flash-latest"
else:
    selected_model_path = "models/gemini-1.5-flash-latest"
    st.sidebar.warning("Введите API Key для загрузки списка моделей")

uploaded_file = st.sidebar.file_uploader("📂 Загрузить реестр заказов", type=['csv', 'xlsx'])

if uploaded_file:
    with st.spinner("Профилирование номенклатуры..."):
        raw_df = load_and_clean_data(uploaded_file)
        
    if not raw_df.empty:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 Фильтры и Прогноз")
        
        horizon_months = st.sidebar.slider("Горизонт прогноза (месяцев)", 1, 12, 3)
        
        all_statuses = raw_df['status'].unique().tolist()
        status_filter = st.sidebar.multiselect("Статус заказа", all_statuses, default=all_statuses)
        df = raw_df[raw_df['status'].isin(status_filter)]
        
        with st.spinner("Генерация прогнозов..."):
            m, forecast, actual = generate_global_forecast(df, horizon_months * 30)
            
            total_rev = df['revenue'].sum()
            
            if forecast is not None and not actual.empty:
                max_date = pd.to_datetime(actual['ds'].max())
                future_forecast = forecast[forecast['ds'] > max_date].copy()
                future_total_rev = future_forecast['yhat'].sum()
                future_total_upper = future_forecast['yhat_upper'].sum()
                
                future_forecast['month_str'] = future_forecast['ds'].dt.strftime('%Y-%m')
                monthly_forecast_totals = future_forecast.groupby('month_str')[['yhat', 'yhat_upper', 'yhat_lower']].sum().reset_index()
                
                historical_total_rev = actual['y'].sum()
                growth_coeff = future_total_rev / historical_total_rev if historical_total_rev > 0 else 1
            else:
                future_total_rev = 0
                future_total_upper = 0
                growth_coeff = 0
                monthly_forecast_totals = pd.DataFrame()

        # UI TABS
        tabs = st.tabs([
            "📊 Dashboard", "🔮 Прогноз продаж", 
            "🎛 Моделирование", "📣 Маркетинг", 
            "🗺️ Геоаналитика", "💰 Продукты", "🧠 AI Директор", "🚀 AI План"
        ])

        # --- 1. DASHBOARD ---
        with tabs[0]:
            st.title("Executive Dashboard")
            unique_orders = df['order_id'].nunique()
            aov = total_rev / unique_orders if unique_orders > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("💰 Историческая Выручка", f"{total_rev:,.0f} ₴")
            c2.metric("🔮 Прогнозная Выручка", f"{future_total_rev:,.0f} ₴", f"+{horizon_months} мес.")
            c3.metric("🛒 Заказов (Факт)", f"{unique_orders:,}")
            c4.metric("🧾 Средний чек", f"{aov:,.0f} ₴")
            
            st.markdown("### Динамика выручки (Факт)")
            fig_trend = px.bar(df.groupby(df['date'].dt.to_period('W').dt.start_time)['revenue'].sum().reset_index(), 
                                x='date', y='revenue', template="plotly_white")
            st.plotly_chart(fig_trend, width='stretch')

        # --- 2. FORECASTING ---
        with tabs[1]:
            st.title(f"Математический прогноз на {horizon_months} мес.")
            if forecast is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=actual['ds'], y=actual['y'], name='Факт', mode='lines', line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], name='Базовый прогноз', line=dict(color='#ff7f0e')))
                fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper'], name='Оптимистичный', line=dict(color='#2ca02c', dash='dot')))
                fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower'], name='Пессимистичный', line=dict(color='#d62728', dash='dot')))
                fig.update_layout(template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig, width='stretch')

                # АВТО-ОБОСНОВАНИЕ ПРОГНОЗА
                st.markdown("### 📋 Авто-обоснование траектории прогноза")
                avg_hist_daily = actual['y'].mean()
                avg_fore_daily = future_forecast['yhat'].mean()
                trend_direction = "восходящий" if avg_fore_daily > avg_hist_daily else "нисходящий или стабилизационный"
                change_pct = abs(avg_fore_daily - avg_hist_daily) / avg_hist_daily * 100
                
                st.info(f"""
                **Математическое заключение (Prophet):**
                * Модель идентифицировала **{trend_direction}** тренд среднесуточных продаж. Смещение относительно исторических данных составляет **{change_pct:.1f}%**.
                * Уровень неопределенности (разница между Оптимистичным и Пессимистичным сценариями) составляет **{((future_total_upper - future_forecast['yhat_lower'].sum()) / future_total_rev * 100):.1f}%**, что указывает на баланс сезонных факторов.
                """)

                # КАЛЕНДАРЬ ПРАЗДНИКОВ
                st.markdown("### 📅 Грядущие ключевые праздники Украины (Влияние на ритейл/спрос 2026)")
                holidays_2026 = pd.DataFrame([
                    {"Дата": "28.06.2026", "Праздник": "День Конституции Украины", "Влияние на продажи": "Краткосрочный спад бизнес-активности, рост в сегменте HoReCa и отдыха."},
                    {"Дата": "15.07.2026", "Праздник": "День Украинской Государственности", "Влияние на продажи": "Локальные колебания спроса, стандартный летний торговый день."},
                    {"Дата": "24.08.2026", "Праздник": "День Независимости Украины", "Влияние на продажи": "Всплеск патриотического спроса, мерча, национальных товаров, продуктов питания."},
                    {"Дата": "01.10.2026", "Праздник": "День защитников и защитниц Украины", "Влияние на продажи": "Высокий спрос на подарки, милитари-сегмент, мужской и специализированный ассортимент."}
                ])
                st.table(holidays_2026)

                st.markdown("### 📅 Агрегированный прогноз по месяцам")
                month_df = monthly_forecast_totals.rename(columns={
                    'month_str': 'Месяц', 
                    'yhat': 'Базовый (₴)', 
                    'yhat_upper': 'Оптимистичный (₴)', 
                    'yhat_lower': 'Пессимистичный (₴)'
                })
                st.dataframe(month_df.style.format({col: "{:,.0f}" for col in month_df.columns if col != 'Месяц'}), width='stretch')

                st.markdown("### 📋 Детализация прогноза по дням")
                future_df_daily = future_forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].copy()
                future_df_daily['Месяц'] = future_df_daily['ds'].dt.strftime('%Y-%m')
                future_df_daily['Дата'] = future_df_daily['ds'].dt.strftime('%d.%m.%Y')
                future_df_daily = future_df_daily.rename(columns={'yhat': 'Базовый прогноз (₴)', 'yhat_upper': 'Оптимистичный (₴)', 'yhat_lower': 'Пессимистичный (₴)'})
                
                styled_df = future_df_daily.set_index(['Месяц', 'Дата'])[['Базовый прогноз (₴)', 'Оптимистичный (₴)', 'Пессимистичный (₴)']]
                st.dataframe(styled_df.style.format("{:,.0f}"), width='stretch', height=300)
            else:
                st.warning("Недостаточно данных для прогноза (нужно минимум 10 дней продаж).")

        # --- 3. WHAT-IF MODELING ---
        with tabs[2]:
            st.title("Стратегическое моделирование (Будущие периоды)")
            st.write(f"Базой для моделирования выступает **Оптимистичный сценарий** на следующие {horizon_months} мес. ({future_total_upper:,.0f} ₴).")
            
            c1, c2 = st.columns(2)
            orders_boost = c1.slider("Рост объема заказов / Трафика (%)", 0.0, 100.0, 15.0, 1.0)
            aov_boost = c2.slider("Рост среднего чека / Цен (%)", 0.0, 100.0, 10.0, 1.0)
            
            if not monthly_forecast_totals.empty:
                model_df = monthly_forecast_totals[['month_str', 'yhat_upper']].copy()
                model_df.rename(columns={'month_str': 'Месяц', 'yhat_upper': 'Оптимистичная база (₴)'}, inplace=True)
                
                # ИСПРАВЛЕНА ОШИБКА: рост чека накладывается на уже увеличенный объем заказов
                # База * Рост Трафика * Рост Чека = Синергетический кумулятивный эффект
                model_df['С учетом изменений (₴)'] = model_df['Оптимистичная база (₴)'] * (1 + orders_boost/100) * (1 + aov_boost/100)
                model_df['Доп. Прибыль от изменений (₴)'] = (model_df['С учетом изменений (₴)'] - model_df['Оптимистичная база (₴)']) * 0.35
                
                # Строка Итого для вывода
                m_total_base = model_df['Оптимистичная база (₴)'].sum()
                m_total_sim = model_df['С учетом изменений (₴)'].sum()
                m_total_profit = model_df['Доп. Прибыль от изменений (₴)'].sum()
                
                total_row = pd.DataFrame([{
                    'Месяц': 'ИТОГО',
                    'Оптимистичная база (₴)': m_total_base,
                    'С учетом изменений (₴)': m_total_sim,
                    'Доп. Прибыль от изменений (₴)': m_total_profit
                }])
                model_df_display = pd.concat([model_df, total_row], ignore_index=True)
                
                st.dataframe(model_df_display.style.format({
                    'Оптимистичная база (₴)': "{:,.0f}", 'С учетом изменений (₴)': "{:,.0f}", 'Доп. Прибыль от изменений (₴)': "{:,.0f}"
                }), width='stretch')

                # АВТО-АНАЛИЗ ЭФФЕКТИВНОСТИ МОДЕЛИРОВАНИЯ
                st.markdown("### 💡 Авто-анализ синергетического плеча")
                pure_volume_gain = m_total_base * (orders_boost / 100)
                pure_aov_gain = m_total_base * (aov_boost / 100)
                synergy_gain = m_total_sim - m_total_base - pure_volume_gain - pure_aov_gain
                
                st.warning(f"""
                **Структура прироста выручки от изменений:**
                * Чистый эффект от притока клиентов/заказов: **+{pure_volume_gain:,.0f} ₴**
                * Чистый эффект от прямого изменения цен/чека: **+{pure_aov_gain:,.0f} ₴**
                * 🔥 **Эффект синергии сложного процента (наложение чека на новый объем): +{synergy_gain:,.0f} ₴**
                * Совокупное масштабирование увеличит оборот на **{((m_total_sim - m_total_base)/m_total_base*100):.1f}%**.
                """)
            else:
                st.warning("Нет данных прогноза для моделирования.")

        # --- 4. MARKETING ---
        with tabs[3]:
            st.title("Маркетинговая аналитика и Прогноз")
            
            fig_sun = px.sunburst(df, path=['utm_source', 'utm_medium', 'utm_campaign'], values='revenue', title="Структура исторических доходов по каналам")
            fig_sun.update_layout(height=500, margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig_sun, width='stretch')
            
            st.markdown("### 💡 Расширенная Авто-Аналитика")
            channel_rev = df.groupby('utm_source')['revenue'].sum().sort_values(ascending=False)
            if not channel_rev.empty:
                top_source = channel_rev.index[0]
                top_val = channel_rev.values[0]
                bottom_source = channel_rev.index[-1]
                
                c_auto1, c_auto2 = st.columns(2)
                with c_auto1:
                    st.success(f"🏆 **Локомотив продаж:** `{top_source}` генерирует {top_val / total_rev * 100:.1f}% всей исторической выручки ({top_val:,.0f} ₴).")
                    st.warning(f"⚠️ **Риск концентрации:** Кассовый разрыв при отключении `{top_source}` составит ~{future_total_rev * (top_val / total_rev):,.0f} ₴ за период прогноза.")
                with c_auto2:
                    st.info(f"🔍 **Зона риска:** `{bottom_source}` показывает минимальную отдачу. Рекомендуется пересмотреть бюджет или креативы этого канала.")

            st.markdown(f"### 📈 Прогноз по источникам (Детализация по месяцам)")
            if not monthly_forecast_totals.empty:
                mkt_dist = get_monthly_distribution(df, 'utm_source', monthly_forecast_totals, total_rev)
                mkt_dist = mkt_dist.rename(columns={'utm_source': 'Источник'})
                format_cols = {col: "{:,.0f}" for col in mkt_dist.columns if col != 'Источник'}
                st.dataframe(mkt_dist.style.format(format_cols), hide_index=True, width='stretch')

        # --- 5. GEO ---
        with tabs[4]:
            st.title("Геоаналитика и Прогноз по регионам")
            city_rev_df = df.groupby('city')['revenue'].sum().sort_values(ascending=False).reset_index()
            
            fig_geo = px.bar(city_rev_df.head(20), x='city', y='revenue', title="Топ-20 городов (Исторический факт)", text_auto='.2s')
            st.plotly_chart(fig_geo, width='stretch')
            
            st.markdown("### 💡 Авто-Аналитика по регионам")
            if not city_rev_df.empty:
                top_city_name = city_rev_df.iloc[0]['city']
                top_city_share = city_rev_df.iloc[0]['revenue'] / total_rev * 100
                st.info(f"📍 **Основной регион:** `{top_city_name}` формирует {top_city_share:.1f}% выручки. Рекомендуется укреплять логистическую цепочку под этот регион.")
            
            st.markdown(f"### 📍 Ожидаемые продажи по городам (Помесячно)")
            if not monthly_forecast_totals.empty:
                geo_dist = get_monthly_distribution(df, 'city', monthly_forecast_totals, total_rev)
                geo_dist = geo_dist.rename(columns={'city': 'Город'})
                format_cols_geo = {col: "{:,.0f}" for col in geo_dist.columns if col != 'Город'}
                st.dataframe(geo_dist.head(25).style.format(format_cols_geo), width='stretch', hide_index=True)

        # --- 6. PRODUCTS ---
        with tabs[5]:
            st.title("Товарная аналитика")
            df['product_full'] = df['art'] + " - " + df['product']
            
            prod_df_base = df.groupby('product_full').agg({'qty':'sum', 'revenue':'sum'}).sort_values(by='revenue', ascending=False).reset_index()
            
            # АВТО АНАЛИТИКА: ABC-анализ номенклатуры
            st.markdown("### 📊 Автоматический ABC-анализ товарного портфеля (Парето)")
            prod_df_base['share'] = prod_df_base['revenue'] / total_rev
            prod_df_base['cum_share'] = prod_df_base['share'].cumsum()
            
            def assign_abc(cum_share):
                if cum_share <= 0.80: return 'A (Ядро продаж, 80% выручки)'
                elif cum_share <= 0.95: return 'B (Поддерживающий класс, 15%)'
                return 'C (Длинный хвост / Неликвид, 5%)'
                
            prod_df_base['Класс ABC'] = prod_df_base['cum_share'].apply(assign_abc)
            
            abc_summary = prod_df_base.groupby('Класс ABC').agg({'product_full':'count', 'revenue':'sum'}).reset_index()
            st.table(abc_summary.style.format({'revenue': "{:,.0f} ₴"}))
            
            st.markdown(f"### 📦 Помесячный математический прогноз по товарам (Топ-50)")
            if not monthly_forecast_totals.empty:
                prod_dist = get_monthly_distribution(df, 'product_full', monthly_forecast_totals, total_rev)
                prod_dist = prod_dist.merge(prod_df_base[['product_full', 'Класс ABC']], on='product_full', how='left')
                prod_dist = prod_dist.rename(columns={'product_full': 'Товар'})
                
                # Перенос класса ABC вперед для наглядности
                cols = ['Товар', 'Класс ABC'] + [c for c in prod_dist.columns if c not in ['Товар', 'Класс ABC']]
                prod_dist = prod_dist[cols]
                
                format_cols_prod = {col: "{:,.0f}" for col in prod_dist.columns if col not in ['Товар', 'Класс ABC']}
                st.dataframe(prod_dist.head(50).style.format(format_cols_prod), width='stretch', hide_index=True)

        # --- 7. AI DIRECTOR ---
        with tabs[6]:
            st.title("🧠 ИИ Финансовый Директор")
            if not api_key: 
                st.warning("⚠️ Введите Google API Key в боковом меню.")
            else:
                if st.button("Генерировать отчет Совета Директоров"):
                    with st.spinner(f"Анализ данных через {selected_model_path}..."):
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel(selected_model_path)
                            top_city = city_rev_df.iloc[0]['city'] if not city_rev_df.empty else ""
                            
                            ai_context = {
                                "finance_fact": total_rev, 
                                "finance_forecast": future_total_rev,
                                "top_city": top_city
                            }
                            prompt = f"Ты CFO. Данные: {json.dumps(ai_context)}. Напиши краткий, жесткий стратегический вывод для руководства."
                            response = model.generate_content(prompt)
                            st.session_state['ai_director_report'] = response.text
                            st.markdown(response.text)
                        except Exception as e:
                            if "429" in str(e) or "ResourceExhausted" in str(e):
                                st.error("🛑 Превышен суточный/минутный лимит запросов Google API. Смените модель в сайдбаре или попробуйте позже.")
                            else:
                                st.error(f"❌ Ошибка ИИ: {e}")
                elif st.session_state['ai_director_report']:
                    st.markdown(st.session_state['ai_director_report'])

        # --- 8. AI PLAN ---
        with tabs[7]:
            st.title("🚀 AI План роста бизнеса")
            if not api_key:
                st.warning("⚠️ Введите Google API Key в боковом меню.")
            else:
                if st.button("Разработать план"):
                    with st.spinner(f"Построение стратегии через {selected_model_path}..."):
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel(selected_model_path)
                            prompt = f"Основываясь на выручке {total_rev} ₴. Напиши лаконичный план масштабирования."
                            response = model.generate_content(prompt)
                            st.session_state['ai_growth_plan'] = response.text
                            st.markdown(response.text)
                        except Exception as e:
                            if "429" in str(e) or "ResourceExhausted" in str(e):
                                st.error("🛑 Превышен лимит запросов Google API. Используйте другую модель.")
                            else:
                                st.error(f"❌ Ошибка ИИ: {e}")
                elif st.session_state['ai_growth_plan']:
                    st.markdown(st.session_state['ai_growth_plan'])

        # --- ГЕНЕРАЦИЯ КРАСИВОГО ПЕЧАТНОГО ОТЧЕТА (HTML / СКАЧАТЬ В PDF) ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🖨️ Экспорт отчетности")
        
        # Сборка HTML строки
        html_report = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; color: #333; }}
                h1 {{ color: #d9534f; border-bottom: 2px solid #d9534f; padding-bottom: 10px; }}
                h2 {{ color: #2c3e50; margin-top: 30px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 12px; }}
                th {{ background-color: #f4f6f7; color: #2c3e50; }}
                .highlight-box {{ background-color: #fcf8e3; padding: 15px; border-left: 4px solid #f0ad4e; margin: 15px 0; font-size: 13px; }}
                .ai-box {{ background-color: #f4f6f7; padding: 15px; border-left: 4px solid #5bc0de; margin: 15px 0; font-size: 13px; font-style: italic; }}
                @media print {{ .no-print {{ display: none; }} }}
            </style>
        </head>
        <body>
            <h1>КОРПОРАТИВНЫЙ АНАЛИТИЧЕСКИЙ ОТЧЕТ BI PLATFORM</h1>
            <p><strong>Период планирования:</strong> На следующие {horizon_months} мес.</p>
            
            <h2>1. Ключевые метрики (Исторический факт)</h2>
            <table>
                <tr><th>Показатель</th><th>Значение</th></tr>
                <tr><td>Общая историческая выручка</td><td>{total_rev:,.2f} ₴</td></tr>
                <tr><td>Всего успешных заказов</td><td>{df['order_id'].nunique():,}</td></tr>
                <tr><td>Средний чек (AOV)</td><td>{(total_rev / df['order_id'].nunique() if df['order_id'].nunique() > 0 else 0):,.2f} ₴</td></tr>
                <tr><td>Ожидаемый базовый оборот за период прогноза</td><td>{future_total_rev:,.2f} ₴</td></tr>
            </table>

            <h2>2. Стратегическое моделирование изменений (Сложный процент)</h2>
            <p>Заложенный рост трафика/заказов: <strong>+{orders_boost}%</strong>. Заложенный рост чека: <strong>+{aov_boost}%</strong>.</p>
            <div class="highlight-box">
                <strong>Эффект синергии:</strong> Рост среднего чека перемножен на новый объем потока клиентов.<br>
                Ожидаемый моделируемый оборот: <strong>{m_total_sim:,.0f} ₴</strong><br>
                Прогнозная доп. прибыль (Маржа 35%): <strong>{m_total_profit:,.0f} ₴</strong>
            </div>

            <h2>3. ABC-анализ товарной матрицы</h2>
            {abc_summary.to_html(index=False, classes='table')}
            
            <h2>4. Структура маркетинговых каналов (Топ источников)</h2>
            {df.groupby('utm_source')['revenue'].sum().sort_values(ascending=False).reset_index().to_html(index=False, classes='table')}
        """
        
        # Добавляем выводы ИИ в отчет, если они существуют в сессии
        if st.session_state['ai_director_report']:
            html_report += f"""
            <h2>5. Стратегическое заключение ИИ Финансового Директора</h2>
            <div class="ai-box">{st.session_state['ai_director_report'].replace('\n', '<br>')}</div>
            """
        if st.session_state['ai_growth_plan']:
            html_report += f"""
            <h2>6. ИИ План стратегического масштабирования</h2>
            <div class="ai-box">{st.session_state['ai_growth_plan'].replace('\n', '<br>')}</div>
            """
            
        html_report += """
            <script>window.onload = function() { window.print(); }</script>
        </body>
        </html>
        """
        
        st.sidebar.download_button(
            label="💾 Скачать отчет для PDF печати",
            data=html_report,
            file_name="Executive_BI_Report.html",
            mime="text/html",
            help="Скачайте файл, откройте его. Браузер автоматически предложит распечатать или сохранить страницу в PDF со всеми стилями и таблицами."
        )

else:
    st.info("👈 Загрузите реестр заказов в боковом меню для старта.")
