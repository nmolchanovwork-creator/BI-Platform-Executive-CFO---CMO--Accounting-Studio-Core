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
    except Exception:
        selected_model_path = "models/gemini-1.5-flash-latest"
else:
    selected_model_path = "models/gemini-1.5-flash-latest"
    st.sidebar.warning("Введите API Key для загрузки списка моделей")

uploaded_file = st.sidebar.file_uploader("📂 Загрузить реестр заказов", type=['csv', 'xlsx'])

# Инициализация переменных для HTML отчета
month_df = pd.DataFrame()
model_df_display = pd.DataFrame()
mkt_dist = pd.DataFrame()
geo_dist = pd.DataFrame()
abc_summary = pd.DataFrame()
prod_dist = pd.DataFrame()
trend_direction = ""
change_pct = 0.0

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
                future_total_rev = future_total_upper = growth_coeff = 0
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

                avg_hist_daily = actual['y'].mean()
                avg_fore_daily = future_forecast['yhat'].mean()
                trend_direction = "восходящий" if avg_fore_daily > avg_hist_daily else "нисходящий/стабилизационный"
                change_pct = abs(avg_fore_daily - avg_hist_daily) / avg_hist_daily * 100 if avg_hist_daily > 0 else 0
                
                st.info(f"**Математическое заключение (Prophet):** Модель идентифицировала **{trend_direction}** тренд. Смещение относительно исторических данных: **{change_pct:.1f}%**.")

                month_df = monthly_forecast_totals.rename(columns={'month_str': 'Месяц', 'yhat': 'Базовый (₴)', 'yhat_upper': 'Оптимистичный (₴)', 'yhat_lower': 'Пессимистичный (₴)'})
                st.dataframe(month_df.style.format({col: "{:,.0f}" for col in month_df.columns if col != 'Месяц'}), width='stretch')

        # --- 3. WHAT-IF MODELING ---
        with tabs[2]:
            st.title("Стратегическое моделирование (Сложный процент)")
            st.write(f"Базой для моделирования выступает **Оптимистичный сценарий** на {horizon_months} мес. ({future_total_upper:,.0f} ₴).")
            
            c1, c2 = st.columns(2)
            orders_boost = c1.slider("Изменение объема заказов / Трафика (%)", -100.0, 100.0, 15.0, 1.0)
            aov_boost = c2.slider("Изменение среднего чека / Цен (%)", -100.0, 100.0, 10.0, 1.0)
            
            if not monthly_forecast_totals.empty:
                model_df = monthly_forecast_totals[['month_str', 'yhat_upper']].copy()
                model_df.rename(columns={'month_str': 'Месяц', 'yhat_upper': 'База (Оптимист.)'}, inplace=True)
                
                # ШАГ 1: Накладываем трафик на Базу
                model_df['База + Трафик'] = model_df['База (Оптимист.)'] * (1 + orders_boost/100)
                model_df['Дельта Трафика'] = model_df['База + Трафик'] - model_df['База (Оптимист.)']
                
                # ШАГ 2: Накладываем чек НА НОВУЮ БАЗУ (База + Трафик)
                model_df['Итог (Трафик + Чек)'] = model_df['База + Трафик'] * (1 + aov_boost/100)
                model_df['Дельта Чека (Синергия)'] = model_df['Итог (Трафик + Чек)'] - model_df['База + Трафик']
                
                # Финансы
                model_df['Совокупный Прирост'] = model_df['Итог (Трафик + Чек)'] - model_df['База (Оптимист.)']
                model_df['Доп. Прибыль (35%)'] = model_df['Совокупный Прирост'] * 0.35
                
                # Строка ИТОГО
                totals = pd.DataFrame([{
                    'Месяц': 'ИТОГО',
                    'База (Оптимист.)': model_df['База (Оптимист.)'].sum(),
                    'База + Трафик': model_df['База + Трафик'].sum(),
                    'Дельта Трафика': model_df['Дельта Трафика'].sum(),
                    'Итог (Трафик + Чек)': model_df['Итог (Трафик + Чек)'].sum(),
                    'Дельта Чека (Синергия)': model_df['Дельта Чека (Синергия)'].sum(),
                    'Совокупный Прирост': model_df['Совокупный Прирост'].sum(),
                    'Доп. Прибыль (35%)': model_df['Доп. Прибыль (35%)'].sum()
                }])
                model_df_display = pd.concat([model_df, totals], ignore_index=True)
                
                format_cols_mod = {col: "{:,.0f}" for col in model_df_display.columns if col != 'Месяц'}
                st.dataframe(model_df_display.style.format(format_cols_mod), width='stretch')

                st.markdown("### 💡 Авто-анализ модели (Декомпозиция прироста)")
                t_base = totals['База (Оптимист.)'].iloc[0]
                t_vol = totals['Дельта Трафика'].iloc[0]
                t_aov = totals['Дельта Чека (Синергия)'].iloc[0]
                t_total = totals['Совокупный Прирост'].iloc[0]
                
                st.success(f"""
                **Разбор факторов изменения (Нарастающим итогом):**
                1. Изначальная оптимистичная база: **{t_base:,.0f} ₴**
                2. Влияние изменения трафика ({orders_boost}%): **{t_vol:+,.0f} ₴**
                3. Влияние изменения чека на новый объем потока ({aov_boost}%): **{t_aov:+,.0f} ₴**
                * Совокупное изменение оборота составит: **{t_total:+,.0f} ₴**. 
                """)

        # --- 4. MARKETING ---
        with tabs[3]:
            st.title("Маркетинговая аналитика")
            fig_sun = px.sunburst(df, path=['utm_source', 'utm_medium', 'utm_campaign'], values='revenue')
            st.plotly_chart(fig_sun, width='stretch')
            
            if not monthly_forecast_totals.empty:
                mkt_dist = get_monthly_distribution(df, 'utm_source', monthly_forecast_totals, total_rev)
                mkt_dist = mkt_dist.rename(columns={'utm_source': 'Источник'})
                st.dataframe(mkt_dist.style.format({col: "{:,.0f}" for col in mkt_dist.columns if col != 'Источник'}), hide_index=True, width='stretch')

        # --- 5. GEO ---
        with tabs[4]:
            st.title("Геоаналитика")
            if not monthly_forecast_totals.empty:
                geo_dist = get_monthly_distribution(df, 'city', monthly_forecast_totals, total_rev).rename(columns={'city': 'Город'})
                st.dataframe(geo_dist.head(25).style.format({col: "{:,.0f}" for col in geo_dist.columns if col != 'Город'}), width='stretch', hide_index=True)

        # --- 6. PRODUCTS ---
        with tabs[5]:
            st.title("Товарная аналитика (ABC-анализ)")
            df['product_full'] = df['art'] + " - " + df['product']
            prod_df_base = df.groupby('product_full').agg({'qty':'sum', 'revenue':'sum'}).sort_values(by='revenue', ascending=False).reset_index()
            
            prod_df_base['share'] = prod_df_base['revenue'] / total_rev
            prod_df_base['cum_share'] = prod_df_base['share'].cumsum()
            prod_df_base['Класс ABC'] = prod_df_base['cum_share'].apply(lambda x: 'A (80% Выручки)' if x <= 0.8 else ('B (15%)' if x <= 0.95 else 'C (5%)'))
            
            abc_summary = prod_df_base.groupby('Класс ABC').agg({'product_full':'count', 'revenue':'sum'}).reset_index().rename(columns={'product_full':'Кол-во товаров'})
            st.table(abc_summary.style.format({'revenue': "{:,.0f} ₴"}))
            
            if not monthly_forecast_totals.empty:
                prod_dist = get_monthly_distribution(df, 'product_full', monthly_forecast_totals, total_rev).rename(columns={'product_full': 'Товар'})
                prod_dist = prod_dist.merge(prod_df_base[['product_full', 'Класс ABC']].rename(columns={'product_full':'Товар'}), on='Товар', how='left')
                cols = ['Товар', 'Класс ABC'] + [c for c in prod_dist.columns if c not in ['Товар', 'Класс ABC']]
                prod_dist = prod_dist[cols]
                st.dataframe(prod_dist.head(50).style.format({col: "{:,.0f}" for col in prod_dist.columns if col not in ['Товар', 'Класс ABC']}), width='stretch', hide_index=True)

        # --- 7 & 8. AI ---
        with tabs[6]:
            st.title("🧠 ИИ Финансовый Директор")
            if st.button("Генерировать отчет Совета Директоров"):
                with st.spinner("Анализ данных..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(selected_model_path)
                        ai_context = {"finance_fact": total_rev, "finance_forecast": future_total_rev}
                        response = model.generate_content(f"Ты CFO. Данные: {json.dumps(ai_context)}. Напиши жесткий стратегический вывод.")
                        st.session_state['ai_director_report'] = response.text
                    except Exception as e: st.error("🛑 Ошибка API. Выберите другую модель.")
            if st.session_state['ai_director_report']: st.markdown(st.session_state['ai_director_report'])

        with tabs[7]:
            st.title("🚀 AI План роста")
            if st.button("Разработать план"):
                with st.spinner("Построение стратегии..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(selected_model_path)
                        response = model.generate_content(f"Основываясь на выручке {total_rev} ₴. Напиши план масштабирования.")
                        st.session_state['ai_growth_plan'] = response.text
                    except Exception as e: st.error("🛑 Ошибка API. Выберите другую модель.")
            if st.session_state['ai_growth_plan']: st.markdown(st.session_state['ai_growth_plan'])

        # --- EXPORT HTML/PDF ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🖨️ Экспорт отчетности")
        
        html_report = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{ size: A4; margin: 15mm; }}
                body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 11px; color: #333; line-height: 1.4; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #e74c3c; padding-bottom: 5px; font-size: 18px; text-transform: uppercase; }}
                h2 {{ color: #2980b9; margin-top: 20px; font-size: 14px; border-bottom: 1px solid #bdc3c7; padding-bottom: 3px; page-break-after: avoid; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 15px; page-break-inside: auto; }}
                tr {{ page-break-inside: avoid; page-break-after: auto; }}
                th, td {{ border: 1px solid #bdc3c7; padding: 6px; text-align: left; }}
                th {{ background-color: #ecf0f1; font-weight: bold; color: #2c3e50; }}
                .page-break {{ page-break-before: always; }}
                .highlight-box {{ background-color: #f9f9f9; padding: 10px; border-left: 4px solid #e74c3c; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Прогноз продаж</h1>
            <p><strong>Горизонт планирования:</strong> {horizon_months} мес.<br>
            <strong>Математический тренд:</strong> {trend_direction} ({change_pct:.1f}%)</p>
            
            <h2>1. Агрегированный прогноз (Базовый алгоритм)</h2>
            {month_df.to_html(index=False, float_format=lambda x: f"{x:,.0f}") if not month_df.empty else "Нет данных"}

            <h2>2. Стратегическое моделирование (Факторный анализ)</h2>
            <p>Заложенные изменения: Объем/Трафик: <strong>{orders_boost}%</strong> | Чек/Цена: <strong>{aov_boost}%</strong></p>
            {model_df_display.to_html(index=False, float_format=lambda x: f"{x:,.0f}") if not model_df_display.empty else "Нет данных"}
            
            <div class="page-break"></div>

            <h2>3. Маркетинговые каналы (Прогноз по месяцам)</h2>
            {mkt_dist.to_html(index=False, float_format=lambda x: f"{x:,.0f}") if not mkt_dist.empty else "Нет данных"}
            
            <h2>4. Геоаналитика (Топ-25 городов)</h2>
            {geo_dist.head(25).to_html(index=False, float_format=lambda x: f"{x:,.0f}") if not geo_dist.empty else "Нет данных"}

            <div class="page-break"></div>

            <h2>5. Структура ассортимента (ABC-анализ)</h2>
            {abc_summary.to_html(index=False, float_format=lambda x: f"{x:,.0f}") if not abc_summary.empty else "Нет данных"}
            
            <h2>6. Товарная аналитика (Топ-50 позиций)</h2>
            {prod_dist.head(50).to_html(index=False, float_format=lambda x: f"{x:,.0f}") if not prod_dist.empty else "Нет данных"}
        """
        
        if st.session_state['ai_director_report']:
            html_report += f"""<div class="page-break"></div><h2>7. Заключение ИИ CFO</h2><div class="highlight-box">{st.session_state['ai_director_report'].replace('\n', '<br>')}</div>"""
        if st.session_state['ai_growth_plan']:
            html_report += f"""<h2>8. ИИ План роста</h2><div class="highlight-box">{st.session_state['ai_growth_plan'].replace('\n', '<br>')}</div>"""
            
        html_report += "<script>window.onload = function() { window.print(); }</script></body></html>"
        
        st.sidebar.download_button(
            label="💾 Скачать PDF/HTML отчет (A4)",
            data=html_report.encode('utf-8'),
            file_name="Sales_Forecast_Report.html",
            mime="text/html"
        )
else:
    st.info("👈 Загрузите реестр заказов в боковом меню для старта.")
