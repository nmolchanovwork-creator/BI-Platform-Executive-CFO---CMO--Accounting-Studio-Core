"""
BI Platform: Executive CFO & CMO / Accounting Studio Core
(Integrated with advanced Nomenclature Parsing & Global Forecasting)
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
        return None, None, daily_rev # Слишком мало данных для прогноза
        
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    try:
        m.add_country_holidays(country_name='UA') 
    except: pass
    
    m.fit(daily_rev)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return m, forecast, daily_rev

# --- UI & SIDEBAR ---
st.sidebar.title("Настройки платформы")

# Выбор ИИ модели
ai_models = {
    "Gemini 1.5 Flash (Быстрая)": "models/gemini-1.5-flash-latest",
    "Gemini 1.5 Pro (Умная)": "models/gemini-1.5-pro-latest"
}
selected_model_name = st.sidebar.selectbox("🤖 Выберите ИИ-модель", list(ai_models.keys()))
system_model_path = ai_models[selected_model_name]

api_key = st.sidebar.text_input("🔑 Google API Key", type="password")

uploaded_file = st.sidebar.file_uploader("📂 Загрузить реестр заказов", type=['csv', 'xlsx'])

if uploaded_file:
    with st.spinner("Профилирование номенклатуры..."):
        raw_df = load_and_clean_data(uploaded_file)
        
    if not raw_df.empty:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 Фильтры и Прогноз")
        
        # Горизонт прогноза перенесен в сайдбар!
        horizon_months = st.sidebar.slider("Горизонт прогноза (месяцев)", 1, 12, 3)
        
        all_statuses = raw_df['status'].unique().tolist()
        status_filter = st.sidebar.multiselect("Статус заказа", all_statuses, default=all_statuses)
        df = raw_df[raw_df['status'].isin(status_filter)]
        
        # ГЛОБАЛЬНЫЙ ПРОГНОЗ (работает в фоне)
        with st.spinner("Генерация глобального прогноза..."):
            m, forecast, actual = generate_global_forecast(df, horizon_months * 30)
            
            # Считаем коэффициенты роста для распределения на другие вкладки
            if forecast is not None:
                max_date = pd.to_datetime(actual['ds'].max())
                future_forecast = forecast[forecast['ds'] > max_date]
                future_total_rev = future_forecast['yhat'].sum()
                historical_total_rev = actual['y'].sum()
                # Коэффициент соотношения будущего периода к историческому
                growth_coeff = future_total_rev / historical_total_rev if historical_total_rev > 0 else 1
            else:
                future_total_rev = 0
                growth_coeff = 0

        # UI TABS
        tabs = st.tabs([
            "📊 Dashboard", "🔮 Прогноз продаж", 
            "🎛 Моделирование", "📣 Маркетинг", 
            "🗺️ Геоаналитика", "💰 Продукты", "🧠 AI Директор", "🚀 AI План"
        ])

        # --- 1. DASHBOARD ---
        with tabs[0]:
            st.title("Executive Dashboard")
            total_rev = df['revenue'].sum()
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
            st.plotly_chart(fig_trend, use_container_width=True)

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
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📋 Таблица будущих продаж (с детализацией по дням)")
                # Подготовка красивой сгруппированной таблицы
                future_df = future_forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].copy()
                future_df['Месяц'] = future_df['ds'].dt.strftime('%Y-%m')
                future_df['Дата'] = future_df['ds'].dt.strftime('%d.%m.%Y')
                future_df = future_df.rename(columns={'yhat': 'Базовый прогноз (₴)', 'yhat_upper': 'Оптимистичный (₴)', 'yhat_lower': 'Пессимистичный (₴)'})
                
                # Устанавливаем MultiIndex для группировки Месяц -> Дата
                styled_df = future_df.set_index(['Месяц', 'Дата'])[['Базовый прогноз (₴)', 'Оптимистичный (₴)', 'Пессимистичный (₴)']]
                
                # Выводим таблицу с форматом валюты
                st.dataframe(styled_df.style.format("{:,.0f}"), use_container_width=True, height=400)
            else:
                st.warning("Недостаточно данных для прогноза (нужно минимум 10 дней продаж).")

        # --- 3. WHAT-IF MODELING ---
        with tabs[2]:
            st.title("Стратегическое моделирование (Будущие периоды)")
            st.write(f"Базой для моделирования выступает **Прогнозная выручка** на следующие {horizon_months} мес. ({future_total_rev:,.0f} ₴).")
            
            c1, c2 = st.columns(2)
            orders_boost = c1.slider("Рост объема заказов / Трафика (%)", 0.0, 100.0, 15.0, 1.0)
            aov_boost = c2.slider("Рост среднего чека / Цен (%)", 0.0, 100.0, 10.0, 1.0)
            
            # ИСПРАВЛЕНО: Теперь рост объема заказов честно умножает всю будущую выручку
            proj_future_rev = future_total_rev * (1 + orders_boost/100) * (1 + aov_boost/100)
            
            c3, c4, c5 = st.columns(3)
            c3.metric("Базовый Прогноз", f"{future_total_rev:,.0f} ₴")
            c4.metric("Моделируемый Прогноз", f"{proj_future_rev:,.0f} ₴", f"+{proj_future_rev - future_total_rev:,.0f} ₴")
            c5.metric("Доп. Прибыль от изменений", f"{(proj_future_rev - future_total_rev) * 0.35:,.0f} ₴")

        # --- 4. MARKETING ---
        with tabs[3]:
            st.title("Маркетинговая аналитика и Прогноз")
            
            c1, c2 = st.columns([2, 1])
            with c1:
                # Увеличена диаграмма
                fig_sun = px.sunburst(df, path=['utm_source', 'utm_medium', 'utm_campaign'], values='revenue')
                fig_sun.update_layout(height=650, margin=dict(t=10, l=10, r=10, b=10))
                st.plotly_chart(fig_sun, use_container_width=True)
            
            with c2:
                st.markdown("### 💡 Авто-Аналитика")
                top_source = df.groupby('utm_source')['revenue'].sum().sort_values(ascending=False).head(1)
                st.info(f"🏆 **Главный канал:** {top_source.index[0]} приносит {top_source.values[0] / total_rev * 100:.1f}% всей выручки.")
                st.warning(f"⚠️ Если отключить источник `{top_source.index[0]}`, кассовый разрыв в будущих периодах может составить до {future_total_rev * (top_source.values[0] / total_rev):,.0f} ₴.")
                
                st.markdown(f"### 📈 Прогноз по каналам на {horizon_months} мес.")
                # Распределяем будущий прогноз пропорционально историческим каналам
                mkt_df = df.groupby('utm_source')['revenue'].sum().reset_index()
                mkt_df['Прогноз (₴)'] = mkt_df['revenue'] * growth_coeff
                mkt_df = mkt_df.rename(columns={'revenue': 'Факт (₴)', 'utm_source': 'Источник'}).sort_values('Прогноз (₴)', ascending=False)
                st.dataframe(mkt_df.style.format({'Факт (₴)': "{:,.0f}", 'Прогноз (₴)': "{:,.0f}"}), hide_index=True)

        # --- 5. GEO ---
        with tabs[4]:
            st.title("Геоаналитика и Прогноз по регионам")
            city_rev = df.groupby('city')['revenue'].sum().sort_values(ascending=False).reset_index()
            city_rev['Прогноз (₴)'] = city_rev['revenue'] * growth_coeff
            
            fig_geo = px.bar(city_rev.head(20), x='city', y='revenue', title="Топ-20 городов (Исторический факт)", text_auto='.2s')
            st.plotly_chart(fig_geo, use_container_width=True)
            
            st.markdown(f"### 📍 Ожидаемые продажи по городам (След. {horizon_months} мес.)")
            city_show = city_rev[['city', 'revenue', 'Прогноз (₴)']].rename(columns={'city': 'Город', 'revenue': 'Факт (₴)'})
            st.dataframe(city_show.head(15).style.format({'Факт (₴)': "{:,.0f}", 'Прогноз (₴)': "{:,.0f}"}), use_container_width=True)

        # --- 6. PRODUCTS ---
        with tabs[5]:
            st.title("Товарная аналитика")
            prod_df = df.groupby(['art', 'product']).agg({'qty':'sum', 'revenue':'sum'}).reset_index()
            prod_df = prod_df.sort_values(by='revenue', ascending=False)
            st.dataframe(prod_df.head(50).style.format({"revenue": "{:,.2f} ₴"}), use_container_width=True)

        # --- 7 & 8. AI DIRECTOR ---
        with tabs[6]:
            st.title("🧠 ИИ Финансовый Директор")
            if not api_key: st.warning("⚠️ Введите Google API Key в боковом меню.")
            else:
                if st.button("Генерировать отчет Совета Директоров"):
                    with st.spinner(f"Анализ данных через {selected_model_name}..."):
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(system_model_path)
                        ai_context = {
                            "finance_fact": total_rev, "finance_forecast": future_total_rev,
                            "top_city": city_rev.iloc[0]['city'] if not city_rev.empty else ""
                        }
                        prompt = f"Ты CFO. Данные: {json.dumps(ai_context)}. Напиши стратегический вывод."
                        st.markdown(model.generate_content(prompt).text)

        with tabs[7]:
            st.title("🚀 AI План роста бизнеса")
            if api_key and st.button("Разработать план"):
                with st.spinner(f"Построение стратегии через {selected_model_name}..."):
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(system_model_path)
                    st.markdown(model.generate_content(f"Основываясь на выручке {total_rev} ₴. Напиши план масштабирования.").text)
else:
    st.info("👈 Загрузите реестр заказов в боковом меню для старта.")
