"""
BI Platform: Executive CFO & CMO / Accounting Studio Core
Production-ready Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import google.generativeai as genai
from datetime import timedelta
import warnings
import json

warnings.filterwarnings('ignore')

# --- CONFIG ---
st.set_page_config(page_title="Executive BI Platform", page_icon="📊", layout="wide")

# --- UTILS & DATA CLEANING ---
@st.cache_data
def load_and_clean_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file)
            
        # Универсальный маппинг колонок (поддержка RU/UA/EN)
        col_mapping = {
            'дата заказа': 'date', 'дата': 'date', 'date': 'date',
            'статус': 'status', 'статус заказа': 'status',
            'номер заказа': 'order_id', 'id': 'order_id',
            'клиент': 'client', 'фио': 'client', 'email': 'client',
            'город': 'city', 'місто': 'city',
            'регион': 'region', 'область': 'region',
            'страна доставки': 'country', 'країна': 'country',
            'название товара': 'product', 'товар': 'product',
            'категория': 'category', 
            'количество': 'qty', 'кількість': 'qty',
            'итого': 'revenue', 'сумма': 'revenue', 'стоимость': 'revenue',
            'себестоимость': 'cogs', 'цена': 'price',
            'utm_source': 'utm_source', 'utm_medium': 'utm_medium', 'utm_campaign': 'utm_campaign'
        }
        
        df.columns = [str(c).lower().strip() for c in df.columns]
        df = df.rename(columns=col_mapping)
        
        # Проверка критических колонок
        if 'date' not in df.columns or 'revenue' not in df.columns:
            st.error("❌ Ошибка: В файле отсутствуют обязательные колонки (Дата, Итого/Сумма).")
            return pd.DataFrame()

        # Очистка дат
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Очистка финансов (удаление нулей и возвратов)
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
        df = df[df['revenue'] > 0]
        
        if 'qty' in df.columns:
            df['qty'] = pd.to_numeric(df['qty'], errors='coerce').fillna(1)
        else:
            df['qty'] = 1
            
        # Заглушка для прибыли, если нет себестоимости (считаем маржу 30% по умолчанию)
        if 'cogs' not in df.columns:
            df['profit'] = df['revenue'] * 0.30
        else:
            df['cogs'] = pd.to_numeric(df['cogs'], errors='coerce').fillna(0)
            df['profit'] = df['revenue'] - df['cogs']

        # Очистка UTM
        for utm in ['utm_source', 'utm_medium', 'utm_campaign']:
            if utm not in df.columns:
                df[utm] = 'organic'
            df[utm] = df[utm].fillna('organic').replace({'(none)': 'direct', 'NaN': 'organic', '': 'organic'})

        # Заполнение пустых городов
        if 'city' in df.columns:
            df['city'] = df['city'].fillna('Не указан')
        else:
            df['city'] = 'Не указан'
            
        if 'status' not in df.columns:
            df['status'] = 'Доставлен'

        return df
    except Exception as e:
        st.error(f"❌ Ошибка при чтении файла: {str(e)}")
        return pd.DataFrame()

# --- PROPHET FORECASTING ---
@st.cache_data
def generate_forecast(df, periods, freq='D'):
    daily_rev = df.groupby(df['date'].dt.date)['revenue'].sum().reset_index()
    daily_rev.columns = ['ds', 'y']
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name='UA') # Праздники Украины
    m.fit(daily_rev)
    
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return m, forecast, daily_rev

# --- SIDEBAR & FILTERS ---
st.sidebar.title("Настройки аналитики")
st.sidebar.markdown("### 🔑 ИИ Директор (Gemini API)")
api_key = st.sidebar.text_input("Введи Google API Key", type="password")
with st.sidebar.expander("ℹ️ Как получить API ключ?"):
    st.markdown("""
    1. Перейди на [Google AI Studio](https://aistudio.google.com/app/apikey)
    2. Нажми **Create API Key**
    3. Скопируй ключ (это бесплатно).
    """)

uploaded_file = st.sidebar.file_uploader("📂 Загрузить реестр (Excel/CSV)", type=['csv', 'xlsx'])

if uploaded_file:
    with st.spinner("Очистка и профилирование данных..."):
        raw_df = load_and_clean_data(uploaded_file)
        
    if not raw_df.empty:
        st.sidebar.markdown("### 🎯 Фильтры")
        
        # Фильтры
        status_filter = st.sidebar.multiselect("Статус заказа", raw_df['status'].unique(), default=raw_df['status'].unique())
        source_filter = st.sidebar.multiselect("Канал трафика", raw_df['utm_source'].unique())
        
        df = raw_df[raw_df['status'].isin(status_filter)]
        if source_filter:
            df = df[df['utm_source'].isin(source_filter)]
            
        # UI TABS
        tabs = st.tabs([
            "📊 Dashboard", "💰 Финансы", "🔮 Прогноз продаж", 
            "🎛 Финансовое моделирование", "📣 Маркетинг", 
            "🗺️ Геоаналитика", "🧠 AI Директор", "🚀 AI План роста"
        ])

        # --- 1. DASHBOARD & FINANCE ---
        with tabs[0]:
            st.title("Executive Dashboard")
            total_rev = df['revenue'].sum()
            total_profit = df['profit'].sum()
            orders = len(df)
            aov = total_rev / orders if orders > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("💰 Выручка", f"{total_rev:,.0f} ₴")
            c2.metric("📈 Валовая прибыль", f"{total_profit:,.0f} ₴", f"{(total_profit/total_rev)*100:.1f}% маржа")
            c3.metric("🛒 Заказов", f"{orders:,}")
            c4.metric("🧾 Средний чек", f"{aov:,.0f} ₴")
            
            st.markdown("### Динамика выручки")
            fig_trend = px.line(df.groupby(df['date'].dt.to_period('W').dt.start_time)['revenue'].sum().reset_index(), 
                                x='date', y='revenue', template="plotly_white")
            st.plotly_chart(fig_trend, use_container_width=True)

        # --- 2. FORECASTING ---
        with tabs[2]:
            st.title("Продвинутое прогнозирование (Prophet)")
            horizon_months = st.slider("Горизонт прогноза (месяцев)", 1, 12, 3)
            
            if st.button("Сгенерировать прогноз"):
                with st.spinner("Обучение математической модели..."):
                    m, forecast, actual = generate_forecast(df, horizon_months * 30)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=actual['ds'], y=actual['y'], name='Факт', mode='lines', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Базовый прогноз', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Оптимистичный', line=dict(color='green', dash='dot')))
                    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Пессимистичный', line=dict(color='red', dash='dot')))
                    
                    fig.update_layout(title=f"Прогноз выручки на {horizon_months} мес.", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    future_rev = forecast.tail(horizon_months * 30)['yhat'].sum()
                    st.success(f"Ожидаемая выручка за выбранный период: **{future_rev:,.0f} ₴**")

        # --- 3. WHAT-IF MODELING ---
        with tabs[3]:
            st.title("Что-если (Анализ чувствительности)")
            c1, c2 = st.columns(2)
            budg_boost = c1.slider("Увеличение трафика/бюджета (%)", 0, 100, 15)
            aov_boost = c2.slider("Рост среднего чека (%)", 0, 100, 10)
            
            proj_rev = total_rev * (1 + budg_boost/100) * (1 + aov_boost/100)
            proj_profit = total_profit * (1 + budg_boost/100) * (1 + aov_boost/100)
            
            c3, c4, c5 = st.columns(3)
            c3.metric("Текущая Выручка", f"{total_rev:,.0f} ₴")
            c4.metric("Моделируемая Выручка", f"{proj_rev:,.0f} ₴", f"+{proj_rev - total_rev:,.0f} ₴")
            c5.metric("Доп. Прибыль (ROI)", f"{proj_profit - total_profit:,.0f} ₴")

        # --- 4. MARKETING & GEO ---
        with tabs[4]:
            st.title("Маркетинговая аналитика")
            fig_sun = px.sunburst(df, path=['utm_source', 'utm_medium'], values='revenue', title="Структура выручки по каналам")
            st.plotly_chart(fig_sun, use_container_width=True)

        with tabs[5]:
            st.title("Геоаналитика")
            city_rev = df.groupby('city')['revenue'].sum().sort_values(ascending=False).head(20).reset_index()
            fig_geo = px.bar(city_rev, x='city', y='revenue', title="Топ-20 городов по выручке", text_auto='.2s')
            st.plotly_chart(fig_geo, use_container_width=True)

        # --- 5. AI DIRECTOR & GROWTH PLAN ---
        with tabs[6]:
            st.title("🧠 ИИ Финансовый Директор")
            if not api_key:
                st.warning("⚠️ Введите Google API Key в боковом меню для активации AI-Директора.")
            else:
                if st.button("Генерировать отчет Совета Директоров"):
                    with st.spinner("Анализ данных нейросетью..."):
                        try:
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            
                            # Формируем JSON-сводку для AI
                            ai_context = {
                                "finance": {"revenue": total_rev, "profit": total_profit, "aov": aov, "orders": orders},
                                "top_cities": city_rev.to_dict('records'),
                                "marketing": df.groupby('utm_source')['revenue'].sum().sort_values(ascending=False).head(5).to_dict()
                            }
                            
                            prompt = f"""Ты — CFO, CMO и CEO компании уровня Big4. Изучи этот JSON массив финансовых данных: {json.dumps(ai_context)}. 
                            Подготовь строгий отчет для совета директоров. Разделы: 1. Executive Summary 2. Основные риски и потери 3. Точки кратного роста 4. План по маркетингу на 30 дней. Пиши профессионально, используй цифры, предлагай жесткие решения."""
                            
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Ошибка API: {str(e)}")

        with tabs[7]:
            st.title("🚀 AI План роста бизнеса")
            if api_key:
                if st.button("Разработать план на 3-6-12 месяцев"):
                    with st.spinner("Построение стратегии..."):
                        try:
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            prompt_growth = f"Основываясь на данных: выручка {total_rev}, средний чек {aov}, лучшие каналы {df['utm_source'].unique()[:3]}. Напиши план масштабирования бизнеса на 3, 6 и 12 месяцев. Включи конкретные гипотезы по рекламе, KPI и прогноз ROI."
                            res_growth = model.generate_content(prompt_growth)
                            st.markdown(res_growth.text)
                        except Exception as e:
                            st.error("Проверьте API ключ.")
            else:
                st.info("Требуется API ключ.")
else:
    st.info("👈 Загрузите реестр заказов в боковом меню для начала аудита.")
