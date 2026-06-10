"""
BI Platform: Executive CFO & CMO / Accounting Studio Core
(Integrated with advanced Nomenclature Parsing)
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
        # 1. Чтение файла (поддержка и CSV, и XLSX)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, dtype=str, low_memory=False)
        else:
            df = pd.read_excel(file, dtype=str)
            
        # 2. Подготовка базовых числовых колонок перед сложной логикой
        num_cols = ['Цена', 'Количество', 'Стоимость', 'Итого']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        def is_razom(name):
            return 'Разом дешевше' in str(name)

        results = []
        
        # 3. Твоя сложная логика парсинга номенклатуры (С сохранением метаданных для BI!)
        for order_num, group in df.groupby('Номер заказа', sort=False):
            rows = group.reset_index(drop=True)
            n = len(rows)
            itogo = rows['Итого'].iloc[0] if 'Итого' in rows.columns else 0
            
            # Сохраняем метаданные всего заказа для аналитики
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

            # Пропорциональное распределение скидки заказа
            raw_sum = sum(x[1] for x in included)
            scale   = (itogo / raw_sum) if raw_sum else 0

            # Сборка финального чистого массива для BI
            for item in included:
                idx, raw_cost, item_art, item_name, qty, price = item
                net_cost = round(raw_cost * scale, 2)
                
                results.append({
                    **meta,
                    'art': item_art,
                    'product': item_name,
                    'qty': qty,
                    'revenue': net_cost  # <-- Это наша истинная чистая выручка
                })

        # 4. Превращаем результат в DataFrame и безопасно парсим даты
        df_clean = pd.DataFrame(results)
        
        if df_clean.empty:
            st.error("Файл не содержит валидных данных или строк.")
            return pd.DataFrame()
            
        # Надежный парсинг дат (исключающий ошибку 'must be 1-d array')
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['date'])
        
        # Добавляем заглушку для прибыли (например, маржинальность 35%)
        df_clean['profit'] = df_clean['revenue'] * 0.35

        return df_clean
        
    except Exception as e:
        st.error(f"❌ Ошибка при профилировании: {str(e)}")
        return pd.DataFrame()

# --- PROPHET FORECASTING ---
@st.cache_data
def generate_forecast(df, periods, freq='D'):
    daily_rev = df.groupby(df['date'].dt.date)['revenue'].sum().reset_index()
    daily_rev.columns = ['ds', 'y']
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name='UA') # Учет праздников
    m.fit(daily_rev)
    
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return m, forecast, daily_rev

# --- UI & SIDEBAR ---
st.sidebar.title("Настройки аналитики")
st.sidebar.markdown("### 🔑 ИИ Директор (Gemini API)")
api_key = st.sidebar.text_input("Введи Google API Key", type="password")
with st.sidebar.expander("ℹ️ Как получить API ключ?"):
    st.markdown("1. Перейди на [Google AI Studio](https://aistudio.google.com/app/apikey)\n2. Нажми **Create API Key**\n3. Скопируй ключ.")

uploaded_file = st.sidebar.file_uploader("📂 Загрузить файл выгрузки", type=['csv', 'xlsx'])

if uploaded_file:
    with st.spinner("Профилирование номенклатуры и очистка..."):
        raw_df = load_and_clean_data(uploaded_file)
        
    if not raw_df.empty:
        st.sidebar.markdown("### 🎯 Фильтры")
        # Фильтры
        all_statuses = raw_df['status'].unique().tolist()
        status_filter = st.sidebar.multiselect("Статус заказа", all_statuses, default=all_statuses)
        
        df = raw_df[raw_df['status'].isin(status_filter)]
        
        # UI TABS
        tabs = st.tabs([
            "📊 Dashboard", "💰 Продукты", "🔮 Прогноз продаж", 
            "🎛 Моделирование", "📣 Маркетинг", 
            "🗺️ Геоаналитика", "🧠 AI Директор", "🚀 AI План роста"
        ])

        # --- 1. DASHBOARD ---
        with tabs[0]:
            st.title("Executive Dashboard")
            total_rev = df['revenue'].sum()
            total_profit = df['profit'].sum()
            unique_orders = df['order_id'].nunique()
            aov = total_rev / unique_orders if unique_orders > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("💰 Чистая Выручка", f"{total_rev:,.0f} ₴")
            c2.metric("📈 Валовая прибыль", f"{total_profit:,.0f} ₴", "35% маржа (расч.)")
            c3.metric("🛒 Заказов", f"{unique_orders:,}")
            c4.metric("🧾 Средний чек", f"{aov:,.0f} ₴")
            
            st.markdown("### Динамика выручки")
            fig_trend = px.bar(df.groupby(df['date'].dt.to_period('W').dt.start_time)['revenue'].sum().reset_index(), 
                                x='date', y='revenue', template="plotly_white")
            st.plotly_chart(fig_trend, use_container_width=True)

        # --- 2. PRODUCT PERFORMANCE ---
        with tabs[1]:
            st.title("Товарная аналитика (С учетом наборов)")
            prod_df = df.groupby(['art', 'product']).agg({'qty':'sum', 'revenue':'sum'}).reset_index()
            prod_df = prod_df.sort_values(by='revenue', ascending=False)
            
            st.dataframe(prod_df.head(50).style.format({"revenue": "{:,.2f} ₴"}), use_container_width=True)

        # --- 3. FORECASTING ---
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

        # --- 4. WHAT-IF MODELING ---
        with tabs[3]:
            st.title("Что-если (Анализ чувствительности)")
            c1, c2 = st.columns(2)
            budg_boost = c1.slider("Рост объема заказов (%)", 0, 100, 15)
            aov_boost = c2.slider("Рост среднего чека (%)", 0, 100, 10)
            
            proj_rev = total_rev * (1 + budg_boost/100) * (1 + aov_boost/100)
            c3, c4 = st.columns(2)
            c3.metric("Текущая Выручка", f"{total_rev:,.0f} ₴")
            c4.metric("Моделируемая Выручка", f"{proj_rev:,.0f} ₴", f"+{proj_rev - total_rev:,.0f} ₴")

        # --- 5. MARKETING & GEO ---
        with tabs[4]:
            st.title("Маркетинговая аналитика")
            fig_sun = px.sunburst(df, path=['utm_source', 'utm_medium'], values='revenue', title="Структура выручки по каналам")
            st.plotly_chart(fig_sun, use_container_width=True)

        with tabs[5]:
            st.title("Геоаналитика")
            city_rev = df.groupby('city')['revenue'].sum().sort_values(ascending=False).head(20).reset_index()
            fig_geo = px.bar(city_rev, x='city', y='revenue', title="Топ-20 городов по выручке", text_auto='.2s')
            st.plotly_chart(fig_geo, use_container_width=True)

        # --- 6. AI DIRECTOR & GROWTH PLAN ---
        with tabs[6]:
            st.title("🧠 ИИ Финансовый Директор")
            if not api_key:
                st.warning("⚠️ Введите Google API Key в боковом меню для активации.")
            else:
                if st.button("Генерировать отчет Совета Директоров"):
                    with st.spinner("Анализ данных нейросетью..."):
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        ai_context = {
                            "finance": {"revenue": total_rev, "orders": unique_orders},
                            "top_products": prod_df.head(3).to_dict('records')
                        }
                        prompt = f"Ты CFO. Данные: {json.dumps(ai_context)}. Напиши стратегический вывод."
                        st.markdown(model.generate_content(prompt).text)

        with tabs[7]:
            st.title("🚀 AI План роста бизнеса")
            if api_key and st.button("Разработать план"):
                with st.spinner("Построение стратегии..."):
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    prompt_growth = f"Основываясь на выручке {total_rev} ₴ и чеке {aov} ₴. Напиши план масштабирования на 3, 6 и 12 месяцев."
                    st.markdown(model.generate_content(prompt_growth).text)
else:
    st.info("👈 Загрузите реестр заказов в боковом меню для старта.")
