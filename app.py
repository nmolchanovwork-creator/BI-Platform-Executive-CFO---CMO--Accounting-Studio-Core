"""
============================================================================================
BUSINESS ANALYTICS PLATFORM
Lead Data Scientist / CFO / CMO / BI Architect Level
============================================================================================
Комплексная финансовая, маркетинговая и коммерческая аналитика бизнеса
============================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import logging
import json
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Business Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0e1117; }

    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-value { font-size: 28px; font-weight: 700; color: #63b3ed; }
    .kpi-label { font-size: 12px; color: #a0aec0; margin-top: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-delta { font-size: 13px; margin-top: 6px; }
    .delta-pos { color: #68d391; }
    .delta-neg { color: #fc8181; }

    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #e2e8f0;
        padding: 12px 0 8px 0;
        border-bottom: 2px solid #2d4a7a;
        margin-bottom: 16px;
    }

    .risk-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .risk-high { background: #742a2a; color: #fc8181; }
    .risk-med  { background: #744210; color: #f6ad55; }
    .risk-low  { background: #1c4532; color: #68d391; }

    .insight-box {
        background: linear-gradient(135deg, #1a2744 0%, #162338 100%);
        border-left: 4px solid #4299e1;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 14px;
        color: #cbd5e0;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #1a1f2e;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        color: #a0aec0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #2d4a7a !important;
        color: #e2e8f0 !important;
    }

    div[data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN MAPPING  (EN / RU / UK)
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_MAP = {
    "date": [
        "date", "дата", "дата заказа", "order date", "дата_замовлення",
        "date_order", "order_date", "замовлення дата", "дата продажи",
        "дата продажу", "created_at", "created at",
    ],
    "status": [
        "status", "статус", "статус заказа", "order status",
        "статус замовлення", "order_status",
    ],
    "order_id": [
        "order_id", "номер заказа", "order id", "order number",
        "номер замовлення", "orderid", "id заказа", "id замовлення",
        "order", "замовлення",
    ],
    "customer": [
        "customer", "клиент", "client", "customer name",
        "клієнт", "покупатель", "покупець", "имя клиента",
    ],
    "city": [
        "city", "город", "місто", "city name", "town",
        "населений пункт", "населенный пункт",
    ],
    "region": [
        "region", "регион", "область", "регіон",
        "district", "territory", "территория",
    ],
    "country": [
        "country", "страна", "країна", "country name",
    ],
    "product": [
        "product", "товар", "product name", "item", "товар назва",
        "продукт", "назва товару", "наименование товара", "sku",
    ],
    "category": [
        "category", "категория", "категорія", "cat", "product category",
        "категория товара", "категорія товару",
    ],
    "quantity": [
        "quantity", "количество", "кількість", "qty", "amount",
        "count", "units", "кол-во",
    ],
    "cost": [
        "cost", "себестоимость", "собівартість", "cost price",
        "cogs", "purchase price", "себестоимость продажи",
    ],
    "price": [
        "price", "цена продажи", "ціна продажу", "sale price",
        "selling price", "unit price", "цена", "ціна",
    ],
    "total": [
        "total", "итого", "підсумок", "revenue", "total sales",
        "sum", "sales amount", "total revenue", "виручка",
        "сумма", "сума", "total_amount", "amount",
    ],
    "margin": [
        "margin", "маржа", "gross margin", "margin %",
        "маржинальность", "маржинальність",
    ],
    "profit": [
        "profit", "прибыль", "прибуток", "net profit",
        "gross profit", "earnings", "прибыль от продаж",
    ],
    "utm_source": [
        "utm_source", "source", "источник", "джерело",
        "traffic source", "источник трафика",
    ],
    "utm_medium": [
        "utm_medium", "medium", "канал", "channel",
        "traffic medium", "медиа",
    ],
    "utm_campaign": [
        "utm_campaign", "campaign", "кампания", "кампанія",
        "рекламная кампания", "рекламна кампанія",
    ],
    "utm_content": [
        "utm_content", "content", "контент", "ad content",
    ],
    "utm_term": [
        "utm_term", "term", "ключевое слово", "keyword",
        "ключове слово",
    ],
}


def detect_column(df: pd.DataFrame, field: str) -> Optional[str]:
    """Автоматически определяет колонку в датафрейме по списку alias."""
    aliases = COLUMN_MAP.get(field, [])
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in df_cols_lower:
            return df_cols_lower[alias.lower()]
    # Fuzzy partial match
    for alias in aliases:
        for col_lower, col_orig in df_cols_lower.items():
            if alias.lower() in col_lower or col_lower in alias.lower():
                return col_orig
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Загрузка данных...")
def load_data(uploaded_file) -> pd.DataFrame:
    """Загружает Excel или CSV файл."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            for enc in ["utf-8", "cp1251", "latin-1"]:
                try:
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except Exception:
                    uploaded_file.seek(0)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Поддерживаются только .csv, .xlsx, .xls файлы")
            return pd.DataFrame()
        logger.info(f"Файл загружен: {df.shape[0]} строк, {df.shape[1]} колонок")
        return df
    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        st.error(f"❌ Ошибка загрузки файла: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="⚙️ Подготовка данных...")
def prepare_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Основная очистка и подготовка данных."""
    df = df_raw.copy()
    col_map: Dict[str, str] = {}

    # Детектируем колонки
    for field in COLUMN_MAP:
        found = detect_column(df, field)
        if found:
            col_map[field] = found

    if "date" not in col_map:
        raise ValueError("Не найдена колонка с датой. Проверьте названия колонок.")
    if "total" not in col_map and "price" not in col_map:
        raise ValueError("Не найдена колонка с выручкой/суммой. Проверьте данные.")

    # Переименовываем
    rename = {v: k for k, v in col_map.items()}
    df = df.rename(columns=rename)

    # Дата
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    before = len(df)
    df = df.dropna(subset=["date"])
    dropped_dates = before - len(df)
    if dropped_dates:
        logger.warning(f"Удалено {dropped_dates} строк с некорректными датами")

    # Числовые поля
    numeric_fields = ["total", "profit", "cost", "price", "quantity", "margin"]
    for f in numeric_fields:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce").fillna(0)

    # Генерируем недостающие поля
    if "total" not in df.columns and "price" in df.columns and "quantity" in df.columns:
        df["total"] = df["price"] * df["quantity"]
    if "profit" not in df.columns and "total" in df.columns and "cost" in df.columns:
        df["profit"] = df["total"] - df["cost"]
    if "margin" not in df.columns and "total" in df.columns and df["total"].sum() > 0:
        if "profit" in df.columns:
            df["margin"] = df["profit"] / df["total"].replace(0, np.nan) * 100
        else:
            df["margin"] = 0.0

    # Удаляем нулевые и отрицательные продажи
    if "total" in df.columns:
        df = df[df["total"] > 0]

    # Дубли
    df = df.drop_duplicates()

    # UTM заполнение
    utm_defaults = {
        "utm_source": "organic",
        "utm_medium": "direct",
        "utm_campaign": "unknown",
        "utm_content": "unknown",
        "utm_term": "unknown",
    }
    for utm_col, default_val in utm_defaults.items():
        if utm_col in df.columns:
            df[utm_col] = df[utm_col].fillna(default_val).replace("", default_val)
        else:
            df[utm_col] = default_val

    # Строковые поля
    for col in ["customer", "city", "region", "country", "product", "category", "status"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        else:
            df[col] = "Unknown"

    # Дополнительные временные признаки
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week"]    = df["date"].dt.isocalendar().week.astype(int)
    df["dow"]     = df["date"].dt.dayofweek
    df["yearmonth"] = df["date"].dt.to_period("M")

    logger.info(f"Данные подготовлены: {len(df)} строк")
    return df, col_map


# ─────────────────────────────────────────────────────────────────────────────
# ФИНАНСОВЫЕ РАСЧЁТЫ
# ─────────────────────────────────────────────────────────────────────────────
def calc_financial_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Рассчитывает все основные финансовые KPI."""
    kpis: Dict[str, Any] = {}

    kpis["revenue"]       = df["total"].sum()
    kpis["profit"]        = df["profit"].sum() if "profit" in df.columns else 0
    kpis["orders"]        = df["order_id"].nunique() if "order_id" in df.columns else len(df)
    kpis["customers"]     = df["customer"].nunique() if "customer" in df.columns else 0
    kpis["avg_check"]     = kpis["revenue"] / kpis["orders"] if kpis["orders"] > 0 else 0
    kpis["margin_pct"]    = kpis["profit"] / kpis["revenue"] * 100 if kpis["revenue"] > 0 else 0
    kpis["arpu"]          = kpis["revenue"] / kpis["customers"] if kpis["customers"] > 0 else 0

    # Retention & Repeat
    if "customer" in df.columns:
        cust_orders = df.groupby("customer")["order_id"].nunique() if "order_id" in df.columns else df.groupby("customer").size()
        repeat_customers = (cust_orders > 1).sum()
        kpis["repeat_customers"] = int(repeat_customers)
        kpis["retention_rate"]   = repeat_customers / kpis["customers"] * 100 if kpis["customers"] > 0 else 0
    else:
        kpis["repeat_customers"] = 0
        kpis["retention_rate"]   = 0

    # LTV (упрощённый = ARPU * Retention)
    months_active = max((df["date"].max() - df["date"].min()).days / 30, 1)
    kpis["ltv"] = kpis["arpu"] * (1 + kpis["retention_rate"] / 100) * min(months_active, 12) / 12

    return kpis


def calc_growth_rates(df: pd.DataFrame) -> Dict[str, float]:
    """Рассчитывает MoM, QoQ, YoY и CAGR."""
    rates: Dict[str, float] = {}
    monthly = df.groupby("yearmonth")["total"].sum().sort_index()

    if len(monthly) >= 2:
        rates["mom"] = (monthly.iloc[-1] / monthly.iloc[-2] - 1) * 100 if monthly.iloc[-2] != 0 else 0
    else:
        rates["mom"] = 0

    quarterly = df.groupby([df["date"].dt.year, df["quarter"]])["total"].sum()
    if len(quarterly) >= 2:
        rates["qoq"] = (quarterly.iloc[-1] / quarterly.iloc[-2] - 1) * 100 if quarterly.iloc[-2] != 0 else 0
    else:
        rates["qoq"] = 0

    yearly = df.groupby("year")["total"].sum()
    if len(yearly) >= 2:
        rates["yoy"] = (yearly.iloc[-1] / yearly.iloc[-2] - 1) * 100 if yearly.iloc[-2] != 0 else 0
    else:
        rates["yoy"] = 0

    # CAGR
    if len(yearly) >= 2:
        n = len(yearly) - 1
        rates["cagr"] = ((yearly.iloc[-1] / yearly.iloc[0]) ** (1 / n) - 1) * 100 if yearly.iloc[0] > 0 else 0
    else:
        rates["cagr"] = 0

    return rates


def calc_risk_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Рассчитывает метрики рисков."""
    risks: Dict[str, Any] = {}

    monthly_rev = df.groupby("yearmonth")["total"].sum()
    if len(monthly_rev) > 1:
        risks["revenue_volatility"] = monthly_rev.std() / monthly_rev.mean() * 100
        risks["sales_decline_risk"] = "Высокий" if monthly_rev.iloc[-1] < monthly_rev.mean() * 0.85 else \
                                      "Средний"  if monthly_rev.iloc[-1] < monthly_rev.mean() * 0.95 else "Низкий"
    else:
        risks["revenue_volatility"] = 0
        risks["sales_decline_risk"] = "Нет данных"

    if "customer" in df.columns:
        cust_share = df.groupby("customer")["total"].sum() / df["total"].sum() * 100
        risks["top_customer_share"] = cust_share.max()
        risks["customer_concentration"] = "Высокий" if risks["top_customer_share"] > 20 else \
                                          "Средний"  if risks["top_customer_share"] > 10 else "Низкий"

    if "city" in df.columns:
        city_share = df.groupby("city")["total"].sum() / df["total"].sum() * 100
        risks["top_city_share"] = city_share.max()
        risks["top_city"] = city_share.idxmax()
        risks["city_concentration"] = "Высокий" if risks["top_city_share"] > 40 else \
                                      "Средний"  if risks["top_city_share"] > 25 else "Низкий"

    channel_share = df.groupby("utm_source")["total"].sum() / df["total"].sum() * 100
    risks["top_channel_share"] = channel_share.max()
    risks["top_channel"] = channel_share.idxmax()
    risks["channel_concentration"] = "Высокий" if risks["top_channel_share"] > 50 else \
                                     "Средний"  if risks["top_channel_share"] > 30 else "Низкий"

    return risks


# ─────────────────────────────────────────────────────────────────────────────
# ПРОГНОЗИРОВАНИЕ
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Обучение модели прогнозирования...")
def train_forecast_model(df_json: str):
    """Обучает модель прогнозирования (Prophet или XGBoost fallback)."""
    df = pd.read_json(df_json)
    df["date"] = pd.to_datetime(df["date"])

    daily = df.groupby("date")["total"].sum().reset_index()
    daily.columns = ["ds", "y"]
    daily = daily.sort_values("ds")

    # Попытка Prophet
    try:
        from prophet import Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
        )
        # Украинские праздники
        ukraine_holidays = pd.DataFrame({
            "holiday": ["New Year", "Christmas Orthodox", "International Womens Day",
                        "Easter", "Labour Day", "Victory Day", "Constitution Day",
                        "Independence Day", "Defenders Day", "Christmas Catholic"],
            "ds": pd.to_datetime([
                "2024-01-01", "2024-01-07", "2024-03-08",
                "2024-05-05", "2024-05-01", "2024-05-09", "2024-06-28",
                "2024-08-24", "2024-10-14", "2024-12-25",
            ]),
            "lower_window": 0,
            "upper_window": 1,
        })
        model.add_country_holidays(country_name="UA")
        model.fit(daily)
        return ("prophet", model, daily)
    except ImportError:
        logger.warning("Prophet не установлен, использую XGBoost")

    # Fallback: простая регрессия на временных признаках
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        daily["t"]   = np.arange(len(daily))
        daily["dow"] = pd.to_datetime(daily["ds"]).dt.dayofweek
        daily["mon"] = pd.to_datetime(daily["ds"]).dt.month
        daily["wk"]  = pd.to_datetime(daily["ds"]).dt.isocalendar().week.astype(int)
        X = daily[["t", "dow", "mon", "wk"]].values
        y = daily["y"].values
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
        model.fit(X, y)
        return ("gbr", model, daily)
    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        return ("naive", None, daily)


def generate_forecast(model_tuple, horizon_days: int, margin_pct: float) -> pd.DataFrame:
    """Генерирует прогноз по обученной модели."""
    model_type, model, daily = model_tuple
    last_date = pd.to_datetime(daily["ds"]).max()

    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon_days, freq="D")

    if model_type == "prophet":
        future = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)
        fc = forecast[forecast["ds"] > last_date][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        fc.columns = ["date", "revenue", "rev_lower", "rev_upper"]
    elif model_type == "gbr":
        n = len(daily)
        rows = []
        for i, d in enumerate(future_dates):
            t   = n + i
            dow = d.dayofweek
            mon = d.month
            wk  = d.isocalendar().week
            pred = max(model.predict([[t, dow, mon, wk]])[0], 0)
            rows.append({"date": d, "revenue": pred,
                         "rev_lower": pred * 0.8, "rev_upper": pred * 1.2})
        fc = pd.DataFrame(rows)
    else:
        # Naive: среднее за последние 30 дней
        avg_daily = daily["y"].tail(30).mean()
        fc = pd.DataFrame({
            "date":      future_dates,
            "revenue":   avg_daily,
            "rev_lower": avg_daily * 0.8,
            "rev_upper": avg_daily * 1.2,
        })

    fc["revenue"]   = fc["revenue"].clip(lower=0)
    fc["rev_lower"] = fc["rev_lower"].clip(lower=0)
    fc["rev_upper"] = fc["rev_upper"].clip(lower=0)

    # Сценарии
    fc["optimistic"]  = fc["rev_upper"]
    fc["realistic"]   = fc["revenue"]
    fc["pessimistic"] = fc["rev_lower"]

    # Прибыль
    fc["profit_base"] = fc["realistic"]  * margin_pct / 100
    fc["profit_opt"]  = fc["optimistic"] * margin_pct / 100
    fc["profit_pess"] = fc["pessimistic"]* margin_pct / 100

    return fc


# ─────────────────────────────────────────────────────────────────────────────
# МАРКЕТИНГОВАЯ АНАЛИТИКА
# ─────────────────────────────────────────────────────────────────────────────
def calc_marketing_metrics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Рассчитывает маркетинговые метрики."""
    mkt: Dict[str, pd.DataFrame] = {}

    mkt["by_source"]   = df.groupby("utm_source").agg(
        revenue=("total","sum"), orders=("total","count"),
        profit=("profit","sum") if "profit" in df.columns else ("total","count")
    ).reset_index().sort_values("revenue", ascending=False)
    mkt["by_source"]["share"] = mkt["by_source"]["revenue"] / mkt["by_source"]["revenue"].sum() * 100

    mkt["by_medium"]   = df.groupby("utm_medium").agg(revenue=("total","sum"), orders=("total","count")).reset_index()
    mkt["by_campaign"] = df.groupby("utm_campaign").agg(
        revenue=("total","sum"), orders=("total","count")
    ).reset_index().sort_values("revenue", ascending=False)

    if "city" in df.columns:
        mkt["by_city"] = df.groupby("city").agg(
            revenue=("total","sum"), orders=("total","count"),
            profit=("profit","sum") if "profit" in df.columns else ("total","count")
        ).reset_index().sort_values("revenue", ascending=False)

    return mkt


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UI
# ─────────────────────────────────────────────────────────────────────────────
def fmt_money(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f}М"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}К"
    return f"{v:.0f}"


def kpi_card(label: str, value: str, delta: Optional[str] = None, delta_pos: bool = True):
    delta_html = ""
    if delta:
        cls = "delta-pos" if delta_pos else "delta-neg"
        arrow = "▲" if delta_pos else "▼"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""


def render_kpi_row(items: List[Tuple]):
    """Рендерит строку KPI карточек (label, value, delta, delta_pos)."""
    cols = st.columns(len(items))
    for i, item in enumerate(items):
        label = item[0]
        value = item[1]
        delta = item[2] if len(item) > 2 else None
        d_pos = item[3] if len(item) > 3 else True
        with cols[i]:
            st.markdown(kpi_card(label, value, delta, d_pos), unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def insight(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 1. SIDEBAR ФИЛЬТРЫ
# ─────────────────────────────────────────────────────────────────────────────
def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## 🔧 Фильтры")

    min_d, max_d = df["date"].min().date(), df["date"].max().date()
    date_range = st.sidebar.date_input("📅 Период", value=(min_d, max_d),
                                        min_value=min_d, max_value=max_d)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        df = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]

    if "status" in df.columns:
        statuses = ["Все"] + sorted(df["status"].unique().tolist())
        sel_status = st.sidebar.selectbox("📦 Статус заказа", statuses)
        if sel_status != "Все":
            df = df[df["status"] == sel_status]

    if "city" in df.columns:
        cities = ["Все"] + sorted(df["city"].unique().tolist())
        sel_city = st.sidebar.selectbox("🏙️ Город", cities)
        if sel_city != "Все":
            df = df[df["city"] == sel_city]

    if "region" in df.columns:
        regions = ["Все"] + sorted(df["region"].unique().tolist())
        sel_region = st.sidebar.selectbox("🗺️ Регион", regions)
        if sel_region != "Все":
            df = df[df["region"] == sel_region]

    sources = ["Все"] + sorted(df["utm_source"].unique().tolist())
    sel_source = st.sidebar.selectbox("📡 Источник трафика", sources)
    if sel_source != "Все":
        df = df[df["utm_source"] == sel_source]

    campaigns = ["Все"] + sorted(df["utm_campaign"].unique().tolist())
    sel_campaign = st.sidebar.selectbox("📢 Кампания", campaigns)
    if sel_campaign != "Все":
        df = df[df["utm_campaign"] == sel_campaign]

    st.sidebar.markdown(f"---\n**Записей:** {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 2. DASHBOARD TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_dashboard(df: pd.DataFrame):
    kpis   = calc_financial_kpis(df)
    growth = calc_growth_rates(df)
    risks  = calc_risk_metrics(df)

    section("📊 Executive Dashboard")

    render_kpi_row([
        ("Выручка",        f"₴{fmt_money(kpis['revenue'])}",
         f"{growth['mom']:+.1f}% MoM", growth['mom'] >= 0),
        ("Валовая прибыль",f"₴{fmt_money(kpis['profit'])}",
         f"{kpis['margin_pct']:.1f}% маржа", kpis['margin_pct'] >= 15),
        ("Заказов",        f"{kpis['orders']:,}",
         f"Средний чек ₴{fmt_money(kpis['avg_check'])}", True),
        ("Клиентов",       f"{kpis['customers']:,}",
         f"Retention {kpis['retention_rate']:.1f}%", kpis['retention_rate'] >= 20),
        ("LTV",            f"₴{fmt_money(kpis['ltv'])}",
         f"ARPU ₴{fmt_money(kpis['arpu'])}", True),
        ("CAGR",           f"{growth['cagr']:.1f}%",
         f"YoY {growth['yoy']:+.1f}%", growth['yoy'] >= 0),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue trend
    monthly = df.groupby("yearmonth").agg(
        revenue=("total", "sum"),
        profit=("profit", "sum") if "profit" in df.columns else ("total", "count"),
        orders=("total", "count"),
    ).reset_index()
    monthly["yearmonth_str"] = monthly["yearmonth"].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly["yearmonth_str"], y=monthly["revenue"],
                             name="Выручка", marker_color="#4299e1"))
        fig.add_trace(go.Scatter(x=monthly["yearmonth_str"], y=monthly["profit"],
                                 name="Прибыль", line=dict(color="#68d391", width=2), yaxis="y2"))
        fig.update_layout(
            title="📈 Выручка и Прибыль по месяцам",
            yaxis=dict(title="Выручка"),
            yaxis2=dict(title="Прибыль", overlaying="y", side="right"),
            template="plotly_dark", height=350,
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "utm_source" in df.columns:
            src = df.groupby("utm_source")["total"].sum().reset_index()
            fig2 = px.pie(src, values="total", names="utm_source",
                          title="🥧 Выручка по каналам",
                          template="plotly_dark", hole=0.4,
                          color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "category" in df.columns:
            cat = df.groupby("category")["total"].sum().nlargest(10).reset_index()
            fig3 = px.bar(cat, x="total", y="category", orientation="h",
                          title="🏆 Топ категорий по выручке",
                          template="plotly_dark", color="total",
                          color_continuous_scale="Blues")
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Риски
        section("⚠️ Карта рисков")
        risk_items = [
            ("Падение продаж",    risks.get("sales_decline_risk", "N/A")),
            ("Концентрация клиентов", risks.get("customer_concentration", "N/A")),
            ("Концентрация городов",  risks.get("city_concentration", "N/A")),
            ("Концентрация каналов",  risks.get("channel_concentration", "N/A")),
        ]
        for name, level in risk_items:
            cls = {"Высокий": "risk-high", "Средний": "risk-med", "Низкий": "risk-low"}.get(level, "risk-med")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin:6px 0;">'
                f'<span style="color:#a0aec0">{name}</span>'
                f'<span class="risk-badge {cls}">{level}</span></div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 2. ФИНАНСЫ TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_finance(df: pd.DataFrame):
    kpis   = calc_financial_kpis(df)
    growth = calc_growth_rates(df)
    risks  = calc_risk_metrics(df)

    section("💰 Финансовая аналитика")

    render_kpi_row([
        ("Выручка",       f"₴{fmt_money(kpis['revenue'])}"),
        ("Прибыль",       f"₴{fmt_money(kpis['profit'])}"),
        ("Маржа",         f"{kpis['margin_pct']:.1f}%"),
        ("Средний чек",   f"₴{fmt_money(kpis['avg_check'])}"),
        ("Заказов",       f"{kpis['orders']:,}"),
        ("Клиентов",      f"{kpis['customers']:,}"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    render_kpi_row([
        ("Повторные продажи", f"{kpis['repeat_customers']:,}"),
        ("Retention Rate",    f"{kpis['retention_rate']:.1f}%"),
        ("LTV",               f"₴{fmt_money(kpis['ltv'])}"),
        ("ARPU",              f"₴{fmt_money(kpis['arpu'])}"),
        ("MoM Growth",        f"{growth['mom']:+.1f}%", None, growth['mom'] >= 0),
        ("YoY Growth",        f"{growth['yoy']:+.1f}%", None, growth['yoy'] >= 0),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Динамика выручки
    monthly = df.groupby("yearmonth").agg(
        revenue=("total","sum"),
        profit=("profit","sum") if "profit" in df.columns else ("total","count"),
    ).reset_index()
    monthly["yearmonth_str"] = monthly["yearmonth"].astype(str)
    monthly["mom_growth"] = monthly["revenue"].pct_change() * 100

    col1, col2 = st.columns(2)
    with col1:
        fig = px.area(monthly, x="yearmonth_str", y="revenue",
                      title="📈 Динамика выручки", template="plotly_dark",
                      color_discrete_sequence=["#4299e1"])
        fig.update_traces(fill="tozeroy")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(monthly, x="yearmonth_str", y="mom_growth",
                      title="📊 MoM рост выручки (%)", template="plotly_dark",
                      color="mom_growth",
                      color_continuous_scale=["#fc8181","#f6ad55","#68d391"])
        st.plotly_chart(fig2, use_container_width=True)

    # Квартальный анализ
    quarterly = df.groupby(["year","quarter"]).agg(
        revenue=("total","sum"),
        profit=("profit","sum") if "profit" in df.columns else ("total","count"),
        orders=("total","count"),
    ).reset_index()
    quarterly["period"] = "Q" + quarterly["quarter"].astype(str) + " " + quarterly["year"].astype(str)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.bar(quarterly, x="period", y=["revenue","profit"],
                      title="📦 Квартальный анализ",
                      template="plotly_dark", barmode="group",
                      color_discrete_sequence=["#4299e1","#68d391"])
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Waterfall прибыли
        fig4 = go.Figure(go.Waterfall(
            x=quarterly["period"].tolist(),
            y=quarterly["profit"].tolist(),
            connector={"line": {"color": "#4a5568"}},
            increasing={"marker": {"color": "#68d391"}},
            decreasing={"marker": {"color": "#fc8181"}},
        ))
        fig4.update_layout(title="💧 Waterfall прибыли", template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)

    # Анализ рисков
    section("⚠️ Анализ рисков")
    col5, col6 = st.columns(2)
    with col5:
        vol = risks.get("revenue_volatility", 0)
        fig5 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=vol,
            title={"text": "Волатильность выручки (%)"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "#4299e1"},
                "steps": [
                    {"range": [0,  20], "color": "#1c4532"},
                    {"range": [20, 50], "color": "#744210"},
                    {"range": [50,100], "color": "#742a2a"},
                ],
            },
        ))
        fig5.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        if "customer" in df.columns:
            top5_cust = df.groupby("customer")["total"].sum().nlargest(5).reset_index()
            total_rev  = df["total"].sum()
            top5_cust["share"] = top5_cust["total"] / total_rev * 100
            fig6 = px.bar(top5_cust, x="customer", y="share",
                          title="👤 Топ-5 клиентов (доля выручки %)",
                          template="plotly_dark", color="share",
                          color_continuous_scale="Reds")
            st.plotly_chart(fig6, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 3. ПРОГНОЗ TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_forecast(df: pd.DataFrame):
    section("🔮 Прогнозирование продаж")

    col_h, col_m = st.columns([2, 1])
    with col_h:
        horizon_label = st.selectbox("Горизонт прогнозирования", [
            "1 месяц", "2 месяца", "3 месяца", "6 месяцев", "12 месяцев"])
    horizon_map = {"1 месяц": 30, "2 месяца": 60, "3 месяца": 90,
                   "6 месяцев": 180, "12 месяцев": 365}
    horizon_days = horizon_map[horizon_label]

    kpis       = calc_financial_kpis(df)
    margin_pct = kpis["margin_pct"]

    df_json   = df[["date", "total"]].to_json()
    model_tup = train_forecast_model(df_json)

    with st.spinner("📡 Генерируем прогноз..."):
        fc = generate_forecast(model_tup, horizon_days, margin_pct)

    # Исторические данные
    hist_daily = df.groupby("date")["total"].sum().reset_index()
    hist_daily.columns = ["date", "revenue"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_daily["date"], y=hist_daily["revenue"],
        name="История", line=dict(color="#a0aec0"), opacity=0.7))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["optimistic"],
        name="Оптимистичный", line=dict(color="#68d391", dash="dot")))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["realistic"],
        name="Реалистичный", line=dict(color="#4299e1", width=2.5)))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["pessimistic"],
        name="Пессимистичный", line=dict(color="#fc8181", dash="dot")))
    fig.add_trace(go.Scatter(
        x=list(fc["date"]) + list(fc["date"])[::-1],
        y=list(fc["rev_upper"]) + list(fc["rev_lower"])[::-1],
        fill="toself", fillcolor="rgba(66,153,225,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Доверительный интервал"))
    fig.update_layout(
        title=f"📈 Прогноз выручки на {horizon_label}",
        template="plotly_dark", height=420,
        xaxis_title="Дата", yaxis_title="Выручка",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Сводная таблица сценариев
    section("📋 Сводка сценариев")
    fc_monthly = fc.copy()
    fc_monthly["month"] = fc_monthly["date"].dt.to_period("M").astype(str)
    summary = fc_monthly.groupby("month").agg(
        optimistic=("optimistic","sum"),
        realistic=("realistic","sum"),
        pessimistic=("pessimistic","sum"),
        profit_opt=("profit_opt","sum"),
        profit_base=("profit_base","sum"),
        profit_pess=("profit_pess","sum"),
    ).reset_index()

    for _, row in summary.iterrows():
        with st.expander(f"📅 {row['month']}"):
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("🟢 Оптимист. выручка", f"₴{fmt_money(row['optimistic'])}")
                st.metric("🟢 Оптимист. прибыль", f"₴{fmt_money(row['profit_opt'])}")
            with r2:
                st.metric("🔵 Реалист. выручка",  f"₴{fmt_money(row['realistic'])}")
                st.metric("🔵 Реалист. прибыль",  f"₴{fmt_money(row['profit_base'])}")
            with r3:
                st.metric("🔴 Пессимист. выручка",f"₴{fmt_money(row['pessimistic'])}")
                st.metric("🔴 Пессимист. прибыль",f"₴{fmt_money(row['profit_pess'])}")

    # Финансовое моделирование "Что если"
    section("🎚️ Финансовое моделирование (Что если)")
    insight("Передвигайте слайдеры, чтобы смоделировать влияние изменений на бизнес")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1: budget_inc  = st.slider("📣 Рекл. бюджет +%", 0, 200, 0, 5)
    with col_s2: check_inc   = st.slider("💳 Средний чек +%", 0, 100, 0, 5)
    with col_s3: conv_inc    = st.slider("🎯 Конверсия +%", 0, 100, 0, 5)
    with col_s4: ret_inc     = st.slider("🔄 Retention +%", 0, 100, 0, 5)

    base_revenue = summary["realistic"].sum()
    base_profit  = summary["profit_base"].sum()

    # Упрощённые коэффициенты влияния
    impact_budget  = 0.30
    impact_check   = 1.00
    impact_conv    = 0.50
    impact_ret     = 0.40

    adj_revenue = base_revenue * (
        1 + budget_inc / 100 * impact_budget
          + check_inc  / 100 * impact_check
          + conv_inc   / 100 * impact_conv
          + ret_inc    / 100 * impact_ret
    )
    adj_profit  = adj_revenue * margin_pct / 100
    extra_profit= adj_profit - base_profit
    roi_pct     = (extra_profit / max(base_profit, 1)) * 100

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("📈 Ожид. выручка",  f"₴{fmt_money(adj_revenue)}", f"+{fmt_money(adj_revenue-base_revenue)}")
    mc2.metric("💰 Ожид. прибыль",  f"₴{fmt_money(adj_profit)}",  f"+{fmt_money(extra_profit)}")
    mc3.metric("➕ Доп. прибыль",   f"₴{fmt_money(extra_profit)}")
    mc4.metric("📊 ROI изменений",   f"{roi_pct:.1f}%")

    # Анализ чувствительности
    sensitivity_params = ["Рекл. бюджет", "Средний чек", "Конверсия", "Retention"]
    sensitivity_impacts = [
        base_revenue * impact_budget * 0.5,
        base_revenue * impact_check  * 0.5,
        base_revenue * impact_conv   * 0.5,
        base_revenue * impact_ret    * 0.5,
    ]
    fig_sens = px.bar(
        x=sensitivity_impacts, y=sensitivity_params, orientation="h",
        title="🎯 Анализ чувствительности (влияние +50% каждого параметра)",
        template="plotly_dark", color=sensitivity_impacts,
        color_continuous_scale="Blues",
        labels={"x": "Доп. выручка", "y": "Параметр"},
    )
    st.plotly_chart(fig_sens, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 8. ГЕОАНАЛИТИКА TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_geo(df: pd.DataFrame):
    section("🗺️ Геоаналитика")

    if "city" not in df.columns or df["city"].nunique() < 2:
        st.info("Недостаточно данных для геоаналитики.")
        return

    city_rev = df.groupby("city").agg(
        revenue=("total","sum"),
        profit=("profit","sum") if "profit" in df.columns else ("total","count"),
        orders=("total","count"),
        customers=("customer","nunique") if "customer" in df.columns else ("total","count"),
    ).reset_index().sort_values("revenue", ascending=False)

    city_rev["ltv"] = city_rev["revenue"] / city_rev["customers"].replace(0, np.nan).fillna(1)
    top20_rev  = city_rev.head(20)
    top20_prof = city_rev.sort_values("profit", ascending=False).head(20)
    top20_ltv  = city_rev.sort_values("ltv", ascending=False).head(20)

    tab_r, tab_p, tab_l = st.tabs(["Топ-20 по выручке","Топ-20 по прибыли","Топ-20 по LTV"])

    with tab_r:
        fig = px.bar(top20_rev, x="revenue", y="city", orientation="h",
                     template="plotly_dark", color="revenue",
                     color_continuous_scale="Blues",
                     title="🏆 Топ-20 городов по выручке")
        st.plotly_chart(fig, use_container_width=True)

    with tab_p:
        fig2 = px.bar(top20_prof, x="profit", y="city", orientation="h",
                      template="plotly_dark", color="profit",
                      color_continuous_scale="Greens",
                      title="💰 Топ-20 городов по прибыли")
        st.plotly_chart(fig2, use_container_width=True)

    with tab_l:
        fig3 = px.bar(top20_ltv, x="ltv", y="city", orientation="h",
                      template="plotly_dark", color="ltv",
                      color_continuous_scale="Purples",
                      title="👥 Топ-20 городов по LTV клиента")
        st.plotly_chart(fig3, use_container_width=True)

    # Scatter по городам
    st.markdown("<br>", unsafe_allow_html=True)
    fig4 = px.scatter(city_rev, x="revenue", y="profit", size="orders",
                      color="ltv", text="city",
                      title="🔵 Матрица город: Выручка vs Прибыль (размер = заказы)",
                      template="plotly_dark", color_continuous_scale="Viridis",
                      labels={"revenue":"Выручка","profit":"Прибыль","ltv":"LTV"})
    fig4.update_traces(textposition="top center")
    st.plotly_chart(fig4, use_container_width=True)

    if "region" in df.columns:
        region_rev = df.groupby("region")["total"].sum().reset_index()
        fig5 = px.treemap(region_rev, path=["region"], values="total",
                          title="🗺️ Выручка по регионам (Treemap)",
                          template="plotly_dark",
                          color="total", color_continuous_scale="Blues")
        st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 5. МАРКЕТИНГ TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_marketing(df: pd.DataFrame):
    section("📣 Маркетинговая аналитика")
    mkt = calc_marketing_metrics(df)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.sunburst(
            df.groupby(["utm_source","utm_medium","utm_campaign"])["total"].sum().reset_index(),
            path=["utm_source","utm_medium","utm_campaign"],
            values="total", title="☀️ Sunburst: Источник → Медиа → Кампания",
            template="plotly_dark", color="total", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "by_campaign" in mkt:
            fig2 = px.treemap(
                mkt["by_campaign"], path=["utm_campaign"], values="revenue",
                title="🌳 Treemap кампаний", template="plotly_dark",
                color="revenue", color_continuous_scale="Blues",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Pareto 80/20
    if "by_campaign" in mkt:
        camp = mkt["by_campaign"].sort_values("revenue", ascending=False).copy()
        camp["cum_share"] = camp["revenue"].cumsum() / camp["revenue"].sum() * 100
        camp["rank"] = range(1, len(camp)+1)

        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(x=camp["utm_campaign"], y=camp["revenue"],
                              name="Выручка", marker_color="#4299e1"), secondary_y=False)
        fig3.add_trace(go.Scatter(x=camp["utm_campaign"], y=camp["cum_share"],
                                  name="Накопл. доля %", line=dict(color="#f6ad55")), secondary_y=True)
        fig3.add_hline(y=80, line_dash="dash", line_color="#fc8181",
                       annotation_text="80%", secondary_y=True)
        fig3.update_layout(title="📐 Pareto 80/20: Кампании", template="plotly_dark", height=380)
        st.plotly_chart(fig3, use_container_width=True)

    # Funnel
    funnel_data = mkt.get("by_source", pd.DataFrame())
    if len(funnel_data) > 0:
        fig4 = go.Figure(go.Funnel(
            y=funnel_data["utm_source"].tolist(),
            x=funnel_data["revenue"].tolist(),
            textinfo="value+percent initial",
            marker=dict(color=px.colors.sequential.Blues_r[:len(funnel_data)]),
        ))
        fig4.update_layout(title="🔻 Funnel выручки по источникам", template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)

    # Таблица каналов
    section("📊 Детальный анализ каналов")
    if "by_source" in mkt:
        df_tbl = mkt["by_source"].copy()
        df_tbl.columns = ["Источник","Выручка","Заказов","Прибыль","Доля %"]
        df_tbl["Выручка"]  = df_tbl["Выручка"].apply(lambda x: f"₴{fmt_money(x)}")
        df_tbl["Прибыль"]  = df_tbl["Прибыль"].apply(lambda x: f"₴{fmt_money(x)}")
        df_tbl["Доля %"]   = df_tbl["Доля %"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(df_tbl, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 7. SALES INTELLIGENCE TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_sales_intelligence(df: pd.DataFrame):
    section("🧠 Sales Intelligence")

    # Товарная аналитика
    if "product" in df.columns:
        prod_monthly = df.groupby(["product","yearmonth"])["total"].sum().reset_index()
        prod_monthly["yearmonth_str"] = prod_monthly["yearmonth"].astype(str)

        prod_stats = df.groupby("product").agg(
            revenue=("total","sum"),
            orders=("total","count"),
            profit=("profit","sum") if "profit" in df.columns else ("total","count"),
        ).reset_index()

        # Рост товаров: сравниваем первую и вторую половину периода
        mid_date = df["date"].min() + (df["date"].max() - df["date"].min()) / 2
        prod_h1 = df[df["date"] <= mid_date].groupby("product")["total"].sum()
        prod_h2 = df[df["date"] >  mid_date].groupby("product")["total"].sum()
        prod_growth = pd.DataFrame({"h1": prod_h1, "h2": prod_h2}).fillna(0)
        prod_growth["growth_pct"] = (prod_growth["h2"] / prod_growth["h1"].replace(0, np.nan) - 1) * 100

        growing = prod_growth.sort_values("growth_pct", ascending=False).head(10)
        declining = prod_growth.sort_values("growth_pct").head(10)

        col1, col2 = st.columns(2)
        with col1:
            insight(f"🚀 Быстрорастущих товаров: {(prod_growth['growth_pct']>20).sum()}")
            fig = px.bar(growing.reset_index(), x="growth_pct", y="product",
                         orientation="h", title="📈 Растущие товары",
                         template="plotly_dark", color="growth_pct",
                         color_continuous_scale="Greens")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            insight(f"📉 Падающих товаров: {(prod_growth['growth_pct']<-10).sum()}")
            fig2 = px.bar(declining.reset_index(), x="growth_pct", y="product",
                          orientation="h", title="📉 Падающие товары",
                          template="plotly_dark", color="growth_pct",
                          color_continuous_scale="Reds_r")
            st.plotly_chart(fig2, use_container_width=True)

    # LTV по каналам
    section("💎 LTV по маркетинговым каналам")
    ltv_channel = df.groupby("utm_source").agg(
        revenue=("total","sum"),
        customers=("customer","nunique") if "customer" in df.columns else ("total","count"),
        orders=("total","count"),
    ).reset_index()
    ltv_channel["ltv"] = ltv_channel["revenue"] / ltv_channel["customers"].replace(0,np.nan).fillna(1)
    ltv_channel["avg_check"] = ltv_channel["revenue"] / ltv_channel["orders"].replace(0,np.nan).fillna(1)
    ltv_channel = ltv_channel.sort_values("ltv", ascending=False)

    fig3 = px.bar(ltv_channel, x="utm_source", y="ltv",
                  title="👥 LTV клиента по источнику трафика",
                  template="plotly_dark", color="ltv",
                  color_continuous_scale="Viridis")
    st.plotly_chart(fig3, use_container_width=True)

    # Retention / повторные покупки
    section("🔄 Анализ повторных покупок")
    if "customer" in df.columns:
        cust_orders = df.groupby("customer")["date"].agg(["min","max","count"]).reset_index()
        cust_orders.columns = ["customer","first_order","last_order","orders"]
        cust_orders["days_between"] = (cust_orders["last_order"] - cust_orders["first_order"]).dt.days

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Одноразовые клиенты",  f"{(cust_orders['orders']==1).sum():,}")
        col_b.metric("Повторные клиенты",    f"{(cust_orders['orders']>1).sum():,}")
        col_c.metric("Среднее заказов/клиент",f"{cust_orders['orders'].mean():.1f}")

        fig4 = px.histogram(cust_orders, x="orders", nbins=20,
                            title="📊 Распределение заказов на клиента",
                            template="plotly_dark", color_discrete_sequence=["#4299e1"])
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 6. GROWTH OPPORTUNITIES TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_growth_opportunities(df: pd.DataFrame):
    section("🚀 Growth Opportunities")

    total_rev = df["total"].sum()

    # Города с потенциалом
    if "city" in df.columns:
        city_stats = df.groupby("city").agg(
            revenue=("total","sum"),
            orders=("total","count"),
            customers=("customer","nunique") if "customer" in df.columns else ("total","count"),
        ).reset_index()
        city_stats["share"] = city_stats["revenue"] / total_rev * 100
        city_stats["avg_check"] = city_stats["revenue"] / city_stats["orders"].replace(0,np.nan).fillna(1)
        avg_check_global = df["total"].mean()

        # Города с высоким средним чеком но малой долей = потенциал роста
        city_stats["growth_potential"] = (city_stats["avg_check"] / avg_check_global) / (city_stats["share"] + 1)
        top_potential_cities = city_stats.sort_values("growth_potential", ascending=False).head(10)

        section("🏙️ Города с высоким потенциалом роста")
        insight("Города с высоким средним чеком, но низкой долей в выручке — приоритет для масштабирования")
        fig = px.scatter(city_stats, x="share", y="avg_check",
                         size="revenue", color="growth_potential",
                         text="city",
                         title="🏙️ Матрица: Доля рынка vs Средний чек",
                         template="plotly_dark",
                         color_continuous_scale="YlOrRd",
                         labels={"share":"Доля %","avg_check":"Средний чек"})
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    # Анализ каналов: недо- и переинвестированные
    section("📡 Анализ маркетинговых каналов")
    channel_stats = df.groupby("utm_source").agg(
        revenue=("total","sum"),
        orders=("total","count"),
        customers=("customer","nunique") if "customer" in df.columns else ("total","count"),
    ).reset_index()
    channel_stats["ltv"] = channel_stats["revenue"] / channel_stats["customers"].replace(0,np.nan).fillna(1)
    channel_stats["share"] = channel_stats["revenue"] / total_rev * 100

    col1, col2 = st.columns(2)
    # Недоинвестированные: высокий LTV, низкая доля
    underinvested = channel_stats[
        (channel_stats["ltv"] > channel_stats["ltv"].median()) &
        (channel_stats["share"] < channel_stats["share"].median())
    ]
    overinvested = channel_stats[
        (channel_stats["ltv"] < channel_stats["ltv"].median()) &
        (channel_stats["share"] > channel_stats["share"].median())
    ]

    with col1:
        st.markdown("#### 🟢 Недоинвестированные каналы")
        insight("Высокий LTV, низкая доля — увеличить инвестиции")
        if not underinvested.empty:
            for _, row in underinvested.iterrows():
                st.markdown(f"- **{row['utm_source']}**: LTV ₴{fmt_money(row['ltv'])}, доля {row['share']:.1f}%")

    with col2:
        st.markdown("#### 🔴 Переинвестированные каналы")
        insight("Низкий LTV, высокая доля — пересмотреть бюджет")
        if not overinvested.empty:
            for _, row in overinvested.iterrows():
                st.markdown(f"- **{row['utm_source']}**: LTV ₴{fmt_money(row['ltv'])}, доля {row['share']:.1f}%")

    # Рейтинг кампаний
    section("📢 Рейтинг рекламных кампаний")
    camp_stats = df.groupby("utm_campaign").agg(
        revenue=("total","sum"),
        orders=("total","count"),
        profit=("profit","sum") if "profit" in df.columns else ("total","count"),
    ).reset_index()
    camp_stats["avg_check"] = camp_stats["revenue"] / camp_stats["orders"].replace(0,np.nan).fillna(1)
    camp_stats["score"] = (camp_stats["revenue"].rank(pct=True) * 0.5 +
                           camp_stats["avg_check"].rank(pct=True) * 0.3 +
                           camp_stats["orders"].rank(pct=True) * 0.2) * 100
    camp_stats = camp_stats.sort_values("score", ascending=False)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🏆 Кампании для масштабирования")
        for _, row in camp_stats.head(5).iterrows():
            potential_uplift = row["revenue"] * 0.3
            st.markdown(
                f'<div class="insight-box">🚀 <b>{row["utm_campaign"]}</b><br>'
                f'Выручка: ₴{fmt_money(row["revenue"])} | Score: {row["score"]:.0f}/100<br>'
                f'Потенциал прироста: +₴{fmt_money(potential_uplift)}</div>',
                unsafe_allow_html=True
            )

    with col4:
        st.markdown("#### ⛔ Кампании для остановки")
        for _, row in camp_stats.tail(5).iterrows():
            st.markdown(
                f'<div class="insight-box" style="border-left-color:#fc8181">⛔ <b>{row["utm_campaign"]}</b><br>'
                f'Выручка: ₴{fmt_money(row["revenue"])} | Score: {row["score"]:.0f}/100</div>',
                unsafe_allow_html=True
            )

    # Top Growth Opportunities Table
    section("🏆 Top Growth Opportunities")
    opps = []
    if "city" in df.columns:
        top_city = city_stats.sort_values("growth_potential", ascending=False).head(3)
        for _, row in top_city.iterrows():
            opps.append({
                "Возможность": f"Масштабирование: {row['city']}",
                "Тип": "Географический",
                "Потенциал прироста выручки":
                    f"₴{fmt_money(row['revenue']*0.4)}",
                "Приоритет": "Высокий" if row["growth_potential"] > 1 else "Средний",
            })

    for _, row in camp_stats.head(3).iterrows():
        opps.append({
            "Возможность": f"Масштаб. кампании: {row['utm_campaign']}",
            "Тип": "Маркетинговый",
            "Потенциал прироста выручки": f"₴{fmt_money(row['revenue']*0.3)}",
            "Приоритет": "Высокий",
        })

    if opps:
        st.dataframe(pd.DataFrame(opps), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 9. AI DIRECTOR TAB (Google Gemini)
# ─────────────────────────────────────────────────────────────────────────────
def render_ai_director(df: pd.DataFrame):
    section("🤖 AI Директор — Анализ уровня Совета Директоров")

    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="Вставьте ваш Gemini API ключ...",
        help="Получите ключ на https://aistudio.google.com/",
    )

    if not api_key:
        st.info("Введите Gemini API ключ для генерации отчёта")
        st.markdown("""
        **Этот блок генерирует детальный отчёт уровня совета директоров:**
        - Executive Summary
        - Финансовое состояние  
        - Прогноз выручки и прибыли
        - Основные риски и потери прибыли
        - Неиспользованные возможности
        - Рекомендации по масштабированию
        - Планы действий: 30 / 90 / 180 дней
        """)
        return

    kpis   = calc_financial_kpis(df)
    growth = calc_growth_rates(df)
    risks  = calc_risk_metrics(df)
    mkt    = calc_marketing_metrics(df)

    top_city = df.groupby("city")["total"].sum().nlargest(5).to_dict() if "city" in df.columns else {}
    top_camp = df.groupby("utm_campaign")["total"].sum().nlargest(5).to_dict()
    top_prod = df.groupby("product")["total"].sum().nlargest(5).to_dict() if "product" in df.columns else {}

    summary_json = {
        "financial_summary": {
            "revenue": round(kpis["revenue"], 2),
            "profit": round(kpis["profit"], 2),
            "margin_pct": round(kpis["margin_pct"], 2),
            "avg_check": round(kpis["avg_check"], 2),
            "orders": kpis["orders"],
            "customers": kpis["customers"],
            "retention_rate": round(kpis["retention_rate"], 2),
            "ltv": round(kpis["ltv"], 2),
            "arpu": round(kpis["arpu"], 2),
        },
        "sales_summary": {
            "top_products": top_prod,
            "top_cities": top_city,
        },
        "marketing_summary": {
            "by_source": mkt["by_source"].to_dict("records") if "by_source" in mkt else [],
            "top_campaigns": top_camp,
        },
        "growth_summary": {
            "mom_growth": round(growth["mom"], 2),
            "qoq_growth": round(growth["qoq"], 2),
            "yoy_growth": round(growth["yoy"], 2),
            "cagr": round(growth["cagr"], 2),
        },
        "risks_summary": {
            "revenue_volatility": round(risks.get("revenue_volatility", 0), 2),
            "sales_decline_risk": risks.get("sales_decline_risk", "N/A"),
            "top_channel_share": round(risks.get("top_channel_share", 0), 2),
            "top_channel": risks.get("top_channel", "N/A"),
            "customer_concentration": risks.get("customer_concentration", "N/A"),
        },
    }

    if st.button("🚀 Сгенерировать AI отчёт", type="primary"):
        with st.spinner("🤖 AI анализирует бизнес..."):
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")

                system_prompt = """Ты одновременно CFO, CMO и CEO компании.
Проанализируй финансовые показатели, продажи, маркетинг и прогноз.
Подготовь детальный отчет уровня совета директоров.

Разделы:
1. Executive Summary
2. Финансовое состояние (детально)
3. Прогноз выручки (3-6-12 мес)
4. Прогноз прибыли (3-6-12 мес)
5. Основные риски (топ-5 с оценкой вероятности и влияния)
6. Потери прибыли (что теряем прямо сейчас)
7. Неиспользованные возможности
8. Рекомендации по масштабированию
9. Рекомендации по маркетингу
10. Рекомендации по продажам
11. План действий на 30 дней
12. План действий на 90 дней
13. План действий на 180 дней

Для каждой рекомендации укажи:
- Влияние на выручку (%)
- Влияние на прибыль (%)
- Сложность внедрения (1-10)
- Срок окупаемости

Данные бизнеса:
"""

                full_prompt = system_prompt + json.dumps(summary_json, ensure_ascii=False, indent=2)
                response = model.generate_content(full_prompt)
                report_text = response.text

                st.markdown("---")
                st.markdown(report_text)

                st.download_button(
                    "📥 Скачать отчёт",
                    data=report_text,
                    file_name=f"ai_director_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                )

            except ImportError:
                st.error("❌ Установите: pip install google-generativeai")
            except Exception as e:
                st.error(f"❌ Ошибка API: {e}")
                logger.error(f"Gemini API error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# БЛОК 10. GROWTH PLAN TAB
# ─────────────────────────────────────────────────────────────────────────────
def render_growth_plan(df: pd.DataFrame):
    section("📋 План роста бизнеса")

    api_key = st.text_input(
        "🔑 Google Gemini API Key (Growth Plan)",
        type="password",
        key="gemini_key_growth",
        placeholder="Вставьте ваш Gemini API ключ...",
    )

    kpis   = calc_financial_kpis(df)
    growth = calc_growth_rates(df)

    # Автоматические расчёты плана роста (без AI)
    section("📊 Базовый план роста (авторасчёт)")

    monthly_rev = df.groupby("yearmonth")["total"].sum()
    avg_monthly = monthly_rev.mean() if len(monthly_rev) > 0 else kpis["revenue"] / 12
    growth_rate = max(growth["cagr"] / 100, 0.02)  # минимум 2% роста в месяц

    periods = {
        "3 месяца": 3,
        "6 месяцев": 6,
        "12 месяцев": 12,
    }

    cols = st.columns(3)
    for idx, (period_label, months) in enumerate(periods.items()):
        expected_rev = avg_monthly * months * (1 + growth_rate) ** months
        expected_profit = expected_rev * kpis["margin_pct"] / 100

        with cols[idx]:
            st.markdown(f"### 🗓️ {period_label}")
            st.metric("Ожид. выручка", f"₴{fmt_money(expected_rev)}")
            st.metric("Ожид. прибыль", f"₴{fmt_money(expected_profit)}")

    if not api_key:
        st.info("Введите Gemini API ключ для генерации детального плана роста с AI")
        return

    if st.button("🚀 Сгенерировать план роста с AI", type="primary"):
        with st.spinner("🤖 AI формирует план роста..."):
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")

                prompt = f"""Ты CEO и CMO компании. На основе данных сформируй детальный план роста бизнеса.

Данные:
- Выручка: ₴{fmt_money(kpis['revenue'])}
- Прибыль: ₴{fmt_money(kpis['profit'])}
- Маржа: {kpis['margin_pct']:.1f}%
- Клиентов: {kpis['customers']:,}
- Retention: {kpis['retention_rate']:.1f}%
- MoM Growth: {growth['mom']:+.1f}%
- CAGR: {growth['cagr']:.1f}%

Для каждого периода (3, 6, 12 месяцев) укажи:
1. Ожидаемую выручку
2. Ожидаемую прибыль
3. Необходимые действия (топ-5)
4. Маркетинговые активности
5. Рекламные гипотезы для тестирования
6. Рекомендуемый маркетинговый бюджет
7. Ключевые KPI
8. Прогнозируемый ROI маркетинга

Будь конкретным, с цифрами и сроками."""

                response = model.generate_content(prompt)
                st.markdown("---")
                st.markdown(response.text)

                st.download_button(
                    "📥 Скачать план роста",
                    data=response.text,
                    file_name=f"growth_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                )

            except ImportError:
                st.error("❌ Установите: pip install google-generativeai")
            except Exception as e:
                st.error(f"❌ Ошибка API: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px 0;">
        <h1 style="font-size:32px;font-weight:800;color:#e2e8f0;margin:0;">
            📊 Business Analytics Platform
        </h1>
        <p style="color:#718096;font-size:15px;margin-top:6px;">
            CFO · CMO · BI Architect · AI Director
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar upload
    st.sidebar.markdown("## 📂 Загрузка данных")
    uploaded_file = st.sidebar.file_uploader(
        "Excel или CSV файл",
        type=["csv", "xlsx", "xls"],
        help="Поддерживаются CSV и Excel файлы с данными о продажах"
    )

    if uploaded_file is None:
        # Landing state
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
            <div style="font-size:72px;margin-bottom:20px;">📈</div>
            <h2 style="color:#e2e8f0;font-size:28px;margin-bottom:12px;">
                Загрузите данные для начала анализа
            </h2>
            <p style="color:#718096;font-size:16px;max-width:600px;margin:0 auto;">
                Поддерживаются Excel (.xlsx, .xls) и CSV файлы с данными о продажах.<br>
                Система автоматически распознаёт колонки на русском, украинском и английском языках.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Поддерживаемые поля данных"):
            fields = [
                ("Дата заказа / Date", "Обязательное"),
                ("Статус заказа", "Опционально"),
                ("Номер заказа", "Опционально"),
                ("Клиент", "Опционально"),
                ("Город / Регион / Страна", "Опционально"),
                ("Товар / Категория", "Опционально"),
                ("Количество", "Опционально"),
                ("Себестоимость", "Опционально"),
                ("Цена продажи", "Рекомендуется"),
                ("Итого (выручка)", "Обязательное"),
                ("Маржа / Прибыль", "Опционально"),
                ("UTM метки (source, medium, campaign...)", "Опционально"),
            ]
            df_fields = pd.DataFrame(fields, columns=["Поле", "Важность"])
            st.dataframe(df_fields, use_container_width=True)
        return

    # Load & prepare data
    df_raw = load_data(uploaded_file)
    if df_raw.empty:
        return

    try:
        df_clean, col_map = prepare_data(df_raw)
    except ValueError as e:
        st.error(f"❌ Ошибка данных: {e}")
        st.info("Проверьте, что файл содержит колонки с датой и суммой продаж.")
        return
    except Exception as e:
        st.error(f"❌ Неожиданная ошибка: {e}")
        logger.exception(e)
        return

    if len(df_clean) == 0:
        st.error("❌ После очистки данных не осталось записей. Проверьте файл.")
        return

    # Sidebar info & filters
    st.sidebar.markdown(f"""
    **✅ Данных загружено:**
    - Строк: {len(df_clean):,}
    - Период: {df_clean['date'].min().strftime('%d.%m.%Y')} – {df_clean['date'].max().strftime('%d.%m.%Y')}
    - Колонок: {len(col_map)}
    """)

    df = apply_sidebar_filters(df_clean)

    if len(df) == 0:
        st.warning("⚠️ Нет данных для выбранных фильтров")
        return

    # Main tabs
    tabs = st.tabs([
        "🏠 Dashboard",
        "💰 Финансы",
        "🔮 Прогноз",
        "🗺️ Геоаналитика",
        "📣 Маркетинг",
        "🧠 Sales Intelligence",
        "🚀 Growth Opportunities",
        "🤖 AI Director",
        "📋 Growth Plan",
    ])

    with tabs[0]: render_dashboard(df)
    with tabs[1]: render_finance(df)
    with tabs[2]: render_forecast(df)
    with tabs[3]: render_geo(df)
    with tabs[4]: render_marketing(df)
    with tabs[5]: render_sales_intelligence(df)
    with tabs[6]: render_growth_opportunities(df)
    with tabs[7]: render_ai_director(df)
    with tabs[8]: render_growth_plan(df)

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;color:#4a5568;font-size:12px;">'
        'Business Analytics Platform · Powered by Streamlit + Plotly + Prophet · '
        f'Данных в работе: {len(df):,} записей</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
