"""
Sales & Inventory Dashboard — SUPR-inspired styling, with predictive insights
------------------------------------------------------------------------------
Install (Windows, using 'python' once installed & on PATH):
    python -m pip install streamlit plotly pandas numpy

Run with:
    python -m streamlit run dashboard.py

(On Mac/Linux, replace 'python -m' with 'python3 -m' or just use 'streamlit run dashboard.py')

This is a MOCKUP using sample data so you can see the visual direction.
Once your real data source is connected (DB / API / export), swap out
the `load_sales_data()` and `load_inventory_data()` functions below —
everything else (styling, layout, charts, forecasting) stays the same.

FORECASTING MODEL (kept dependency-light — NumPy/Pandas only):
  For each product: fit a linear trend on daily units sold, then compute a
  day-of-week seasonal adjustment (average residual per weekday). The forecast
  = trend + seasonal offset, projected forward. This is a simple but genuinely
  useful baseline (trend + seasonality decomposition). For production-grade
  forecasting later, swap `forecast_units()` for Prophet, ARIMA/SARIMA, or a
  gradient-boosted model (XGBoost/LightGBM) — the rest of the dashboard code
  doesn't need to change since it just consumes a (date, predicted_units)
  dataframe.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sales & Inventory Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# BRAND-INSPIRED THEME (soft cream / blush / sage — SUPR-style wellness look)
# ----------------------------------------------------------------------
PRIMARY = "#C9A28A"      # warm blush/terracotta accent
SECONDARY = "#8FA88A"    # soft sage green
ALERT = "#B4543A"        # for critical stockout warnings
BG = "#FDF8F3"           # cream background
CARD_BG = "#FFFFFF"
TEXT_DARK = "#2E2A26"
TEXT_MUTED = "#8A8178"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
        color: {TEXT_DARK};
    }}

    .stApp {{
        background-color: {BG};
    }}

    .dashboard-header {{
        padding: 1.5rem 0 0.5rem 0;
    }}
    .dashboard-header h1 {{
        font-weight: 600;
        color: {TEXT_DARK};
        margin-bottom: 0.1rem;
    }}
    .dashboard-header p {{
        color: {TEXT_MUTED};
        font-size: 0.95rem;
    }}

    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.03);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
        font-weight: 500;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT_DARK} !important;
        font-weight: 600;
    }}

    .section-card {{
        background-color: {CARD_BG};
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.2rem;
        border: 1px solid rgba(0,0,0,0.03);
    }}
    .section-card h3 {{
        margin-top: 0;
        font-weight: 600;
        color: {TEXT_DARK};
    }}
    .section-card p.subtle {{
        color: {TEXT_MUTED};
        font-size: 0.85rem;
        margin-top: -0.6rem;
    }}

    section[data-testid="stSidebar"] {{
        background-color: #FBF3EC;
    }}

    .stDataFrame {{
        border-radius: 14px;
        overflow: hidden;
    }}

    .badge-low, .badge-critical {{
        background-color: #F6DCD1;
        color: #B4543A;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 500;
    }}
    .badge-warning {{
        background-color: #FBEBC9;
        color: #9A7717;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 500;
    }}
    .badge-ok {{
        background-color: #E4EDE1;
        color: #5A7A54;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 500;
    }}

    .insight-banner {{
        background: linear-gradient(135deg, #F6EFE8, #EFE6DC);
        border-radius: 16px;
        padding: 1rem 1.4rem;
        margin-bottom: 1rem;
        border-left: 4px solid {PRIMARY};
    }}
    .insight-banner b {{ color: {TEXT_DARK}; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# DATA LOADING  (⚠️ REPLACE THESE WITH YOUR REAL DATA SOURCE ⚠️)
# ----------------------------------------------------------------------
@st.cache_data(ttl=60)  # auto re-fetches every 60 seconds — tune as needed
def load_sales_data():
    """
    Replace this with a real query, e.g.:
        engine = sqlalchemy.create_engine("postgresql://user:pw@host/db")
        return pd.read_sql("SELECT * FROM sales", engine)

    Sample data below includes a trend + weekend seasonality + one festive
    spike so the forecasting section has something meaningful to detect.
    """
    rng = np.random.default_rng(42)
    n_days = 120
    dates = pd.date_range(end=datetime.today(), periods=n_days)
    products = ["Choco Fudge Protein", "Malai Kulfi Protein", "SUPR Tumbler",
                "SUPR Shaker", "Protein Cookies", "Special Bundle"]
    prices = {"Choco Fudge Protein": 2986, "Malai Kulfi Protein": 2886,
              "SUPR Tumbler": 1499, "SUPR Shaker": 899,
              "Protein Cookies": 619, "Special Bundle": 3186}

    # a synthetic "festive" spike window to demonstrate peak detection
    spike_start = n_days - 18
    spike_end = n_days - 10

    rows = []
    for i, d in enumerate(dates):
        weekday = d.weekday()
        weekend_boost = 6 if weekday >= 5 else 0
        trend = i * 0.08                      # gentle upward trend over time
        spike = 18 if spike_start <= i <= spike_end else 0
        for p in products:
            base = rng.normal(15, 4)
            units = max(0, int(base + weekend_boost + trend + spike))
            rows.append({"date": d, "product": p, "units_sold": units,
                         "revenue": units * prices[p]})
    return pd.DataFrame(rows)

@st.cache_data(ttl=60)
def load_inventory_data():
    """
    Replace this with your real inventory query / API call.
    """
    data = [
        {"product": "Choco Fudge Protein", "stock_units": 220, "reorder_level": 50},
        {"product": "Malai Kulfi Protein", "stock_units": 480, "reorder_level": 50},
        {"product": "SUPR Tumbler", "stock_units": 40, "reorder_level": 20},
        {"product": "SUPR Shaker", "stock_units": 300, "reorder_level": 30},
        {"product": "Protein Cookies", "stock_units": 90, "reorder_level": 40},
        {"product": "Special Bundle", "stock_units": 260, "reorder_level": 25},
    ]
    return pd.DataFrame(data)

sales_df = load_sales_data()
inv_df = load_inventory_data()

# ----------------------------------------------------------------------
# FORECASTING HELPERS
# ----------------------------------------------------------------------
def forecast_units(daily_series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    daily_series: columns ['date', 'units_sold'], one row per day, sorted.
    Returns a dataframe of ['date', 'predicted_units'] for the next `horizon` days.

    Model: linear trend (NumPy polyfit) + day-of-week seasonal offset
    (average residual per weekday). Simple, transparent, no extra ML deps.
    """
    d = daily_series.copy().sort_values("date").reset_index(drop=True)
    d["t"] = np.arange(len(d))
    d["dow"] = d["date"].dt.dayofweek

    if len(d) < 7 or d["units_sold"].sum() == 0:
        # not enough signal — flat forecast at recent average
        avg = d["units_sold"].tail(7).mean() if len(d) else 0
        future_dates = pd.date_range(d["date"].max() + timedelta(days=1), periods=horizon) \
            if len(d) else pd.date_range(datetime.today(), periods=horizon)
        return pd.DataFrame({"date": future_dates, "predicted_units": [max(avg, 0)] * horizon})

    coeffs = np.polyfit(d["t"], d["units_sold"], 1)
    trend_fn = np.poly1d(coeffs)
    d["trend_pred"] = trend_fn(d["t"])
    d["residual"] = d["units_sold"] - d["trend_pred"]
    seasonal = d.groupby("dow")["residual"].mean()

    last_t = d["t"].max()
    last_date = d["date"].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
    future_t = np.arange(last_t + 1, last_t + 1 + horizon)
    future_trend = trend_fn(future_t)
    future_dow = future_dates.dayofweek
    future_seasonal = np.array([seasonal.get(dw, 0.0) for dw in future_dow])
    predicted = np.clip(future_trend + future_seasonal, 0, None)

    return pd.DataFrame({"date": future_dates, "predicted_units": predicted})


def estimate_stockout(current_stock: float, forecast_df: pd.DataFrame):
    """
    Walks forward through the forecast, cumulatively subtracting predicted
    daily sales from current stock. Returns (stockout_date_or_None, days_until).
    None means stock lasts beyond the forecast horizon.
    """
    remaining = current_stock
    for _, row in forecast_df.iterrows():
        remaining -= row["predicted_units"]
        if remaining <= 0:
            days_until = (row["date"] - pd.Timestamp.today().normalize()).days
            return row["date"], max(days_until, 0)
    return None, None


@st.cache_data(ttl=60)
def build_forecasts(sales_df: pd.DataFrame, inv_df: pd.DataFrame, horizon: int = 45):
    """Builds per-product forecasts + stockout estimates + avg price lookup."""
    avg_price = (
        sales_df.groupby("product")
        .apply(lambda g: g["revenue"].sum() / max(g["units_sold"].sum(), 1))
        .to_dict()
    )

    forecasts = {}
    stockouts = []
    for product in sales_df["product"].unique():
        daily = sales_df[sales_df["product"] == product].groupby("date", as_index=False)["units_sold"].sum()
        fc = forecast_units(daily, horizon)
        fc["product"] = product
        fc["predicted_revenue"] = fc["predicted_units"] * avg_price.get(product, 0)
        forecasts[product] = fc

        stock_row = inv_df[inv_df["product"] == product]
        current_stock = stock_row["stock_units"].iloc[0] if not stock_row.empty else 0
        stockout_date, days_until = estimate_stockout(current_stock, fc)
        avg_daily_rate = fc["predicted_units"].head(14).mean()
        stockouts.append({
            "product": product,
            "current_stock": current_stock,
            "avg_daily_sales_forecast": round(avg_daily_rate, 1),
            "stockout_date": stockout_date,
            "days_until_stockout": days_until,
        })

    all_forecasts = pd.concat(forecasts.values(), ignore_index=True)
    stockout_df = pd.DataFrame(stockouts)
    return all_forecasts, stockout_df, avg_price


FORECAST_HORIZON_DEFAULT = 45
all_forecasts, stockout_df, avg_price_lookup = build_forecasts(sales_df, inv_df, FORECAST_HORIZON_DEFAULT)

# Predicted peak sales period: best 7-day rolling window of total predicted revenue
peak_totals = all_forecasts.groupby("date", as_index=False)["predicted_revenue"].sum().sort_values("date")
peak_totals["rolling_7d"] = peak_totals["predicted_revenue"].rolling(7).sum()
if peak_totals["rolling_7d"].notna().any():
    peak_end_idx = peak_totals["rolling_7d"].idxmax()
    peak_end_date = peak_totals.loc[peak_end_idx, "date"]
    peak_start_date = peak_end_date - timedelta(days=6)
    peak_window_revenue = peak_totals.loc[peak_end_idx, "rolling_7d"]
else:
    peak_start_date = peak_end_date = None
    peak_window_revenue = 0

# ----------------------------------------------------------------------
# SIDEBAR — FILTERS
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 Filters")
    date_range = st.date_input(
        "Date range (historical view)",
        value=(sales_df["date"].min().date(), sales_df["date"].max().date()),
    )
    selected_products = st.multiselect(
        "Products",
        options=sales_df["product"].unique(),
        default=sales_df["product"].unique(),
    )
    st.markdown("---")
    st.markdown("### 🔮 Forecast settings")
    horizon = st.slider("Forecast horizon (days)", min_value=14, max_value=90,
                         value=FORECAST_HORIZON_DEFAULT, step=1)
    forecast_product = st.selectbox("Forecast product (chart below)",
                                     options=sales_df["product"].unique())
    st.markdown("---")
    st.caption("Dashboard auto-refreshes every 60s once connected to a live source. "
               "Forecasts use a trend + weekly-seasonality model on the historical data.")

# Rebuild forecasts if horizon changed from default
if horizon != FORECAST_HORIZON_DEFAULT:
    all_forecasts, stockout_df, avg_price_lookup = build_forecasts(sales_df, inv_df, horizon)
    peak_totals = all_forecasts.groupby("date", as_index=False)["predicted_revenue"].sum().sort_values("date")
    peak_totals["rolling_7d"] = peak_totals["predicted_revenue"].rolling(7).sum()
    if peak_totals["rolling_7d"].notna().any():
        peak_end_idx = peak_totals["rolling_7d"].idxmax()
        peak_end_date = peak_totals.loc[peak_end_idx, "date"]
        peak_start_date = peak_end_date - timedelta(days=6)
        peak_window_revenue = peak_totals.loc[peak_end_idx, "rolling_7d"]

# Filter historical data
mask = (
    (sales_df["date"].dt.date >= date_range[0])
    & (sales_df["date"].dt.date <= date_range[1])
    & (sales_df["product"].isin(selected_products))
)
filtered = sales_df[mask]

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.markdown(f"""
<div class="dashboard-header">
    <h1>Sales & Inventory Overview</h1>
    <p>Live snapshot + predictive insights across your catalog · Last updated {datetime.now().strftime('%d %b %Y, %I:%M %p')}</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TOP-LEVEL PREDICTIVE INSIGHT BANNER
# ----------------------------------------------------------------------
critical_products = stockout_df[stockout_df["days_until_stockout"].notna() &
                                 (stockout_df["days_until_stockout"] <= 14)]

banner_col1, banner_col2 = st.columns(2)
with banner_col1:
    if peak_start_date is not None:
        st.markdown(f"""
        <div class="insight-banner">
            📈 <b>Predicted peak sales period:</b> {peak_start_date.strftime('%d %b')} – {peak_end_date.strftime('%d %b')}
            &nbsp;·&nbsp; ~₹{peak_window_revenue:,.0f} projected across catalog that week
        </div>
        """, unsafe_allow_html=True)
with banner_col2:
    if len(critical_products) > 0:
        names = ", ".join(critical_products["product"].tolist())
        st.markdown(f"""
        <div class="insight-banner" style="border-left-color:{ALERT};">
            ⚠️ <b>{len(critical_products)} product(s) may stock out within 14 days:</b> {names}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-banner" style="border-left-color:{SECONDARY};">
            ✅ <b>No products predicted to stock out in the next 14 days.</b>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# KPI ROW
# ----------------------------------------------------------------------
total_revenue = filtered["revenue"].sum()
total_units = filtered["units_sold"].sum()
avg_order_value = total_revenue / max(total_units, 1)
low_stock_count = (inv_df["stock_units"] < inv_df["reorder_level"]).sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
c2.metric("Units Sold", f"{total_units:,}")
c3.metric("Avg. Unit Value", f"₹{avg_order_value:,.0f}")
c4.metric("Low Stock Alerts", f"{low_stock_count}", delta_color="inverse")

st.write("")

# ----------------------------------------------------------------------
# REVENUE TREND (historical)
# ----------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### Revenue Trend")
daily = filtered.groupby("date", as_index=False)["revenue"].sum()
fig = px.area(daily, x="date", y="revenue")
fig.update_traces(line_color=PRIMARY, fillcolor="rgba(201,162,138,0.15)")
fig.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    font_family="Poppins", margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title=None, xaxis_title=None,
)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# PREDICTIVE: SALES FORECAST CHART (historical + forecast for selected product)
# ----------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(f"### 🔮 Sales Forecast — {forecast_product}")
st.markdown('<p class="subtle">Trend + weekly-seasonality model. Shaded region is the forecast.</p>', unsafe_allow_html=True)

hist_p = sales_df[sales_df["product"] == forecast_product].groupby("date", as_index=False)["units_sold"].sum()
fc_p = all_forecasts[all_forecasts["product"] == forecast_product][["date", "predicted_units"]]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=hist_p["date"], y=hist_p["units_sold"], mode="lines",
                           name="Actual", line=dict(color=TEXT_DARK, width=2)))
fig3.add_trace(go.Scatter(x=fc_p["date"], y=fc_p["predicted_units"], mode="lines",
                           name="Forecast", line=dict(color=PRIMARY, width=2, dash="dash")))
# highlight peak forecast day for this product
if not fc_p.empty:
    peak_row = fc_p.loc[fc_p["predicted_units"].idxmax()]
    fig3.add_trace(go.Scatter(x=[peak_row["date"]], y=[peak_row["predicted_units"]],
                               mode="markers+text", name="Peak day",
                               marker=dict(color=ALERT, size=10),
                               text=["Peak"], textposition="top center"))
fig3.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    font_family="Poppins", margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="Units", xaxis_title=None,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig3, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TWO-COLUMN: PRODUCT MIX + INVENTORY STATUS
# ----------------------------------------------------------------------
col1, col2 = st.columns([1.1, 1])

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Revenue by Product")
    by_product = filtered.groupby("product", as_index=False)["revenue"].sum().sort_values("revenue")
    fig2 = px.bar(by_product, x="revenue", y="product", orientation="h",
                  color_discrete_sequence=[SECONDARY])
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Poppins", margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title=None, xaxis_title=None,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Inventory Status")
    for _, row in inv_df.iterrows():
        is_low = row["stock_units"] < row["reorder_level"]
        badge_class = "badge-low" if is_low else "badge-ok"
        badge_text = "Reorder" if is_low else "In Stock"
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid rgba(0,0,0,0.05);">
            <span>{row['product']}</span>
            <span style="display:flex; align-items:center; gap:8px;">
                <b>{row['stock_units']} units</b>
                <span class="{badge_class}">{badge_text}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# PREDICTIVE: STOCK-OUT RISK TABLE
# ----------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### ⚠️ Stock-Out Predictions")
st.markdown(f'<p class="subtle">Based on forecasted demand over the next {horizon} days.</p>', unsafe_allow_html=True)

display_stockout = stockout_df.copy().sort_values(
    by="days_until_stockout", na_position="last"
)

def risk_badge(days):
    if days is None or (isinstance(days, float) and np.isnan(days)):
        return '<span class="badge-ok">Stock sufficient</span>'
    elif days <= 7:
        return '<span class="badge-critical">Critical</span>'
    elif days <= 21:
        return '<span class="badge-warning">Watch</span>'
    else:
        return '<span class="badge-ok">Healthy</span>'

for _, row in display_stockout.iterrows():
    date_str = row["stockout_date"].strftime("%d %b %Y") if pd.notna(row["stockout_date"]) else f"Beyond {horizon}-day forecast"
    days_str = f"{int(row['days_until_stockout'])} days" if pd.notna(row["days_until_stockout"]) else "—"
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid rgba(0,0,0,0.05);">
        <span style="flex:1.4;">{row['product']}</span>
        <span style="flex:1; color:{TEXT_MUTED}; font-size:0.85rem;">{row['current_stock']} in stock · ~{row['avg_daily_sales_forecast']}/day</span>
        <span style="flex:1; color:{TEXT_MUTED}; font-size:0.85rem;">{date_str} ({days_str})</span>
        <span style="flex:0.6; text-align:right;">{risk_badge(row['days_until_stockout'])}</span>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# RAW DATA TABLE
# ----------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("### Detailed Sales Log")
st.dataframe(filtered.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)