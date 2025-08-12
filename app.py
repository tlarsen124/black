import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm

st.set_page_config(page_title="Black-Scholes Dashboard", layout="wide")

def black_scholes_price(S, K, r, q, sigma, T, option_type="call"):
    S = np.array(S, dtype=float)
    sigma = np.array(sigma, dtype=float)

    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    T = np.maximum(T, 0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if np.isscalar(T):
        if T == 0:
            if option_type == "call":
                return np.maximum(S - K, 0.0)
            else:
                return np.maximum(K - S, 0.0)

    disc_factor = np.exp(-r * T)
    div_factor = np.exp(-q * T)

    if option_type == "call":
        price = S * div_factor * norm.cdf(d1) - K * disc_factor * norm.cdf(d2)
    else:
        price = K * disc_factor * norm.cdf(-d2) - S * div_factor * norm.cdf(-d1)

    if isinstance(T, np.ndarray):
        zeroT = (T == 0)
        if np.any(zeroT):
            payoff = np.where(option_type == "call", np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
            price = np.where(zeroT, payoff, price)

    return price

# -------------------------
# Sidebar: Model inputs & inspect values combined
# -------------------------
st.sidebar.header("Model inputs")

K = st.sidebar.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
r = st.sidebar.number_input("Risk-free rate (annual) r", value=0.02, format="%.4f")
q = st.sidebar.number_input("Dividend yield (annual) q", value=0.00, format="%.4f")
T = st.sidebar.number_input("Time to expiry (years) T", value=0.5, min_value=0.0, step=0.01, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.subheader("Inspect values")
S_inspect = st.sidebar.number_input("Underlying asset price (S)", value=100.0, format="%.4f")
sigma_inspect = st.sidebar.number_input("Volatility (σ)", value=0.20, min_value=0.0001, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.write("Spot / Volatility grid")

S_min = st.sidebar.number_input("Spot min", value=50.0, format="%.2f")
S_max = st.sidebar.number_input("Spot max", value=150.0, format="%.2f")
S_steps = st.sidebar.slider("Spot steps", min_value=20, max_value=500, value=121, step=1)

sigma_min = st.sidebar.number_input("Volatility min (σ)", value=0.05, min_value=0.0001, format="%.4f")
sigma_max = st.sidebar.number_input("Volatility max (σ)", value=0.6, min_value=0.0001, format="%.4f")
sigma_steps = st.sidebar.slider("Vol steps", min_value=10, max_value=200, value=60, step=1)

st.sidebar.markdown("---")
st.sidebar.write("Visualization options")
log_spot = st.sidebar.checkbox("Plot spot on log scale (x-axis)", value=False)
show_contours = st.sidebar.checkbox("Show contour lines", value=True)
colormap = st.sidebar.selectbox("Color map (plotly)", options=["viridis","plasma","inferno","magma","cividis"], index=0)

# -------------------------
# Compute prices for heatmaps
# -------------------------
S_values = np.linspace(S_min, S_max, S_steps)
sigma_values = np.linspace(sigma_min, sigma_max, sigma_steps)
S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

# Call and Put price grids for heatmaps 1 & 2
call_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, T, option_type="call")
put_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, T, option_type="put")

# Summary table data
call_inspect_price = black_scholes_price(S_inspect, K, r, q, sigma_inspect, T, option_type="call")
put_inspect_price = black_scholes_price(S_inspect, K, r, q, sigma_inspect, T, option_type="put")

# -------------------------
# Price decay grid (heatmap 3)
# Price on y-axis, time (from T to 0) on x-axis
# -------------------------
price_min = max(0, S_min*0.8)
price_max = S_max*1.2
price_steps = 100
time_steps = 100

price_values = np.linspace(price_min, price_max, price_steps)
time_values = np.linspace(T, 0, time_steps)  # From input T down to 0

Price_grid_decay = black_scholes_price(price_values[:, None], K, r, q, sigma_inspect, time_values[None, :], option_type="call")  # Use call for decay plot

# -------------------------
# Display title and variables summary table
# -------------------------
st.title("Black-Scholes Option Pricing Dashboard")

summary_df = pd.DataFrame({
    "Variable": [
        "Strike (K)",
        "Risk-free rate (r)",
        "Dividend yield (q)",
        "Time to expiry (T)",
        "Underlying asset price (S)",
        "Volatility (σ)"
    ],
    "Value": [K, r, q, T, S_inspect, sigma_inspect]
})

# Format float values nicely for display
summary_df["Value"] = summary_df["Value"].map(lambda x: f"{x:.4f}")

# Show summary table
st.table(summary_df)

# Display Call and Put prices with big bold text, Put in red
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<p style="font-size:40px; font-weight:bold; color:green;">Call Price: {call_inspect_price:.4f}</p>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<p style="font-size:40px; font-weight:bold; color:red;">Put Price: {put_inspect_price:.4f}</p>', unsafe_allow_html=True)

# -------------------------
# Heatmaps row 1: Call and Put prices (spot vs volatility)
# -------------------------
col1, col2 = st.columns(2)

with col1:
    fig_call = px.imshow(
        call_price_grid,
        x=np.round(S_values, 4),
        y=np.round(sigma_values, 6),
        labels={"x": "Spot (S)", "y": "Volatility σ", "color": "Call Price"},
        origin="lower",
        aspect="auto",
        color_continuous_scale=colormap,
    )
    fig_call.update_layout(height=600, margin=dict(t=30, l=10, r=10, b=40))
    if log_spot:
        fig_call.update_xaxes(type="log")
    
    import plotly.graph_objects as go

if show_contours:
    contour = go.Contour(
        z=call_price_grid,
        x=np.round(S_values, 4),
        y=np.round(sigma_values, 6),
        contours=dict(
            coloring="none",
            showlabels=True,
            start=np.min(call_price_grid),
            end=np.max(call_price_grid),
            size=(np.max(call_price_grid) - np.min(call_price_grid)) / 8,
        ),
        line_width=1,
        showscale=False,
    )
    fig_call.add_trace(contour)
    # Flip y axis so volatility increases upwards (lower volatility at bottom)
    fig_call.update_yaxes(autorange="reversed")

    # Show price numbers inside heatmap cells
    fig_call.update_traces(text=np.round(call_price_grid,2), texttemplate="%{text}", textfont={"size":8}, hoverinfo='none')

    st.plotly_chart(fig_call, use_container_width=True)

with col2:
    fig_put = px.imshow(
        put_price_grid,
        x=np.round(S_values, 4),
        y=np.round(sigma_values, 6),
        labels={"x": "Spot (S)", "y": "Volatility σ", "color": "Put Price"},
        origin="lower",
        aspect="auto",
        color_continuous_scale=colormap,
    )
    fig_put.update_layout(height=600, margin=dict(t=30, l=10, r=10, b=40))
    if log_spot:
        fig_put.update_xaxes(type="log")
    if show_contours:
        fig_contour = px.contour(
            z=put_price_grid,
            x=np.round(S_values, 4),
            y=np.round(sigma_values, 6),
            labels={"x": "Spot (S)", "y": "Volatility σ", "color": "Put Price"},
            contours=dict(showlabels=True, coloring="none", start=put_price_grid.min(), end=put_price_grid.max(), size=(put_price_grid.max()-put_price_grid.min())/8),
        )
        for trace in fig_contour.data:
            fig_put.add_trace(trace)

    # Flip y axis so volatility increases upwards
    fig_put.update_yaxes(autorange="reversed")

    # Show price numbers inside heatmap cells
    fig_put.update_traces(text=np.round(put_price_grid,2), texttemplate="%{text}", textfont={"size":8}, hoverinfo='none')

    st.plotly_chart(fig_put, use_container_width=True)

# -------------------------
# Heatmap row 2: Option price decay (price vs time)
# -------------------------
st.markdown("---")
st.subheader("Option price decay over time to expiry (Call option)")

fig_decay = px.imshow(
    Price_grid_decay,
    x=np.round(time_values, 4),
    y=np.round(price_values, 4),
    labels={"x": "Time to Expiry (Years)", "y": "Underlying Price (S)", "color": "Call Price"},
    origin="lower",
    aspect="auto",
    color_continuous_scale=colormap,
)

fig_decay.update_layout(height=600, margin=dict(t=30, l=10, r=10, b=40))

# Time to expiry goes from T (left) to 0 (right), so reverse x-axis:
fig_decay.update_xaxes(autorange="reversed")

# Show price numbers inside heatmap cells
fig_decay.update_traces(text=np.round(Price_grid_decay, 2), texttemplate="%{text}", textfont={"size":8}, hoverinfo='none')

st.plotly_chart(fig_decay, use_container_width=True)

# -------------------------
# Notes
# -------------------------
st.markdown(
    """
**Notes & tips**

- The summary table above shows model input variables including inspect values.

- Call and Put prices are shown in large colored text above.

- Heatmaps 1 & 2 show call and put prices over spot and volatility.

- The third heatmap shows time decay effect on call option price over underlying price and time to expiry (from input T to 0).

- Contour lines can be toggled on/off.

- Spot axis can be log scaled.
"""
)
