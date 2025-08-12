import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Black-Scholes Dashboard", layout="wide")

# -------------------------
# Black-Scholes formula (vectorized)
# -------------------------
def black_scholes_price(S, K, r, q, sigma, T, option_type="call"):
    S = np.array(S, dtype=float)
    sigma = np.array(sigma, dtype=float)

    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    T = np.maximum(T, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
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
# Sidebar inputs (Model inputs + Inspect values)
# -------------------------
st.sidebar.header("Model inputs")

K = st.sidebar.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
r = st.sidebar.number_input("Risk-free rate (annual) r", value=0.02, format="%.4f")
q = st.sidebar.number_input("Dividend yield (annual) q", value=0.00, format="%.4f")
T = st.sidebar.number_input("Time to expiry (years) T", value=1.0, min_value=0.0, step=0.01, format="%.4f")

# Inspect values moved up but no header now
underlying_asset_price = st.sidebar.number_input("Underlying asset price (Spot S)", value=100.0, format="%.4f")
volatility_inspect = st.sidebar.number_input("Volatility σ", value=0.20, min_value=0.0001, format="%.4f")

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
colormap = st.sidebar.selectbox("Color map (plotly)", options=["viridis", "plasma", "inferno", "magma", "cividis"], index=0)

# -------------------------
# Calculate prices grids
# -------------------------
S_values = np.linspace(S_min, S_max, S_steps)
sigma_values = np.linspace(sigma_min, sigma_max, sigma_steps)

S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

call_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, T, option_type="call")
put_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, T, option_type="put")

time_steps = np.linspace(T, 0, 100)
price_range_min = max(0.01, underlying_asset_price * 0.5)
price_range_max = underlying_asset_price * 1.5
price_range_steps = 100
spot_time_grid = np.linspace(price_range_min, price_range_max, price_range_steps)
T_grid, S_time_grid = np.meshgrid(time_steps, spot_time_grid)

time_decay_price_grid_call = black_scholes_price(S_time_grid, K, r, q, volatility_inspect, T_grid, option_type="call")
time_decay_price_grid_put = black_scholes_price(S_time_grid, K, r, q, volatility_inspect, T_grid, option_type="put")

# -------------------------
# Display variable table and prices
# -------------------------
st.title("Black-Scholes Option Pricing Dashboard")

variables = {
    "Strike (K)": K,
    "Risk-free rate (r)": r,
    "Dividend yield (q)": q,
    "Time to expiry (T)": T,
    "Underlying asset price": underlying_asset_price,
    "Volatility (σ)": volatility_inspect,
}

summary_df = pd.DataFrame(variables.items(), columns=["Variable", "Value"])

call_price_inspect = black_scholes_price(underlying_asset_price, K, r, q, volatility_inspect, T, "call")
put_price_inspect = black_scholes_price(underlying_asset_price, K, r, q, volatility_inspect, T, "put")

st.dataframe(summary_df.style.format({"Value": "{:.4f}"}), width=600, height=200)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<h1 style='color:green; font-weight:bold;'>Call Price: ${call_price_inspect:.4f}</h1>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<h1 style='color:red; font-weight:bold;'>Put Price: ${put_price_inspect:.4f}</h1>", unsafe_allow_html=True)

# -------------------------
# Heatmaps with price annotations
# -------------------------
col_call, col_put = st.columns(2)

with col_call:
    st.subheader("Call Option Price Heatmap (Spot vs Volatility)")
    fig_call = px.imshow(
        call_price_grid[::-1],
        x=np.round(S_values, 4),
        y=np.round(sigma_values[::-1], 6),
        labels={"x": "Spot (S)", "y": "Volatility σ", "color": "Call Price"},
        origin="lower",
        aspect="auto",
        color_continuous_scale=colormap,
        text=np.round(call_price_grid[::-1], 2),
    )
    fig_call.update_traces(textfont_size=8)
    fig_call.update_layout(height=650, margin=dict(t=30, l=10, r=10, b=40))
    if log_spot:
        fig_call.update_xaxes(type="log")
    if show_contours:
        contour_call = go.Contour(
            z=call_price_grid[::-1],
            x=np.round(S_values, 4),
            y=np.round(sigma_values[::-1], 6),
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
        fig_call.add_trace(contour_call)
    st.plotly_chart(fig_call, use_container_width=True)

with col_put:
    st.subheader("Put Option Price Heatmap (Spot vs Volatility)")
    fig_put = px.imshow(
        put_price_grid[::-1],
        x=np.round(S_values, 4),
        y=np.round(sigma_values[::-1], 6),
        labels={"x": "Spot (S)", "y": "Volatility σ", "color": "Put Price"},
        origin="lower",
        aspect="auto",
        color_continuous_scale=colormap,
        text=np.round(put_price_grid[::-1], 2),
    )
    fig_put.update_traces(textfont_size=8)
    fig_put.update_layout(height=650, margin=dict(t=30, l=10, r=10, b=40))
    if log_spot:
        fig_put.update_xaxes(type="log")
    if show_contours:
        contour_put = go.Contour(
            z=put_price_grid[::-1],
            x=np.round(S_values, 4),
            y=np.round(sigma_values[::-1], 6),
            contours=dict(
                coloring="none",
                showlabels=True,
                start=np.min(put_price_grid),
                end=np.max(put_price_grid),
                size=(np.max(put_price_grid) - np.min(put_price_grid)) / 8,
            ),
            line_width=1,
            showscale=False,
        )
        fig_put.add_trace(contour_put)
    st.plotly_chart(fig_put, use_container_width=True)

# -------------------------
# Third heatmap: Time decay price (Spot vs Time to expiry)
# -------------------------
st.subheader("Time Decay Heatmap (Spot vs Time to Expiry)")

fig_time_decay = px.imshow(
    time_decay_price_grid_call,
    x=np.round(time_steps[::-1], 6),
    y=np.round(spot_time_grid, 4),
    labels={"x": "Time to Expiry (Years)", "y": "Spot Price", "color": "Call Price"},
    origin="lower",
    aspect="auto",
    color_continuous_scale=colormap,
    text=np.round(time_decay_price_grid_call, 2),
)
fig_time_decay.update_traces(textfont_size=7)
fig_time_decay.update_layout(height=400, margin=dict(t=30, l=10, r=10, b=40))
if show_contours:
    contour_time = go.Contour(
        z=time_decay_price_grid_call,
        x=np.round(time_steps, 6),
        y=np.round(spot_time_grid, 4),
        contours=dict(
            coloring="none",
            showlabels=True,
            start=np.min(time_decay_price_grid_call),
            end=np.max(time_decay_price_grid_call),
            size=(np.max(time_decay_price_grid_call) - np.min(time_decay_price_grid_call)) / 8,
        ),
        line_width=1,
        showscale=False,
    )
    fig_time_decay.add_trace(contour_time)

st.plotly_chart(fig_time_decay, use_container_width=True)

# -------------------------
# Notes
# -------------------------
st.markdown(
    """
**Notes & tips**
- The heatmaps show option prices with colors and numbers inside each box.
- The first two heatmaps show Call and Put prices vs Spot price and Volatility.
- The third heatmap shows how the Call price changes over time to expiry and spot price, demonstrating time decay.
- Time axis in the third heatmap starts from the input expiry time on the left and goes down to 0 on the right.
"""
)
