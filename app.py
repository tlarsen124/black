import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Black-Scholes Dashboard", layout="wide")

# -------------------------
# Black-Scholes vectorized price function
# -------------------------
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
# Sidebar inputs (moved all to sidebar)
# -------------------------
st.sidebar.header("Model Inputs")

K = st.sidebar.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
r = st.sidebar.number_input("Risk-free rate (annual) r", value=0.02, format="%.4f")
q = st.sidebar.number_input("Dividend yield (annual) q", value=0.00, format="%.4f")
T = st.sidebar.number_input("Time to expiry (years) T", value=0.5, min_value=0.0, step=0.01, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.write("Spot / Volatility grid")

S_min = st.sidebar.number_input("Spot min", value=50.0, format="%.2f")
S_max = st.sidebar.number_input("Spot max", value=150.0, format="%.2f")
S_steps = st.sidebar.slider("Spot steps", min_value=20, max_value=500, value=121, step=1)

sigma_min = st.sidebar.number_input("Volatility min (σ)", value=0.05, min_value=0.0001, format="%.4f")
sigma_max = st.sidebar.number_input("Volatility max (σ)", value=0.6, min_value=0.0001, format="%.4f")
sigma_steps = st.sidebar.slider("Vol steps", min_value=10, max_value=200, value=60, step=1)

st.sidebar.markdown("---")
st.sidebar.write("Inspect Values")

underlying_price = st.sidebar.number_input("Underlying asset price (Spot)", value=(S_min + S_max) / 2, format="%.4f")
volatility_inspect = st.sidebar.number_input("Volatility for inspection σ", value=0.20, min_value=0.0001, max_value=10.0, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.write("Visualization options")
log_spot = st.sidebar.checkbox("Plot spot on log scale (x-axis)", value=False)
show_contours = st.sidebar.checkbox("Show contour lines", value=True)
colormap = st.sidebar.selectbox("Color map (plotly)", options=["viridis", "plasma", "inferno", "magma", "cividis"], index=0)

# -------------------------
# Build grids
# -------------------------
S_values = np.linspace(S_min, S_max, S_steps)
sigma_values = np.linspace(sigma_min, sigma_max, sigma_steps)

S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

# Call and Put price grids
call_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, T, option_type="call")
put_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, T, option_type="put")

# -------------------------
# Compute single option price for inspect values
# -------------------------
call_price_single = black_scholes_price(underlying_price, K, r, q, volatility_inspect, T, option_type="call")
put_price_single = black_scholes_price(underlying_price, K, r, q, volatility_inspect, T, option_type="put")

# -------------------------
# Display variables & prices at top
# -------------------------
st.title("Black-Scholes Option Pricing Dashboard")

cols_top = st.columns(6)

cols_top[0].write(f"**Strike (K):** {K:.2f}")
cols_top[1].write(f"**Risk-free rate (r):** {r:.4f}")
cols_top[2].write(f"**Dividend yield (q):** {q:.4f}")
cols_top[3].write(f"**Time to expiry (T):** {T:.4f} years")
cols_top[4].write(f"**Underlying price:** {underlying_price:.4f}")
cols_top[5].write(f"**Volatility (σ):** {volatility_inspect:.4f}")

st.markdown("---")

st.markdown(
    f'<p style="font-size:40px; font-weight:bold; color:green;">Call Price: {call_price_single:.4f}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p style="font-size:40px; font-weight:bold; color:red;">Put Price: {put_price_single:.4f}</p>',
    unsafe_allow_html=True,
)

# -------------------------
# Helper for reversing volatility axis (lowest vol bottom)
# -------------------------
def reverse_y_axis(arr):
    return arr[::-1]

# -------------------------
# Call Price Heatmap (Spot vs Volatility)
# -------------------------
fig_call = go.Figure(
    data=go.Heatmap(
        z=reverse_y_axis(call_price_grid),
        x=np.round(S_values, 4),
        y=np.round(reverse_y_axis(sigma_values), 6),
        colorscale=colormap,
        text=np.round(reverse_y_axis(call_price_grid), 2),
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar=dict(title="Call Price"),
    )
)
fig_call.update_layout(
    height=600,
    margin=dict(t=30, l=10, r=10, b=40),
    xaxis_title="Spot (S)",
    yaxis_title="Volatility σ",
)
if log_spot:
    fig_call.update_xaxes(type="log")

# -------------------------
# Put Price Heatmap (Spot vs Volatility)
# -------------------------
fig_put = go.Figure(
    data=go.Heatmap(
        z=reverse_y_axis(put_price_grid),
        x=np.round(S_values, 4),
        y=np.round(reverse_y_axis(sigma_values), 6),
        colorscale=colormap,
        text=np.round(reverse_y_axis(put_price_grid), 2),
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar=dict(title="Put Price"),
    )
)
fig_put.update_layout(
    height=600,
    margin=dict(t=30, l=10, r=10, b=40),
    xaxis_title="Spot (S)",
    yaxis_title="Volatility σ",
)
if log_spot:
    fig_put.update_xaxes(type="log")

# -------------------------
# Time decay heatmap (Spot vs Time to expiry)
# -------------------------
# Build time grid for decay plot
time_steps = 100
time_values = np.linspace(T, 0, time_steps)  # from T down to 0

S_grid_time, time_grid = np.meshgrid(S_values, time_values)

# Prices for the inspected volatility, varying spot and time to expiry
price_decay_grid = black_scholes_price(S_grid_time, K, r, q, volatility_inspect, time_grid, option_type="call")

fig_decay = go.Figure(
    data=go.Heatmap(
        z=reverse_y_axis(price_decay_grid),
        x=np.round(time_values, 4),
        y=np.round(reverse_y_axis(S_values), 4),
        colorscale=colormap,
        text=np.round(reverse_y_axis(price_decay_grid), 2),
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar=dict(title="Call Price"),
    )
)

fig_decay.update_layout(
    height=600,
    margin=dict(t=30, l=10, r=10, b=40),
    xaxis_title="Time to Expiry (Years)",
    yaxis_title="Spot (S)",
)
fig_decay.update_xaxes(autorange="reversed")  # Show time from T (left) to 0 (right)

# -------------------------
# Layout: Show plots
# -------------------------
st.subheader("Call Option Price Heatmap (Spot vs Volatility)")
st.plotly_chart(fig_call, use_container_width=True)

st.subheader("Put Option Price Heatmap (Spot vs Volatility)")
st.plotly_chart(fig_put, use_container_width=True)

st.subheader("Time Decay Heatmap (Spot vs Time to Expiry)")
st.plotly_chart(fig_decay, use_container_width=True)
