import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Black-Scholes Dashboard", layout="wide")

# Black-Scholes pricing function (vectorized)
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

# Sidebar controls
st.sidebar.header("Model inputs")

K = st.sidebar.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
r = st.sidebar.number_input("Risk-free rate (annual) r", value=0.02, format="%.4f")
q = st.sidebar.number_input("Dividend yield (annual) q", value=0.00, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.write("Spot / Volatility / Time grid")

S_min = st.sidebar.number_input("Spot min", value=50.0, format="%.2f")
S_max = st.sidebar.number_input("Spot max", value=150.0, format="%.2f")
S_steps = st.sidebar.slider("Spot steps", min_value=20, max_value=50, value=20, step=1)

sigma_min = st.sidebar.number_input("Volatility min (σ)", value=0.05, min_value=0.0001, format="%.4f")
sigma_max = st.sidebar.number_input("Volatility max (σ)", value=0.6, min_value=0.0001, format="%.4f")
sigma_steps = st.sidebar.slider("Vol steps", min_value=10, max_value=50, value=20, step=1)

T_min = 0.01  # minimum positive time to expiry to avoid divide by zero in time decay plot
T_max = 2.0   # max years for time decay plot
T_steps = 50  # time steps for time decay plot

st.sidebar.markdown("---")
st.sidebar.write("Visualization options")
log_spot = st.sidebar.checkbox("Plot spot on log scale (x-axis for grids 1 & 2)", value=False)
show_contours = st.sidebar.checkbox("Show contour lines (all heatmaps)", value=True)
colormap = st.sidebar.selectbox("Color map (plotly)", options=["Viridis","Plasma","Inferno","Magma","Cividis"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Inspect values")
chosen_spot = st.sidebar.number_input("Underlying Asset Price", value=(S_min + S_max) / 2, format="%.4f")
chosen_sigma = st.sidebar.number_input("Volatility σ", value=0.20, format="%.4f")
chosen_T = st.sidebar.number_input("Time to expiry (years) T", value=0.5, min_value=0.0, format="%.4f")

# Compute grids
S_values = np.linspace(S_min, S_max, S_steps)
sigma_values = np.linspace(sigma_min, sigma_max, sigma_steps)
S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

call_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, chosen_T, option_type="call")
put_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, chosen_T, option_type="put")

# Inspect prices at chosen values
call_price_inspect = black_scholes_price(chosen_spot, K, r, q, chosen_sigma, chosen_T, option_type="call")
put_price_inspect = black_scholes_price(chosen_spot, K, r, q, chosen_sigma, chosen_T, option_type="put")

# Summary table (without call/put prices)
summary_data = {
    "Variable": ["Strike (K)", "Risk-free rate (r)", "Dividend yield (q)", 
                 "Inspect Underlying Asset Price", "Inspect Volatility σ", "Inspect Time to expiry (T)"],
    "Value": [f"{K:.2f}", f"{r:.4f}", f"{q:.4f}", f"{chosen_spot:.4f}", f"{chosen_sigma:.4f}", f"{chosen_T:.4f}"],
}
summary_df = pd.DataFrame(summary_data)

# Style function for summary table
def style_table(s):
    return ['' for _ in s]

st.title("Black-Scholes Option Pricing Dashboard")

st.markdown("### Model Inputs and Inspect Values")
st.dataframe(summary_df.style.apply(style_table, axis=1), width=700, height=180)

# Display Call and Put price at inspect values, big and colored
col_call_put = st.columns(2)
with col_call_put[0]:
    st.markdown(f'<div style="font-size:40px; font-weight:bold; color:#2e7d32;">Call Price: {call_price_inspect:.4f}</div>', unsafe_allow_html=True)
with col_call_put[1]:
    st.markdown(f'<div style="font-size:40px; font-weight:bold; color:#c62828;">Put Price: {put_price_inspect:.4f}</div>', unsafe_allow_html=True)

# Prepare DataFrames for heatmaps
call_df = pd.DataFrame(call_price_grid, index=np.round(sigma_values,8), columns=np.round(S_values,4))
call_df.index.name = "Volatility σ"
call_df.columns.name = "Spot S"

put_df = pd.DataFrame(put_price_grid, index=np.round(sigma_values,8), columns=np.round(S_values,4))
put_df.index.name = "Volatility σ"
put_df.columns.name = "Spot S"

# Heatmap 1 & 2: Call and Put prices vs Spot and Volatility
col1, col2, col3 = st.columns([2, 2, 2])

def add_heatmap(fig, z, x, y, title, colorbar_title):
    heatmap = go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colormap.lower(),
        colorbar=dict(title=colorbar_title),
        reversescale=False,
        hovertemplate=f"{title}<br>Spot: %{{x}}<br>Volatility σ: %{{y}}<br>Price: %{{z:.4f}}<extra></extra>",
        text=np.round(z, 2).astype(str),
        texttemplate="%{text}",
        textfont={"size":9},
    )
    fig.add_trace(heatmap)
    if show_contours:
        fig.add_trace(go.Contour(
            z=z,
            x=x,
            y=y,
            contours=dict(
                coloring="none",
                showlabels=True,
                start=np.nanmin(z),
                end=np.nanmax(z),
                size=(np.nanmax(z) - np.nanmin(z))/8,
            ),
            line_width=1,
            colorscale="Greys",
            showscale=False,
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Spot (S)",
        yaxis_title="Volatility σ",
        yaxis_autorange="reversed",
        height=650,
        margin=dict(t=50, l=50, r=50, b=50),
    )
    if log_spot:
        fig.update_xaxes(type="log")
    return fig

with col1:
    fig_call = go.Figure()
    fig_call = add_heatmap(fig_call, call_price_grid, np.round(S_values, 2), np.round(sigma_values, 4), "Call Option Price Heatmap", "Call Price")
    st.plotly_chart(fig_call, use_container_width=True)

with col2:
    fig_put = go.Figure()
    fig_put = add_heatmap(fig_put, put_price_grid, np.round(S_values, 2), np.round(sigma_values, 4), "Put Option Price Heatmap", "Put Price")
    st.plotly_chart(fig_put, use_container_width=True)

# Heatmap 3: Underlying Asset Price (y-axis) vs Time to Expiry (x-axis), showing Call price time decay
# Fix spot range and time range grid
S_time_values = np.linspace(S_min, S_max, S_steps)
T_time_values = np.linspace(T_min, T_max, T_steps)
S_time_grid, T_time_grid = np.meshgrid(S_time_values, T_time_values)

call_time_decay_grid = black_scholes_price(S_time_grid, K, r, q, chosen_sigma, T_time_grid, option_type="call")

# Calculate breakeven line (where call price >= 0)
# Breakeven approx for calls: spot where intrinsic value >= 0 at expiry, here just price > 0
# We'll shade prices <= 0 as "no profit" zone

with col3:
    fig_time_decay = go.Figure()

    heatmap = go.Heatmap(
        z=call_time_decay_grid,
        x=np.round(T_time_values, 4),
        y=np.round(S_time_values, 2),
        colorscale=colormap.lower(),
        colorbar=dict(title="Call Price"),
        reversescale=False,
        hovertemplate="Time to expiry: %{x:.4f}<br>Spot: %{y}<br>Call Price: %{z:.4f}<extra></extra>",
    )
    fig_time_decay.add_trace(heatmap)

    if show_contours:
        fig_time_decay.add_trace(go.Contour(
            z=call_time_decay_grid,
            x=np.round(T_time_values, 4),
            y=np.round(S_time_values, 2),
            contours=dict(
                coloring="none",
                showlabels=True,
                start=np.nanmin(call_time_decay_grid),
                end=np.nanmax(call_time_decay_grid),
                size=(np.nanmax(call_time_decay_grid) - np.nanmin(call_time_decay_grid))/8,
            ),
            line_width=1,
            colorscale="Greys",
            showscale=False,
        ))

    fig_time_decay.update_layout(
        title="Call Option Price vs Underlying Asset Price and Time to Expiry",
        xaxis_title="Time to expiry (years)",
        yaxis_title="Underlying Asset Price (S)",
        height=650,
        margin=dict(t=50, l=50, r=50, b=50),
    )

    st.plotly_chart(fig_time_decay, use_container_width=True)

st.markdown("---")

with st.expander("Show raw call price grid (Spot vs Volatility)"):
    st.dataframe(call_df.style.format("{:.4f}"), height=300)

with st.expander("Show raw put price grid (Spot vs Volatility)"):
    st.dataframe(put_df.style.format("{:.4f}"), height=300)

st.markdown(
    """
**Notes & tips**

- Top table shows model inputs plus inspect values.

- Call and Put prices at inspect values are shown big and colored.

- Heatmaps 1 & 2 show Call and Put prices vs Spot and Volatility.

- Heatmap 3 shows Call price vs Underlying Asset Price and Time to Expiry — illustrating time decay.

- Contour lines are optional.

- Want Greeks or Put time decay plot? Just ask!
"""
)

