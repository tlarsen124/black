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
T = st.sidebar.number_input("Time to expiry (years) T", value=0.5, min_value=0.0, step=0.01, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.write("Spot / Volatility grid")

S_min = st.sidebar.number_input("Spot min", value=50.0, format="%.2f")
S_max = st.sidebar.number_input("Spot max", value=150.0, format="%.2f")
S_steps = st.sidebar.slider("Spot steps", min_value=20, max_value=50, value=20, step=1)

sigma_min = st.sidebar.number_input("Volatility min (σ)", value=0.05, min_value=0.0001, format="%.4f")
sigma_max = st.sidebar.number_input("Volatility max (σ)", value=0.6, min_value=0.0001, format="%.4f")
sigma_steps = st.sidebar.slider("Vol steps", min_value=10, max_value=50, value=20, step=1)

st.sidebar.markdown("---")
st.sidebar.write("Visualization options")
log_spot = st.sidebar.checkbox("Plot spot on log scale (x-axis)", value=False)
show_contours = st.sidebar.checkbox("Show contour lines", value=True)
colormap = st.sidebar.selectbox("Color map (plotly)", options=["Viridis","Plasma","Inferno","Magma","Cividis"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Inspect values")
chosen_spot = st.sidebar.number_input("Pick a spot price to inspect", value=(S_min + S_max) / 2, format="%.4f")
chosen_sigma = st.sidebar.number_input("Pick a volatility to inspect", value=0.20, format="%.4f")

# Compute grid and prices
S_values = np.linspace(S_min, S_max, S_steps)
sigma_values = np.linspace(sigma_min, sigma_max, sigma_steps)
S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

call_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, T, option_type="call")
put_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, T, option_type="put")

# Calculate inspected prices at chosen spot and sigma
call_price_inspect = black_scholes_price(chosen_spot, K, r, q, chosen_sigma, T, option_type="call")
put_price_inspect = black_scholes_price(chosen_spot, K, r, q, chosen_sigma, T, option_type="put")

# Build summary table data
summary_data = {
    "Variable": ["Strike (K)", "Risk-free rate (r)", "Dividend yield (q)", "Time to expiry (T)", "Inspect Spot", "Inspect Volatility", "Call Price", "Put Price"],
    "Value": [f"{K:.2f}", f"{r:.4f}", f"{q:.4f}", f"{T:.4f}", f"{chosen_spot:.4f}", f"{chosen_sigma:.4f}", f"{call_price_inspect:.4f}", f"{put_price_inspect:.4f}"],
}

summary_df = pd.DataFrame(summary_data)

# Style call/put price cells with colors and bold text
def highlight_call_put(s):
    is_call = s.name == 6  # row index for Call Price
    is_put = s.name == 7   # row index for Put Price
    if is_call:
        return ['background-color: #b6d7a8; font-weight: bold' if col == 'Value' else '' for col in s.index]
    elif is_put:
        return ['background-color: #f4cccc; font-weight: bold' if col == 'Value' else '' for col in s.index]
    else:
        return ['' for _ in s.index]

st.title("Black-Scholes Option Pricing Dashboard")

st.markdown("### Model Inputs and Prices Summary")
st.dataframe(summary_df.style.apply(highlight_call_put, axis=1), width=600, height=220)

# Prepare DataFrames for hover + download
call_df = pd.DataFrame(call_price_grid, index=np.round(sigma_values,8), columns=np.round(S_values,4))
call_df.index.name = "Volatility σ"
call_df.columns.name = "Spot S"

put_df = pd.DataFrame(put_price_grid, index=np.round(sigma_values,8), columns=np.round(S_values,4))
put_df.index.name = "Volatility σ"
put_df.columns.name = "Spot S"

# Layout heatmaps side-by-side
col1, col2 = st.columns([2,2])

with col1:
    # Heatmap for Call Prices with text annotations
    call_text = np.round(call_price_grid, 2).astype(str)
    fig_call = go.Figure(data=go.Heatmap(
        z=call_price_grid,
        x=np.round(S_values, 2),
        y=np.round(sigma_values, 4),
        colorscale=colormap.lower(),
        colorbar=dict(title="Call Price"),
        text=call_text,
        texttemplate="%{text}",
        textfont={"size":9},
        hovertemplate="Spot: %{x}<br>Volatility: %{y}<br>Call Price: %{z:.4f}<extra></extra>",
        reversescale=False,
    ))
    fig_call.update_layout(
        title="Call Option Price Heatmap",
        xaxis_title="Spot (S)",
        yaxis_title="Volatility σ",
        yaxis_autorange="reversed",
        height=650,
        margin=dict(t=50, l=50, r=50, b=50)
    )
    if log_spot:
        fig_call.update_xaxes(type="log")
    if show_contours:
        fig_call.add_trace(go.Contour(
            z=call_price_grid,
            x=np.round(S_values, 2),
            y=np.round(sigma_values, 4),
            contours=dict(
                coloring="none",
                showlabels=True,
                start=np.nanmin(call_price_grid),
                end=np.nanmax(call_price_grid),
                size=(np.nanmax(call_price_grid) - np.nanmin(call_price_grid))/8,
            ),
            line_width=1,
            colorscale="Greys",
            showscale=False,
        ))
    st.plotly_chart(fig_call, use_container_width=True)

with col2:
    # Heatmap for Put Prices with text annotations
    put_text = np.round(put_price_grid, 2).astype(str)
    fig_put = go.Figure(data=go.Heatmap(
        z=put_price_grid,
        x=np.round(S_values, 2),
        y=np.round(sigma_values, 4),
        colorscale=colormap.lower(),
        colorbar=dict(title="Put Price"),
        text=put_text,
        texttemplate="%{text}",
        textfont={"size":9},
        hovertemplate="Spot: %{x}<br>Volatility: %{y}<br>Put Price: %{z:.4f}<extra></extra>",
        reversescale=False,
    ))
    fig_put.update_layout(
        title="Put Option Price Heatmap",
        xaxis_title="Spot (S)",
        yaxis_title="Volatility σ",
        yaxis_autorange="reversed",
        height=650,
        margin=dict(t=50, l=50, r=50, b=50)
    )
    if log_spot:
        fig_put.update_xaxes(type="log")
    if show_contours:
        fig_put.add_trace(go.Contour(
            z=put_price_grid,
            x=np.round(S_values, 2),
            y=np.round(sigma_values, 4),
            contours=dict(
                coloring="none",
                showlabels=True,
                start=np.nanmin(put_price_grid),
                end=np.nanmax(put_price_grid),
                size=(np.nanmax(put_price_grid) - np.nanmin(put_price_grid))/8,
            ),
            line_width=1,
            colorscale="Greys",
            showscale=False,
        ))
    st.plotly_chart(fig_put, use_container_width=True)

st.markdown("---")

with st.expander("Show raw call price grid (sample)"):
    st.dataframe(call_df.style.format("{:.4f}"), height=300)

with st.expander("Show raw put price grid (sample)"):
    st.dataframe(put_df.style.format("{:.4f}"), height=300)

st.markdown(
    """
**Notes & tips**
- The heatmaps show Call and Put prices separately.
- Numbers in each box show the price rounded to 2 decimals.
- The top table shows your chosen parameters plus the option prices at your selected spot and volatility.
- If `T=0` the prices reduce to payoffs.
- Want Greeks (Delta/Gamma/Vega/theta)? I can add them quickly.
"""
)
