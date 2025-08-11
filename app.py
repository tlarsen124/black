import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm
from io import StringIO

st.set_page_config(page_title="Black-Scholes Dashboard", layout="wide")

# -------------------------
# Black-Scholes implementation (vectorized)
# -------------------------
def black_scholes_price(S, K, r, q, sigma, T, option_type="call"):
    """
    Vectorized Black-Scholes price.
    S, sigma may be numpy arrays (same shapes or broadcastable).
    K, r, q, T scalars (or arrays broadcastable).
    option_type: "call" or "put"
    """
    # convert to numpy arrays
    S = np.array(S, dtype=float)
    sigma = np.array(sigma, dtype=float)

    # guard small sigma or T to avoid divide by zero
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    T = np.maximum(T, 0.0)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    # when T == 0, option value is payoff
    # handle T == 0 separately to avoid numerical issues
    if np.isscalar(T):
        if T == 0:
            if option_type == "call":
                return np.maximum(S - K, 0.0)
            else:
                return np.maximum(K - S, 0.0)
    else:
        # vector T case: create mask where T==0
        pass

    disc_factor = np.exp(-r * T)
    div_factor = np.exp(-q * T)

    if option_type == "call":
        price = S * div_factor * norm.cdf(d1) - K * disc_factor * norm.cdf(d2)
    else:  # put
        price = K * disc_factor * norm.cdf(-d2) - S * div_factor * norm.cdf(-d1)

    # If T is an array, enforce payoff for zero times
    if isinstance(T, np.ndarray):
        zeroT = (T == 0)
        if np.any(zeroT):
            payoff = np.where(option_type == "call", np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
            # need to broadcast shapes properly
            price = np.where(zeroT, payoff, price)

    return price

# -------------------------
# Sidebar: controls
# -------------------------
st.sidebar.header("Model inputs")

option_type = st.sidebar.selectbox("Option type", ["call", "put"])
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
st.sidebar.write("Visualization options")
log_spot = st.sidebar.checkbox("Plot spot on log scale (x-axis)", value=False)
show_contours = st.sidebar.checkbox("Show contour lines", value=True)
colormap = st.sidebar.selectbox("Color map (plotly)", options=["viridis","plasma","inferno","magma","cividis"], index=0)

# quick selector for a volatility to inspect cross-section
inspect_sigma = st.sidebar.number_input("Inspect volatility σ (for cross-section)", value=0.20, format="%.4f")

# -------------------------
# Build grid and compute prices
# -------------------------
S_values = np.linspace(S_min, S_max, S_steps)
sigma_values = np.linspace(sigma_min, sigma_max, sigma_steps)

# Create 2D grid: rows: volatility, cols: spot
S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, T, option_type=option_type)

# prepare DataFrame for download and optional display
df = pd.DataFrame(price_grid, index=np.round(sigma_values,8), columns=np.round(S_values,4))
df.index.name = "sigma"
df.columns.name = "spot"

# -------------------------
# Layout: main area
# -------------------------
st.title("Black-Scholes Option Pricing — Spot vs Volatility heatmap")
st.markdown(
    "Interactive dashboard: change inputs on the left. Heatmap shows option price across Spot (x) and Volatility (y)."
)

col1, col2 = st.columns([2, 1])

with col1:
    # heatmap with plotly
    # plotly heatmap expects z with shape (y, x) where y corresponds to rows -> sigma
    fig = px.imshow(
        price_grid,
        x=np.round(S_values, 4),
        y=np.round(sigma_values, 6),
        labels={"x": "Spot (S)", "y": "Volatility σ", "color": "Price"},
        origin="lower",  # so low sigma at bottom
        aspect="auto",
        color_continuous_scale=colormap,
    )
    fig.update_layout(height=650, margin=dict(t=30, l=10, r=10, b=40))
    if log_spot:
        # annotate that x axis is log scaled by converting tick labels only
        fig.update_xaxes(type="log")

    if show_contours:
        # add contour lines on top
        fig_contour = px.contour(
            z=price_grid,
            x=np.round(S_values, 4),
            y=np.round(sigma_values, 6),
            labels={"x": "Spot (S)", "y": "Volatility σ", "color": "Price"},
            contours=dict(showlabels=True, coloring="none", start=np.nanmin(price_grid), end=np.nanmax(price_grid), size=(np.nanmax(price_grid)-np.nanmin(price_grid))/8),
        )
        # overlay contour traces
        for trace in fig_contour.data:
            fig.add_trace(trace)

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Inspect values")
    # show price at chosen spot/vol pairs
    # show price at mid spot and selected sigma
    chosen_spot = st.number_input("Pick a spot price to inspect", value=float((S_min + S_max) / 2), format="%.4f")
    chosen_sigma = st.number_input("Pick a volatility to inspect", value=float(inspect_sigma), format="%.4f")
    single_price = black_scholes_price(chosen_spot, K, r, q, chosen_sigma, T, option_type=option_type)
    st.metric("Option price", f"{single_price:.4f}")

    # show cross-section price vs Spot for chosen sigma
    st.markdown("**Cross-section: Price vs Spot**")
    sigma_closest_idx = np.abs(sigma_values - chosen_sigma).argmin()
    cross_prices = price_grid[sigma_closest_idx, :]
    df_cross = pd.DataFrame({"spot": S_values, "price": cross_prices})
    fig2 = px.line(df_cross, x="spot", y="price", title=f"σ ≈ {sigma_values[sigma_closest_idx]:.4f}")
    fig2.update_layout(height=300, margin=dict(t=30, l=10, r=10, b=30))
    st.plotly_chart(fig2, use_container_width=True)

    # summary stats
    st.markdown("**Grid summary**")
    st.write(f"Price min: {np.nanmin(price_grid):.4f}")
    st.write(f"Price max: {np.nanmax(price_grid):.4f}")
    st.write(f"Mean price: {np.nanmean(price_grid):.4f}")

    st.markdown("---")
    st.download_button(
        "Download grid as CSV",
        data=df.to_csv(),
        file_name=f"bs_grid_K{K}_T{T}.csv",
        mime="text/csv",
    )

# -------------------------
# Optional Data preview and notes
# -------------------------
with st.expander("Show raw price grid (sample)"):
    st.dataframe(df.style.format("{:.4f}"), height=300)

st.markdown(
    """
**Notes & tips**
- The heatmap X axis is Spot (S) and Y axis is volatility σ. Color = option price.
- If `T=0` the prices reduce to payoffs.
- For calls: price increases in S and σ (generally). For puts: different shape — explore!
- Want Greeks (Delta/Gamma/Vega/theta)? I can add them quickly.
"""
)