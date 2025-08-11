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

option_type = st.sidebar.selectbox("Option type", ["call", "put"])
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

inspect_sigma = st.sidebar.number_input("Inspect volatility σ (for cross-section)", value=0.20, format="%.4f")

# Compute grid and prices
S_values = np.linspace(S_min, S_max, S_steps)
sigma_values = np.linspace(sigma_min, sigma_max, sigma_steps)

S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)
price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, T, option_type=option_type)

# Display option price summary at top
st.title("Black-Scholes Option Pricing Dashboard")
st.markdown(f"### {option_type.capitalize()} option price summary for inputs:")
st.write(f"- Strike (K): {K:.2f}")
st.write(f"- Risk-free rate (r): {r:.4f}")
st.write(f"- Dividend yield (q): {q:.4f}")
st.write(f"- Time to expiry (T): {T:.4f} years")
st.write(f"---")
mid_spot = (S_min + S_max) / 2
mid_sigma = inspect_sigma
mid_price = black_scholes_price(mid_spot, K, r, q, mid_sigma, T, option_type=option_type)
st.markdown(f"**Price at Spot = {mid_spot:.2f} and Volatility = {mid_sigma:.4f} is:**  **{mid_price:.4f}**")

# Prepare DataFrame for hover + download
df = pd.DataFrame(price_grid, index=np.round(sigma_values,8), columns=np.round(S_values,4))
df.index.name = "Volatility σ"
df.columns.name = "Spot S"

# Layout with heatmap & cross-section
col1, col2 = st.columns([2,1])

with col1:
    # Create heatmap with text annotations (price numbers in each cell)
    # Plotly heatmap requires text to be strings with same shape as z
    text_values = np.round(price_grid, 2).astype(str)

    fig = go.Figure(data=go.Heatmap(
        z=price_grid,
        x=np.round(S_values, 2),
        y=np.round(sigma_values, 4),
        colorscale=colormap.lower(),
        colorbar=dict(title="Option Price"),
        text=text_values,
        texttemplate="%{text}",
        textfont={"size":9},
        hovertemplate="Spot: %{x}<br>Volatility: %{y}<br>Price: %{z:.4f}<extra></extra>",
        reversescale=False,
    ))

    fig.update_layout(
        title=f"Option Price Heatmap ({option_type.capitalize()})",
        xaxis_title="Spot (S)",
        yaxis_title="Volatility σ",
        yaxis_autorange="reversed",
        height=650,
        margin=dict(t=50, l=50, r=50, b=50)
    )

    if log_spot:
        fig.update_xaxes(type="log")

    if show_contours:
        fig.add_trace(go.Contour(
            z=price_grid,
            x=np.round(S_values, 2),
            y=np.round(sigma_values, 4),
            contours=dict(
                coloring="none",
                showlabels=True,
                start=np.nanmin(price_grid),
                end=np.nanmax(price_grid),
                size=(np.nanmax(price_grid) - np.nanmin(price_grid))/8,
            ),
            line_width=1,
            colorscale="Greys",
            showscale=False,
        ))

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Inspect values")

    chosen_spot = st.number_input("Pick a spot price to inspect", value=mid_spot, format="%.4f")
    chosen_sigma = st.number_input("Pick a volatility to inspect", value=mid_sigma, format="%.4f")

    single_price = black_scholes_price(chosen_spot, K, r, q, chosen_sigma, T, option_type=option_type)
    st.metric("Option price", f"{single_price:.4f}")

    st.markdown("**Cross-section: Price vs Spot**")
    sigma_closest_idx = np.abs(sigma_values - chosen_sigma).argmin()
    cross_prices = price_grid[sigma_closest_idx, :]
    df_cross = pd.DataFrame({"Spot": S_values, "Price": cross_prices})
    fig2 = go.Figure(data=go.Scatter(x=df_cross["Spot"], y=df_cross["Price"], mode="lines+markers"))
    fig2.update_layout(
        title=f"Price vs Spot (Volatility σ ≈ {sigma_values[sigma_closest_idx]:.4f})",
        xaxis_title="Spot (S)",
        yaxis_title="Option Price",
        height=300,
        margin=dict(t=40, l=40, r=20, b=30)
    )
    st.plotly_chart(fig2, use_container_width=True)

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

with st.expander("Show raw price grid (sample)"):
    st.dataframe(df.style.format("{:.4f}"), height=300)

st.markdown(
    """
**Notes & tips**
- The heatmap X axis is Spot (S) and Y axis is volatility σ. Color = option price.
- Numbers in each box show the price rounded to 2 decimals.
- If `T=0` the prices reduce to payoffs.
- For calls: price increases in S and σ (generally). For puts: different shape — explore!
- Want Greeks (Delta/Gamma/Vega/theta)? I can add them quickly.
"""
)
