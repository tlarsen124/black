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

def black_scholes_greeks(S, K, r, q, sigma, T, option_type="call"):
    S = np.array(S, dtype=float)
    sigma = np.array(sigma, dtype=float)
    T = np.maximum(T, 1e-12)  # Avoid division by zero
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    exp_neg_qT = np.exp(-q * T)
    exp_neg_rT = np.exp(-r * T)
    
    if option_type == "call":
        delta = exp_neg_qT * cdf_d1
        theta = (- (S * sigma * exp_neg_qT * pdf_d1) / (2 * np.sqrt(T))
                 - r * K * exp_neg_rT * cdf_d2
                 + q * S * exp_neg_qT * cdf_d1)
        rho = K * T * exp_neg_rT * cdf_d2
    else:
        delta = -exp_neg_qT * norm.cdf(-d1)
        theta = (- (S * sigma * exp_neg_qT * pdf_d1) / (2 * np.sqrt(T))
                 + r * K * exp_neg_rT * norm.cdf(-d2)
                 - q * S * exp_neg_qT * norm.cdf(-d1))
        rho = -K * T * exp_neg_rT * norm.cdf(-d2)
    
    gamma = (exp_neg_qT * pdf_d1) / (S * sigma * np.sqrt(T))
    vega = S * exp_neg_qT * pdf_d1 * np.sqrt(T)

    theta /= 365.0  # per day

    greeks = {
        "Delta": delta,
        "Gamma": gamma,
        "Theta (per day)": theta,
        "Vega": vega / 100,   # per 1% vol change
        "Rho": rho / 100,     # per 1% rate change
    }
    return greeks

# ---------------------------
# Sidebar inputs - All under Model Inputs
# ---------------------------
st.sidebar.header("Model Inputs")

# Strike, risk-free, dividend
K = st.sidebar.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
r = st.sidebar.number_input("Risk-free rate (annual) r", value=0.04, format="%.2f")
q = st.sidebar.number_input("Dividend yield (annual) q", value=0.00, format="%.2f")

# Inspect values for pricing
chosen_spot = st.sidebar.number_input("Underlying Asset Price", value=100.00, format="%.2f")
chosen_sigma = st.sidebar.number_input("Volatility σ", value=0.20, format="%.4f")
chosen_T = st.sidebar.number_input("Time to expiry (years) T", value=0.5, min_value=0.0, format="%.2f")


st.sidebar.header("Heatmap Parameters")
# Spot range and steps
S_min = st.sidebar.number_input("Spot min", value=50.0, format="%.2f")
S_max = st.sidebar.number_input("Spot max", value=150.0, format="%.2f")
S_steps = st.sidebar.slider("Spot steps", min_value=20, max_value=50, value=20, step=1)



# Volatility range and steps
sigma_min = st.sidebar.number_input("Volatility min (σ)", value=0.05, min_value=0.0001, format="%.2f")
sigma_max = st.sidebar.number_input("Volatility max (σ)", value=0.6, min_value=0.0001, format="%.2f")
sigma_steps = st.sidebar.slider("Vol steps", min_value=10, max_value=50, value=20, step=1)


# Visualization options
st.sidebar.markdown("---")
st.sidebar.write("Visualization options")
show_greeks = st.sidebar.checkbox("Show Greeks", value=False)
log_spot = st.sidebar.checkbox("Plot spot on log scale (x-axis for grids 1 & 2)", value=False)
show_contours = st.sidebar.checkbox("Show contour lines (all heatmaps)", value=True)
colormap = st.sidebar.selectbox("Color map (plotly)", options=["Viridis","Plasma","Inferno","Magma","Cividis"], index=0)


# ---------------------------
# Generate grids for heatmaps 1 & 2 (call & put)
# ---------------------------
S_values = np.linspace(S_min, S_max, S_steps)
sigma_values = np.linspace(sigma_min, sigma_max, sigma_steps)
S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

call_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, chosen_T, option_type="call")
put_price_grid = black_scholes_price(S_grid, K, r, q, sigma_grid, chosen_T, option_type="put")

# Inspect price at chosen values for selected option type
inspect_call_price = black_scholes_price(chosen_spot, K, r, q, chosen_sigma, chosen_T, option_type="call")
inspect_put_price = black_scholes_price(chosen_spot, K, r, q, chosen_sigma, chosen_T, option_type="put")

# Summary table (variables)
# ---------------------------
summary_data = {
    "Variable": ["Strike (K)", "Risk-free rate (r)", "Dividend yield (q)", 
                 "Underlying Asset Price", "Volatility σ", "Time to expiry (T)"],
    "Value": [f"{K:.2f}", f"{r:.2f}", f"{q:.2f}", f"{chosen_spot:.2f}", f"{chosen_sigma:.4f}", f"{chosen_T:.2f}"]}
summary_df = pd.DataFrame(summary_data)

# Safe formatter for summary table
def safe_format(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return x

styled_df = summary_df.style.format({"Value": safe_format})

# ---------------------------
# Display header, inputs, and price
# ---------------------------
st.title("Black-Scholes Option Pricing Dashboard")

st.markdown("### Model Inputs and Inspect Values")
st.dataframe(styled_df, width=700, height=180)

# Show inspected option price big & bold below table with color

st.markdown(
    """
    <div style="display: flex; justify-content: space-between; width: 100%; margin-top: 10px;">
        <div style="font-size: 30px; font-weight: bold; color: #2e7d32; padding-left: 80px;">
            Call Price: {:.4f}
        </div>
        <div style="font-size: 30px; font-weight: bold; color: #c62828; padding-right: 80px;">
            Put Price: {:.4f}
        </div>
    </div>
    """.format(inspect_call_price, inspect_put_price),
    unsafe_allow_html=True
)



if show_greeks:
    call_greeks = black_scholes_greeks(chosen_spot, K, r, q, chosen_sigma, chosen_T, option_type="call")
    put_greeks = black_scholes_greeks(chosen_spot, K, r, q, chosen_sigma, chosen_T, option_type="put")

    def format_greeks(greeks_dict):
        return {k: f"{v:.4f}" for k, v in greeks_dict.items()}

    call_greeks_fmt = format_greeks(call_greeks)
    put_greeks_fmt = format_greeks(put_greeks)

    st.markdown("### Greeks at Chosen Inputs")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Call Option Greeks")
        greeks_df = pd.DataFrame.from_dict(call_greeks_fmt, orient='index', columns=['Value'])
        st.table(greeks_df)

    with col2:
        st.markdown("#### Put Option Greeks")
        greeks_df = pd.DataFrame.from_dict(put_greeks_fmt, orient='index', columns=['Value'])
        st.table(greeks_df)

# ---------------------------
# ---------------------------
# Helper to add heatmaps with contours and text
# ---------------------------
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
        height=650,
        margin=dict(t=50, l=50, r=50, b=50),
    )
    if log_spot:
        fig.update_xaxes(type="log")
    return fig

# ---------------------------
# Show call and put heatmaps side by side
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    fig_call = go.Figure()
    fig_call = add_heatmap(fig_call, call_price_grid, np.round(S_values, 2), np.round(sigma_values, 4), "Call Option Price Heatmap", "Call Price")
    st.plotly_chart(fig_call, use_container_width=True)

with col2:
    fig_put = go.Figure()
    fig_put = add_heatmap(fig_put, put_price_grid, np.round(S_values, 2), np.round(sigma_values, 4), "Put Option Price Heatmap", "Put Price")
    st.plotly_chart(fig_put, use_container_width=True)

# ---------------------------
# Heatmap 3: Option price decay for selected option type (Spot vs Time)
# ---------------------------
T_decay_values = np.linspace(chosen_T, 0, 50)
S_decay_values = np.linspace(S_min, S_max, S_steps)
S_decay_grid, T_decay_grid = np.meshgrid(S_decay_values, T_decay_values)

price_decay_grid = black_scholes_price(S_decay_grid, K, r, q, chosen_sigma, T_decay_grid, "call")

st.markdown("---")
st.markdown("### Option Price vs Underlying Asset Price and Time to Expiry (Time Decay)")

fig_decay = go.Figure()

heatmap_decay = go.Heatmap(
    z=price_decay_grid,
    x=np.round(T_decay_values, 4),
    y=np.round(S_decay_values, 2),
    colorscale=colormap.lower(),
    reversescale=False,
    hovertemplate="Time to expiry: %{x:.4f}<br>Spot: %{y}<br>Price: %{z:.4f}<extra></extra>",
    text=np.round(price_decay_grid, 2).astype(str),
    texttemplate="%{text}",
    textfont={"size":9},
)
fig_decay.add_trace(heatmap_decay)

if show_contours:
    fig_decay.add_trace(go.Contour(
        z=price_decay_grid,
        x=np.round(T_decay_values, 4),
        y=np.round(S_decay_values, 2),
        contours=dict(
            coloring="none",
            showlabels=True,
            start=np.nanmin(price_decay_grid),
            end=np.nanmax(price_decay_grid),
            size=(np.nanmax(price_decay_grid) - np.nanmin(price_decay_grid))/8,
        ),
        line_width=1,
        colorscale="Greys",
        showscale=False,
    ))

fig_decay.update_layout(
    height=650,
    xaxis_autorange='reversed',
    margin=dict(t=50, l=50, r=50, b=50),
    xaxis_title="Time to expiry (years)",
    yaxis_title="Underlying Asset Price (S)",
)

st.plotly_chart(fig_decay, use_container_width=True)

st.markdown("---")

# ---------------------------
# Show raw grids as tables for reference
# ---------------------------
call_df = pd.DataFrame(call_price_grid, index=np.round(sigma_values,8), columns=np.round(S_values,4))
call_df.index.name = "Volatility σ"
call_df.columns.name = "Spot S"

put_df = pd.DataFrame(put_price_grid, index=np.round(sigma_values,8), columns=np.round(S_values,4))
put_df.index.name = "Volatility σ"
put_df.columns.name = "Spot S"

with st.expander("Show raw call price grid (Spot vs Volatility)"):
    st.dataframe(call_df.style.format("{:.4f}"), height=300)

with st.expander("Show raw put price grid (Spot vs Volatility)"):
    st.dataframe(put_df.style.format("{:.4f}"), height=300)

# ---------------------------
# Notes
# ---------------------------
st.markdown(
    """
**Notes & tips**

- The top table shows model inputs and inspect values.

- The large colored price below the table is the option price at the chosen inspect inputs.

- Heatmaps 1 & 2 show call and put prices over spot and volatility for the fixed time to expiry.

- Heatmap 3 shows option price decay over time and spot for the selected option type.

- Contour lines can be toggled on/off.

- Let me know if you want Greeks or more features!
"""
)

