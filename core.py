import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
import plotly.graph_objects as go
from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="Stock analysis ", layout="wide")
st.title("📈 Stock Analyzer :)")

LOCAL_CSV_PATH = "/mnt/data/T20-GL-gainers-allSec-22-Nov-2025.csv"

uploaded_file = st.file_uploader("Upload your NSE stock CSV (column: 'symbol' or 'ticker')", type=["csv"])
use_local = False
if uploaded_file is None and os.path.exists(LOCAL_CSV_PATH):
    use_local = st.checkbox(f"Use previously uploaded file: {os.path.basename(LOCAL_CSV_PATH)}", value=True)

if uploaded_file is None and not use_local:
    st.info("Upload a CSV or choose the previously uploaded file to proceed.")
    st.stop()

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded.")
    else:
        df = pd.read_csv(LOCAL_CSV_PATH)
        st.success(f"Loaded local CSV: {os.path.basename(LOCAL_CSV_PATH)}")
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

df.columns = df.columns.str.strip().str.lower()

if "ticker" not in df.columns and "symbol" in df.columns:
    df = df.rename(columns={"symbol": "ticker"})


if "ticker" not in df.columns:
    st.error("CSV must contain a 'ticker' or 'symbol' column. Detected columns: " + ", ".join(df.columns))
    st.stop()

# Clean tickers
df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

def to_yf_symbol(t):
    t = t.strip().upper()
    return t if "." in t else f"{t}.NS"

df["yf_symbol"] = df["ticker"].apply(to_yf_symbol)

st.subheader("CSV preview (first 10 rows)")
st.dataframe(df.head(10), width="stretch")

# -------------------------
# Controls
# -------------------------
left_col, right_col = st.columns([1, 3])
with left_col:
    st.header("Controls")

    horizon_map = {
        "1 Day": "1d",
        "5 Days": "5d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
    }
    horizon_label = st.selectbox("Time horizon (for peer charts)", options=list(horizon_map.keys()), index=4)
    period = horizon_map[horizon_label]

    all_symbols = df["yf_symbol"].tolist()
    selected = st.multiselect("Select tickers to analyze (from CSV)", options=all_symbols, default=all_symbols[:6] if len(all_symbols) >= 6 else all_symbols)

    if not selected:
        st.warning("Select at least one ticker from CSV.")
        st.stop()

    run_models = st.button("Run Tomorrow Predictions (LR, MA, RF, LSTM)")

# -------------------------
# Fetch OHLCV data (cached)
# -------------------------
@st.cache_resource(show_spinner=False, ttl=3600*6)
def download_history(symbols_list, period_label):
    # yfinance Tickers accepts space-separated string
    joined = " ".join(symbols_list)
    tickers_obj = yf.Tickers(joined)
    hist = tickers_obj.history(period=period_label, auto_adjust=False)
    return hist

with st.spinner("Downloading price data from Yahoo Finance..."):
    hist = download_history(selected, period)

if hist is None or hist.empty:
    st.error("No data returned from Yahoo Finance. Symbols may be unsupported or delisted.")
    st.stop()

# Normalize closes dataframe for multi/single symbol responses
if isinstance(hist.columns, pd.MultiIndex):
    closes = hist["Close"].copy()
else:
    closes = pd.DataFrame({selected[0]: hist["Close"]})

fetched_symbols = list(closes.columns)
failed = [s for s in selected if s not in fetched_symbols]
if failed:
    st.warning("Failed to fetch data for: " + ", ".join(failed))

# -------------------------
# Overview & normalized chart
# -------------------------
with right_col:
    st.header("Overview & Normalized Price Comparison")
    # normalized (start=1)
    normalized = closes.copy()
    for col in normalized.columns:
        first = normalized[col].dropna().iloc[0]
        if first != 0:
            normalized[col] = normalized[col] / first
        else:
            normalized[col] = normalized[col]

    norm_df = normalized.reset_index().melt(id_vars=["Date"], var_name="Stock", value_name="NormalizedPrice")
    line_chart = (
        alt.Chart(norm_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("NormalizedPrice:Q", title="Normalized price", axis=alt.Axis(format=".3f")),
            color=alt.Color("Stock:N", title="Stock"),
            tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Stock:N", title="Stock"), alt.Tooltip("NormalizedPrice:Q", format=".3f", title="Normalized")]
        )
        .properties(height=420)
    )
    st.altair_chart(line_chart, width="stretch")

# -------------------------
# Peer comparisons grid
# -------------------------
st.header("Peer comparisons — each vs peer average")
if len(fetched_symbols) <= 1:
    st.info("Need 2+ tickers for peer comparison.")
else:
    NUM_COLS = 4
    grid_cols = st.columns(NUM_COLS)
    for i, tick in enumerate(fetched_symbols):
        peers = normalized.drop(columns=[tick])
        peer_avg = peers.mean(axis=1)

        df_line = pd.DataFrame({
            "Date": normalized.index,
            tick: normalized[tick],
            "Peer avg": peer_avg
        }).melt(id_vars=["Date"], var_name="Series", value_name="Price")

        chart = (
            alt.Chart(df_line)
            .mark_line()
            .encode(
                x=alt.X("Date:T"),
                y=alt.Y("Price:Q", axis=alt.Axis(format=".3f")),
                color=alt.Color("Series:N", scale=alt.Scale(domain=[tick, "Peer avg"], range=["red", "gray"])),
                tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Series:N"), alt.Tooltip("Price:Q", format=".3f")]
            )
            .properties(title=f"{tick} vs peer avg", height=260)
        )

        cell = grid_cols[(i * 2) % NUM_COLS].container(border=True)
        cell.altair_chart(chart, width="stretch")

        # delta area chart
        df_delta = pd.DataFrame({"Date": normalized.index, "Delta": normalized[tick] - peer_avg})
        chart_delta = (
            alt.Chart(df_delta)
            .mark_area()
            .encode(
                x=alt.X("Date:T"),
                y=alt.Y("Delta:Q", axis=alt.Axis(format=".3f")),
                tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Delta:Q", format=".3f")]
            )
            .properties(title=f"{tick} - peer avg", height=260)
        )

        cell2 = grid_cols[(i * 2 + 1) % NUM_COLS].container(border=True)
        cell2.altair_chart(chart_delta, width="stretch")

# -------------------------
# Technical dashboard for a single symbol
# -------------------------
st.header("Technical dashboard — single stock analysis")
selected_single = st.selectbox("Choose a single ticker for technical analysis", options=fetched_symbols, index=0)

if selected_single:
    with st.spinner(f"Downloading full history for {selected_single}..."):
        t = yf.Ticker(selected_single)
        hist_single = t.history(period="1y")
    if hist_single is None or hist_single.empty:
        st.error("No history for selected ticker.")
    else:
        hist_single["EMA20"] = hist_single["Close"].ewm(span=20).mean()
        hist_single["EMA50"] = hist_single["Close"].ewm(span=50).mean()
        hist_single["SMA200"] = hist_single["Close"].rolling(200).mean()

        def compute_rsi(series, period=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        hist_single["RSI"] = compute_rsi(hist_single["Close"], 14)
        hist_single["EMA12"] = hist_single["Close"].ewm(span=12).mean()
        hist_single["EMA26"] = hist_single["Close"].ewm(span=26).mean()
        hist_single["MACD"] = hist_single["EMA12"] - hist_single["EMA26"]
        hist_single["Signal"] = hist_single["MACD"].ewm(span=9).mean()

        hist_single["BUY"] = np.where((hist_single["EMA20"] > hist_single["EMA50"]) & (hist_single["EMA20"].shift() <= hist_single["EMA50"].shift()), hist_single["Close"], np.nan)
        hist_single["SELL"] = np.where((hist_single["EMA20"] < hist_single["EMA50"]) & (hist_single["EMA20"].shift() >= hist_single["EMA50"].shift()), hist_single["Close"], np.nan)

        # Candlestick + EMAs + signals
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=hist_single.index, open=hist_single["Open"], high=hist_single["High"], low=hist_single["Low"], close=hist_single["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=hist_single.index, y=hist_single["EMA20"], name="EMA20"))
        fig.add_trace(go.Scatter(x=hist_single.index, y=hist_single["EMA50"], name="EMA50"))
        fig.add_trace(go.Scatter(x=hist_single.index, y=hist_single["SMA200"], name="SMA200"))
        fig.add_trace(go.Scatter(x=hist_single.index, y=hist_single["BUY"], mode="markers", marker=dict(size=8, color="green"), name="Buy"))
        fig.add_trace(go.Scatter(x=hist_single.index, y=hist_single["SELL"], mode="markers", marker=dict(size=8, color="red"), name="Sell"))
        fig.update_layout(title=f"{selected_single} — Price & Moving Averages", height=520)
        st.plotly_chart(fig, width="stretch")

        # RSI / MACD
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=hist_single.index, y=hist_single["RSI"], name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title="RSI (14)", height=260)
        st.plotly_chart(fig_rsi, width="stretch")

        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=hist_single.index, y=hist_single["MACD"], name="MACD"))
        fig_macd.add_trace(go.Scatter(x=hist_single.index, y=hist_single["Signal"], name="Signal"))
        fig_macd.update_layout(title="MACD (12,26,9)", height=260)
        st.plotly_chart(fig_macd, width="stretch")

        # KPIs
        last_close = hist_single["Close"].iloc[-1]
        prev_close = hist_single["Close"].iloc[-2] if len(hist_single) >= 2 else last_close
        pct = (last_close - prev_close) / prev_close * 100 if prev_close != 0 else 0
        vol_last = int(hist_single["Volume"].iloc[-1]) if "Volume" in hist_single.columns and not hist_single["Volume"].dropna().empty else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Last Price", f"{last_close:.2f}", delta=f"{pct:.2f}%")
        k2.metric("6M High", f"{hist_single['High'].max():.2f}")
        k3.metric("6M Low", f"{hist_single['Low'].min():.2f}")
        rsi_val = hist_single["RSI"].iloc[-1] if not hist_single["RSI"].dropna().empty else None
        k4.metric("RSI (14)", f"{rsi_val:.2f}" if rsi_val is not None else "N/A")

# -------------------------
# Tomorrow prediction models (LR, MA, RF, LSTM)
# -------------------------
st.header("🔮 Tomorrow Predictions — Linear, MA, RandomForest, LSTM (if available)")

def feature_engineering(series_close, series_high=None, series_low=None, series_vol=None):
    """
    Build DataFrame with features for modeling from close series and optional high/low/vol series.
    Returns X (features), y (target next-day close).
    """
    df_feat = pd.DataFrame({"close": series_close})
    if series_high is not None:
        df_feat["high"] = series_high
    if series_low is not None:
        df_feat["low"] = series_low
    if series_vol is not None:
        df_feat["volume"] = series_vol

    # lags
    for lag in [1,2,3,5,7]:
        df_feat[f"lag_{lag}"] = df_feat["close"].shift(lag)

    # rolling stats
    df_feat["sma_5"] = df_feat["close"].rolling(5).mean()
    df_feat["sma_10"] = df_feat["close"].rolling(10).mean()
    df_feat["ema_12"] = df_feat["close"].ewm(span=12).mean()
    df_feat["ema_26"] = df_feat["close"].ewm(span=26).mean()
    df_feat["roc_5"] = df_feat["close"].pct_change(periods=5)
    df_feat["roc_10"] = df_feat["close"].pct_change(periods=10)
    df_feat = df_feat.dropna()

    # target: next-day close
    df_feat["target"] = df_feat["close"].shift(-1)
    df_feat = df_feat.dropna()
    y = df_feat["target"]
    X = df_feat.drop(columns=["target"])
    return X, y

def train_linear_regression(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return model, scaler, rmse

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return model, rmse

def predict_ma(series_close, window=5):
    window_vals = series_close.tail(window).values
    if len(window_vals) < 2:
        return float(series_close.iloc[-1])
    slope = (window_vals[-1] - window_vals[0]) / (len(window_vals)-1)
    return float(window_vals[-1] + slope)

def train_lstm(series_close, n_steps=20, epochs=20, batch_size=16):
    if not TF_AVAILABLE:
        return None, None, None
    data = series_close.values.reshape(-1,1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    Xs, ys = [], []
    for i in range(n_steps, len(data_scaled)-1):
        Xs.append(data_scaled[i-n_steps:i, 0])
        ys.append(data_scaled[i, 0])
    Xs, ys = np.array(Xs), np.array(ys)
    if len(Xs) < 10:
        return None, None, None
    Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))
    split = int(len(Xs)*0.8)
    X_train, X_test = Xs[:split], Xs[split:]
    y_train, y_test = ys[:split], ys[split:]
    model = Sequential()
    model.add(LSTM(64, input_shape=(Xs.shape[1],1), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = model.predict(X_test).flatten()
    # inverse transform
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    preds_inv = scaler.inverse_transform(preds.reshape(-1,1)).flatten()
    rmse = mean_squared_error(y_test_inv, preds_inv, squared=False)
    return model, scaler, rmse

# Run models only when user clicks button
if run_models:
    st.info("Running models across selected tickers. This may take time depending on number of tickers and LSTM availability.")
    lr_results = []
    ma_results = []
    rf_results = []
    lstm_results = []

    for sym in fetched_symbols:
        try:
            # extract series for symbol from hist (multiindex) or single
            if isinstance(hist.columns, pd.MultiIndex):
                close_s = hist["Close"][sym].dropna()
                high_s = hist["High"][sym].dropna()
                low_s  = hist["Low"][sym].dropna()
                vol_s  = hist["Volume"][sym].dropna()
            else:
                close_s = hist["Close"].dropna()
                high_s = hist["High"].dropna()
                low_s  = hist["Low"].dropna()
                vol_s  = hist["Volume"].dropna()

            if len(close_s) < 30:
                # skip if not enough data
                continue

            # Features and target
            X, y = feature_engineering(close_s, high_s, low_s, vol_s)
            if X.empty or y.empty:
                continue

            # Linear Regression
            try:
                lr_model, lr_scaler, lr_rmse = train_linear_regression(X, y)
                last_features = X.tail(1)
                last_scaled = lr_scaler.transform(last_features)
                lr_pred = lr_model.predict(last_scaled)[0]
                lr_results.append({"Symbol": sym, "LR_Pred": round(float(lr_pred),3), "LR_RMSE": round(lr_rmse,4)})
            except Exception:
                lr_results.append({"Symbol": sym, "LR_Pred": None, "LR_RMSE": None})

            # Moving Average prediction
            try:
                ma_pred = predict_ma(close_s, window=5)
                ma_results.append({"Symbol": sym, "MA_Pred": round(float(ma_pred),3)})
            except Exception:
                ma_results.append({"Symbol": sym, "MA_Pred": None})

            # Random Forest
            try:
                rf_model, rf_rmse = train_random_forest(X, y)
                last_row = X.tail(1)
                rf_pred = rf_model.predict(last_row)[0]
                rf_results.append({"Symbol": sym, "RF_Pred": round(float(rf_pred),3), "RF_RMSE": round(rf_rmse,4)})
            except Exception:
                rf_results.append({"Symbol": sym, "RF_Pred": None, "RF_RMSE": None})

            # LSTM (optional)
            if TF_AVAILABLE:
                try:
                    lstm_model, lstm_scaler, lstm_rmse = train_lstm(close_s, n_steps=20, epochs=25, batch_size=16)
                    if lstm_model is not None:
                        seq = close_s.values[-20:].reshape(-1,1)
                        seq_s = lstm_scaler.transform(seq).reshape(1,20,1)
                        lstm_pred_scaled = lstm_model.predict(seq_s).flatten()[0]
                        lstm_pred = lstm_scaler.inverse_transform(np.array([[lstm_pred_scaled]])).flatten()[0]
                        lstm_results.append({"Symbol": sym, "LSTM_Pred": round(float(lstm_pred),3), "LSTM_RMSE": round(lstm_rmse,4)})
                    else:
                        lstm_results.append({"Symbol": sym, "LSTM_Pred": None, "LSTM_RMSE": None})
                except Exception:
                    lstm_results.append({"Symbol": sym, "LSTM_Pred": None, "LSTM_RMSE": None})
            else:
                lstm_results.append({"Symbol": sym, "LSTM_Pred": None, "LSTM_RMSE": None})

        except Exception:
            # symbol-level exception: skip
            continue

    # Convert lists to DataFrames and set index
    lr_df = pd.DataFrame(lr_results)
    rf_df = pd.DataFrame(rf_results)
    ma_df = pd.DataFrame(ma_results)
    lstm_df = pd.DataFrame(lstm_results)

    # Ensure Symbol column exists in each
    for d in (lr_df, rf_df, ma_df, lstm_df):
        if "Symbol" not in d.columns:
            d["Symbol"] = None

    # Set index and add suffixes to avoid overlap, then join
    if not lr_df.empty:
        lr_df = lr_df.set_index("Symbol").add_suffix("_LR")
    else:
        lr_df = pd.DataFrame()

    if not rf_df.empty:
        rf_df = rf_df.set_index("Symbol").add_suffix("_RF")
    else:
        rf_df = pd.DataFrame()

    if not ma_df.empty:
        ma_df = ma_df.set_index("Symbol").add_suffix("_MA")
    else:
        ma_df = pd.DataFrame()

    if not lstm_df.empty:
        lstm_df = lstm_df.set_index("Symbol").add_suffix("_LSTM")
    else:
        lstm_df = pd.DataFrame()

    # Join safely (start from lr_df or empty DF)
    merged = pd.DataFrame()
    frames = [dfi for dfi in [lr_df, rf_df, ma_df, lstm_df] if not dfi.empty]
    if frames:
        merged = frames[0].join(frames[1:], how="outer") if len(frames) > 1 else frames[0].copy()
        merged = merged.reset_index().rename(columns={"index": "Symbol"})
    else:
        merged = pd.DataFrame(columns=["Symbol"])

    # Show merged and per-model tables
    st.subheader("Model Predictions — Combined Table")
    st.dataframe(merged, width="stretch")

    st.subheader("Linear Regression — Predictions & RMSE")
    st.dataframe(lr_df.reset_index().rename(columns=lambda c: c.replace("_LR","")) if not lr_df.empty else pd.DataFrame(), width="stretch")

    st.subheader("Moving Average (MA-5) — Predictions")
    st.dataframe(ma_df.reset_index().rename(columns=lambda c: c.replace("_MA","")) if not ma_df.empty else pd.DataFrame(), width="stretch")

    st.subheader("Random Forest — Predictions & RMSE")
    st.dataframe(rf_df.reset_index().rename(columns=lambda c: c.replace("_RF","")) if not rf_df.empty else pd.DataFrame(), width="stretch")

    if TF_AVAILABLE:
        st.subheader("LSTM — Predictions & RMSE")
        st.dataframe(lstm_df.reset_index().rename(columns=lambda c: c.replace("_LSTM","")) if not lstm_df.empty else pd.DataFrame(), width="stretch")
    else:
        st.info("TensorFlow not available — LSTM predictions skipped.")

# -------------------------
# Raw data & download
# -------------------------
st.header("Raw close prices (tail)")
st.dataframe(closes.tail(50), width="stretch")

st.subheader("Download processed CSV")
proc = df.copy()
proc["yf_symbol"] = proc["yf_symbol"]
csv_bytes = proc.to_csv(index=False).encode("utf-8")
st.download_button("Download processed CSV", data=csv_bytes, file_name="processed_tickers.csv", mime="text/csv")

st.success("App ready — models run (if you clicked 'Run Tomorrow Predictions').")
