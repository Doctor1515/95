import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from prediction_model import (
    generate_historical_data,
    calculate_technical_indicators,
    train_model,
    predict_future,
    get_signal,
    get_market_summary,
    CURRENCY_PAIRS,
    load_excel_data,
    analyze_custom_data,
)
from chatbot import CurrencyChatbot

st.set_page_config(
    page_title="CurrencyX AI - Exchange Predictions",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E7D32;
    }
    .metric-card {
        background: linear-gradient(135deg, #1B2838 0%, #0D1B2A 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2E7D32;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .chat-container {
        background: #1B2838;
        border-radius: 12px;
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .user-message {
        background: #1E3A5F;
        padding: 0.75rem 1rem;
        border-radius: 12px 12px 0 12px;
        margin: 0.5rem 0;
        color: #E0E6ED;
    }
    .ai-message {
        background: #2E7D32;
        padding: 0.75rem 1rem;
        border-radius: 12px 12px 12px 0;
        margin: 0.5rem 0;
        color: #E0E6ED;
    }
    .signal-buy { color: #2E7D32; font-weight: bold; }
    .signal-sell { color: #FF6B35; font-weight: bold; }
    .signal-hold { color: #FFC107; font-weight: bold; }
    .stApp {
        background: #0D1B2A;
    }
    .stMarkdown {
        color: #E0E6ED;
    }
    div[data-testid="stMetric"] {
        background: #1B2838;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2E7D32;
    }
    div[data-testid="stMetricLabel"] {
        color: #E0E6ED;
    }
    div[data-testid="stMetricValue"] {
        color: #2E7D32;
    }
    .css-1d391kg {
        background: #0D1B2A;
    }
    section[data-testid="stSidebar"] {
        background: #1B2838;
    }
    .chat-input input {
        background: #0D1B2A;
        color: #E0E6ED;
    }
</style>
""",
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chatbot" not in st.session_state:
    st.session_state.chatbot = CurrencyChatbot()
if "selected_pair" not in st.session_state:
    st.session_state.selected_pair = "EUR/USD"
if "custom_data" not in st.session_state:
    st.session_state.custom_data = None


def create_chart(df: pd.DataFrame, predictions: list, pair: str):
    """Create interactive chart with historical data and predictions."""
    fig = go.Figure()

    dates = pd.to_datetime(df["Date"])

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["Close"],
            mode="lines",
            name="Historical",
            line=dict(color="#1E3A5F", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["SMA_20"],
            mode="lines",
            name="SMA 20",
            line=dict(color="#FF6B35", width=1, dash="dot"),
            opacity=0.7,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["SMA_50"],
            mode="lines",
            name="SMA 50",
            line=dict(color="#2E7D32", width=1, dash="dot"),
            opacity=0.7,
        )
    )

    if predictions:
        pred_dates = [
            dates.iloc[-1] + timedelta(days=i + 1) for i in range(len(predictions))
        ]

        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=predictions,
                mode="lines+markers",
                name="Predicted",
                line=dict(color="#FF6B35", width=2),
                marker=dict(size=6, symbol="diamond"),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{pair} - Price History & Predictions",
            font=dict(size=18, color="#E0E6ED"),
        ),
        xaxis=dict(title="Date", gridcolor="#1B2838", tickfont=dict(color="#E0E6ED")),
        yaxis=dict(title="Price", gridcolor="#1B2838", tickfont=dict(color="#E0E6ED")),
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        font=dict(color="#E0E6ED"),
        legend=dict(bgcolor="#1B2838", bordercolor="#2E7D32", borderwidth=1),
        height=450,
        margin=dict(l=60, r=30, t=60, b=60),
    )

    return fig


def create_rsi_chart(df: pd.DataFrame):
    """Create RSI chart."""
    fig = go.Figure()

    dates = pd.to_datetime(df["Date"])

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color="#9C27B0", width=2),
        )
    )

    fig.add_hline(
        y=70, line_dash="dash", line_color="#FF6B35", annotation_text="Overbought"
    )
    fig.add_hline(
        y=30, line_dash="dash", line_color="#2E7D32", annotation_text="Oversold"
    )

    fig.update_layout(
        title=dict(text="RSI (14)", font=dict(size=14, color="#E0E6ED")),
        xaxis=dict(gridcolor="#1B2838", tickfont=dict(color="#E0E6ED")),
        yaxis=dict(range=[0, 100], gridcolor="#1B2838", tickfont=dict(color="#E0E6ED")),
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        font=dict(color="#E0E6ED"),
        height=250,
        margin=dict(l=40, r=20, t=40, b=30),
    )

    return fig


def create_macd_chart(df: pd.DataFrame):
    """Create MACD chart."""
    fig = go.Figure()

    dates = pd.to_datetime(df["Date"])

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["MACD"],
            mode="lines",
            name="MACD",
            line=dict(color="#2196F3", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["MACD_Signal"],
            mode="lines",
            name="Signal",
            line=dict(color="#FF6B35", width=2),
        )
    )

    fig.add_bar(
        x=dates,
        y=df["MACD"] - df["MACD_Signal"],
        name="Histogram",
        marker_color="#2E7D32",
        opacity=0.5,
    )

    fig.update_layout(
        title=dict(text="MACD", font=dict(size=14, color="#E0E6ED")),
        xaxis=dict(gridcolor="#1B2838", tickfont=dict(color="#E0E6ED")),
        yaxis=dict(gridcolor="#1B2838", tickfont=dict(color="#E0E6ED")),
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        font=dict(color="#E0E6ED"),
        height=250,
        margin=dict(l=40, r=20, t=40, b=30),
    )

    return fig


def handle_chat(message: str):
    """Handle chat message input."""
    if message.strip():
        st.session_state.chat_history.append({"role": "user", "content": message})

        response = st.session_state.chatbot.process_message(
            message, st.session_state.selected_pair
        )

        time.sleep(0.3)

        st.session_state.chat_history.append({"role": "ai", "content": response})


def main():
    st.markdown(
        '<div class="main-header">💱 CurrencyX AI Agent</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align:center; color:#E0E6ED;">AI-Powered Currency Exchange Predictions & Analysis</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        data_source = st.radio(
            "Data Source", ["Demo Data", "Upload Excel"], index=0, horizontal=True
        )

        if data_source == "Upload Excel":
            st.markdown("#### 📁 Upload Your Data")
            uploaded_file = st.file_uploader(
                "Upload Excel file with columns: Date, Close (required), Open, High, Low, Volume (optional)",
                type=["xlsx", "xls"],
            )

            if uploaded_file:
                try:
                    df = load_excel_data(uploaded_file)
                    st.session_state.custom_data = df
                    st.success(f"Loaded {len(df)} rows successfully!")

                    with st.expander("Preview Data"):
                        st.dataframe(df.head(10))
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.custom_data = None
            else:
                if st.session_state.custom_data is None:
                    st.info("Please upload an Excel file to analyze your data.")
                    pair_options = list(CURRENCY_PAIRS.keys())
                    try:
                        selected_idx = pair_options.index(
                            st.session_state.selected_pair
                        )
                    except ValueError:
                        selected_idx = 0
                    st.session_state.selected_pair = st.selectbox(
                        "Select Currency Pair",
                        options=pair_options,
                        index=selected_idx,
                    )
                    st.markdown("---")
                    prediction_days = st.slider("Prediction Period (days)", 1, 30, 7)
                    st.stop()
        else:
            st.session_state.custom_data = None
            pair_options = list(CURRENCY_PAIRS.keys())
            try:
                selected_idx = pair_options.index(st.session_state.selected_pair)
            except ValueError:
                selected_idx = 0
            st.session_state.selected_pair = st.selectbox(
                "Select Currency Pair",
                options=pair_options,
                index=selected_idx,
            )

        st.markdown("---")

        prediction_days = st.slider("Prediction Period (days)", 1, 30, 7)

        st.markdown("---")

        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **CurrencyX AI** uses ensemble ML models (Random Forest + Gradient Boosting + Linear Regression) for currency predictions.
            
            **Indicators Used:**
            - SMA 20, SMA 50
            - RSI (14)
            - MACD
            - Bollinger Bands
            
            ⚠️ *Disclaimer: This is for educational purposes only. Not financial advice.*
            """)

    if st.session_state.custom_data is not None:
        df = st.session_state.custom_data
        df = calculate_technical_indicators(df)

        pair = "Custom Data"

        with st.spinner("Training model on your data..."):
            model = train_model(df)
            predictions, ensemble_pred, std_error = predict_future(
                df, model, prediction_days
            )

        analysis = analyze_custom_data(df)
        signal = get_signal(
            df, predictions[-1] if predictions else df["Close"].iloc[-1]
        )

        st.markdown(f"### 📊 Analysis of Your Data")
        st.markdown(f"**Date Range:** {analysis['date_range']}")
        st.markdown(f"**Data Points:** {analysis['data_points']}")

        current_price = analysis["current"]
        current_change = analysis["total_change"]
        rsi_val = analysis["rsi"]
        trend_val = analysis["trend"]

    else:
        pair = st.session_state.selected_pair

        with st.spinner("Loading market data..."):
            df = generate_historical_data(pair, 365)
            df = calculate_technical_indicators(df)
            model = train_model(df)
            predictions, ensemble_pred, std_error = predict_future(
                df, model, prediction_days
            )
            signal = get_signal(
                df, predictions[-1] if predictions else df["Close"].iloc[-1]
            )
            summary = get_market_summary(pair)

        current_price = summary["current"]
        current_change = summary["change_pct"]
        rsi_val = summary["rsi"]
        trend_val = summary["trend"]

    pred_change = (
        ((predictions[-1] - current_price) / current_price) * 100 if predictions else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Rate", f"{current_price:.5f}", f"{current_change:.2f}%")

    with col2:
        st.metric(
            f"{prediction_days}-Day Prediction",
            f"{predictions[-1]:.5f}" if predictions else "N/A",
            f"{pred_change:.2f}%",
        )

    with col3:
        signal_colors = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        st.metric(
            "Trading Signal", f"{signal_colors[signal['signal']]} {signal['signal']}"
        )

    with col4:
        confidence = signal["confidence"]
        st.metric("Confidence", f"{confidence:.0f}%")

    tab1, tab2, tab3 = st.tabs(
        ["📈 Price Chart", "📊 Technical Indicators", "💬 AI Chatbot"]
    )

    with tab1:
        st.plotly_chart(create_chart(df, predictions, pair), use_container_width=True)

        st.markdown("### Prediction Details")

        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <h4 style="color:#2E7D32;">Price Forecast</h4>
                <p><strong>Current:</strong> {current_price:.5f}</p>
                <p><strong>{prediction_days}-Day Target:</strong> {predictions[-1]:.5f}</p>
                <p><strong>Expected Change:</strong> {pred_change:+.2f}%</p>
                <p><strong>Standard Error:</strong> ±{std_error:.5f}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with pred_col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <h4 style="color:#2E7D32;">Signal Analysis</h4>
                <p><strong>Signal:</strong> <span class="signal-{signal["color"].lower()}">{signal["signal"]}</span></p>
                <p><strong>Confidence:</strong> {signal["confidence"]:.0f}%</p>
                <p><strong>RSI:</strong> {rsi_val:.1f}</p>
                <p><strong>Trend:</strong> {trend_val}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("### Forecast Table")
        forecast_df = pd.DataFrame(
            {
                "Day": range(1, prediction_days + 1),
                "Predicted Price": predictions,
                "Lower Bound": [
                    p - std_error * (i + 1) ** 0.5 for i, p in enumerate(predictions)
                ],
                "Upper Bound": [
                    p + std_error * (i + 1) ** 0.5 for i, p in enumerate(predictions)
                ],
            }
        )
        st.dataframe(
            forecast_df.style.format(
                {
                    "Predicted Price": "{:.5f}",
                    "Lower Bound": "{:.5f}",
                    "Upper Bound": "{:.5f}",
                }
            ),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(create_rsi_chart(df.tail(90)), use_container_width=True)
        st.plotly_chart(create_macd_chart(df.tail(90)), use_container_width=True)

        with st.expander("View All Technical Indicators"):
            st.dataframe(
                df[
                    [
                        "Date",
                        "Close",
                        "SMA_20",
                        "SMA_50",
                        "RSI",
                        "MACD",
                        "MACD_Signal",
                        "BB_Upper",
                        "BB_Lower",
                    ]
                ]
                .tail(30)
                .style.format(
                    {
                        "Close": "{:.5f}",
                        "SMA_20": "{:.5f}",
                        "SMA_50": "{:.5f}",
                        "RSI": "{:.1f}",
                        "MACD": "{:.5f}",
                        "MACD_Signal": "{:.5f}",
                        "BB_Upper": "{:.5f}",
                        "BB_Lower": "{:.5f}",
                    }
                ),
                use_container_width=True,
            )

    with tab3:
        st.markdown("### 💬 Chat with CurrencyX AI")
        st.markdown("*Ask me about predictions, trends, trading signals, and more!*")

        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-message">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="ai-message">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("---")

        col_input, col_send = st.columns([5, 1])

        with col_input:
            user_input = st.text_input(
                "Type your message...",
                placeholder="e.g., What will EUR/USD be tomorrow?",
                label_visibility="collapsed",
                key="chat_input",
            )

        with col_send:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Send", type="primary"):
                handle_chat(user_input)
                st.rerun()

        st.markdown("---")

        quick_actions = st.columns(4)

        with quick_actions[0]:
            if st.button("📊 Predictions", use_container_width=True):
                handle_chat(f"What is the prediction for {pair}?")
                st.rerun()

        with quick_actions[1]:
            if st.button("📈 Current Rate", use_container_width=True):
                handle_chat(f"What is the current {pair} rate?")
                st.rerun()

        with quick_actions[2]:
            if st.button("🎯 Trading Signal", use_container_width=True):
                handle_chat(f"Should I buy or sell {pair}?")
                st.rerun()

        with quick_actions[3]:
            if st.button("📋 Full Analysis", use_container_width=True):
                handle_chat(f"Give me a full analysis of {pair}")
                st.rerun()

    st.markdown("---")
    st.markdown(
        """
    <div style="text-align:center; color:#666; font-size:0.8rem;">
        <p>⚠️ Disclaimer: This application is for educational and demonstration purposes only.</p>
        <p>CurrencyX AI does not provide financial advice. All predictions are simulated.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
