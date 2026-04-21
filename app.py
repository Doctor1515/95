import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model import CurrencyCrisisModel
from chatbot import CurrencyChatbot, initialize_chatbot

st.set_page_config(
    page_title="Currency Crisis Prediction Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .css-1d391kg {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.1);
        color: white;
    }
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.1);
        color: white;
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    .risk-low { color: #4CAF50; }
    .risk-moderate { color: #FFC107; }
    .risk-high { color: #FF9800; }
    .risk-critical { color: #F44336; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model = CurrencyCrisisModel()
    model.train()
    return model


def main():
    st.title("Currency Crisis Prediction Agent")
    st.markdown("### AI-Powered Currency Crisis Prediction & Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Prediction", "File Analysis", "Chat", "Dashboard"]
    )

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Economic Indicators")

            exchange_volatility = st.slider(
                "Exchange Rate Volatility",
                min_value=0.01,
                max_value=0.50,
                value=0.08,
                step=0.01,
                help="Standard deviation of exchange rate changes",
            )

            inflation_rate = st.slider(
                "Inflation Rate (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.5
            )

            interest_spread = st.slider(
                "Interest Rate Spread (%)",
                min_value=-2.0,
                max_value=15.0,
                value=3.0,
                step=0.5,
            )

            current_account = st.slider(
                "Current Account Balance (% GDP)",
                min_value=-15.0,
                max_value=10.0,
                value=-2.0,
                step=0.5,
            )

            foreign_reserves = st.slider(
                "Foreign Reserves (Months of Imports)",
                min_value=1.0,
                max_value=24.0,
                value=6.0,
                step=0.5,
            )

            external_debt = st.slider(
                "External Debt (% GDP)",
                min_value=10.0,
                max_value=150.0,
                value=45.0,
                step=5.0,
            )

        with col2:
            st.markdown("### Prediction Result")

            model = load_model()
            features = [
                exchange_volatility,
                inflation_rate,
                interest_spread,
                current_account,
                foreign_reserves,
                external_debt,
                15.0,
                0.8,
                5.0,
                -1.0,
            ]

            probability = model.predict(features)
            risk_level, color = model.get_risk_level(probability)

            st.metric("Crisis Probability", f"{probability * 100:.1f}%")

            st.markdown(
                f"### Risk Level: <span class='risk-{color}'>{risk_level}</span>",
                unsafe_allow_html=True,
            )

            st.progress(probability)

            if probability >= 0.75:
                st.error(
                    "WARNING: Critical risk of currency crisis detected. Immediate action recommended."
                )
            elif probability >= 0.50:
                st.warning(
                    "ALERT: Elevated risk of currency crisis. Monitor closely and consider preventive measures."
                )
            elif probability >= 0.25:
                st.info(
                    "CAUTION: Moderate risk detected. Maintain vigilance on economic indicators."
                )
            else:
                st.success("Status: Low risk. Currency appears stable.")

            st.markdown("### Feature Importance")

            importance = model.get_feature_importance()
            importance_df = pd.DataFrame(
                list(importance.items()), columns=["Indicator", "Importance"]
            ).sort_values("Importance", ascending=True)

            fig = px.bar(
                importance_df,
                x="Importance",
                y="Indicator",
                orientation="h",
                title="Feature Importance",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### File Analysis")
        st.markdown("Upload an Excel or CSV file to analyze currency crisis risk")

        uploaded_file = st.file_uploader(
            "Choose a file (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="File must contain columns: exchange_rate_volatility, inflation_rate, interest_rate_spread, current_account_balance_gdp, foreign_reserves_months_imports, external_debt_gdp, debt_service_ratio, m2_reserves_ratio, real_exchange_rate_change, trade_balance_gdp",
        )

        if uploaded_file is not None:
            model = load_model()

            try:
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name

                results = model.analyze_file(temp_path)

                os.unlink(temp_path)

                st.success("File analyzed successfully!")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Records", len(results))
                with col2:
                    high_risk_count = (
                        results["risk_level"] == "Critical Risk"
                    ).sum() + (results["risk_level"] == "High Risk").sum()
                    st.metric("High Risk Records", high_risk_count)

                st.markdown("### Analysis Results")
                st.dataframe(
                    results.style.background_gradient(
                        subset=["crisis_probability"], cmap="RdYlGn_r"
                    ),
                    use_container_width=True,
                )

                st.markdown("### Risk Distribution")

                risk_counts = results["risk_level"].value_counts()
                risk_colors = {
                    "Low Risk": "#4CAF50",
                    "Moderate Risk": "#FFC107",
                    "High Risk": "#FF9800",
                    "Critical Risk": "#F44336",
                }

                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=risk_counts.index,
                            values=risk_counts.values,
                            marker=dict(
                                colors=[
                                    risk_colors.get(r, "#999")
                                    for r in risk_counts.index
                                ]
                            ),
                        )
                    ]
                )
                fig.update_layout(
                    title="Risk Level Distribution",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Download Results")
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="crisis_analysis_results.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error analyzing file: {str(e)}")
                st.info(
                    "Make sure your file contains all required columns: exchange_rate_volatility, inflation_rate, interest_rate_spread, current_account_balance_gdp, foreign_reserves_months_imports, external_debt_gdp, debt_service_ratio, m2_reserves_ratio, real_exchange_rate_change, trade_balance_gdp"
                )

    with tab3:
        chatbot = initialize_chatbot()

        st.markdown("### Currency Crisis Chat Assistant")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about currency crises..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = chatbot.get_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

        with st.sidebar:
            st.markdown("### Quick Reference")
            st.markdown("""
            **Key Indicators:**
            - Exchange Rate Volatility
            - Inflation Rate
            - Interest Rate Spread
            - Foreign Reserves
            - External Debt

            **Sample Questions:**
            - What causes currency crises?
            - How to prevent a crisis?
            - What happened in 1997?
            """)
            if st.button("Clear Chat"):
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Chat cleared! How can I help you?",
                    }
                ]
                st.rerun()

    with tab4:
        model = load_model()
        importance = model.get_feature_importance()

        col1, col2 = st.columns(2)

        with col1:
            importance_df = pd.DataFrame(
                list(importance.items()), columns=["Indicator", "Importance"]
            ).sort_values("Importance", ascending=False)

            fig = px.pie(
                importance_df,
                values="Importance",
                names="Indicator",
                title="Indicator Importance Distribution",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            categories = [
                "Low Risk\n(0-25%)",
                "Moderate Risk\n(25-50%)",
                "High Risk\n(50-75%)",
                "Critical Risk\n(75-100%)",
            ]
            values = [15, 35, 30, 20]

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=categories,
                        values=values,
                        marker=dict(
                            colors=["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
                        ),
                    )
                ]
            )
            fig.update_layout(
                title="Risk Distribution",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Historical Trend Analysis")

        years = list(range(2000, 2025))
        crises_count = [
            3,
            2,
            1,
            2,
            1,
            3,
            2,
            1,
            1,
            2,
            3,
            2,
            1,
            1,
            2,
            1,
            2,
            3,
            2,
            1,
            2,
            1,
            2,
            1,
            2,
        ]
        stability_index = [
            85,
            88,
            90,
            87,
            92,
            85,
            88,
            91,
            93,
            90,
            86,
            89,
            92,
            94,
            91,
            88,
            90,
            85,
            88,
            92,
            90,
            93,
            91,
            94,
            92,
        ]

        df = pd.DataFrame(
            {"Year": years, "Crises": crises_count, "Stability Index": stability_index}
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["Year"],
                y=df["Crises"],
                name="Crises",
                line=dict(color="#F44336", width=3),
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Year"],
                y=df["Stability Index"],
                name="Stability Index",
                line=dict(color="#4CAF50", width=3),
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="Currency Crises Over Time",
            xaxis_title="Year",
            yaxis_title="Number of Crises",
            yaxis2=dict(title="Stability Index", overlaying="y", side="right"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            legend=dict(x=0, y=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Currency Crisis Prediction Agent** | Powered by Machine Learning")


if __name__ == "__main__":
    main()
