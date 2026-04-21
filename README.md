# Currency Crisis Prediction AI Agent

**HBF2212 Capstone Project**
*Great Zimbabwe University - School of Business Sciences*

An AI-powered web application that predicts currency crises using economic indicators. Features a machine learning model, interactive chatbot, and data file analysis.

## Features

- **Prediction**: Input economic indicators to get crisis probability (0-100%)
- **File Analysis**: Upload Excel/CSV files for batch analysis
- **Chat**: AI assistant for currency crisis questions
- **Dashboard**: Visualizations and historical trends

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

## Required Data Columns

For file analysis, your CSV/Excel file must contain:

```
exchange_rate_volatility         - Standard deviation of exchange rate changes
inflation_rate                   - Annual inflation percentage (%)
interest_rate_spread            - Domestic minus foreign interest rate (%)
current_account_balance_gdp      - Current account balance (% GDP)
foreign_reserves_months_imports - Foreign reserves coverage (months)
external_debt_gdp               - External debt (% GDP)
debt_service_ratio              - Debt service ratio (%)
m2_reserves_ratio              - M2 to reserves ratio
real_exchange_rate_change       - Real exchange rate change (%)
trade_balance_gdp             - Trade balance (% GDP)
```

## Sample Data

See `data/sample_data.csv` for reference.

## Tech Stack

- Python 3.9+
- Streamlit (UI)
- scikit-learn (ML)
- pandas, numpy
- plotly (Visualizations)

## Project Structure

```
CURRENCY CRISIS PREDICTION/
├── app.py              # Main Streamlit app
├── model.py            # ML prediction model
├── chatbot.py          # AI chatbot
├── requirements.txt   # Dependencies
├── README.md          # This file
├── SPEC.md             # Project specification
├── .gitignore         # Git ignore
└── data/
    └── sample_data.csv # Sample data file
```

## License

**HBF2212 Capstone Project**
**Great Zimbabwe University - School of Business Sciences**

MIT