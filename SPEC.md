# Currency Crisis Prediction AI Agent

## Project Overview

- **Project Name**: Currency Crisis Prediction Agent
- **Type**: AI-powered web application with ML prediction and chatbot
- **Core Functionality**: Predicts currency crises using economic indicators, provides conversational interface for analysis, deployable on Streamlit
- **Target Users**: Financial analysts, economists, policymakers, traders

## Technical Stack

- **Language**: Python 3.9+
- **ML Framework**: scikit-learn, XGBoost, pandas
- **Chatbot**: LlamaIndex (RAG-based conversation)
- **UI**: Streamlit
- **Version Control**: Git/GitHub

## Functionality Specification

### 1. Currency Crisis Prediction Model
- Uses historical currency data with indicators:
  - Exchange rate volatility
  - Inflation rate
  - Interest rate spreads
  - Current account balance
  - Foreign reserves
  - External debt
- Binary classification: crisis vs stable
- Ensemble model (Random Forest + XGBoost)

### 2. Chatbot Interface
- RAG-based conversational AI
- Context-aware responses about:
  - Currency crisis indicators
  - Risk assessment
  - Historical crises analysis
  - Prevention strategies
- Integrates prediction model insights

### 3. Streamlit Dashboard
- **Tab 1: Prediction** - Input economic indicators, get crisis probability
- **Tab 2: Chat** - Interactive chatbot
- **Tab 3: Dashboard** - Visualize historical data and trends

### 4. GitHub Integration
- Git repository initialization
- Proper commit history tracking
- README and documentation

## File Structure

```
CURRENCY CRISIS PREDICTION/
├── app.py                 # Streamlit app
├── model.py              # ML prediction model
├── chatbot.py            # LlamaIndex chatbot
├── data/
│   └── sample_data.csv   # Sample currency data
├── requirements.txt      # Dependencies
└── .gitignore           # Git ignore
```

## Acceptance Criteria

1. Streamlit app runs without errors
2. Prediction model accepts input and outputs crisis probability
3. Chatbot responds to currency crisis queries
4. Dashboard displays data visualizations
5. GitHub repository initialized with proper structure