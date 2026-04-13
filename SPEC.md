# Currency Exchange Predictions AI Agent

## Project Overview
- **Project Name**: CurrencyX AI Agent
- **Type**: Web Application (AI/ML + Chatbot)
- **Core Functionality**: AI-powered currency exchange rate predictions with interactive chatbot interface
- **Target Users**: Traders, financial analysts, and anyone interested in currency exchange predictions

## Functionality Specification

### Core Features

1. **Currency Prediction Engine**
   - Support for major currency pairs: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD
   - Uses historical data patterns and machine learning (Random Forest + Linear Regression ensemble)
   - Prediction horizons: 1 day, 7 days, 30 days
   - Technical indicators: Moving averages, RSI, MACD

2. **Interactive Chatbot**
   - Natural language queries about currency rates
   - Ask for predictions and analysis
   - Market sentiment queries
   - Conversational interface for non-technical users

3. **Dashboard UI**
   - Real-time exchange rate display
   - Historical charts with predictions
   - Signal indicators (Buy/Sell/Hold)
   - Portfolio overview section

### User Interactions
- Select currency pair from dropdown
- View current rate and predicted changes
- Chat with AI assistant for insights
- View historical charts with predictions overlay

### Data Handling
- Generate synthetic historical data for demo (can be replaced with real API)
- Store predictions in session state
- Cache model results for performance

## UI/UX Specification

### Layout Structure
- **Header**: Logo, title, navigation
- **Main Content**: 
  - Left sidebar: Currency pair selector, prediction controls
  - Center: Charts and visualizations
  - Right: Chatbot panel
- **Footer**: Disclaimer and data sources

### Visual Design
- **Color Palette**:
  - Primary: #1E3A5F (Deep Navy)
  - Secondary: #2E7D32 (Forest Green - for positive)
  - Accent: #FF6B35 (Coral Orange - for alerts)
  - Background: #0D1B2A (Dark Blue)
  - Surface: #1B2838 (Slate)
  - Text: #E0E6ED (Light Gray)
- **Typography**:
  - Headings: Roboto Bold
  - Body: Roboto Regular
  - Monospace for numbers: JetBrains Mono
- **Spacing**: 16px base unit, 24px section gaps
- **Effects**: Subtle shadows on cards, smooth transitions

### Components
- Currency pair selector (dropdown)
- Prediction cards with confidence intervals
- Line charts with prediction overlay
- Chat message bubbles (user vs AI)
- Signal indicators (colored badges)

## Acceptance Criteria
1. App loads without errors
2. Currency pairs can be selected and predictions displayed
3. Charts render correctly with historical and predicted data
4. Chatbot responds to queries about currencies
5. Responsive layout works on desktop
6. No critical errors in console
