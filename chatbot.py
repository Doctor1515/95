import streamlit as st
from model import CurrencyCrisisModel


class CurrencyChatbot:
    def __init__(self):
        self.crisis_model = CurrencyCrisisModel()
        self.crisis_model.train()
        self.context = self._build_context()

    def _build_context(self):
        return """
CURRENCY CRISIS REFERENCE GUIDE

A currency crisis occurs when a sudden depreciation of a country's currency creates severe 
economic instability. Key indicators include:

1. EXCHANGE RATE VOLATILITY: Measures fluctuations in the currency's value. High volatility 
   (above 15%) often signals market instability.

2. INFLATION RATE: Persistent high inflation (above 10%) erodes currency purchasing power 
   and can trigger crisis conditions.

3. INTEREST RATE SPREAD: The difference between domestic and foreign interest rates. Large 
   spreads (above 8%) can lead to capital flight.

4. CURRENT ACCOUNT BALANCE: A sustained deficit (below -5% of GDP) indicates external 
   vulnerabilities.

5. FOREIGN RESERVES: Low reserves (below 3 months of imports) leave countries vulnerable 
   to speculative attacks.

6. EXTERNAL DEBT: High external debt (above 100% of GDP) increases refinancing risk.

Historical Crises:
- 1997 Asian Financial Crisis: Thailand, Indonesia, South Korea
- 1998 Russian Financial Crisis: Ruble collapse
- 2001 Argentine Crisis: Peso devaluation
- 2022 Sri Lankan Crisis: Complete currency collapse

Prevention Strategies:
- Maintain adequate foreign reserves
- Manage external debt levels
- Implement flexible exchange rate policies
- Strengthen banking sector regulations
- Diversify economy and exports
"""

    def get_response(self, query: str) -> str:
        query_lower = query.lower()

        if "risk" in query_lower and "predict" in query_lower:
            return "To get a risk prediction, go to the 'Prediction' tab and enter the economic indicators. The model will calculate the probability of a currency crisis based on current data."

        if "predict" in query_lower or "probability" in query_lower:
            return "Use the Prediction tab to input current economic indicators. The model analyzes exchange rate volatility, inflation, interest rate spreads, foreign reserves, and external debt to calculate crisis probability."

        if "indicator" in query_lower:
            return "Key currency crisis indicators are:\n1. Exchange Rate Volatility\n2. Inflation Rate\n3. Interest Rate Spread\n4. Current Account Balance (% GDP)\n5. Foreign Reserves (months of imports)\n6. External Debt (% GDP)\n7. Debt Service Ratio\n8. M2/Reserves Ratio\n9. Real Exchange Rate Change\n10. Trade Balance (% GDP)"

        if "indicator" in query_lower or "factor" in query_lower:
            return "Key currency crisis indicators are:\n1. Exchange Rate Volatility\n2. Inflation Rate\n3. Interest Rate Spread\n4. Current Account Balance (% GDP)\n5. Foreign Reserves (months of imports)\n6. External Debt (% GDP)\n7. Debt Service Ratio\n8. M2/Reserves Ratio\n9. Real Exchange Rate Change\n10. Trade Balance (% GDP)"

        if "crisis" in query_lower and "historical" in query_lower:
            return "Major currency crises include:\n- 1997 Asian Financial Crisis (Thailand, Indonesia, South Korea)\n- 1998 Russian Financial Crisis (Ruble collapse)\n- 2001 Argentine Crisis (Peso devaluation)\n- 2022 Sri Lankan Crisis (Complete currency collapse)"

        if "asian" in query_lower or "1997" in query_lower:
            return "The 1997 Asian Financial Crisis started in Thailand when the baht was devalued. It spread to Indonesia, South Korea, Malaysia, and Philippines. Causes included high external debt, fixed exchange rates, and weak banking sectors."

        if (
            "prevent" in query_lower
            or "avoid" in query_lower
            or "protect" in query_lower
        ):
            return "Currency crisis prevention strategies:\n1. Maintain adequate foreign reserves (6+ months of imports)\n2. Manage external debt levels below 60% of GDP\n3. Implement flexible exchange rate policies\n4. Strengthen banking sector regulations\n5. Diversify economy and exports\n6. Maintain reasonable inflation rates"

        if "inflation" in query_lower:
            return "Inflation is a key crisis indicator. High inflation (above 10%) erodes currency purchasing power. Central banks should maintain inflation targeting to preserve currency stability."

        if "reserve" in query_lower or "foreign reserve" in query_lower:
            return "Foreign reserves are critical for defending currency. Countries should maintain at least 6 months of import coverage. Low reserves (below 3 months) indicate high vulnerability to crisis."

        if "debt" in query_lower:
            return "External debt is a major vulnerability factor. High external debt (above 100% of GDP) increases refinancing risk. The debt service ratio should remain below 20% of exports."

        if "exchange" in query_lower:
            return "Exchange rate volatility above 15% often signals market instability. Fixed exchange rates can be particularly vulnerable to speculative attacks. Many experts recommend managed floating regimes."

        if "model" in query_lower or "how" in query_lower and "work" in query_lower:
            return "Our model uses Gradient Boosting ensemble with 10 economic indicators. It was trained on synthetic data mimicking historical crisis conditions. The model achieves high accuracy in identifying crisis vulnerability."

        if "hello" in query_lower or "hi" in query_lower or "hey" in query_lower:
            return "Hello! I'm your Currency Crisis Prediction Assistant. I can help you understand currency crisis indicators, historical events, risk assessment, and prevention strategies. What would you like to know?"

        if "help" in query_lower:
            return "I can help you with:\n- Understanding currency crisis indicators\n- Interpreting risk predictions\n- Historical crisis analysis\n- Prevention strategies\n- Model explanations\n\nJust ask me a question!"

        return f"I understand you're asking about: '{query}'. For detailed analysis, try the Prediction tab to input current economic indicators. I can also explain key currency crisis indicators, historical events, or prevention strategies. Can you be more specific?"

    def get_indicator_guidance(self) -> dict:
        return {
            "exchange_rate_volatility": "Current value as standard deviation of exchange rate changes. Values > 15% indicate high volatility.",
            "inflation_rate": "Annual inflation percentage. Values > 10% indicate high inflation risk.",
            "interest_rate_spread": "Difference between domestic and foreign rates. Values > 8% may cause capital flight.",
            "current_account_balance_gdp": "Current account as % of GDP. Values < -5% indicate external vulnerability.",
            "foreign_reserves_months_imports": "Months of imports covered by reserves. Values < 3 indicate critical shortage.",
            "external_debt_gdp": "External debt as % of GDP. Values > 100% indicate high debt vulnerability.",
        }


def initialize_chatbot():
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = CurrencyChatbot()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your Currency Crisis Prediction Assistant. I can help you understand currency crisis indicators, analyze risk factors, and answer questions about historical crises. What would you like to know?",
            }
        ]
    return st.session_state.chatbot
