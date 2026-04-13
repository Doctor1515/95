import re
import random
from prediction_model import CURRENCY_PAIRS, get_market_summary


class CurrencyChatbot:
    """AI Chatbot for currency exchange queries."""

    def __init__(self):
        self.context = {}

    def process_message(self, message: str, selected_pair: str = "EUR/USD") -> str:
        """Process user message and generate response."""
        message = message.lower().strip()

        if any(word in message for word in ["hello", "hi", "hey", "greetings"]):
            return self._greeting()

        if any(word in message for word in ["help", "what can you do", "commands"]):
            return self._help()

        pair = self._detect_currency_pair(message)
        if not pair:
            pair = selected_pair

        if any(
            word in message
            for word in ["predict", "prediction", "forecast", "will", "future"]
        ):
            return self._prediction_response(pair, message)

        if any(
            word in message for word in ["current", "now", "today", "price", "rate"]
        ):
            return self._current_rate_response(pair)

        if any(
            word in message
            for word in ["trend", "bullish", "bearish", "going", "direction"]
        ):
            return self._trend_response(pair)

        if any(
            word in message for word in ["buy", "sell", "signal", "trade", "should i"]
        ):
            return self._signal_response(pair)

        if any(word in message for word in ["rsi", "macd", "technical", "indicator"]):
            return self._technical_response(pair, message)

        if any(word in message for word in ["analysis", "analyze", "opinion"]):
            return self._analysis_response(pair)

        if any(word in message for word in ["risk", "volatility", "volatile"]):
            return self._risk_response(pair)

        return self._fallback_response()

    def _greeting(self) -> str:
        greetings = [
            "Hello! I'm CurrencyX AI. I can help you with currency exchange predictions and market analysis. What would you like to know?",
            "Hi there! I specialize in currency exchange analysis. Ask me about predictions, trends, or trading signals!",
            "Welcome! I'm here to help with your currency exchange questions. Try asking about predictions or market analysis.",
        ]
        return random.choice(greetings)

    def _help(self) -> str:
        return """
I can help you with:

• **Predictions** - Ask "What will EUR/USD be tomorrow?"
• **Current Rates** - Ask "What's the current EUR/USD rate?"
• **Trends** - Ask "What's the trend for GBP/USD?"
• **Trading Signals** - Ask "Should I buy or sell EUR/USD?"
• **Technical Analysis** - Ask about RSI, MACD, or indicators
• **Market Analysis** - Ask for a full analysis of any pair

Just type your question naturally!"""

    def _detect_currency_pair(self, message: str) -> str | None:
        """Detect currency pair mentioned in message."""
        pairs = {
            "eur/usd": "EUR/USD",
            "eurusd": "EUR/USD",
            "gbp/usd": "GBP/USD",
            "gbpusd": "GBP/USD",
            "usd/jpy": "USD/JPY",
            "usdjpy": "USD/JPY",
            "usd/chf": "USD/CHF",
            "usdchf": "USD/CHF",
            "aud/usd": "AUD/USD",
            "audusd": "AUD/USD",
            "usd/cad": "USD/CAD",
            "usdcad": "USD/CAD",
        }

        for key, value in pairs.items():
            if key in message:
                return value
        return None

    def _prediction_response(self, pair: str, message: str) -> str:
        summary = get_market_summary(pair)

        days = 7
        if "tomorrow" in message or "1 day" in message:
            days = 1
        elif "week" in message or "7 day" in message:
            days = 7
        elif "month" in message or "30 day" in message:
            days = 30

        change = (
            random.uniform(-2.5, 2.5)
            if "random" in message
            else summary["change_pct"] * 0.5
        )
        predicted = summary["current"] * (1 + change / 100)

        direction = "up" if change > 0 else "down"

        return f"""
Based on my analysis of {pair}:

📊 **Prediction for next {days} day(s)**:
• Current: {summary["current"]:.5f}
• Predicted: {predicted:.5f}
• Expected change: {change:.2f}% ({direction})

The model shows a {abs(change):.1f}% {"potential increase" if change > 0 else "potential decrease"} over the forecast period.

*Note: This is a simulated prediction for demonstration purposes."""

    def _current_rate_response(self, pair: str) -> str:
        summary = get_market_summary(pair)

        emoji = "📈" if summary["change_pct"] > 0 else "📉"

        return f"""
Current {pair} Rate:
━━━━━━━━━━━━━━━━━━━━
• Rate: {summary["current"]:.5f}
• Change: {summary["change_pct"]:.2f}% {emoji}
• Trend: {summary["trend"]}
• RSI (14): {summary["rsi"]:.1f}
• MACD: {summary["macd"]:.5f}

Data is simulated for demonstration."""

    def _trend_response(self, pair: str) -> str:
        summary = get_market_summary(pair)

        trend_desc = (
            "upward momentum" if summary["trend"] == "Bullish" else "downward pressure"
        )

        return f"""
{pair} Trend Analysis:
━━━━━━━━━━━━━━━━━━━━━
• Current Trend: **{summary["trend"]}**
• Direction: {trend_desc}
• Recent Change: {summary["change_pct"]:.2f}%
• RSI: {summary["rsi"]:.1f} ({"Overbought" if summary["rsi"] > 70 else "Oversold" if summary["rsi"] < 30 else "Neutral"})

The market is showing {"positive" if summary["trend"] == "Bullish" else "negative"} signals."""

    def _signal_response(self, pair: str) -> str:
        summary = get_market_summary(pair)

        if summary["rsi"] > 70:
            signal = "SELL"
            reason = "RSI indicates overbought conditions"
        elif summary["rsi"] < 30:
            signal = "BUY"
            reason = "RSI indicates oversold conditions"
        elif summary["change_pct"] > 2:
            signal = "SELL"
            reason = "Significant upward movement may reverse"
        elif summary["change_pct"] < -2:
            signal = "BUY"
            reason = "Significant drop may present buying opportunity"
        else:
            signal = "HOLD"
            reason = "Market appears stable, wait for clearer signals"

        return f"""
{pair} Trading Signal:
━━━━━━━━━━━━━━━━━━━━━
🔔 **Signal: {signal}**

Reason: {reason}

Current RSI: {summary["rsi"]:.1f}
Recent Change: {summary["change_pct"]:.2f}%

*This is for educational purposes only. Not financial advice."""

    def _technical_response(self, pair: str, message: str) -> str:
        summary = get_market_summary(pair)

        if "rsi" in message:
            rsi_val = summary["rsi"]
            interpretation = (
                "overbought"
                if rsi_val > 70
                else "oversold"
                if rsi_val < 30
                else "neutral"
            )
            return f"""
RSI (Relative Strength Index) for {pair}:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Current RSI: {rsi_val:.1f}
• Interpretation: **{interpretation}**

RSI Scale:
• Above 70 = Overbought (potential sell signal)
• Below 30 = Oversold (potential buy signal)
• 30-70 = Neutral"""

        if "macd" in message:
            return f"""
MACD for {pair}:
━━━━━━━━━━━━━━━━
• Current MACD: {summary["macd"]:.5f}

Interpretation:
• MACD > Signal Line = Bullish
• MACD < Signal Line = Bearish"""

        return f"""
Technical Indicators for {pair}:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• RSI (14): {summary["rsi"]:.1f}
• MACD: {summary["macd"]:.5f}
• Trend: {summary["trend"]}

RSI > 70 suggests overbought, RSI < 30 suggests oversold."""

    def _analysis_response(self, pair: str) -> str:
        summary = get_market_summary(pair)

        return f"""
{pair} Market Analysis:
━━━━━━━━━━━━━━━━━━━━━━
📊 **Overview**:
• Current Rate: {summary["current"]:.5f}
• Change: {summary["change_pct"]:.2f}%
• Trend: {summary["trend"]}

📈 **Technical Signals**:
• RSI: {summary["rsi"]:.1f} ({"Overbought" if summary["rsi"] > 70 else "Oversold" if summary["rsi"] < 30 else "Neutral"})
• MACD: {summary["macd"]:.5f}

⚡ **Summary**: The pair is showing {"bullish" if summary["trend"] == "Bullish" else "bearish"} momentum with {"strong" if abs(summary["rsi"] - 50) > 20 else "moderate"} indicators.

*This is simulated data for demonstration."""

    def _risk_response(self, pair: str) -> str:
        config = CURRENCY_PAIRS.get(pair, {"volatility": 0.01})
        vol = config["volatility"] * 100

        risk_level = "Low" if vol < 1 else "Medium" if vol < 1.5 else "High"

        return f"""
{pair} Risk Assessment:
━━━━━━━━━━━━━━━━━━━━━━
• Volatility: {vol:.2f}%
• Risk Level: **{risk_level}**

Volatility Guide:
• < 1%: Low Risk
• 1-1.5%: Medium Risk
• > 1.5%: High Risk

Higher volatility means larger potential swings in either direction."""

    def _fallback_response(self) -> str:
        responses = [
            "I'm not sure about that. Try asking about predictions, current rates, or trading signals!",
            "Could you rephrase that? I can help with currency predictions, trends, and analysis.",
            "I specialize in currency exchange analysis. Ask me something like 'What's the EUR/USD prediction?' or 'Should I buy GBP/USD?'",
        ]
        return random.choice(responses)
