import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

CURRENCY_PAIRS = {
    "EUR/USD": {"base": 1.08, "volatility": 0.008},
    "GBP/USD": {"base": 1.26, "volatility": 0.012},
    "USD/JPY": {"base": 148.5, "volatility": 0.7},
    "USD/CHF": {"base": 0.88, "volatility": 0.006},
    "AUD/USD": {"base": 0.66, "volatility": 0.009},
    "USD/CAD": {"base": 1.35, "volatility": 0.008},
}


def generate_historical_data(pair: str, days: int = 365) -> pd.DataFrame:
    """Generate synthetic historical currency data."""
    config = CURRENCY_PAIRS[pair]
    base_price = config["base"]
    volatility = config["volatility"]

    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(days, 0, -1)]

    np.random.seed(hash(pair) % 2**32)

    returns = np.random.normal(0.0001, volatility, days)
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices,
            "High": [p * (1 + np.random.uniform(0, volatility / 2)) for p in prices],
            "Low": [p * (1 - np.random.uniform(0, volatility / 2)) for p in prices],
            "Close": [
                p * (1 + np.random.normal(0, volatility / 4)) for p in prices[:-1]
            ]
            + [prices[-1]],
            "Volume": np.random.randint(1000000, 10000000, days),
        }
    )

    df["Close"] = df["Open"] * (
        1 + np.cumsum(np.random.normal(0, volatility / 3, days))
    )

    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the data."""
    df = df.copy()

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (std * 2)

    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=20).std() * np.sqrt(252)

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for the model."""
    df = calculate_technical_indicators(df)

    feature_cols = [
        "Close",
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_Upper",
        "BB_Lower",
        "Volatility",
    ]

    df_features = df[feature_cols].dropna()

    return df_features


def train_model(df: pd.DataFrame):
    """Train the prediction model."""
    df = calculate_technical_indicators(df)

    df_model = df.dropna()

    X = df_model[
        [
            "SMA_20",
            "SMA_50",
            "RSI",
            "MACD",
            "MACD_Signal",
            "BB_Upper",
            "BB_Lower",
            "Volatility",
        ]
    ].values
    y = df_model["Close"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    lr_model = LinearRegression()

    rf_model.fit(X_scaled, y)
    gb_model.fit(X_scaled, y)
    lr_model.fit(X_scaled, y)

    return {
        "rf": rf_model,
        "gb": gb_model,
        "lr": lr_model,
        "scaler": scaler,
        "feature_cols": [
            "SMA_20",
            "SMA_50",
            "RSI",
            "MACD",
            "MACD_Signal",
            "BB_Upper",
            "BB_Lower",
            "Volatility",
        ],
    }


def predict_future(df: pd.DataFrame, model_dict: dict, days: int = 7) -> tuple:
    """Predict future prices."""
    df = calculate_technical_indicators(df)
    last_row = df.dropna().iloc[-1:].copy()

    features = model_dict["feature_cols"]
    X = last_row[features].values
    X_scaled = model_dict["scaler"].transform(X)

    rf_pred = model_dict["rf"].predict(X_scaled)[0]
    gb_pred = model_dict["gb"].predict(X_scaled)[0]
    lr_pred = model_dict["lr"].predict(X_scaled)[0]

    ensemble_pred = rf_pred * 0.4 + gb_pred * 0.4 + lr_pred * 0.2

    config = CURRENCY_PAIRS.get(
        df["Close"].iloc[0] if "EUR/USD" in str(df.columns) else "EUR/USD",
        CURRENCY_PAIRS["EUR/USD"],
    )
    volatility = config.get("volatility", 0.01)

    last_price = df["Close"].iloc[-1]
    std_error = last_price * volatility * np.sqrt(days / 365)

    predictions = []
    current_price = last_price

    for i in range(days):
        trend = (ensemble_pred - last_price) / (days + 1)
        pred = current_price + trend * (i + 1)

        noise = np.random.normal(0, std_error / (i + 1) ** 0.5)
        pred += noise

        predictions.append(pred)
        current_price = pred

    return predictions, ensemble_pred, std_error


def get_signal(df: pd.DataFrame, prediction: float) -> dict:
    """Generate trading signal based on analysis."""
    last_close = df["Close"].iloc[-1]
    last_rsi = df["RSI"].iloc[-1]

    change_pct = ((prediction - last_close) / last_close) * 100

    if change_pct > 1.5:
        signal = "BUY"
        color = "green"
    elif change_pct < -1.5:
        signal = "SELL"
        color = "red"
    elif last_rsi > 70:
        signal = "SELL"
        color = "red"
    elif last_rsi < 30:
        signal = "BUY"
        color = "green"
    else:
        signal = "HOLD"
        color = "yellow"

    return {
        "signal": signal,
        "color": color,
        "change_pct": change_pct,
        "confidence": min(95, max(55, 75 + abs(change_pct) * 5)),
    }


def get_market_summary(pair: str) -> dict:
    """Get quick market summary."""
    df = generate_historical_data(pair, 30)
    df = calculate_technical_indicators(df)

    current = df["Close"].iloc[-1]
    change = df["Close"].iloc[-1] - df["Close"].iloc[-30]
    change_pct = (change / df["Close"].iloc[-30]) * 100

    return {
        "pair": pair,
        "current": current,
        "change": change,
        "change_pct": change_pct,
        "rsi": df["RSI"].iloc[-1],
        "macd": df["MACD"].iloc[-1],
        "trend": "Bullish"
        if df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1]
        else "Bearish",
    }


def load_excel_data(file) -> pd.DataFrame:
    """Load and validate Excel data for analysis."""
    try:
        df = pd.read_excel(file)

        required_columns = ["Date", "Close"]
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        if "Open" not in df.columns:
            df["Open"] = df["Close"]
        if "High" not in df.columns:
            df["High"] = df["Close"] * 1.01
        if "Low" not in df.columns:
            df["Low"] = df["Close"] * 0.99
        if "Volume" not in df.columns:
            df["Volume"] = 1000000

        return df

    except Exception as e:
        raise ValueError(f"Error loading Excel file: {str(e)}")


def analyze_custom_data(df: pd.DataFrame) -> dict:
    """Analyze custom uploaded data."""
    df = calculate_technical_indicators(df)

    current = df["Close"].iloc[-1]
    start = df["Close"].iloc[0]
    total_change = ((current - start) / start) * 100

    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]

    trend = "Bullish" if df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1] else "Bearish"

    volume_avg = df["Volume"].mean() if "Volume" in df.columns else 0
    volatility = df["Volatility"].iloc[-1] if "Volatility" in df.columns else 0

    return {
        "current": current,
        "start": start,
        "total_change": total_change,
        "rsi": rsi,
        "macd": macd,
        "trend": trend,
        "volume_avg": volume_avg,
        "volatility": volatility,
        "data_points": len(df),
        "date_range": f"{df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}",
    }
