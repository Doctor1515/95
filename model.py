import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os


class CurrencyCrisisModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            "exchange_rate_volatility",
            "inflation_rate",
            "interest_rate_spread",
            "current_account_balance_gdp",
            "foreign_reserves_months_imports",
            "external_debt_gdp",
            "debt_service_ratio",
            "m2_reserves_ratio",
            "real_exchange_rate_change",
            "trade_balance_gdp",
        ]
        self.trained = False

    def generate_sample_data(self):
        np.random.seed(42)
        n_samples = 1000

        data = {
            "exchange_rate_volatility": np.random.uniform(0.01, 0.25, n_samples),
            "inflation_rate": np.random.uniform(1, 25, n_samples),
            "interest_rate_spread": np.random.uniform(-2, 15, n_samples),
            "current_account_balance_gdp": np.random.uniform(-10, 5, n_samples),
            "foreign_reserves_months_imports": np.random.uniform(1, 24, n_samples),
            "external_debt_gdp": np.random.uniform(10, 150, n_samples),
            "debt_service_ratio": np.random.uniform(5, 40, n_samples),
            "m2_reserves_ratio": np.random.uniform(0.1, 2.0, n_samples),
            "real_exchange_rate_change": np.random.uniform(-20, 20, n_samples),
            "trade_balance_gdp": np.random.uniform(-8, 8, n_samples),
        }

        df = pd.DataFrame(data)

        risk_score = (
            df["exchange_rate_volatility"] * 2
            + df["inflation_rate"] * 0.5
            + df["interest_rate_spread"] * 0.3
            + (50 - df["foreign_reserves_months_imports"]) * 0.3
            + df["external_debt_gdp"] * 0.1
            + df["debt_service_ratio"] * 0.2
        )

        df["crisis"] = (risk_score > 30).astype(int)

        return df

    def train(self):
        df = self.generate_sample_data()
        X = df[self.feature_names]
        y = df["crisis"]

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.report = classification_report(y_test, y_pred, output_dict=True)
        self.trained = True

        return self.report

    def predict(self, features):
        if not self.trained:
            self.train()

        feature_array = np.array([features])
        feature_scaled = self.scaler.transform(feature_array)

        probability = self.model.predict_proba(feature_scaled)[0][1]

        return probability

    def get_risk_level(self, probability):
        if probability < 0.25:
            return "Low Risk", "green"
        elif probability < 0.50:
            return "Moderate Risk", "yellow"
        elif probability < 0.75:
            return "High Risk", "orange"
        else:
            return "Critical Risk", "red"

    def get_feature_importance(self):
        if not self.trained:
            self.train()

        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

    def predict_batch(self, df):
        if not self.trained:
            self.train()
        X = df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        return probabilities

    def analyze_file(self, file_path):
        ext = file_path.split(".")[-1].lower()
        if ext in ["csv"]:
            df = pd.read_csv(file_path)
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        df = df[self.feature_names]
        probabilities = self.predict_batch(df)

        results = df.copy()
        results["crisis_probability"] = probabilities
        results["risk_level"] = [self.get_risk_level(p)[0] for p in probabilities]

        return results

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path="model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    model = CurrencyCrisisModel()
    report = model.train()
    print("Model trained successfully!")
    print(f"Accuracy: {report['accuracy']:.2f}")
    print(f"Crisis detection rate: {report['1.0']['recall']:.2f}")
