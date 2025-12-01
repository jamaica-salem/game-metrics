# model_training.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from sqlalchemy import create_engine
from pathlib import Path

# Load cleaned data
db_path = Path("data/gaming_analytics.db")
engine = create_engine(f"sqlite:///{db_path}")
df = pd.read_sql("SELECT * FROM cleaned_games", engine)

# Features & target
X = df[["na_sales", "eu_sales", "jp_sales", "other_sales"]] 
y = df["global_sales"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² score:", model.score(X_test, y_test))

# Save model
#joblib.dump(model, "models/linear_regression_model.pkl")
