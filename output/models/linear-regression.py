import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

db_path = Path(__file__).parent.parent / "data" / "gaming_analytics.db"
engine = create_engine(f"sqlite:///{db_path}")

df = pd.read_sql("SELECT * FROM raw_games", engine)
X = df[["na_sales","eu_sales","jp_sales","other_sales"]]
y = df["global_sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model R² score:", model.score(X_test, y_test))
print("Coefficients:", model.coef_)

