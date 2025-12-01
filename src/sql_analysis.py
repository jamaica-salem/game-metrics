import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

db_path = Path(__file__).parent.parent / "data" / "gaming_analytics.db"
engine = create_engine(f"sqlite:///{db_path}")

df = pd.read_sql("SELECT * FROM raw_games", engine)

# Top 10 Publishers by Global Sales
q1 = """
SELECT publisher, SUM(global_sales) AS total_sales
FROM raw_games
GROUP BY publisher
ORDER BY total_sales DESC
LIMIT 10;
"""

df_publishers = pd.read_sql(q1, engine)
print(df_publishers)

# Best Selling Genre
q2 = """
SELECT genre, SUM(global_sales) AS total_sales
FROM raw_games
GROUP BY genre
ORDER BY total_sales DESC;
"""

df_genres = pd.read_sql(q2, engine)
print(df_genres)

# Total Sales per Platform
q3 = """
SELECT platform, SUM(global_sales) AS total_sales
FROM raw_games
GROUP BY platform
ORDER BY total_sales DESC;
"""

df_platforms = pd.read_sql(q3, engine)
print(df_platforms)

# Region that drives the most sales
q4 = """
SELECT
    SUM(na_sales) AS na,
    SUM(eu_sales) AS eu,
    SUM(jp_sales) AS jp,
    SUM(other_sales) AS other
FROM raw_games;
"""

df_region_sales = pd.read_sql(q4, engine)
print(df_region_sales)

# Visualizations
df_platforms.plot(kind="bar", x="platform", y="total_sales", figsize=(12,6))
plt.title("Total Global Sales Per Platform")
plt.ylabel("Total Sales (Millions)")
plt.show()

df_publishers.plot(kind="barh", x="publisher", y="total_sales")
plt.title("Top 10 Publishers by Global Sales")
plt.show()

df_genres.plot(kind="bar", x="genre", y="total_sales", figsize=(10,5), color='orange')
plt.title("Best Selling Genre")
plt.show()

X = df[["na_sales","eu_sales","jp_sales","other_sales"]]
y = df["global_sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model R² score:", model.score(X_test, y_test))
print("Coefficients:", model.coef_)
