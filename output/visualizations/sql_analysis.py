import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
import matplotlib.pyplot as plt

db_path = Path(__file__).parent.parent.parent / "data" / "gaming_analytics.db"
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


