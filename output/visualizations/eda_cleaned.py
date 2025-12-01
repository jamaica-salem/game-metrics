import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.engine import create_engine
from pathlib import Path 

#Database connection
db_path = Path(__file__).parent.parent.parent / "data" / "gaming_analytics.db"
engine = create_engine(f"sqlite:///{db_path}")

# Verify connection
print(db_path.resolve())
print(db_path.exists())

#Load data
df = pd.read_sql("SELECT *  FROM cleaned_games", engine)

# Inspect data
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualizations

# Top 10 Games by Global Sales
top_games = df.groupby("name")["global_sales"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=top_games.values, y=top_games.index, palette="viridis")
plt.title("Top 10 Games by Global Sales")
plt.xlabel("Global Sales (Millions)")
plt.ylabel("Game")
plt.show()

# Sales by Region
sales_by_region = df[["na_sales", "eu_sales", "jp_sales", "other_sales"]].sum()

sales_by_region.plot(kind="pie", autopct="%1.1f%%", figsize=(8,8), colors=sns.color_palette("pastel"))
plt.title("Sales Distribution by Region")
plt.ylabel("")
plt.show()

# Genre Popularity Over Time
df['year'] = pd.to_numeric(df['year'], errors='coerce')
plt.figure(figsize=(12,6))
sns.countplot(data=df[df['year'].notnull()], x='year', hue='genre', palette="Set2")
plt.title("Genre Popularity Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Games Released")
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()  

#Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df[["na_sales","eu_sales","jp_sales","other_sales","global_sales"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Regional and Global Sales")
plt.show()
