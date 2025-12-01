import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path


db_path = Path(__file__).parent.parent / "data" / "gaming_analytics.db"
engine = create_engine(f"sqlite:///{db_path}")

raw_df = pd.read_sql("SELECT * FROM raw_games", engine)

# Data cleaning
cleaned_df = raw_df.copy()
cleaned_df["publisher"] = cleaned_df["publisher"].fillna("Unknown")
cleaned_df["publisher"] = cleaned_df["publisher"].str.strip()

median_year = int(cleaned_df["year"].median(skipna=True))
cleaned_df["year"] = cleaned_df["year"].fillna(median_year)
cleaned_df["year"] = cleaned_df["year"].astype(int)

cleaned_df["global_sales"] = cleaned_df[["na_sales", "eu_sales", "jp_sales", "other_sales"]].sum(axis=1)

cleaned_df["name"] = cleaned_df["name"].str.strip()
cleaned_df["platform"] = cleaned_df["platform"].str.strip().str.upper()
cleaned_df["genre"] = cleaned_df["genre"].str.strip().str.title()

# Verify cleaning
cleaned_df.isnull().sum()
cleaned_df.describe()
cleaned_df.info()
cleaned_df.head()


# Save cleaned table
cleaned_df.to_sql("cleaned_games", engine, if_exists="replace", index=False)
cleaned_df.to_csv("data/processed/vgsales_cleaned.csv", index=False)
print("Data cleaning complete. Cleaned data saved to 'cleaned_games' table.")
