import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

db_path = Path(__file__).parent.parent / "data" / "gaming_analytics.db"
engine = create_engine(f"sqlite:///{db_path}")

# Query top 5 rows
df = pd.read_sql("SELECT * FROM raw_games LIMIT 5", engine)
print(df)
