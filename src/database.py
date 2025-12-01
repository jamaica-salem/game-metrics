"""
Database setup and connection management for gaming analytics project.
Creates SQLite database with raw and cleaned data tables.
"""

import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Import configuration
import sys
# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from config import DATABASE_PATH, RAW_TABLE, CLEANED_TABLE


def create_database_connection(db_path: Optional[Path] = None) -> Engine:
    """
    Create SQLAlchemy engine for database connection.
    
    Args:
        db_path: Path to SQLite database file. If None, uses default from config.
        
    Returns:
        SQLAlchemy Engine object
        
    Example:
        >>> engine = create_database_connection()
        >>> with engine.connect() as conn:
        ...     result = conn.execute(text("SELECT * FROM raw_games LIMIT 5"))
    """
    if db_path is None:
        db_path = DATABASE_PATH
    
    # Create SQLAlchemy engine
    # check_same_thread=False allows multi-threaded access
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False  # Set to True to see SQL queries in console (useful for debugging)
    )
    
    print(f"✓ Database connection created: {db_path}")
    return engine


def initialize_database(engine: Engine) -> None:
    """
    Create database tables if they don't exist.
    
    Args:
        engine: SQLAlchemy Engine object
        
    Creates two tables:
        - raw_games: Original data from CSV
        - cleaned_games: Processed data with feature engineering
    """
    
    # SQL for raw_games table
    create_raw_table_sql = """
    CREATE TABLE IF NOT EXISTS raw_games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rank INTEGER,
        name TEXT,
        platform TEXT,
        year TEXT,
        genre TEXT,
        publisher TEXT,
        na_sales REAL,
        eu_sales REAL,
        jp_sales REAL,
        other_sales REAL,
        global_sales REAL,
        loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # SQL for cleaned_games table
    create_cleaned_table_sql = """
    CREATE TABLE IF NOT EXISTS cleaned_games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_id INTEGER,
        name TEXT NOT NULL,
        platform TEXT NOT NULL,
        year INTEGER,
        genre TEXT NOT NULL,
        publisher TEXT NOT NULL,
        na_sales REAL NOT NULL,
        eu_sales REAL NOT NULL,
        jp_sales REAL NOT NULL,
        other_sales REAL NOT NULL,
        global_sales REAL NOT NULL,
        decade INTEGER,
        na_sales_ratio REAL,
        eu_sales_ratio REAL,
        jp_sales_ratio REAL,
        platform_generation TEXT,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (raw_id) REFERENCES raw_games(id)
    );
    """
    
    # Execute table creation
    with engine.connect() as conn:
        conn.execute(text(create_raw_table_sql))
        conn.execute(text(create_cleaned_table_sql))
        conn.commit()
    
    print(f"✓ Tables created: {RAW_TABLE}, {CLEANED_TABLE}")


def create_indexes(engine: Engine) -> None:
    """
    Create indexes on frequently queried columns for performance optimization.
    
    Args:
        engine: SQLAlchemy Engine object
        
    Indexes improve query speed for:
        - Filtering by genre, platform, year
        - Sorting by sales figures
    """
    
    indexes_sql = [
        "CREATE INDEX IF NOT EXISTS idx_raw_genre ON raw_games(genre);",
        "CREATE INDEX IF NOT EXISTS idx_raw_platform ON raw_games(platform);",
        "CREATE INDEX IF NOT EXISTS idx_raw_year ON raw_games(year);",
        "CREATE INDEX IF NOT EXISTS idx_cleaned_genre ON cleaned_games(genre);",
        "CREATE INDEX IF NOT EXISTS idx_cleaned_platform ON cleaned_games(platform);",
        "CREATE INDEX IF NOT EXISTS idx_cleaned_year ON cleaned_games(year);",
        "CREATE INDEX IF NOT EXISTS idx_cleaned_global_sales ON cleaned_games(global_sales);",
    ]
    
    with engine.connect() as conn:
        for idx_sql in indexes_sql:
            conn.execute(text(idx_sql))
        conn.commit()
    
    print(f"✓ Indexes created for optimized query performance")


def get_table_info(engine: Engine, table_name: str) -> pd.DataFrame:
    """
    Get schema information for a table.
    
    Args:
        engine: SQLAlchemy Engine object
        table_name: Name of table to inspect
        
    Returns:
        DataFrame with column information
    """
    query = f"PRAGMA table_info({table_name});"
    
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)
    
    return result


def get_row_count(engine: Engine, table_name: str) -> int:
    """
    Get number of rows in a table.
    
    Args:
        engine: SQLAlchemy Engine object
        table_name: Name of table
        
    Returns:
        Number of rows
    """
    query = f"SELECT COUNT(*) as count FROM {table_name};"
    
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)
    
    return result['count'].iloc[0]


if __name__ == "__main__":
    """
    Run this script directly to set up the database.
    """
    print("=" * 50)
    print("Setting up Gaming Analytics Database")
    print("=" * 50)
    
    # Create connection
    engine = create_database_connection()
    
    # Initialize tables
    initialize_database(engine)
    
    # Create indexes
    create_indexes(engine)
    
    # Display table info
    print("\n" + "=" * 50)
    print("RAW_GAMES Table Schema:")
    print("=" * 50)
    print(get_table_info(engine, RAW_TABLE))
    
    print("\n" + "=" * 50)
    print("CLEANED_GAMES Table Schema:")
    print("=" * 50)
    print(get_table_info(engine, CLEANED_TABLE))
    
    print("\n✓ Database setup complete!")