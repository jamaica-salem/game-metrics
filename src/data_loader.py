"""
Data loading utilities for gaming analytics project.
Handles CSV ingestion into SQLite database with validation and error handling.
"""

import pandas as pd
from sqlalchemy.engine import Engine
from pathlib import Path
from typing import Tuple, Optional
import sys

# Import configuration and database utilities
sys.path.append(str(Path(__file__).parent.parent))
from config import VGSALES_CSV, RAW_TABLE, CLEANED_TABLE
from src.database import create_database_connection, get_row_count


def load_csv_to_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame with basic validation.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame containing the CSV data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        pd.errors.EmptyDataError: If CSV is empty
    """
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading CSV from: {csv_path}")
    
    # Read CSV with pandas
    # low_memory=False to prevent dtype warnings for mixed-type columns
    df = pd.read_csv(csv_path, low_memory=False)
    
    print(f"✓ CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate DataFrame structure matches expected video game sales schema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid: bool, issues: list of strings)
        
    Checks:
        - Required columns present
        - Data types are reasonable
        - No completely empty rows
    """
    
    issues = []
    
    # Expected columns from video game sales dataset
    expected_columns = [
        'Rank', 'Name', 'Platform', 'Year', 'Genre', 
        'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 
        'Other_Sales', 'Global_Sales'
    ]
    
    # Check for missing columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for completely empty DataFrame
    if df.empty:
        issues.append("DataFrame is empty (0 rows)")
    
    # Check for all-null rows
    all_null_rows = df.isnull().all(axis=1).sum()
    if all_null_rows > 0:
        issues.append(f"Found {all_null_rows} completely empty rows")

    all_duplicated_rows = df.duplicated().sum()
    if all_duplicated_rows > 0:
        issues.append(f"Found {all_duplicated_rows} duplicated rows")
    
    # Check sales columns are numeric
    sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    for col in sales_columns:
        if col in df.columns:
            # Try to convert to numeric, see if it fails
            try:
                pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                issues.append(f"Column {col} cannot be converted to numeric: {e}")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        print("✓ DataFrame validation passed")
    else:
        print("✗ DataFrame validation found issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    return is_valid, issues


def prepare_dataframe_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for database insertion by standardizing column names.
    
    Args:
        df: Original DataFrame from CSV
        
    Returns:
        DataFrame with database-friendly column names
        
    Changes:
        - Converts column names to lowercase with underscores
        - Example: 'NA_Sales' -> 'na_sales'
    """
    
    # Create a copy to avoid modifying original
    df_prepared = df.copy()
    
    # Convert column names to lowercase with underscores (SQL convention)
    # 'NA_Sales' -> 'na_sales'
    df_prepared.columns = df_prepared.columns.str.lower().str.replace(' ', '_')
    
    print(f"✓ Column names standardized: {list(df_prepared.columns)}")
    
    return df_prepared


def load_dataframe_to_database(
    df: pd.DataFrame, 
    engine: Engine, 
    table_name: str,
    if_exists: str = 'replace'
) -> int:
    """
    Load DataFrame into SQLite database table.
    
    Args:
        df: DataFrame to load
        engine: SQLAlchemy Engine object
        table_name: Name of target table
        if_exists: What to do if table exists ('replace', 'append', 'fail')
        
    Returns:
        Number of rows inserted
        
    Note:
        - 'replace' drops existing table and recreates it
        - 'append' adds rows to existing table
        - 'fail' raises error if table exists
    """
    
    print(f"Loading {len(df)} rows into table '{table_name}'...")
    
    # Use pandas to_sql for easy loading
    # index=False means don't write DataFrame index as a column
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=False,
        method='multi',  # Insert multiple rows at once (faster)
        chunksize=1000   # Insert in batches of 1000 rows
    )
    
    # Verify rows were inserted
    row_count = get_row_count(engine, table_name)
    
    print(f"✓ {row_count} rows successfully loaded into '{table_name}'")
    
    return row_count


def display_sample_data(engine: Engine, table_name: str, n: int = 5) -> None:
    """
    Display first N rows from database table.
    
    Args:
        engine: SQLAlchemy Engine object
        table_name: Name of table to query
        n: Number of rows to display
    """
    
    query = f"SELECT * FROM {table_name} LIMIT {n};"
    
    print(f"\nFirst {n} rows from '{table_name}':")
    print("=" * 80)
    
    df_sample = pd.read_sql(query, engine)
    print(df_sample.to_string())
    print("=" * 80)


def get_data_summary(engine: Engine, table_name: str) -> None:
    """
    Display summary statistics about the loaded data.
    
    Args:
        engine: SQLAlchemy Engine object
        table_name: Name of table to analyze
    """
    
    print(f"\nData Summary for '{table_name}':")
    print("=" * 80)
    
    # Get total rows
    total_rows = get_row_count(engine, table_name)
    print(f"Total rows: {total_rows:,}")
    
    # Get unique counts for categorical columns
    queries = {
        "Unique games": "SELECT COUNT(DISTINCT name) as count FROM raw_games;",
        "Unique platforms": "SELECT COUNT(DISTINCT platform) as count FROM raw_games;",
        "Unique genres": "SELECT COUNT(DISTINCT genre) as count FROM raw_games;",
        "Unique publishers": "SELECT COUNT(DISTINCT publisher) as count FROM raw_games;",
    }
    
    for label, query in queries.items():
        result = pd.read_sql(query, engine)
        print(f"{label}: {result['count'].iloc[0]:,}")
    
    # Get year range
    year_query = """
        SELECT 
            MIN(CAST(year AS INTEGER)) as min_year,
            MAX(CAST(year AS INTEGER)) as max_year
        FROM raw_games 
        WHERE year IS NOT NULL AND year != '';
    """
    year_result = pd.read_sql(year_query, engine)
    print(f"Year range: {year_result['min_year'].iloc[0]} - {year_result['max_year'].iloc[0]}")
    
    # Get sales statistics
    sales_query = """
        SELECT 
            ROUND(SUM(global_sales), 2) as total_sales,
            ROUND(AVG(global_sales), 2) as avg_sales,
            ROUND(MAX(global_sales), 2) as max_sales
        FROM raw_games;
    """
    sales_result = pd.read_sql(sales_query, engine)
    print(f"Total global sales: {sales_result['total_sales'].iloc[0]:,.2f} million")
    print(f"Average game sales: {sales_result['avg_sales'].iloc[0]:.2f} million")
    print(f"Highest selling game: {sales_result['max_sales'].iloc[0]:.2f} million")
    
    # Get top 5 best-selling games
    top_games_query = """
        SELECT name, platform, year, global_sales
        FROM raw_games
        ORDER BY global_sales DESC
        LIMIT 5;
    """
    print("\nTop 5 Best-Selling Games:")
    top_games = pd.read_sql(top_games_query, engine)
    print(top_games.to_string(index=False))
    
    print("=" * 80)


def ingest_video_game_sales_data(csv_path: Optional[Path] = None) -> None:
    """
    Main function to orchestrate the complete data ingestion process.
    
    Args:
        csv_path: Path to CSV file. If None, uses default from config.
        
    Process:
        1. Load CSV into DataFrame
        2. Validate data structure
        3. Prepare for database insertion
        4. Create database connection
        5. Load into raw_games table
        6. Display sample and summary
    """
    
    print("\n" + "=" * 80)
    print("VIDEO GAME SALES DATA INGESTION")
    print("=" * 80 + "\n")
    
    # Use default path if none provided
    if csv_path is None:
        csv_path = VGSALES_CSV
    
    try:
        # Step 1: Load CSV
        df = load_csv_to_dataframe(csv_path)
        
        # Step 2: Validate
        is_valid, issues = validate_dataframe(df)
        if not is_valid:
            print("\n⚠ Validation issues found, but continuing with data load...")
            print("These issues will be addressed in the data cleaning phase.\n")
        
        # Step 3: Prepare for DB
        df_prepared = prepare_dataframe_for_db(df)
        
        # Step 4: Create database connection
        engine = create_database_connection()
        
        # Step 5: Load into database
        # Using 'replace' to overwrite if data already exists
        row_count = load_dataframe_to_database(
            df_prepared, 
            engine, 
            RAW_TABLE,
            if_exists='replace'
        )
        
        # Step 6: Display sample data
        display_sample_data(engine, RAW_TABLE, n=5)
        
        # Step 7: Show summary statistics
        get_data_summary(engine, RAW_TABLE)
        
        print("\n" + "=" * 80)
        print("✓ DATA INGESTION COMPLETE!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during data ingestion: {e}")
        raise


if __name__ == "__main__":
    """
    Run this script directly to load video game sales data into database.
    """
    ingest_video_game_sales_data()