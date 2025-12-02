# eCommerce Gaming Product Performance Analytics & Recommendation Engine

## Project Overview
An end-to-end data analytics and machine learning system for analyzing gaming product sales performance across global markets. This project demonstrates the complete data development lifecycle: data ingestion, preprocessing, exploratory analysis, predictive modeling, and stakeholder-facing visualization.

**Business Context**: Simulates analytics workflows for gaming eCommerce platforms like Play-Asia, analyzing product performance across regions, building sales forecasting models, and creating recommendation systems to increase cross-sell revenue.

## Tech Stack
- **Language**: Python 3.9+
- **Data Manipulation**: pandas, NumPy
- **Database**: SQLite with SQLAlchemy ORM
- **Visualization**: Matplotlib, Seaborn (statistical), Plotly (interactive)
- **Machine Learning**: scikit-learn (regression, clustering, recommendations)
- **Development**: Jupyter Notebook, Git

## Project Structure
```
playasia-gaming-analytics/
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned data
├── src/                  # Python modules
├── output/
│   ├── visualizations/   # Generated charts
│   └── models/           # Trained ML models
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd playasia-gaming-analytics
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
- Download from [Kaggle Video Game Sales Dataset](https://www.kaggle.com/datasets/gregorut/videogamesales)
- Place `vgsales.csv` in `data/raw/` directory

### 5. Run Jupyter Notebook
```bash
jupyter notebook
```
Navigate to `notebooks/` and open the analysis notebook.

## Skills Demonstrated
✅ Python programming with modular design  
✅ Data cleaning and preprocessing (handling missing values, feature engineering)  
✅ SQL database design and querying  
✅ Exploratory Data Analysis with statistical testing  
✅ Machine Learning (regression, clustering, recommendations)  
✅ Data visualization for technical and business audiences  
✅ Version control and documentation best practices  

## Author
Jamaica Salem
Built as a portfolio project demonstrating Junior Data Developer skills
