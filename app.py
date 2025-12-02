import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="Game Analytics Dashboard and Recommendation Engine",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Database connection
db_path = Path(__file__).parent / "data" / "gaming_analytics.db"
engine = create_engine(f"sqlite:///{db_path}")

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS styling"""
    
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@300;400;500;600;700;800&display=swap');
        
        /* All text elements */
        * {
            font-family: 'Figtree', sans-serif !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Figtree', sans-serif !important;
            font-weight: 700 !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-family: 'Figtree', sans-serif !important;
            font-weight: 600;
            font-size: 1rem;
            padding: 10px 20px;
            border-radius: 8px 8px 0 0;
        }
        
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #1f77b4 !important;
        }
        
        /* KPI Cards */
        .kpi-container {
            background: linear-gradient(135deg, #f8f9fa 0%, #f8f9fa 100%);
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .kpi-container:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .kpi-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4 !important;
            margin: 10px 0;
        }
        
        .kpi-label {
            font-size: 0.9rem;
            font-weight: 500;
            opacity: 0.7;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Metric containers */
        [data-testid="stMetric"] {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
        }
        
        /* Fix dataframe column menu overlapping text */
        [data-testid="stDataFrameResizable"] button[kind="header"] {
            min-width: 120px !important;
            padding: 8px 16px !important;
            white-space: nowrap !important;
            display: block !important;
            clear: both !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
        }
        
        /* Column header menu popup */
        div[role="tooltip"] {
            z-index: 9999 !important;
        }
        
        /* Menu items in column header dropdown */
        div[role="tooltip"] div[role="option"] {
            padding: 8px 16px !important;
            white-space: nowrap !important;
            min-width: 150px !important;
            display: block !important;
            clear: both !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
        }
        
        /* Fix menu text rendering */
        div[role="tooltip"] span {
            display: inline-block !important;
            white-space: nowrap !important;
            overflow: visible !important;
        }
        
        /* Ensure proper spacing in dropdown */
        div[data-baseweb="menu"] {
            padding: 8px 0 !important;
        }
        
        div[data-baseweb="menu"] li {
            padding: 8px 16px !important;
            line-height: 1.5 !important;
            white-space: nowrap !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_data():
    """Load game sales data from SQLite database"""
    try:
        df = pd.read_sql("SELECT * FROM cleaned_games", engine)
        # Handle missing values
        df = df.fillna({
            'genre': 'Unknown',
            'publisher': 'Unknown',
            'platform': 'Unknown',
            'year': 0,
            'global_sales': 0
        })
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_similarity_matrix(_df):
    """Generate similarity matrix for game recommendations"""
    try:
        df_encoded = pd.get_dummies(_df, columns=["genre", "platform", "publisher"])
        numerical_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col not in 
                         ["year", "na_sales", "eu_sales", "jp_sales", "other_sales", "global_sales"]]
        return cosine_similarity(df_encoded[numerical_cols])
    except Exception as e:
        st.error(f"Error creating similarity matrix: {e}")
        return None

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def apply_filters(df, genre, publisher, platform):
    """Apply selected filters to dataframe"""
    filtered = df.copy()
    
    if genre != "All":
        filtered = filtered[filtered["genre"] == genre]
    
    if publisher != "All":
        filtered = filtered[filtered["publisher"] == publisher]
    
    if platform != "All":
        filtered = filtered[filtered["platform"] == platform]
    
    return filtered

def calculate_kpis(df):
    """Calculate key performance indicators"""
    total_sales = df["global_sales"].sum()
    top_genre = df.groupby("genre")["global_sales"].sum().idxmax() if len(df) > 0 else "N/A"
    top_platform = df.groupby("platform")["global_sales"].sum().idxmax() if len(df) > 0 else "N/A"
    total_games = len(df)
    
    return {
        "total_sales": total_sales,
        "top_genre": top_genre,
        "top_platform": top_platform,
        "total_games": total_games
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_genre_sales_chart(df):
    """Create genre sales bar chart"""
    genre_sales = df.groupby("genre")["global_sales"].sum().reset_index()
    genre_sales = genre_sales.sort_values("global_sales", ascending=False)
    
    fig = px.bar(
        genre_sales,
        x="genre",
        y="global_sales",
        title="Global Sales by Genre (in Millions)",
        labels={"global_sales": "Sales (M)", "genre": "Genre"},
        color="global_sales",
        color_continuous_scale="blues"
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Sales: $%{y:.2f}M<extra></extra>"
    )
    
    return fig

def create_publisher_chart(df):
    """Create top publishers horizontal bar chart"""
    publisher_sales = (
        df.groupby("publisher")["global_sales"].sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    
    fig = px.bar(
        publisher_sales,
        x="global_sales",
        y="publisher",
        orientation="h",
        title="Top 10 Publishers by Global Sales",
        labels={"global_sales": "Sales (M)", "publisher": "Publisher"},
        color="global_sales",
        color_continuous_scale="viridis"
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Sales: $%{x:.2f}M<extra></extra>"
    )
    
    return fig

def create_platform_chart(df):
    """Create platform sales bar chart"""
    platform_sales = (
        df.groupby("platform")["global_sales"].sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    
    fig = px.bar(
        platform_sales,
        x="platform",
        y="global_sales",
        title="Top 10 Platforms by Global Sales",
        labels={"global_sales": "Sales (M)", "platform": "Platform"},
        color="global_sales",
        color_continuous_scale="teal"
    )
    
    fig.update_layout(
        height=500,
        showlegend=False
    )
    
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Sales: $%{y:.2f}M<extra></extra>"
    )
    
    return fig

def create_top_games_chart(df):
    """Create top 10 selling games chart"""
    top_games = df.nlargest(10, "global_sales")[["name", "global_sales", "genre", "platform"]]
    
    fig = px.bar(
        top_games,
        x="global_sales",
        y="name",
        orientation="h",
        title="Top 10 Best-Selling Games",
        labels={"global_sales": "Sales (M)", "name": "Game"},
        color="genre",
        hover_data=["platform"]
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Sales: $%{x:.2f}M<br>Genre: %{customdata[0]}<extra></extra>"
    )
    
    return fig

def create_sales_by_year_chart(df):
    """Create sales trend by year line chart"""
    year_sales = df[df["year"] > 0].groupby("year")["global_sales"].sum().reset_index()
    
    fig = px.line(
        year_sales,
        x="year",
        y="global_sales",
        title="Global Sales Trend Over Time",
        labels={"global_sales": "Sales (M)", "year": "Year"},
        markers=True
    )
    
    fig.update_layout(
        height=400,
        hovermode='x unified'
    )
    
    fig.update_traces(
        line_color='#1f77b4',
        line_width=3,
        hovertemplate="Year: %{x}<br>Sales: $%{y:.2f}M<extra></extra>"
    )
    
    return fig

# ============================================================================
# RECOMMENDATION FUNCTIONS
# ============================================================================

def get_top_games_by_filter(df, genre=None, platform=None, top_n=10):
    """Get top selling games by filter criteria"""
    filtered = df.copy()
    
    if genre:
        filtered = filtered[filtered["genre"] == genre]
    
    if platform:
        filtered = filtered[filtered["platform"] == platform]
    
    top_games = filtered.nlargest(top_n, "global_sales")[
        ["name", "genre", "platform", "publisher", "year", "global_sales"]
    ]
    
    return top_games

def recommend_similar_games(game_name, df, similarity_matrix, top_n=5):
    """Recommend similar games based on cosine similarity"""
    try:
        game_idx = df[df["name"] == game_name].index[0]
        sim_scores = list(enumerate(similarity_matrix[game_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        
        game_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        recommended = df.iloc[game_indices][
            ["name", "genre", "platform", "publisher", "year", "global_sales"]
        ].copy()
        
        recommended["similarity_score"] = [f"{score:.2%}" for score in similarity_scores]
        
        return recommended
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Apply custom styling
    apply_custom_css()
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("Failed to load data. Please check your database connection.")
        return
    
    # Title and description
    st.title("🎮 Game Analytics Dashboard and Recommendation Engine")
    st.markdown("**Interactive insights and AI-powered recommendations from global video game sales data**")
    
    st.markdown("---")
    
    # ========================================================================
    # FILTERS SECTION
    # ========================================================================
    
    st.markdown("### Filters & Controls")
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        genre_filter = st.selectbox(
            "Genre",
            ["All"] + sorted(df["genre"].unique().tolist()),
            key="genre_filter"
        )
    
    with filter_col2:
        publisher_filter = st.selectbox(
            "Publisher",
            ["All"] + sorted(df["publisher"].unique().tolist()),
            key="publisher_filter"
        )
    
    with filter_col3:
        platform_filter = st.selectbox(
            "Platform",
            ["All"] + sorted(df["platform"].unique().tolist()),
            key="platform_filter"
        )
    
    with filter_col4:
        top_n = st.slider(
            "Top N Results",
            min_value=5,
            max_value=50,
            value=10,
            step=1,
            key="top_n_slider"
        )
    
    # Apply filters
    filtered_df = apply_filters(df, genre_filter, publisher_filter, platform_filter)
    
    st.markdown("---")
    
    # ========================================================================
    # KPI SECTION
    # ========================================================================
    
    kpis = calculate_kpis(filtered_df)
    
    st.markdown("### Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-label">Total Sales</div>
                <div class="kpi-value">${kpis['total_sales']:.1f}M</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with kpi_col2:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-label">Top Genre</div>
                <div class="kpi-value">{kpis['top_genre']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with kpi_col3:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-label">Top Platform</div>
                <div class="kpi-value">{kpis['top_platform']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with kpi_col4:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-label">Total Games</div>
                <div class="kpi-value">{kpis['total_games']:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # ========================================================================
    # TABBED CONTENT SECTIONS
    # ========================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Sales Analytics",
        "🏆 Top Performers",
        "🎯 Recommendations",
        "📋 Data Explorer",
        "📊 Trends"
    ])
    
    # TAB 1: Sales Analytics
    with tab1:
        st.markdown("### Sales Performance by Category")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig_genre = create_genre_sales_chart(filtered_df)
            st.plotly_chart(fig_genre, use_container_width=True)
        
        with chart_col2:
            fig_platform = create_platform_chart(filtered_df)
            st.plotly_chart(fig_platform, use_container_width=True)
        
        st.markdown("---")
        
        fig_publisher = create_publisher_chart(filtered_df)
        st.plotly_chart(fig_publisher, use_container_width=True)
    
    # TAB 2: Top Performers
    with tab2:
        st.markdown("### Best-Selling Games")
        
        top_games_df = filtered_df.nlargest(top_n, "global_sales")[
            ["name", "genre", "platform", "publisher", "year", "global_sales"]
        ].reset_index(drop=True)
        
        # Add rank column
        top_games_df.insert(0, "Rank", range(1, len(top_games_df) + 1))
        
        # Format sales column
        top_games_df["global_sales"] = top_games_df["global_sales"].apply(lambda x: f"${x:.2f}M")
        
        # Display with conditional formatting
        st.dataframe(
            top_games_df,
            use_container_width=True,
            height=600,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "name": st.column_config.TextColumn("Game Title", width="large"),
                "genre": st.column_config.TextColumn("Genre", width="medium"),
                "platform": st.column_config.TextColumn("Platform", width="small"),
                "publisher": st.column_config.TextColumn("Publisher", width="medium"),
                "year": st.column_config.NumberColumn("Year", width="small"),
                "global_sales": st.column_config.TextColumn("Global Sales", width="small")
            },
            hide_index=True
        )
        
        st.markdown("---")
        
        fig_top_games = create_top_games_chart(filtered_df)
        st.plotly_chart(fig_top_games, use_container_width=True)
    
    # TAB 3: Recommendations
    with tab3:
        st.markdown("### AI-Powered Game Recommendations")
        
        rec_col1, rec_col2 = st.columns([2, 3])
        
        with rec_col1:
            recommendation_type = st.radio(
                "Select Recommendation Type",
                ["Top Games by Filter", "Similar Games"],
                key="rec_type"
            )
        
        with rec_col2:
            if recommendation_type == "Similar Games":
                selected_game = st.selectbox(
                    "Select a Game to Find Similar Titles",
                    sorted(df["name"].unique().tolist()),
                    key="game_selector"
                )
        
        st.markdown("---")
        
        if recommendation_type == "Top Games by Filter":
            genre_sel = None if genre_filter == "All" else genre_filter
            platform_sel = None if platform_filter == "All" else platform_filter
            
            recommendations = get_top_games_by_filter(df, genre_sel, platform_sel, top_n)
            
            st.markdown(f"#### Top {top_n} Games" + 
                       (f" in **{genre_sel}** genre" if genre_sel else "") +
                       (f" on **{platform_sel}** platform" if platform_sel else ""))
            
            recommendations_display = recommendations.reset_index(drop=True)
            recommendations_display.insert(0, "Rank", range(1, len(recommendations_display) + 1))
            recommendations_display["global_sales"] = recommendations_display["global_sales"].apply(
                lambda x: f"${x:.2f}M"
            )
            
            st.dataframe(
                recommendations_display,
                use_container_width=True,
                height=500,
                hide_index=True
            )
        
        else:  # Similar Games
            similarity_matrix = load_similarity_matrix(df)
            
            if similarity_matrix is not None:
                recommendations = recommend_similar_games(selected_game, df, similarity_matrix, top_n)
                
                st.markdown(f"#### Games Similar to **{selected_game}**")
                
                if not recommendations.empty:
                    recommendations_display = recommendations.reset_index(drop=True)
                    recommendations_display.insert(0, "Rank", range(1, len(recommendations_display) + 1))
                    recommendations_display["global_sales"] = recommendations_display["global_sales"].apply(
                        lambda x: f"${x:.2f}M"
                    )
                    
                    st.dataframe(
                        recommendations_display,
                        use_container_width=True,
                        height=500,
                        column_config={
                            "similarity_score": st.column_config.TextColumn("Match Score", width="small")
                        },
                        hide_index=True
                    )
                else:
                    st.warning("No similar games found.")
    
    # TAB 4: Data Explorer
    with tab4:
        st.markdown("### Dataset Explorer")
        
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            search_term = st.text_input(
                "🔎 Search games by name",
                placeholder="Enter game title...",
                key="search_input"
            )
        
        with search_col2:
            sort_by = st.selectbox(
                "Sort by",
                ["global_sales", "year", "name"],
                key="sort_selector"
            )
        
        # Apply search filter
        display_df = filtered_df.copy()
        
        if search_term:
            display_df = display_df[
                display_df["name"].str.contains(search_term, case=False, na=False)
            ]
        
        # Sort data
        display_df = display_df.sort_values(sort_by, ascending=False)
        
        st.markdown(f"**Showing {len(display_df):,} of {len(df):,} games**")
        
        # Format for display
        display_df_formatted = display_df.copy()
        display_df_formatted["global_sales"] = display_df_formatted["global_sales"].apply(
            lambda x: f"${x:.2f}M"
        )
        
        st.dataframe(
            display_df_formatted,
            use_container_width=True,
            height=600,
            column_config={
                "name": st.column_config.TextColumn("Game Title", width="large"),
                "genre": st.column_config.TextColumn("Genre", width="medium"),
                "platform": st.column_config.TextColumn("Platform", width="small"),
                "publisher": st.column_config.TextColumn("Publisher", width="medium"),
                "year": st.column_config.NumberColumn("Year", width="small"),
                "global_sales": st.column_config.TextColumn("Global Sales", width="small")
            },
            hide_index=True
        )
    
    # TAB 5: Trends
    with tab5:
        st.markdown("### Sales Trends Over Time")
        
        fig_trend = create_sales_by_year_chart(filtered_df)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### Regional Sales Distribution")
        
        regional_col1, regional_col2 = st.columns(2)
        
        with regional_col1:
            # Regional sales pie chart
            regional_sales = pd.DataFrame({
                "Region": ["North America", "Europe", "Japan", "Other"],
                "Sales": [
                    filtered_df["na_sales"].sum(),
                    filtered_df["eu_sales"].sum(),
                    filtered_df["jp_sales"].sum(),
                    filtered_df["other_sales"].sum()
                ]
            })
            
            fig_regional = px.pie(
                regional_sales,
                values="Sales",
                names="Region",
                title="Sales Distribution by Region",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig_regional.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>Sales: $%{value:.2f}M<br>Percentage: %{percent}<extra></extra>"
            )
            
            st.plotly_chart(fig_regional, use_container_width=True)
        
        with regional_col2:
            # Top genres by year (heatmap-style)
            if len(filtered_df[filtered_df["year"] > 0]) > 0:
                genre_year = filtered_df[filtered_df["year"] > 0].groupby(
                    ["year", "genre"]
                )["global_sales"].sum().reset_index()
                
                # Get top 10 genres
                top_genres = filtered_df.groupby("genre")["global_sales"].sum().nlargest(10).index
                genre_year = genre_year[genre_year["genre"].isin(top_genres)]
                
                fig_genre_trend = px.line(
                    genre_year,
                    x="year",
                    y="global_sales",
                    color="genre",
                    title="Genre Trends Over Time",
                    labels={"global_sales": "Sales (M)", "year": "Year", "genre": "Genre"}
                )
                
                fig_genre_trend.update_layout(
                    hovermode='x unified',
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.01
                    )
                )
                
                st.plotly_chart(fig_genre_trend, use_container_width=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()