import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/processed/vgsales_cleaned.csv")

features = ["genre", "platform", "publisher"]
df_encoded = pd.get_dummies(df, columns=features)

# Compute similarity matrix
similarity_matrix = cosine_similarity(df_encoded.drop(["name", "year", "na_sales", "eu_sales",
                                                      "jp_sales", "other_sales", "global_sales"], axis=1))
def recommend_games(game_name, df, similarity_matrix, top_n=5):
    # Find the index of the selected game
    idx = df[df['name'] == game_name].index[0]
    
    # Get similarity scores for this game
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get indices of top N similar games (skip the first, which is the game itself)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    return df.iloc[top_indices][["name", "genre", "platform", "publisher", "global_sales"]]

def top_games_by_filter(df, genre=None, platform=None, top_n=10):
    filtered_df = df.copy()
    if genre:
        filtered_df = filtered_df[filtered_df['genre'] == genre]
    if platform:
        filtered_df = filtered_df[filtered_df['platform'] == platform]
    
    return filtered_df.sort_values(by="global_sales", ascending=False).head(top_n)
