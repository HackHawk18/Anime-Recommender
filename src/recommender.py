import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz

def load_data(anime_path):
     df = pd.read_csv(anime_path)
     
     
     df = df.dropna(subset=['genre'])
     df['type'] = df['type'].fillna("Unknown")
     df = df.dropna(subset=['rating'])
     df['name'] = df['name'].str.replace('&#039;', "'", regex=False)
     df['name'] = df['name'].str.replace('&amp;', '&', regex=False)
     df['episodes'] = df['episodes'].replace('Unknown', 'Ongoing')
     
     df = df.reset_index(drop=True)  
     return df

def prepare_features(df):
    
    df['genre'] = df['genre'].str.split(", ")
    
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genre'])
    
    scaler = MinMaxScaler()

    rating_scaled = scaler.fit_transform(df[['rating']])
    members_scaled = scaler.fit_transform(df[['members']])

    combined_matrix = np.hstack([
     genre_matrix,          
     rating_scaled*2 ,
     members_scaled*1 
    ])
    
    return df,combined_matrix

def recommend_anime(anime_title, df, combined_matrix, n=10):
    try:
            idx = df[df['name'] == anime_title].index[0]
    except:
            print(f"{anime_title} not found!")
            return None
        
    anime_vector = combined_matrix[idx].reshape(1, -1)
    sim_score = cosine_similarity(anime_vector, combined_matrix)[0]
    sim_score[idx] = -1
    sim_indices = np.argsort(sim_score)[::-1][:n*3]  # get more first
        
    result = df[['name', 'genre', 'type', 'rating',
                     'members', 'episodes']].iloc[sim_indices].copy()
    result['similarity'] = sim_score[sim_indices]
        
        # Filter by minimum members
    result = result[result['members'] >= 800]
        
    return result.head(n)

def search_anime(anime,df,n=20):
    anime = anime.lower().strip()
    scores = []
    for idx, title in enumerate(df['name']):
        title_lower = title.lower()
        score = max(
            fuzz.token_sort_ratio(anime, title_lower),
            fuzz.partial_ratio(anime, title_lower)
        )
        if score >= 85:
            scores.append((idx, score))
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    indices = [i[0] for i in scores[:n]]
    return df[['name', 'genre', 'members',
               'type']].iloc[indices]