from src.recommender import load_data, prepare_features
from src.recommender import recommend_anime, search_anime

ANIME_PATH = 'data/anime.csv'

print("Loading data...")
df = load_data(ANIME_PATH)
df, combined_matrix = prepare_features(df)   
print("Ready!\n")

while True:
    anime = input("Enter anime name (or 'search:name' to search, 'quit' to exit): ")
    
    if anime.lower() in ['quit', 'exit', 'q']:
        print("Bye!")
        break
    
    elif anime.lower().startswith('search:'):      
        query = anime[7:]
        results = search_anime(query, df)
        if len(results) == 0:
            print("No Anime found!")
        else:
            print("\nMatching Anime:")
            print(results.to_string(index=False))
    
    else:                                         
        print("Finding recommendations...")
        results = recommend_anime(anime, df, combined_matrix)
        if results is not None:
            print(f"\nTop 10 recommendations for '{anime}':")
            print(results.to_string(index=False))
    
    print()