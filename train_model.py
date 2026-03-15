import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

# Create movie-user matrix
movie_matrix = data.pivot_table(index="title", columns="userId", values="rating")

# Replace NaN with 0
movie_matrix = movie_matrix.fillna(0)

print("Movie Matrix Shape:", movie_matrix.shape)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(movie_matrix)

# Convert to dataframe
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=movie_matrix.index,
    columns=movie_matrix.index
)

print("Similarity matrix created!")

# Recommendation function
def recommend_movies(movie_name, num_recommendations=5):

    if movie_name not in similarity_df.index:
        print("Movie not found!")
        return

    similar_scores = similarity_df[movie_name].sort_values(ascending=False)

    # Skip the first movie (itself)
    recommendations = similar_scores.iloc[1:num_recommendations+1]

    print("\nRecommended Movies:")
    for movie in recommendations.index:
        print(movie)


# Test recommendation
recommend_movies("Toy Story (1995)")