import pandas as pd

# Load datasets
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Merge datasets
data = pd.merge(ratings, movies, on="movieId")

# Create movie-user matrix
movie_matrix = data.pivot_table(index="title", columns="userId", values="rating")

# Replace NaN with 0
movie_matrix = movie_matrix.fillna(0)

print("Movie User Matrix:")
print(movie_matrix.head())

print("\nShape of matrix:", movie_matrix.shape)