import pandas as pd

# Load the datasets
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Display first few rows
print("Movies Dataset:")
print(movies.head())

print("\nRatings Dataset:")
print(ratings.head())

# Merge the datasets
data = pd.merge(ratings, movies, on="movieId")

print("\nMerged Dataset:")
print(data.head())