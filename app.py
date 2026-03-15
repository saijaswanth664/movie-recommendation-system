import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load datasets
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

data = pd.merge(ratings, movies, on="movieId")

movie_matrix = data.pivot_table(index="title", columns="userId", values="rating").fillna(0)

similarity = cosine_similarity(movie_matrix)

similarity_df = pd.DataFrame(similarity, index=movie_matrix.index, columns=movie_matrix.index)


def recommend_movies(movie_name):
    if movie_name not in similarity_df.index:
        return []

    similar_scores = similarity_df[movie_name].sort_values(ascending=False)
    recommendations = similar_scores.iloc[1:6]

    return list(recommendations.index)


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    if request.method == "POST":
        movie_name = request.form["movie"]
        recommendations = recommend_movies(movie_name)

    return render_template("index.html", recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)