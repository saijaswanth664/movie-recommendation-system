from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Create user-movie matrix
movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Fill missing values
movie_matrix = movie_matrix.fillna(0)

# Calculate similarity
movie_similarity = cosine_similarity(movie_matrix.T)

similarity_df = pd.DataFrame(movie_similarity,
                             index=movie_matrix.columns,
                             columns=movie_matrix.columns)


def recommend_movies(movie_title):

    movie_id = movies[movies['title'] == movie_title]['movieId']

    if movie_id.empty:
        return []

    movie_id = movie_id.values[0]

    similar_scores = similarity_df[movie_id].sort_values(ascending=False)

    recommended_ids = similar_scores.iloc[1:6].index

    recommendations = movies[movies['movieId'].isin(recommended_ids)]['title']

    return recommendations.tolist()


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    if request.method == "POST":

        movie_name = request.form["movie"]

        recommendations = recommend_movies(movie_name)

    return render_template("index.html", recommendations=recommendations)


@app.route("/search")
def search():

    query = request.args.get("q")

    if not query:
        return jsonify({"movies": []})

    suggestions = movies[movies['title'].str.contains(query, case=False)].head(5)

    return jsonify({"movies": suggestions['title'].tolist()})


if __name__ == "__main__":
    app.run(debug=True)