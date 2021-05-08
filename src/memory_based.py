#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import operator


def filter_ratings(df, n1=30, n2=30):
    ratings_per_user = df.groupby("user_id")["rating"].count()
    ratings_per_recipe = df.groupby("recipe_id")["rating"].count()
    
    ratings_per_recipe_df = pd.DataFrame(ratings_per_recipe)
    filtered_ratings_per_recipe_df = ratings_per_recipe_df[ratings_per_recipe_df["rating"] >= n1]
    popular_recipe = filtered_ratings_per_recipe_df.index.tolist()

    ratings_per_user_df = pd.DataFrame(ratings_per_user)
    filtered_ratings_per_user_df = ratings_per_user_df[ratings_per_user_df["rating"] >= n2]
    prolific_users = filtered_ratings_per_user_df.index.tolist()

    filtered_ratings = df[df["recipe_id"].isin(popular_recipe)]
    filtered_ratings = filtered_ratings[filtered_ratings["user_id"].isin(prolific_users)]
    
    return filtered_ratings


def get_rating_matrix(df, rows, cols, n1=30, n2=30):
    rating_matrix = df.pivot_table(index=rows, columns=cols, values="rating")
    rating_matrix.fillna(0, inplace=True)
    rating_matrix = rating_matrix.reset_index()
    rating_matrix.columns.name = None
    
    return rating_matrix


def user_recommend(user_id, matrix, recipes, n_items=5):
    def get_similar_users(user_id, matrix, k=5):
        user = matrix[matrix["user_id"] == user_id]
        other_users = matrix[matrix["user_id"] != user_id]

        similarities = cosine_similarity(user, other_users)[0].tolist() # Calculate cosine similarity for each user
        indices = other_users["user_id"].tolist()
        index_similarity = dict(zip(indices, similarities))
        index_similarity_sorted = sorted(index_similarity.items(),
                                         key=operator.itemgetter(1))
        index_similarity_sorted.reverse()

        top_users_similarities = index_similarity_sorted[:k]
        users = [u[0] for u in top_users_similarities]

        return users
    
    similar_user_indices = get_similar_users(user_id, matrix)
    
    user_row = matrix[matrix["user_id"] == user_id] # Get all recipes rated by user
    user_row_transposed = user_row.transpose()
    user_row_transposed.columns = ["rating"]
    user_row_transposed.drop("user_id", axis=0, inplace=True)
    user_row_transposed = user_row_transposed[user_row_transposed["rating"] == 0] # Only keep recipes user has not rated
    unrated_recipes = user_row_transposed.index.tolist()

    similar_users = matrix[matrix["user_id"].isin(similar_user_indices)]
    similar_users = similar_users.mean(axis=0)
    
    similar_users_df = pd.DataFrame(similar_users, columns=["mean"]) # Mean rating across all similar users
    similar_users_df.drop("user_id", axis=0, inplace=True)
    similar_users_df = similar_users_df[similar_users_df.index.isin(unrated_recipes)] # Get recipes used by similar users but not used by user of interest
    similar_users_df = similar_users_df.sort_values(by=["mean"], ascending=False) # Get highest rated items from similar users

    top_recipes = similar_users_df.head(n_items)
    top_recipes_indices = top_recipes.index.tolist()

    recommended_recipes = recipes[recipes["id"].isin(top_recipes_indices)]
    
    print("The top {} recipes recommended for user {} are:".format(n_items, user_id))
    return recommended_recipes


def get_recipe(df, user_id, top_only=True):
    rated = df[(df["user_id"] == user_id) & (df["rating"] == 5)]
    
    if top_only == False:
        rated = df[df["user_id"] == user_id]
    
    recipe_id = np.random.choice(rated["recipe_id"])

    return recipe_id


def item_recommend(user_id, recipe_id, matrix, recipes, k=5):
    current_name = recipes[recipes["id"] == recipe_id]["name"].to_string(index=False)
    
    user_col = matrix[["recipe_id", user_id]]
    user_col.columns = ["recipe_id", "rating"]
    user_col = user_col[user_col["rating"] == 0] # Only keep recipes user has not rated
    unrated_recipes = set(user_col["recipe_id"])

    recipe = matrix[matrix["recipe_id"] == recipe_id]
    other_recipes = matrix[matrix["recipe_id"] != recipe_id]
    other_recipes_set = set(matrix[matrix["recipe_id"] != recipe_id]["recipe_id"])
    other_recipes_unrated = unrated_recipes.intersection(other_recipes_set)
    other_recipes = other_recipes[other_recipes["recipe_id"].isin(other_recipes_unrated)]

    similarities = cosine_similarity(recipe, other_recipes)[0].tolist() # Calculate cosine similarity for each recipe
    similarities
    indices = other_recipes["recipe_id"].tolist()
    index_similarity = dict(zip(indices, similarities))
    index_similarity_sorted = sorted(index_similarity.items(),
                                     key=operator.itemgetter(1))
    index_similarity_sorted.reverse()

    top_recipes_similarities = index_similarity_sorted[:k]
    top_recipes = [u[0] for u in top_recipes_similarities]
    
    recommended_recipes = recipes[recipes["id"].isin(top_recipes)]
    
    print("User {} is currently viewing:{}". format(user_id, current_name))
    print("The top {} recipes recommended for user {} are:".format(k, user_id))
    return recommended_recipes