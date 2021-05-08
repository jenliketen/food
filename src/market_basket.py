#!/usr/bin/env python3

import pandas as pd


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
    filtered_ratings = filtered_ratings.reset_index(drop=True)
    
    return filtered_ratings


def map_recipes(recipes, interactions):
    id_name = dict(zip(recipes["id"], recipes["name"]))
    interactions["recipe_name"] = interactions["recipe_id"].map(id_name)
    return interactions[["user_id", "recipe_name"]]