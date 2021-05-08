#!/usr/bin/env python3

import pandas as pd


def get_most_popular(n, interactions, recipes, best_only=True):
    top_ratings = interactions[interactions["rating"] == 5]
    top_ids = top_ratings["recipe_id"].value_counts()[:n].index.tolist()
    
    if best_only == False:
        top_ids = interactions["recipe_id"].value_counts()[:n].index.tolist()
    
    top_recipes = recipes[recipes["id"].isin(top_ids)]
    
    return top_recipes