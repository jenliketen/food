#!/usr/bin/env python3

import pandas as pd
import contractions
import emoji
import string
import re
from rake_nltk import Rake


def preprocess(df, cols):
    def remove_emojis(text):
            return text.encode("ascii", "ignore").decode("ascii")
    def remove_punctuation(text):
        punct = []
        punct += list(string.punctuation)
        punct += '’'
        #punct.remove("'")
        for punctuation in punct:
            text = text.replace(punctuation, " ")
        return text

    # General preprocessing 
    for col in cols:
        df[col] = df[col].apply(lambda x: contractions.fix(x)) # Expand contractions
        df[col] = df[col].apply(lambda x: x.replace("\n", " "))
        df[col] = df[col].str.replace("http\S+|www.\S+", "", case=False) # Remove hyperlinks
        df[col] = df[col].apply(lambda x: x.replace("&gt;", "")) # Remove "&gt;"
        df[col] = df[col].apply(lambda x: remove_emojis(x))
        df[col] = df[col].apply(remove_punctuation)
        df[col] = df[col].apply(lambda x: str(x).replace(" s ", " ")) # Remove "s" after removing possessive apostrophe
        df[col] = df[col].apply(lambda x: x.lower()) # Bring to lowercase
        df[col] = df[col].apply(lambda x: " ".join(x.split())) # Replace multiple spaces with a single space
    
    return df


def get_keywords(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords_dict_scores = r.get_word_degrees()
    keywords = list(keywords_dict_scores.keys())
    keywords = " ".join(str(keyword) for keyword in keywords)
    
    return keywords


def recommend(words, recipes, name, similarities, k=5):
    indices = pd.Series(words.index)
    idx = indices[indices == name].index[0]

    # Find similar recipes
    score_series = pd.Series(similarities[idx]).sort_values(ascending=False)

    # Select most similar recipes
    top_recommend = list(score_series.iloc[1:k+1].index)
    
    # Get recipe details
    recommended_list = [list(words.index)[i] for i in top_recommend]
    recommended_recipes = recipes[recipes["name"].isin(recommended_list)]
    recommended_recipes.drop("bag_of_words", axis=1, inplace=True)
    
        
    return recommended_recipes