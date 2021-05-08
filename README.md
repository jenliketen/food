# FOOD RECIPE RECOMMENDATION ENGINE

## Introduction

Recipes and menus, like all other things, are moving online. When we browse one recipe, we are most likely interested in seeing other recipes that we can use in the future. This is why we need to have recommendation engines to predict our browsing pattern and feed us new recipes to try. Most major e-commerce stores and streaming services have recommendation engines. This project aims to build a recommendation engine for recipes.

## How to use

### Data

The data we use is downloaded from Kaggle. There is a data frame for [recipes](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv) and another data frame for [user interactions](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_interactions.csv). We do not provide the data for download here as they are too large for GitHub, but you can download them by clicking on the links.

### Source code

All source coce scripts are stored in the `src` folder and executable from the command line. They contain functions called by the code in the Jupyter Notebooks.

### Jupyter Notebooks

The Jupyter Notebooks are numbered in order of execution. You can just read through them in the order presented!

## Recommendation engine

We have used the following algorithms to build the recommendation engine:

* Most popular items
* Memory-based recommeder
* Model-based recommender
* Market basket analysis
* Content-based recommender

## Conclusion

Each of our algorithms would give the optimal performance in different contexts, and there is no best algorithm. Here are some pointers:

* **Most popular items.** The most popular items algorithm is best for new restaurants/websites with very few ratings.

* **Memory-based recommender.** Memory-based recommenders are very interpretable and good to use when most of the user have rated most of the recipes. However, it is very memory-intensive.

* **Model-based recommender.** Model-based recommenders do not need as much memory as memory-based recommenders, but they are the least interpretable out of all of our approaches.

* **Market basket analysis.** Market basket analysis is a very popular recommendation method for restaurants and e-commerce platforms. It uses the Bayesian apriori algorithm and recommends what usually goes in the "cart" alongside particular items. This works well for food. However, our data in particular does not have enough instances where multiple recipes are used together to create association rules.

* **Content-based recommender.** The content-based recommender is an NLP approach that recommends recipes based on keywords in the recipe's tags, steps, and ingredients. It does not depend on user ratings and as a result, we do not need to worry about a rating matrix - sparse or otherwise. The content-based recommender works very well for our data in terms of recommending similar recipes. However, the recommended recipes may be too similar to take diversity in user taste into consideration.