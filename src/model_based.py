#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import coo_matrix
from surprise.model_selection import cross_validate
from surprise import SVD, NMF, NormalPredictor, BaselineOnly


def map_dfs(recipes, interactions):
    def get_new_id(df, col):
        values = df[col]
        unique_values = np.unique(values)
        new_id = dict([(x, y) for y, x in enumerate(unique_values)])
        
        return new_id

    new_recipe_id = get_new_id(recipes, "id")
    new_user_id = get_new_id(interactions, "user_id")
    
    recipes.drop(["level_0", "index"], axis=1, inplace=True)
    recipes["id"] = recipes["id"].map(new_recipe_id)
    recipes = recipes[~recipes["name"].isna()]
    interactions["user_id"] = interactions["user_id"].map(new_user_id)
    interactions["recipe_id"] = interactions["recipe_id"].map(new_recipe_id)
    interactions = interactions[interactions["rating"] != 0]
    
    return recipes, interactions


def get_benchmark(data, algorithms):
    benchmark = []
    
    for algorithm in algorithms:
        results = cross_validate(algorithm, data, measures=["RMSE"], cv=3, verbose=False)

        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(" ")[0].split(".")[-1]], index=["Algorithm"]))
        benchmark.append(tmp)

    benchmarks = pd.DataFrame(benchmark).set_index("Algorithm").sort_values("test_rmse")
    return benchmarks


def make_predictions_df(predictions, trainset):
    def get_Iu(uid, trainset):
        """
        args: 
        uid: the id of the user
        returns:
        the number of items rated by the user
        """
        try:
            return len(trainset.ur[trainset.to_inner_uid(uid)])
        except ValueError: # user was not part of the trainset
            return 0

    def get_Ui(iid, trainset):
        """
        args:
        iid: the raw id of the item
        returns:
        the number of users that have rated the item.
        """
        try:
            return len(trainset.ir[trainset.to_inner_iid(iid)])
        except ValueError:
            return 0

    df_predictions = pd.DataFrame(predictions, columns=["uid", "iid", "rui", "est", "details"])
    df_predictions["Iu"] = df_predictions["uid"].apply(get_Iu, args=(trainset, ))
    df_predictions["Ui"] = df_predictions["iid"].apply(get_Ui, args=(trainset, ))
    df_predictions["err"] = abs(df_predictions["est"] - df_predictions["rui"])

    return df_predictions


def inspect_results(df, recipe_id, user_id):
    user = df[(df["recipe_id"] == recipe_id) & (df["user_id"] == user_id)]
    other_users = df[(df["recipe_id"] == recipe_id) & (df["user_id"] != user_id)]
    
    rating = df[df["recipe_id"] == recipe_id]["rating"]
    
    figure = plt.figure(figsize=(5, 5))
    plt.hist(rating)
    plt.xlabel("Rating")
    plt.ylabel("Number of ratings")
    plt.title("Distribution of ratings received by {}".format(recipe_id))
    
    return user, other_users


# def get_threshold(predictions):
#     final = []
#     for threshold in np.arange(0, 5.5, 0.5):
#         tp = 0
#         fn = 0
#         fp = 0
#         tn = 0

#         temp = []
#         for uid, _, true_r, est, _ in predictions:
#             if(true_r >= threshold):
#                 if(est >= threshold):
#                     tp += 1
#                 else:
#                     fn += 1
#             else:
#                 if(est>=threshold):
#                     fp += 1
#                 else:
#                     tn += 1

#             if tp == 0:
#                 precision = 0
#                 recall = 0
#                 f1 = 0
#             else:
#                 precision = tp / (tp + fp)
#                 recall = tp / (tp + fn)
#                 f1 = 2 * (precision * recall) / (precision + recall)

#         temp = [threshold, tp,fp,tn ,fn, precision, recall, f1]
#         final.append(temp)

#     results = pd.DataFrame(final)
#     results.rename(columns={0: "threshold", 1: "tp", 2: "fp", 3: "tn", 4:"fn",
#                             5: "precision", 6: "recall", 7: "f1"}, inplace=True)
    
#     return results


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # Map the predictions to each user
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@k: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@k: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


def get_pr_ks(predictions):
    results = []

    for i in range(2, 11):
        precisions, recalls = precision_recall_at_k(predictions, k=i, threshold=3.5)

        prec = sum(prec for prec in precisions.values()) / len(precisions)
        rec = sum(rec for rec in recalls.values()) / len(recalls)           

        results.append({"k": i, "precision": prec, "recall": rec})    
        results_df = pd.DataFrame(results)     

    return results_df


def plot_prs(pr_ks):
    x = pr_ks["k"]
    y1 = pr_ks["precision"]
    y2 = pr_ks["recall"]

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(x, y1, "darkorange")
    ax2.plot(x, y2, "purple")

    ax1.set_xlabel(r"$k$")
    ax1.set_ylabel("Precision", color="darkorange")
    ax2.set_ylabel("Recall", color="purple")
    ax1.set_title(r"Precision and recall at various $k$ values")

    ax2.spines["left"].set_color("darkorange")
    ax2.spines["right"].set_color("purple")

    ax1.tick_params(axis="y", colors="darkorange", which="major")
    ax2.tick_params(axis="y", colors="purple", which="major")


def recommend(uid, iids, recipes, algo, rui=3.5, k=3):
    iid_to_test = [i for i in recipes["id"] if i not in iids]

    test_set = [[uid ,iid, rui] for iid in iid_to_test]
    predictions = algo.test(test_set)
    pred_uid = [pred.uid for pred in predictions]
    pred_iid = [pred.iid for pred in predictions]
    pred_est = [pred.est for pred in predictions]
    pred_df = pd.DataFrame({"uid": pred_uid,
                            "iid": pred_iid,
                            "est": pred_est})
    pred_df = pred_df.sort_values(by="est", ascending=False)
    top_k = pred_df.head(k)["iid"].tolist()

    results = recipes[recipes["id"].isin(top_k)]
    return results