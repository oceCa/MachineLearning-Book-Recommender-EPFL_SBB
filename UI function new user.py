# Recommender code for new user

import pandas as pd
import numpy as np

items=pd.read_csv("UI-items.csv")
item_sim=np.load("NPYs/UI-item_similarity.npy")
content_sim=np.load("NPYs/UI-content_similarity.npy")
# user_sim=np.load("kaggle_data/UI-user_similarity.npy")

def recommend_new_user(selected_books, item_similarity, content_sim, top_k=10, alpha=0.5):
    """
    selected_books:
        list of item indices already read by the user

    alpha:
        weight for item-item CF
        (1-alpha) for content similarity
    """

    n_items = item_similarity.shape[0]
    
    #interaction matrix for new user
    user_vector = np.zeros(n_items)
    user_vector[selected_books] = 1


    item_scores = (user_vector.dot(item_similarity) / (np.abs(item_similarity).sum(axis=1) + 1e-9))

    content_scores = (user_vector.dot(content_sim) / (np.abs(content_sim).sum(axis=1) + 1e-9))

    final_scores = (alpha * item_scores+ (1 - alpha) * content_scores)

    top_items = np.argsort(final_scores)[-top_k:][::-1]

    return top_items
