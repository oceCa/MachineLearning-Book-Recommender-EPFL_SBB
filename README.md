# BookMatch AI — Personalized Book Recommendation System

BookMatch AI is a book recommendation project developed for the Machine Learning course. The goal is to recommend relevant books to users based on their previous interactions, while also providing an interactive Streamlit interface to explore the results in a clear and user-friendly way.

The project combines collaborative filtering, content-based recommendation, metadata enrichment, and a visual web interface. It supports both existing users, who already appear in the dataset, and new users, who can receive recommendations by selecting books they like.

The deployed Streamlit app is available here: https://bookmatch-ai-epfl-sbb.streamlit.app/

```text
Add Streamlit app link here
```

The project video is available here:

```text
Add YouTube video link here:
```

---

# I. Project Objective and Data Description

This project aims to recommend books to users based on their preferences. To do so, we build and evaluate several recommender models and keep the best-performing one as the final recommendation system.

The data used to train the model is provided in:

```text
kaggle_data/interactions_train.csv
```

This file contains the historical user-item interactions and is used to build the interaction matrix indicating which books each user has read or interacted with.

The book metadata is provided in:

```text
kaggle_data/items.csv
```

This file is used to improve the model by drawing similarity between books based on their metadata, including title, author, publisher, ISBN, and subjects.

In some trials, the data was also augmented using external APIs, especially to retrieve book descriptions. However, augmented metadata was not retained for the final recommender model because it did not improve the leaderboard score. Descriptions are still used in the Streamlit interface to improve user experience.

Most of the methods applied in this project are based on the course lectures and recommender lab code. Additional methods, such as the recency decay function and sentence-transformer embeddings, were explored as extensions.

---

# II. Exploratory Data Analysis

Before building the recommender system, an exploratory data analysis was conducted in:

```text
EDA.py
```

The purpose of the EDA was to understand the structure, quality, and main patterns of the dataset before designing the recommendation pipeline.

The EDA focuses on two main files:

```text
kaggle_data/interactions_train.csv
kaggle_data/items.csv
```

The interaction file contains historical user-item interactions, while the item file contains book metadata such as title, author, ISBN, publisher, subjects, and item ID.

All EDA outputs are saved in:

```text
eda_outputs/
```

---

## Purpose of the EDA

The EDA investigates:

```text
dataset size
missing values
duplicates
user activity
item popularity
matrix sparsity
time evolution of interactions
metadata quality
coverage between interactions and metadata
long-tail effects
```

This step is important because recommender systems are strongly affected by data sparsity, popularity concentration, missing metadata, and repeated interactions.

---

## Dataset Overview

The dataset contains:

```text
87,047 interactions
7,838 users
15,291 books
```

The interaction table contains three variables:

```text
u    user ID
i    item ID
t    timestamp
```

The item metadata contains descriptive information about books:

```text
Title
Author
ISBN Valid
Publisher
Subjects
i
```

The EDA confirms that the interaction data and metadata are well aligned. All items appearing in the interaction table are present in the metadata table, and only 182 books in the metadata have never been interacted with. This strong coverage is useful because it allows the recommender to combine collaborative filtering and content-based methods on a consistent catalog.

---

## Data Quality Checks

The interaction dataset is very clean. It contains no missing values and only two exact duplicate rows.

However, there are:

```text
23,044 repeated user-item pairs
```

This means that the same user may have interacted several times with the same item. These repeated interactions are not necessarily errors, but they show that interactions should not always be interpreted as strictly unique binary events.

The item metadata contains more missing values. The main missing fields are:

```text
Author: 17.35% missing
Subjects: 14.54% missing
ISBN Valid: 4.73% missing
```

Titles and publishers are almost complete. Missing authors and subjects are important because they can reduce the quality of content-based recommendation methods.

The missing value plot is saved as:

```text
eda_outputs/items_missing_values_percentage.png
```

---

## Interaction Matrix Sparsity

A key result of the EDA is the extreme sparsity of the user-item matrix.

The matrix density is:

```text
0.000726
```

This corresponds to a sparsity level of approximately:

```text
99.93%
```

This means that only a very small fraction of all possible user-item pairs has been observed. Such sparsity is typical in recommender systems, but it makes the task more difficult, especially for methods relying only on collaborative filtering.

This result supports the choice of combining collaborative methods with content-based information.

---

## User Activity and Item Popularity

User activity is highly right-skewed. Most users interact with only a few books, while a small number of users are much more active.

The average number of interactions per user is:

```text
11.1
```

while the median is only:

```text
6
```

The most active user has:

```text
385 interactions
```

Item popularity follows a similar pattern. Most books receive few interactions, while a small number of books are much more popular.

Books receive on average:

```text
5.76 interactions
```

with a median of:

```text
4 interactions
```

and a maximum of:

```text
380 interactions
```

The corresponding plots are saved as:

```text
eda_outputs/hist_interactions_per_user.png
eda_outputs/hist_interactions_per_user_log.png
eda_outputs/hist_interactions_per_item.png
```

---

## Most Popular Books

The EDA identifies the most interacted books and merges them with the item metadata to retrieve titles and authors.

Examples of highly interacted books include:

```text
Le Petit Robert
Demon Slayer
Vagabond
Spy x Family
L'Arabe du futur
```

This confirms that the catalog is heterogeneous and includes reference books, manga, fiction, academic books, and professional material.

The top popular items are saved in:

```text
eda_outputs/top_popular_items.csv
```

---

## Time Analysis

The timestamp column is converted into datetime format to analyze how interactions evolve over time.

The interactions span the period:

```text
January 2023 to October 2024
```

The monthly interaction plot shows that activity is relatively stable during 2023, usually between 5,000 and 6,000 interactions per month, before declining during 2024. This decline may reflect a real decrease in activity or incomplete coverage of the latest months.

The monthly time evolution plot is saved as:

```text
eda_outputs/interactions_over_time_monthly.png
```

---

## Metadata Analysis

The EDA also investigates the richness of the item metadata.

The publisher analysis shows that the catalog includes literary, academic, legal, and professional publishers. Some of the most frequent publishers include:

```text
Gallimard
Flammarion
Albin Michel
Dunod
Stämpfli
```

The corresponding plot is saved as:

```text
eda_outputs/top_publishers.png
```

The subjects field is also informative. Books have on average:

```text
3.23 subjects
```

with a median of:

```text
2 subjects
```

The most frequent subjects include:

```text
Bandes dessinées
Schweiz
Suisse
Guides pratiques
Mangas
Roman
Droit
```

This shows that the collection is multilingual and diverse, with strong representation of comics, practical guides, fiction, legal books, and Swiss-related topics.

The corresponding plots are saved as:

```text
eda_outputs/subjects_per_book.png
eda_outputs/top_subjects.png
```

---

## Long-Tail Analysis

The EDA includes a long-tail analysis to study how concentrated user attention is across books.

The popularity concentration curve shows that interactions are not evenly distributed across the catalog. The top 20% most popular books account for:

```text
46.07% of all interactions
```

This confirms a clear long-tail effect. A relatively small number of books receives a large share of all interactions, while most books receive limited attention.

The long-tail curve is saved as:

```text
eda_outputs/long_tail_curve.png
```

This result is important because a recommender relying too heavily on popularity may reinforce popularity bias and reduce exposure to less visible books.

---

## Summary of EDA Findings

The EDA shows that the dataset is coherent, rich, and suitable for building a recommender system.

However, it also reveals several classic challenges:

```text
extreme user-item matrix sparsity
highly unequal user activity
strong item popularity concentration
long-tail effect
missing values in author and subject metadata
repeated user-item interactions
```

These findings justify the use of a hybrid recommendation approach combining collaborative filtering and content-based information.

---

# III. Best Model Description

The best model is a hybrid recommender combining:

```text
user-user collaborative filtering
item-item collaborative filtering
content-based filtering
```

The final weights are:

```text
0.3 user-user
0.3 item-item
0.4 content-based
```

This model first creates an interaction matrix from:

```text
kaggle_data/interactions_train.csv
```

It then applies an exponential decay function to give more importance to recent interactions.

The metadata from:

```text
kaggle_data/items.csv
```

is cleaned and transformed into a content matrix using TF-IDF on titles, authors, and subjects.

Once the data is prepared, the code calculates cosine similarities for:

```text
user-user matrix
item-item matrix
content matrix
```

The final hybrid prediction score is computed as:

```python
final_prediction = alpha * user_prediction + beta * item_prediction + (1 - alpha - beta) * content_prediction
```

The final model gives the best Kaggle leaderboard score:

```text
0.1655
```

The final recommendation file used by the interface is:

```text
Submission/Hybrid_0.3_0.3_SBB_R08_final.csv
```

---

# IV. Alternative Models

## Summary of Results

| Model | Precision @ k=10 | Recall @ k=10 |
|---|---:|---:|
| User-user | 0.05653 | 0.29065 |
| Item-item | 0.05561 | 0.26399 |
| Hybrid without content | 0.06082 | 0.29224 |
| Hybrid with content | 0.06120 | 0.29597 |
| Nearest neighbor k=150 without content | 0.06078 | 0.29213 |
| Nearest neighbor k=150 with content | 0.06110 | 0.29482 |
| Decay - Final | 0.06137 | 0.29596 |
| Embedding | 0.06095 | 0.29336 |

---

## User-user and Item-item Models

The user-user and item-item models are the basis of the recommender pipeline. They are built separately using collaborative filtering formulas from the recommender lab.

The user-user model compares users based on their interaction profiles, while the item-item model compares books based on the users who interacted with them.

---

## Hybrid Model Without Metadata

After building user-user and item-item models separately, the next step was to combine them into a hybrid collaborative model.

This model provides recommendations based on both user similarity and item similarity. It improves the leaderboard score compared with the standalone models.

---

## Hybrid Model With Metadata

Another layer of complexity was added using the metadata from:

```text
kaggle_data/items.csv
```

The content-based part uses:

```text
title
author
subjects
```

These fields are cleaned, combined into one text field, and vectorized using TF-IDF.

The TF-IDF parameters include:

```text
max_features = 10000
ngram_range = (1, 3)
min_df = 3
max_df = 0.7
```

The content-based model improves the precision and recall compared with the hybrid model without content. However, the leaderboard score varies depending on the weights.

---

## Recency Decay

The interaction file contains timestamps, which provide information about when a user interacted with a book.

To account for changing user interests over time, the model applies an exponential decay function:

$$
w = e^{-\lambda (t_{\max} - t)}
$$

where:

```text
lambda = optimized decay rate
t_max = most recent timestamp in the dataset
t = timestamp of the interaction
```

This gives more importance to recent interactions and improves the final performance.

---

## Nearest Neighbor

A nearest-neighbor version of the user-user recommender was also tested. The idea was to compare a user only with the k most similar users instead of the full user database.

The best value tested was:

```text
k = 150
```

However, this approach did not improve the results compared with the full hybrid model. For this reason, it was not retained in the final recommender.

---

## Sentence Embeddings

SentenceTransformer embeddings were tested as an alternative to TF-IDF for representing book metadata.

The goal was to capture semantic similarity beyond keyword matching. However, the embedding-based model did not outperform the TF-IDF-based model.

For this reason, the final model uses TF-IDF.

---

## Discussion of Results

The final recommender combines collaborative filtering, content-based similarity, and recency weighting.

The results suggest that interaction data is the strongest signal in this dataset. Metadata improves interpretability and adds useful information, but embeddings and augmented metadata did not improve the final leaderboard score.

The final model therefore keeps a balance between:

```text
user-user collaborative filtering
item-item collaborative filtering
TF-IDF content similarity
recency decay
```

---

# V. User Interface

The user interface is implemented with Streamlit in:

```text
UI_ML.py
```

Its role is to make the recommender system easier to explore, test, and present through an interactive web application.

The interface supports:

```text
recommendations for existing users
cold-start recommendations for new users
seen items for existing users
popular items
similarity exploration between books
book cards with covers
book descriptions
```

---

## Existing-User Recommendation Flow

For existing users, the app does not recompute the recommender.

Instead, it directly loads the final precomputed recommendation file:

```text
Submission/Hybrid_0.3_0.3_SBB_R08_final.csv
```

Each row contains a user ID and a ranked list of recommended item IDs.

When a user selects an existing user ID, the app retrieves the corresponding recommendation list, merges the item IDs with the book metadata, and displays the results as visual book cards.

---

## Seen Items

The app also displays books already seen by the selected user.

These seen items come directly from:

```text
kaggle_data/interactions_train.csv
```

To make the app compatible with Streamlit Cloud memory limits, the deployed version does not build a dense user-item matrix. Instead, it creates a lightweight lookup dictionary mapping each user to the set of books they have already interacted with.

This avoids creating a large users-by-items matrix in memory.

---

## New-User Recommendation Flow

For new users, no historical interaction profile exists in the dataset.

The interface therefore implements a cold-start recommendation workflow. A new user can search for books by:

```text
title
author
subject
category keyword
```

The user can then select books they have read and enjoyed. These selected books are stored temporarily in Streamlit session state.

This allows a user to search across several categories and gradually build a temporary preference profile.

---

## Lightweight New-User Recommendation

The original local version of the interface used full `.npy` similarity matrices:

```text
NPYs/UI-item_similarity.npy
NPYs/UI-content_similarity.npy
```

However, these files were too large for Streamlit Cloud deployment.

The deployed version therefore uses lightweight precomputed top-similarity CSV files:

```text
kaggle_data/top_item_similarities.csv
kaggle_data/top_content_similarities.csv
```

For each selected book, the app retrieves its most similar books according to:

```text
item-item collaborative similarity
content-based similarity
```

The final score combines both sources:

```python
score = alpha * item_similarity_score + (1 - alpha) * content_similarity_score
```

where `alpha` controls the balance between collaborative similarity and content similarity.

This makes the new-user recommender much lighter and suitable for deployment.

---

## Popular Items

The interface includes a popular-items section.

This section displays books ranked by the number of historical interactions.

It serves as a simple non-personalized baseline and helps compare personalized recommendations with popularity-based recommendations.

---

## Similarity Exploration

The similarity-exploration section allows users to select any book from the catalog and display the most similar books.

This feature is not personalized. Instead, it answers the question:

```text
Which books are most similar to this selected book?
```

The deployed version uses:

```text
kaggle_data/top_item_similarities.csv
```

instead of loading the full item-item similarity matrix.

---

## Book Cards and Visual Design

The UI displays recommendations as visual book cards.

Each card can include:

```text
rank
cover image or placeholder cover
title
author
score or similarity score
description expander
```

If a real cover is available, it is displayed using the local `cover_path` from:

```text
kaggle_data/items_with_covers.csv
```

If no real cover is available, the UI generates a CSS-based placeholder cover so that the interface remains visually consistent.

The decorative library background image used in the UI comes from:

```text
https://www.elaee.com/2017/08/21/28089-plus-belles-bibliotheques-monde-quil-ny-a-linternet-vie
```

---

## Book Descriptions

Book descriptions are stored separately in:

```text
kaggle_data/item_descriptions.csv
```

This file is generated by:

```text
download_descriptions.py
```

Descriptions are retrieved using external sources such as Open Library and Google Books when available. When no external description is found, the script creates a fallback description from the local metadata, such as title, author, publisher, and subjects.

Descriptions are merged with the item metadata in the Streamlit app using the item ID `i`.

Each book card includes a `View description` expander. If a description is available, it is displayed inside the expander. Otherwise, the UI displays:

```text
No description available for this book.
```

---

# VI. Project Structure

The main files and folders are:

```text
UI_ML.py
Recommender Main Code.py
EDA.py
download_covers_2.py
download_descriptions.py
requirements.txt

kaggle_data/
    items.csv
    items_with_covers.csv
    interactions_train.csv
    item_descriptions.csv
    top_item_similarities.csv
    top_content_similarities.csv
    sample_submission.csv
    covers/

Submission/
    Hybrid_0.3_0.3_SBB_R08_final.csv

eda_outputs/
    figures and summary tables

clean_items.csv
```

The full `.npy` similarity matrices were used during local experimentation, but the deployed Streamlit app uses lightweight top-similarity CSV files to avoid memory issues on Streamlit Cloud.

---

# VII. Main Files

## `EDA.py`

This file performs the exploratory data analysis.

It loads:

```text
kaggle_data/interactions_train.csv
kaggle_data/items.csv
```

and generates figures and summary tables in:

```text
eda_outputs/
```

Run it with:

```bash
python EDA.py
```

---

## `Recommender Main Code.py`

This file contains the main recommender pipeline.

It loads the original book metadata and user interaction data, builds recommender models, evaluates them, and generates the final recommendation CSV.

Main inputs:

```text
kaggle_data/items.csv
kaggle_data/interactions_train.csv
```

Main output:

```text
Submission/Hybrid_0.3_0.3_SBB_R08_final.csv
```

---

## `UI_ML.py`

This is the main Streamlit application.

It loads the recommender outputs and provides the interactive interface.

Run it locally with:

```bash
streamlit run UI_ML.py
```

---

## `download_covers_2.py`

This script downloads book covers and stores them locally.

The output file is:

```text
kaggle_data/items_with_covers.csv
```

The cover images are stored in:

```text
kaggle_data/covers/
```

This script can take a long time and does not need to be rerun if the covers are already downloaded.

---

## `download_descriptions.py`

This script retrieves book descriptions without downloading covers.

The output file is:

```text
kaggle_data/item_descriptions.csv
```

The script uses external sources when available and fallback descriptions when no external description is found.

---

# VIII. Data Files

## `kaggle_data/items.csv`

Original book metadata.

Used for content-based recommendation and metadata display.

---

## `kaggle_data/interactions_train.csv`

Historical user-item interactions.

Used to build collaborative filtering models and identify seen items.

---

## `kaggle_data/items_with_covers.csv`

Book metadata enriched with local cover paths.

Used by the Streamlit UI to display book covers.

---

## `kaggle_data/item_descriptions.csv`

Book descriptions used in the Streamlit UI.

---

## `kaggle_data/top_item_similarities.csv`

Precomputed top item-item similar books for each item.

Used by the deployed Streamlit app for:

```text
new-user recommendations
similarity exploration
```

---

## `kaggle_data/top_content_similarities.csv`

Precomputed top content-based similar books for each item.

Used by the deployed Streamlit app for:

```text
new-user recommendations
```

---

## `Submission/Hybrid_0.3_0.3_SBB_R08_final.csv`

Final precomputed recommendations for existing users.

Each row contains:

```text
user_id
recommendation
```

where `recommendation` is a space-separated list of recommended item IDs.

---

## `clean_items.csv`

Cleaned metadata used to improve title and author display in the interface.

---

# IX. How to Run the Project

## 1. Install dependencies

Install the required packages with:

```bash
pip install -r requirements.txt
```

---

## 2. Run the EDA

```bash
python EDA.py
```

This creates the `eda_outputs/` folder and saves the EDA figures and summary tables.

---

## 3. Run the recommender

```bash
python "Recommender Main Code.py"
```

This generates or updates the final recommendation file:

```text
Submission/Hybrid_0.3_0.3_SBB_R08_final.csv
```

---

## 4. Download covers if needed

```bash
python download_covers_2.py
```

This generates:

```text
kaggle_data/items_with_covers.csv
kaggle_data/covers/
```

This step can take a long time and does not need to be repeated if covers are already available.

---

## 5. Download descriptions if needed

```bash
python download_descriptions.py
```

This generates:

```text
kaggle_data/item_descriptions.csv
```

---

## 6. Launch the Streamlit app locally

```bash
streamlit run UI_ML.py
```

The app will open in the browser.

---

# X. Required Files Before Running the UI

Before launching the interface, the following files should be available:

```text
kaggle_data/items_with_covers.csv
kaggle_data/item_descriptions.csv
kaggle_data/interactions_train.csv
kaggle_data/top_item_similarities.csv
kaggle_data/top_content_similarities.csv
clean_items.csv
Submission/Hybrid_0.3_0.3_SBB_R08_final.csv
```

The `.npy` similarity matrices are not required for the deployed Streamlit app. They were replaced by lightweight CSV files containing only the top similar items for each book.

---

# XI. Streamlit Cloud Deployment

The Streamlit app is deployed from the GitHub repository.

The main Streamlit file is:

```text
UI_ML.py
```

Because Streamlit Cloud has memory limits, the deployed version avoids loading full dense similarity matrices. Instead, it uses:

```text
kaggle_data/top_item_similarities.csv
kaggle_data/top_content_similarities.csv
```

This makes the app lighter and more stable online.

When changes are made locally, they must be pushed to GitHub:

```bash
git add .
git commit -m "Update Streamlit app"
git push
```

Streamlit Cloud then pulls the latest version from GitHub and updates the app.

---

# XII. Cache Note

The Streamlit app uses caching to make loading faster.

If a CSV file is updated, for example the recommendation CSV or the description CSV, it may be necessary to clear the Streamlit cache or reboot the app from Streamlit Cloud.

---

# XIII. Summary

This project combines an offline recommender model with an interactive Streamlit interface.

The recommender model computes hybrid recommendations based on collaborative filtering, content-based similarity, and recency weighting. The Streamlit interface then makes the recommender accessible through a visual and interactive app.

Existing users receive recommendations from the precomputed hybrid recommendation CSV. New users receive cold-start recommendations based on selected books and lightweight top-similarity files. The interface also improves interpretability by showing seen items, popular books, similar books, covers, and descriptions.

Overall, BookMatch AI turns the recommender system into an interactive tool that is easier to understand, test, present, and use.