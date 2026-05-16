# I. Project Objective and Data Description

This project aims to make book recommendations to users based on their preferences. To do so, we build a model that gives the most accurate recommendations.

The data used to train the model is provided in `user_preferences.csv`. This data serves to build a binary interaction matrix indicating which books each user has read.

The `items.csv` matrix is used to improve the model by drawing similarity between books based on their metadata. This yields better results in some cases, including the final recommender model.

In some trials, the data is also augmented using the Google Books API, but this is not retained for the final model as explained in later sections.

Most of the methods applied in building this model are based on the class lectures and lab session code. Some other methods such as the decay function and sentence-transform embeddings are attempted based on more extended research assisted by AI tools.

---
## Exploratory Data Analysis

Before building the recommender system, an exploratory data analysis was conducted in `EDA.py` to understand the structure, quality, and main patterns of the dataset. The EDA focuses on two main files:

```text
kaggle_data/interactions_train.csv
kaggle_data/items.csv
```

The interaction file contains the historical user-item interactions, while the item file contains the book metadata such as title, author, ISBN, publisher, subjects, and item ID.

The EDA script generates printed summaries, diagnostic checks, plots, and summary tables. All generated outputs are saved in:

```text
eda_outputs/
```

---

### Purpose of the EDA

The goal of the EDA is to understand the dataset before designing the recommendation pipeline. In particular, the analysis investigates:

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

### Dataset Overview

The dataset contains:

```text
87,047 interactions
7,838 users
15,291 items
```

The interaction table contains three main variables:

```text
u    user ID
i    item ID
t    timestamp
```

The item metadata contains descriptive information about books, including:

```text
Title
Author
ISBN Valid
Publisher
Subjects
i
```

The EDA confirms that the two datasets are well aligned. All items appearing in the interactions table are present in the metadata table, and only 182 books in the metadata have never been interacted with. This strong coverage is useful because it allows both collaborative filtering and content-based methods to rely on a consistent item catalog.

---

### Data Quality Checks

The script checks missing values, duplicate rows, duplicate user-item pairs, and duplicate item IDs.

The interaction dataset is very clean: it contains no missing values and only two exact duplicate rows. However, there are 23,044 repeated user-item pairs. This means that the same user may have interacted several times with the same item. These repeated interactions are not necessarily errors, but they show that interactions should not be interpreted only as unique binary events without reflection.

The item metadata contains more missing values. The main missing fields are:

```text
Author: 17.35% missing
Subjects: 14.54% missing
ISBN Valid: 4.73% missing
```

Titles and publishers are almost complete. The missing author and subject fields are important because they can reduce the quality of content-based recommendation methods.

One of the EDA plots visualizes the percentage of missing values by metadata field:

```text
eda_outputs/items_missing_values_percentage.png
```

---

### Interaction Matrix Sparsity

A key result of the EDA is the extreme sparsity of the user-item matrix. Based on the number of users, items, and observed interactions, the matrix density is:

```text
0.000726
```

This corresponds to a sparsity level of approximately:

```text
99.93%
```

This means that only a very small fraction of all possible user-item pairs has been observed. Such sparsity is typical in recommender systems, but it makes the task more difficult, especially for methods relying only on collaborative filtering.

The EDA therefore supports the choice of combining collaborative methods with content-based information.

---

### User Activity and Item Popularity

The EDA analyzes how many interactions each user has and how many interactions each item receives.

User activity is highly right-skewed. Most users interact with only a few books, while a small number of users are much more active. The average number of interactions per user is approximately:

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

Item popularity follows a similar pattern. Most books receive few interactions, while a small number of books are much more popular. Books receive on average:

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

The following plots are generated to visualize these distributions:

```text
eda_outputs/hist_interactions_per_user.png
eda_outputs/hist_interactions_per_user_log.png
eda_outputs/hist_interactions_per_item.png
```

These plots show a strong right-skewed pattern, which is typical of recommender datasets.

---

### Most Popular Books

The EDA identifies the most interacted items and merges them with the item metadata to retrieve the corresponding titles and authors. The most popular books include a diverse set of content, such as reference books, manga, fiction, and academic or professional material.

Examples of highly interacted books include:

```text
Le Petit Robert
Demon Slayer
Vagabond
Spy x Family
L'Arabe du futur
```

This confirms that the catalog is heterogeneous and that the recommender must handle different types of books rather than a single homogeneous domain.

The top popular items are saved in:

```text
eda_outputs/top_popular_items.csv
```

---

### Time Analysis

The timestamp column is converted into datetime format to analyze how interactions evolve over time.

The interactions span the period from:

```text
January 2023 to October 2024
```

The monthly interaction plot shows that activity is relatively stable during 2023, usually between 5,000 and 6,000 interactions per month, before declining during 2024. This decline may reflect a real decrease in activity or incomplete coverage for the latest months.

The monthly time evolution plot is saved as:

```text
eda_outputs/interactions_over_time_monthly.png
```

---

### Metadata Analysis

The EDA also investigates the quality and richness of the item metadata.

For text fields such as title, author, publisher, and subjects, the script computes length statistics. It also identifies the most frequent authors, publishers, and subjects.

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

The subjects field is also informative. Books have on average around:

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

The corresponding subject plots are saved as:

```text
eda_outputs/subjects_per_book.png
eda_outputs/top_subjects.png
```

---

### Joined Analysis: Interactions and Metadata

The script merges the interaction data with the item metadata to analyze which authors and publishers are most interacted with.

This joined analysis checks whether metadata is correctly attached to the interaction records and identifies the most interacted authors and publishers.

The merge is successful, with no missing titles after joining the interaction table with the metadata.

Generated plots include:

```text
eda_outputs/most_interacted_authors.png
eda_outputs/most_interacted_publishers.png
```

One limitation is that the author field contains many missing values. As a result, the category `Unknown` can dominate author-level statistics, which reduces the interpretability of author-based analysis unless missing authors are filtered out.

---

### Long-Tail Analysis

The EDA includes a long-tail analysis to study how concentrated user attention is across items.

The popularity concentration curve shows that interactions are not evenly distributed across the catalog. The top 20% most popular items account for:

```text
46.07% of all interactions
```

This confirms a clear long-tail effect: a relatively small number of books receives a large share of all interactions, while most books receive limited attention.

The long-tail curve is saved as:

```text
eda_outputs/long_tail_curve.png
```

This result is important for the recommender design because models that rely too heavily on popularity may reinforce popularity bias and reduce exposure to less visible books.

---

### Output Files Generated by the EDA

The EDA script saves both figures and summary tables in the `eda_outputs/` folder.

Main figures include:

```text
hist_interactions_per_user.png
hist_interactions_per_user_log.png
hist_interactions_per_item.png
interactions_over_time_monthly.png
items_missing_values_percentage.png
top_authors.png
top_publishers.png
subjects_per_book.png
top_subjects.png
most_interacted_authors.png
most_interacted_publishers.png
long_tail_curve.png
```

Main summary tables include:

```text
user_activity_describe.csv
item_popularity_describe.csv
top_popular_items.csv
items_missing_percentage.csv
```

---

### How to Run the EDA

The EDA can be run with:

```bash
python EDA.py
```

Before running it, the following files must be available:

```text
kaggle_data/interactions_train.csv
kaggle_data/items.csv
```

The script will automatically create the output folder if it does not already exist:

```text
eda_outputs/
```

---

### Summary of EDA Findings

The EDA shows that the dataset is coherent, rich, and suitable for building a recommender system. The interaction and metadata files are well aligned, and the metadata provides useful content information through titles, publishers, authors, and subjects.

However, the analysis also reveals several classic challenges of recommender systems:

```text
extreme user-item matrix sparsity
highly unequal user activity
strong item popularity concentration
long-tail effect
missing values in author and subject metadata
repeated user-item interactions
```

These findings justify the use of a hybrid recommendation approach combining collaborative filtering and content-based information. The EDA also motivates the inclusion of popularity baselines, seen-item filtering, and metadata enrichment in the final interface.

# II. Best Model Description

The best model is a combination between a user-user, item-item, and content-based recommender system. They are weighted `0.3`, `0.3`, and `0.4` respectively in the final recommendation.

This model first creates an interaction matrix from the `interactions_train.csv` provided in the data files. It weights each interaction with an exponential decay function that gives the highest weight to the most recent entries as compared to the most recent one across the entire dataset (Section III.6).

Then, the code cleans the metadata for each book as given in `items.csv`, and creates a content matrix using the title, author, and subject of each book with a TF-IDF approach (Section III.4).

Once the data is prepared, the code calculates the cosine similarity for the:
- user-user matrix
- item-item matrix
- content matrix

and computes the likelihood that the user will interact with an item based on the following formulas provided in the recommender lab:

## Item-item collaborative filtering

$$
p_u(i)=\frac{\sum_{i'} sim(i,i') \cdot R_u(i')}{\sum_{i'} sim(i,i')}
$$

## User-user collaborative filtering

$$
p_u(i)=\frac{\sum_{u'} sim(u,u') \cdot R_{u'}(i)}{\sum_{u'} sim(u,u')}
$$

Where:

- $p_u(i)$ is the likelihood of user $u$ interacting with item $i$
- $sim(i,i')$ is the cosine similarity between items $i$ and $i'$
- $sim(u,u')$ is the cosine similarity between users $u$ and $u'$
- $R_u(i')$ is 1 if user $u$ has already interacted with item $i'$, and otherwise 0
- $R_{u'}(i)$ is 1 if user $u'$ has already interacted with item $i$, and otherwise 0

Finally, the code gives a final prediction combining all three models, weighted:
- `0.3` for user-user
- `0.3` for item-item
- `0.4` for content-based

This is the weight combination that gives the best score on the Kaggle leaderboard: **0.1655**.

Sample recommendations based on this model can be collected on our website designed specifically to run this recommender.

Many other approaches were considered while building this model. The following section details the most relevant ones, as well as the precision and recall of the results.

---

# III. Alternative Models

## 1. Summary of Results

The following table presents a summary of the precision and recall for several attempted models. The components and operation of each model are further detailed in the rest of the section.

| Model | Precision @ k=10 | Recall @ k=10 |
|---|---|---|
| User-user (R00) | 0.05653 | 0.29065 |
| Item-item (R00) | 0.05561 | 0.26399 |
| Hybrid without content (R01) | 0.06082 | 0.29224 |
| Hybrid with content (R01-with) | 0.06142 | 0.29726 |
| Nearest neighbor k=150 (R07) without content (0.55, 0.45) | 0.06078 | 0.29213 |
| Nearest neighbor k=150 (R07) with content (0.55, 0.45) | 0.06110 | 0.29482 |
| Decay (R08-decay) - Final | 0.06137 | 0.29596 |
| Embedding | 0.06095 | 0.29336 |

> Rerun nearest neighbor as a save-as of hybrid with content to check if results for neighbor are accurate.

---

## 2. User-user and Item-item Models

These two models are the basis of the final code. They are built separately using the same probability formulas presented above, as per the recommender lab.

If run on the full data, the user-user model results in the baseline score of `0.1452` on the leaderboard.

---

## 3. Hybrid Model Without Metadata

Once a first simple model was built, the immediate idea was to combine them.

This model provides recommendations based on both:
- the user-user model
- the item-item model

For a weight of `0.55` for the user-user model, it provides the score of `0.1643` on the leaderboard.

---

## 4. Hybrid Model With Metadata

Another layer of complexity was added to the model with the metadata in the `items.csv` file.

The data included in this model is:
- title
- author
- subject

The subjects are cleaned into distinct words, and punctuation as well as stop words are removed.

All three fields are then combined into one single content field, which is converted into vectors using TF-IDF with the following arguments:

- `max_features = 10000`
- `ngram_range = (1, 3)`
- `min_df = 3`
- `max_df = 0.7`

The precision and recall of this model are superior to the one without content for a combination of weights where the user-user model is `0.5`.

However, the score on the Kaggle leaderboard remains between `0.16` and `0.1638` depending on the weights, which is not an improvement over the previous model.

This content model based on metadata is implemented in the final code nonetheless, as its precision and recall are superior, and the final model performs better with it than without it on the leaderboard.

The `items.csv` metadata was also augmented using the Google Books API. This provided:
- descriptions of certain books
- categories for most books

However, other data such as rankings could not be obtained, and the model did not yield a better result than the non-augmented metadata.

For this reason, only non-augmented metadata is used in the final model.

---

## 5. Nearest Neighbor

In the interest of improving the user-user model, we attempt to find a similarity between user \(u\) and only \(k\) of its nearest neighbors.

A series of trials performed to obtain the optimal number of neighbors for precision and recall yields:

\[
k = 150
\]

However, adding this to the model does not improve the precision and recall compared to the hybrid model.

We assume that:
- a too-small \(k\), such as `25` or `50`, deprives the model of relevant data
- hence producing lower precision and recall than larger values such as `150`

But \(k=150\) did not yield a better result because neighbors beyond 150 might not have been relevant enough to improve performance even if included.

For this reason, the final model does not include the k-neighbor approach.

---

## 6. Decay

The `interactions_train.csv` file provides information on when a book was read by a specific user.

We use this information to build a decay function that weights the interaction matrix based on the following formula:

$$
w = e^{-\lambda (t_{\max} - t)}
$$

Where:

- $\lambda$ is the optimized decay rate
- $t_{\max}$ is the most recent timestamp in the dataset
- $t$ is the time at which user $u$ read item $i$

This approach combined with the non-augmented metadata hybrid model results in the best leaderboard score:

$$
0.1655
$$

---

## 7. Embedding

In an attempt to improve the metadata matrix of the final model, we use SentenceTransformer embeddings instead of the TF-IDF method.

Sentence Transformers generate dense semantic embeddings of the metadata, enabling the recommender system to capture contextual similarity between books beyond the simple keyword matching of TF-IDF.

The model yields a maximum leaderboard score of `0.1643`, for weights shifted more towards the content matrix, with the following distribution:

- `0.2` user-user
- `0.2` item-item
- `0.6` content matrix

all other elements of the final code included as described in Section II.

This is not better than the code without embeddings. For this reason, the final recommender model operates on TF-IDF.

---

## 8. Discussion of Results

The final code builds recommendations based on a hybrid model of:
- user-user collaborative filtering
- item-item collaborative filtering
- content-based filtering using TF-IDF metadata

Enhancements of this hybrid model were made using a decay function to give more importance to books that were read more recently.

Fine-tuning parameters such as:
- TF-IDF arguments
- weight distributions

allowed further optimization of the overall performance.

As a result of the other attempted approaches, we also conclude that the interaction data is more informative than semantic content features.

This is supported by the fact that:
- embeddings
- augmented metadata

decreased the performance of the model.

---

# IV. User Interface

The user interface is designed using Streamlit.

First, it allows users to obtain recommendations based on `user_preferences.csv`. In other words, the users in the provided dataset act as registered users of the website who can immediately obtain recommendations.

The user interface also provides an option for new users to obtain recommendations.

This option is based on the same recommender, called as a function to compute the similarity of the new user to the already-computed similarity matrices.

The new user can enter as many books as they like.

However, because there is no field allowing the user to indicate when they read the books (as this is not usual information to request from users), this onboarding model does not use the decay function for the new user, while still maintaining this feature in the pretrained recommendation model.


# BookMatch AI — Interactive Book Recommender

This project implements a book recommendation system and an interactive Streamlit interface called **BookMatch AI**. The objective is to recommend books to both existing users, who already appear in the interaction dataset, and new users, who do not yet have historical interactions.

The project combines several recommendation strategies, including collaborative filtering, content-based similarity, and a hybrid recommender. The final outputs are made accessible through a user-friendly interface that displays recommendations as book cards, with covers, descriptions, seen items, popular items, and similarity exploration.

---

## Project Structure

The main files and folders used in the project are:

```text
UI_ML.py
Recommender Main Code.py
download_covers_2.py
download_descriptions.py

kaggle_data/
    items.csv
    items_with_covers.csv
    interactions_train.csv
    item_descriptions.csv
    covers/

Submission/
    Hybrid_0.3_0.3_SBB_R08_final.csv

NPYs/
    UI-item_similarity.npy
    UI-content_similarity.npy

clean_items.csv
```

---

## Main Files

### `Recommender Main Code.py`

This file contains the main recommender pipeline. It loads the original book metadata and user interaction data, builds the user-item interaction matrix, computes similarity matrices, evaluates recommendation quality, and generates the final recommendation CSV for existing users.

The main input files are:

```text
kaggle_data/items.csv
kaggle_data/interactions_train.csv
```

The main output file is:

```text
Submission/Hybrid_0.3_0.3_SBB_R08_final.csv
```

This CSV contains, for each existing user, a ranked list of recommended item IDs.

The recommender also saves the similarity matrices used by the UI:

```text
NPYs/UI-item_similarity.npy
NPYs/UI-content_similarity.npy
```

These `.npy` files allow the Streamlit interface to use the recommender outputs without recomputing the full model.

---

### `UI_ML.py`

This is the main Streamlit application. It loads the outputs produced by the recommender model and provides an interactive front-end for exploring recommendations.

The interface supports:

- recommendations for existing users;
- cold-start recommendations for new users;
- seen items for existing users;
- popular items;
- similarity exploration between books;
- visual book cards with real covers when available;
- generated placeholder covers when no real cover exists;
- book descriptions displayed through expandable sections.

The app can be launched with:

```bash
streamlit run UI_ML.py
```

---

### `download_covers_2.py`

This script downloads book covers and stores them locally. It uses ISBN when available and falls back to title-author search when needed.

The output is:

```text
kaggle_data/items_with_covers.csv
```

This file contains the original item metadata enriched with:

```text
cover_path
cover_source
```

The actual cover images are stored in:

```text
kaggle_data/covers/
```

This script does not need to be rerun once the covers have already been downloaded.

---

### `download_descriptions.py`

This script downloads book descriptions separately from the covers. It queries the Google Books API using ISBN, title, and author information.

The output is:

```text
kaggle_data/item_descriptions.csv
```

This file contains textual metadata such as:

```text
i
Title
Author
description
api_title
api_authors
api_source
api_query
```

Descriptions are kept in a separate CSV to avoid redownloading or recomputing book covers. The Streamlit UI later merges this file with the existing item metadata using the item ID `i`.

---

## Data Files

### `kaggle_data/items.csv`

This is the original item metadata file. It contains information about the books, such as title, author, ISBN, publisher, and subjects.

It is used by the recommender model to build content-based representations of books.

---

### `kaggle_data/interactions_train.csv`

This file contains the historical user-item interactions.

It includes columns such as:

```text
u
i
t
```

where:

- `u` is the user ID;
- `i` is the item ID;
- `t` is the interaction timestamp.

This file is used to build the user-item interaction matrix.

---

### `kaggle_data/items_with_covers.csv`

This file contains the item metadata enriched with local cover paths. It is used by the Streamlit UI to display real book covers when available.

---

### `kaggle_data/item_descriptions.csv`

This file contains book descriptions downloaded from the Google Books API. It is merged with the item metadata in the UI.

---

### `clean_items.csv`

This file contains cleaned metadata, especially cleaned titles and authors. It is merged with `items_with_covers.csv` in the UI to improve display quality.

---

### `Submission/Hybrid_0.3_0.3_SBB_R08_final.csv`

This file contains the final precomputed recommendations for existing users. Each row corresponds to one user and contains a space-separated list of recommended item IDs.

Example structure:

```text
user_id,recommendation
0,123 456 789 ...
1,42 91 302 ...
```

The UI uses this file directly for existing-user recommendations.

---

### `NPYs/UI-item_similarity.npy`

This file stores the item-item collaborative filtering similarity matrix. It is computed offline in the recommender code and loaded by the UI.

It is used for:

- new-user recommendations;
- similarity exploration.

---

### `NPYs/UI-content_similarity.npy`

This file stores the content-based similarity matrix. It is computed offline from the book metadata.

It is used for:

- new-user recommendations.

---

## Recommender Logic

The recommender uses a hybrid approach combining three sources of information:

1. user-user collaborative filtering;
2. item-item collaborative filtering;
3. content-based similarity.

The final hybrid prediction score is computed as:

```python
final_prediction = alpha * user_prediction + beta * item_prediction + (1 - alpha - beta) * content_prediction
```

In the final version used for the UI, the recommendations are generated offline and saved into:

```text
Submission/Hybrid_0.3_0.3_SBB_R08_final.csv
```

This design avoids recomputing the full recommender inside the Streamlit app.

---

## Existing-User Recommendation Flow

For users already present in the training dataset, the interface does not recompute recommendations. Instead, it loads the precomputed CSV:

```text
Submission/Hybrid_0.3_0.3_SBB_R08_final.csv
```

The user selects a user ID in the interface. The app then retrieves the recommendation list corresponding to that user and merges the recommended item IDs with book metadata.

This is handled by:

```python
get_recommendations_from_csv(user_id, recommendations_df, items, top_k)
```

The recommendations are then displayed as book cards.

---

## Seen Items

For existing users, the UI also displays the books that the user has already interacted with.

These are not recommendations. They come directly from:

```text
kaggle_data/interactions_train.csv
```

The app reconstructs a user-item interaction matrix:

```python
def create_data_matrix(data, n_users, n_items):
    matrix = np.zeros((n_users, n_items))
    matrix[data["u"].values, data["i"].values] = 1
    return matrix
```

For a selected user, seen items are identified with:

```python
seen_items = np.where(matrix[int(user_id)] == 1)[0]
```

The resulting item IDs are merged with the book metadata and displayed in the `Seen items` tab.

The interface also includes a `Remove seen items` option. When enabled, books already seen by the selected user are removed from the displayed recommendation list.

---

## New-User Recommendation Flow

For new users, no historical interaction profile exists in the training data. Therefore, the interface implements a cold-start recommendation flow.

The user can search for a book category, title, author, or subject, and then select books they have already read and enjoyed.

The selected books are stored in Streamlit session state:

```python
st.session_state.liked_item_ids_new_user
```

This allows the user to search across multiple categories without losing previously selected books.

For example, a user can first select romance books, then search for politics or history books, and keep all selected books in the same temporary profile.

---

## New-User Scoring

For new users, the recommendation score combines two similarity sources:

```python
final_scores = alpha * item_scores + (1 - alpha) * content_scores
```

where:

- `item_scores` are based on item-item collaborative filtering;
- `content_scores` are based on content-based similarity;
- `alpha` controls the balance between collaborative similarity and content similarity.

The two similarity matrices are loaded from:

```text
NPYs/UI-item_similarity.npy
NPYs/UI-content_similarity.npy
```

They are loaded using:

```python
item_sim = np.load("NPYs/UI-item_similarity.npy", mmap_mode="r")
content_sim = np.load("NPYs/UI-content_similarity.npy", mmap_mode="r")
```

The use of `mmap_mode="r"` avoids loading the full matrices directly into memory, which is useful because the matrices are large.

The new-user recommendation function creates a temporary user vector where selected books are marked as liked. The selected books are then excluded from the final recommendation list so that the app does not recommend books the user has already selected.

---

## Book Search for New Users

The new-user section includes a flexible search feature. Instead of searching only by title, the UI searches across several metadata fields when available, including:

```text
Title
Author
Subjects
concepts
```

This allows users to search for broad categories such as:

```text
romance
politics
history
science
business
```

If a user enters a query that returns no result, the UI displays the following warning:

```text
There may be a typo in your search. Please try again. If the problem persists, try another category.
```

This improves usability by helping users understand why no books are displayed.

---

## Book Cards and Visual Design

The UI displays books as visual cards.

Each card can include:

```text
rank
cover image or placeholder cover
title
author
score or similarity score
description expander
```

If a real cover is available, the UI displays it using the local `cover_path` from:

```text
kaggle_data/items_with_covers.csv
```

If no cover is available, the UI generates a custom placeholder cover using CSS. The placeholder is designed to look like an old book cover, with a cream background, decorative borders, and the book title displayed in the center.

This ensures that all books are displayed in a visually consistent way, even when real covers are missing.

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

Inside the UI, descriptions are merged with the item metadata using the item ID `i`:

```python
items = items.merge(
    descriptions[description_cols],
    on="i",
    how="left"
)
```

Each book card includes a `View description` expander.

If a description is available, it is displayed when the user opens the expander. Otherwise, the UI displays:

```text
No description available for this book.
```

The relevant UI logic is:

```python
description_text = (
    str(row["description"])
    if "description" in row
    and pd.notna(row["description"])
    and str(row["description"]).strip()
    else "No description available for this book."
)

with st.expander("View description"):
    st.write(description_text)
```

---

## Popular Items

The UI includes a `Popular items` section.

This section provides a simple non-personalized baseline by ranking books according to the number of historical interactions.

Popularity is computed from the interaction matrix:

```python
popularity = matrix.sum(axis=0)
```

The most popular books are then displayed as book cards.

This section is useful for comparing personalized recommendations with a simple popularity-based recommendation strategy.

---

## Similarity Exploration

The `Similarity exploration` section allows users to select any book from the database and display the most similar books according to the item-item similarity matrix.

This feature is different from personalized recommendation. It is not based on a user profile. Instead, it answers the question:

```text
Which books are most similar to this selected book?
```

The selected book is compared to all other books using:

```python
sim_scores = item_sim[int(i)].copy()
sim_scores[int(i)] = -np.inf
```

The selected item itself is excluded by setting its similarity score to negative infinity. The app then ranks all other books by similarity score and displays the top results.

To make this feature easier to interpret, the dropdown menu displays:

```text
item_id - title — author
```

The UI also displays an explanatory sentence such as:

```text
Here are the books most similar to [selected title] by [selected author] (item [item_id] in the database).
```

---

## How to Run the Project

### 1. Generate or update recommendations

Run the recommender script:

```bash
python "Recommender Main Code.py"
```

This generates or updates:

```text
Submission/Hybrid_0.3_0.3_SBB_R08_final.csv
NPYs/UI-item_similarity.npy
NPYs/UI-content_similarity.npy
```

---

### 2. Download covers if needed

If the cover file does not already exist, run:

```bash
python download_covers_2.py
```

This generates:

```text
kaggle_data/items_with_covers.csv
kaggle_data/covers/
```

This step can take a long time and does not need to be repeated if covers are already downloaded.

---

### 3. Download descriptions if needed

Run:

```bash
python download_descriptions.py
```

This generates:

```text
kaggle_data/item_descriptions.csv
```

This script only downloads book descriptions and does not redownload covers.

---

### 4. Launch the Streamlit app

Run:

```bash
streamlit run UI_ML.py
```

The app will open in the browser.

---

## Required Files Before Running the UI

Before launching the interface, the following files should be available:

```text
kaggle_data/items_with_covers.csv
kaggle_data/item_descriptions.csv
kaggle_data/interactions_train.csv
clean_items.csv
Submission/Hybrid_0.3_0.3_SBB_R08_final.csv
NPYs/UI-item_similarity.npy
NPYs/UI-content_similarity.npy
```

If one of these files is missing, the UI may not run correctly or some features may be unavailable.

---

## Cache Note

The Streamlit app uses caching to make loading faster.

If a CSV file is updated, for example the recommendation CSV or the description CSV, it may be necessary to clear the Streamlit cache or restart the Streamlit app so that the new file is loaded correctly.

---

## Summary

This project combines an offline recommender model with an interactive Streamlit interface.

The recommender model computes hybrid recommendations and similarity matrices. The UI then uses these precomputed outputs to provide a fast and user-friendly application.

Existing users receive recommendations from the precomputed hybrid recommendation CSV. New users receive cold-start recommendations based on selected books and precomputed similarity matrices. The interface also improves interpretability by showing seen items, popular items, similar books, book covers, and book descriptions.

Overall, BookMatch AI turns the recommender system into an interactive tool that is easier to understand, test, and present.