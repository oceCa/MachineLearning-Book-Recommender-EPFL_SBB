# ============================================================
# EDA.py
# Exploratory Data Analysis for:
# - interactions_train.csv
# - items.csv
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

INTERACTIONS_PATH = "kaggle_data/interactions_train.csv"
ITEMS_PATH = "kaggle_data/items.csv"

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Helper functions
# ============================================================

def save_fig(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches="tight")
    plt.show()


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def count_subjects(subject_string):
    if pd.isna(subject_string):
        return 0
    parts = [x.strip() for x in str(subject_string).split(";") if x.strip()]
    return len(parts)


def split_subjects(subject_string):
    if pd.isna(subject_string):
        return []
    return [x.strip() for x in str(subject_string).split(";") if x.strip()]


# ============================================================
# 3. Load data
# ============================================================

interactions = pd.read_csv(INTERACTIONS_PATH)
items = pd.read_csv(ITEMS_PATH)


# ============================================================
# 4. Basic overview
# ============================================================

print_section("DATA OVERVIEW")

print("Interactions shape:", interactions.shape)
print("Items shape:", items.shape)

print("\nInteractions columns:")
print(interactions.columns.tolist())

print("\nItems columns:")
print(items.columns.tolist())

print("\nFirst rows of interactions:")
print(interactions.head())

print("\nFirst rows of items:")
print(items.head())


# ============================================================
# 5. Data quality checks
# ============================================================

print_section("DATA QUALITY CHECKS")

print("\nMissing values in interactions:")
print(interactions.isna().sum())

print("\nMissing values in items:")
print(items.isna().sum())

print("\nDuplicate rows in interactions:", interactions.duplicated().sum())
print("Duplicate rows in items:", items.duplicated().sum())

if {"u", "i", "t"}.issubset(interactions.columns):
    print("\nDuplicate (u, i, t) rows:", interactions.duplicated(subset=["u", "i", "t"]).sum())
    print("Duplicate (u, i) pairs:", interactions.duplicated(subset=["u", "i"]).sum())

if "i" in items.columns:
    print("\nDuplicate item IDs in items:", items.duplicated(subset=["i"]).sum())


# ============================================================
# 6. Interactions EDA
# ============================================================

print_section("INTERACTIONS EDA")

n_interactions = len(interactions)
n_users = interactions["u"].nunique()
n_items_interacted = interactions["i"].nunique()
n_items_total = items["i"].nunique()

density = n_interactions / (n_users * n_items_total)
sparsity = 1 - density

print(f"Number of interactions: {n_interactions:,}")
print(f"Number of users: {n_users:,}")
print(f"Number of unique interacted items: {n_items_interacted:,}")
print(f"Number of total items in metadata: {n_items_total:,}")
print(f"Matrix density: {density:.6f}")
print(f"Matrix sparsity: {sparsity:.6f}")

# interactions per user
user_activity = interactions.groupby("u").size().sort_values(ascending=False)
print("\nInteractions per user:")
print(user_activity.describe())

# interactions per item
item_popularity = interactions.groupby("i").size().sort_values(ascending=False)
print("\nInteractions per item:")
print(item_popularity.describe())

# plot: interactions per user
plt.figure(figsize=(8, 5))
plt.hist(user_activity, bins=50, edgecolor="black")
plt.title("Distribution of interactions per user")
plt.xlabel("Number of interactions")
plt.ylabel("Number of users")
save_fig("hist_interactions_per_user.png")

# plot: log scale for user activity
plt.figure(figsize=(8, 5))
plt.hist(user_activity, bins=50, edgecolor="black", log=True)
plt.title("Distribution of interactions per user (log y-scale)")
plt.xlabel("Number of interactions")
plt.ylabel("Number of users (log scale)")
save_fig("hist_interactions_per_user_log.png")

# plot: interactions per item
plt.figure(figsize=(8, 5))
plt.hist(item_popularity, bins=50, edgecolor="black")
plt.title("Distribution of interactions per item")
plt.xlabel("Number of interactions")
plt.ylabel("Number of items")
save_fig("hist_interactions_per_item.png")

# top active users
print("\nTop 10 most active users:")
print(user_activity.head(10))

# top popular items
print("\nTop 10 most popular item IDs:")
print(item_popularity.head(10))

top_popular_items = (
    item_popularity.head(10)
    .rename("interaction_count")
    .reset_index()
    .merge(items[["i", "Title", "Author"]], on="i", how="left")
)

print("\nTop 10 most popular books:")
print(top_popular_items)


# ============================================================
# 7. Time analysis
# ============================================================

print_section("TIME ANALYSIS")

if "t" in interactions.columns:
    interactions["datetime"] = pd.to_datetime(interactions["t"], unit="s", errors="coerce")
    interactions["date"] = interactions["datetime"].dt.date
    interactions["year_month"] = interactions["datetime"].dt.to_period("M").astype(str)

    print("Min timestamp:", interactions["datetime"].min())
    print("Max timestamp:", interactions["datetime"].max())

    monthly_interactions = interactions.groupby("year_month").size()

    print("\nMonthly interaction counts:")
    print(monthly_interactions.head())

    plt.figure(figsize=(12, 5))
    monthly_interactions.plot(kind="line")
    plt.title("Interactions over time (monthly)")
    plt.xlabel("Month")
    plt.ylabel("Number of interactions")
    plt.xticks(rotation=45)
    save_fig("interactions_over_time_monthly.png")


# ============================================================
# 8. Coverage checks
# ============================================================

print_section("COVERAGE CHECKS")

items_in_interactions_not_in_metadata = set(interactions["i"].unique()) - set(items["i"].unique())
items_in_metadata_not_in_interactions = set(items["i"].unique()) - set(interactions["i"].unique())

print("Items appearing in interactions but missing in items metadata:",
      len(items_in_interactions_not_in_metadata))
print("Items appearing in metadata but never interacted with:",
      len(items_in_metadata_not_in_interactions))

user_item_counts = interactions.groupby("u")["i"].nunique()
print("\nUnique items per user:")
print(user_item_counts.describe())


# ============================================================
# 9. Items metadata EDA
# ============================================================

print_section("ITEMS METADATA EDA")

print("\nColumn types:")
print(items.dtypes)

# Text lengths
for col in ["Title", "Author", "Publisher", "Subjects"]:
    if col in items.columns:
        items[f"{col}_len"] = items[col].fillna("").astype(str).str.len()
        print(f"\n{col} length stats:")
        print(items[f"{col}_len"].describe())

# Missingness summary in percentage
missing_pct = (items.isna().mean() * 100).sort_values(ascending=False)
print("\nMissing values percentage in items:")
print(missing_pct)

# Plot missingness
plt.figure(figsize=(10, 5))
missing_pct.plot(kind="bar")
plt.title("Missing value percentage by column (items)")
plt.ylabel("Percentage")
plt.xlabel("Column")
plt.xticks(rotation=45)
save_fig("items_missing_values_percentage.png")

# Top authors
if "Author" in items.columns:
    top_authors = items["Author"].fillna("Unknown").value_counts().head(15)
    print("\nTop 15 authors:")
    print(top_authors)

    plt.figure(figsize=(10, 6))
    top_authors.sort_values().plot(kind="barh")
    plt.title("Top 15 authors by number of books")
    plt.xlabel("Number of books")
    save_fig("top_authors.png")

# Top publishers
if "Publisher" in items.columns:
    top_publishers = items["Publisher"].fillna("Unknown").value_counts().head(15)
    print("\nTop 15 publishers:")
    print(top_publishers)

    plt.figure(figsize=(10, 6))
    top_publishers.sort_values().plot(kind="barh")
    plt.title("Top 15 publishers by number of books")
    plt.xlabel("Number of books")
    save_fig("top_publishers.png")

# Subject count per book
if "Subjects" in items.columns:
    items["n_subjects"] = items["Subjects"].apply(count_subjects)
    print("\nNumber of subjects per book:")
    print(items["n_subjects"].describe())

    plt.figure(figsize=(8, 5))
    plt.hist(items["n_subjects"], bins=30, edgecolor="black")
    plt.title("Distribution of number of subjects per book")
    plt.xlabel("Number of subjects")
    plt.ylabel("Number of books")
    save_fig("subjects_per_book.png")

    # explode subjects
    all_subjects = items["Subjects"].apply(split_subjects).explode()
    top_subjects = all_subjects.value_counts().head(20)

    print("\nTop 20 subjects:")
    print(top_subjects)

    plt.figure(figsize=(10, 7))
    top_subjects.sort_values().plot(kind="barh")
    plt.title("Top 20 most frequent subjects")
    plt.xlabel("Count")
    save_fig("top_subjects.png")

# ISBN presence
if "ISBN Valid" in items.columns:
    has_isbn = items["ISBN Valid"].notna().mean() * 100
    print(f"\nPercentage of books with non-missing ISBN Valid: {has_isbn:.2f}%")

# Title length plot
if "Title_len" in items.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(items["Title_len"], bins=50, edgecolor="black")
    plt.title("Distribution of title length")
    plt.xlabel("Title length (characters)")
    plt.ylabel("Number of books")
    save_fig("title_length_distribution.png")


# ============================================================
# 10. Join interactions with metadata
# ============================================================

print_section("JOINED ANALYSIS: INTERACTIONS + METADATA")

merged = interactions.merge(items, on="i", how="left")

print("Merged shape:", merged.shape)
print("Missing titles after merge:", merged["Title"].isna().sum())

# Most interacted authors
if "Author" in merged.columns:
    popular_authors = merged["Author"].fillna("Unknown").value_counts().head(15)
    print("\nMost interacted authors:")
    print(popular_authors)

    plt.figure(figsize=(10, 6))
    popular_authors.sort_values().plot(kind="barh")
    plt.title("Most interacted authors")
    plt.xlabel("Number of interactions")
    save_fig("most_interacted_authors.png")

# Most interacted publishers
if "Publisher" in merged.columns:
    popular_publishers = merged["Publisher"].fillna("Unknown").value_counts().head(15)
    print("\nMost interacted publishers:")
    print(popular_publishers)

    plt.figure(figsize=(10, 6))
    popular_publishers.sort_values().plot(kind="barh")
    plt.title("Most interacted publishers")
    plt.xlabel("Number of interactions")
    save_fig("most_interacted_publishers.png")


# ============================================================
# 11. Long tail analysis
# ============================================================

print_section("LONG-TAIL ANALYSIS")

item_popularity_df = item_popularity.reset_index()
item_popularity_df.columns = ["i", "interaction_count"]
item_popularity_df = item_popularity_df.sort_values("interaction_count", ascending=False).reset_index(drop=True)
item_popularity_df["cum_share_interactions"] = item_popularity_df["interaction_count"].cumsum() / item_popularity_df["interaction_count"].sum()
item_popularity_df["cum_share_items"] = (np.arange(len(item_popularity_df)) + 1) / len(item_popularity_df)

plt.figure(figsize=(8, 6))
plt.plot(item_popularity_df["cum_share_items"], item_popularity_df["cum_share_interactions"])
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Long-tail / concentration curve of item popularity")
plt.xlabel("Cumulative share of items")
plt.ylabel("Cumulative share of interactions")
save_fig("long_tail_curve.png")

top_20pct_items = int(0.2 * len(item_popularity_df))
share_top_20pct = item_popularity_df.iloc[:top_20pct_items]["interaction_count"].sum() / item_popularity_df["interaction_count"].sum()
print(f"Top 20% most popular items account for {share_top_20pct:.2%} of all interactions.")


# ============================================================
# 12. Save summary tables
# ============================================================

print_section("SAVING SUMMARY TABLES")

user_activity.describe().to_csv(os.path.join(OUTPUT_DIR, "user_activity_describe.csv"))
item_popularity.describe().to_csv(os.path.join(OUTPUT_DIR, "item_popularity_describe.csv"))
top_popular_items.to_csv(os.path.join(OUTPUT_DIR, "top_popular_items.csv"), index=False)
missing_pct.to_csv(os.path.join(OUTPUT_DIR, "items_missing_percentage.csv"))

print(f"EDA outputs saved in: {OUTPUT_DIR}")


# ============================================================
# 13. Final remarks
# ============================================================

print_section("EDA COMPLETED")

print("Main things to report in your notebook/report:")
print("1. Dataset size: users, items, interactions")
print("2. Sparsity of the user-item matrix")
print("3. User activity and item popularity distributions")
print("4. Time evolution of interactions")
print("5. Missing values and duplicates")
print("6. Metadata quality: titles, authors, publishers, subjects")
print("7. Long-tail effect in item popularity")
print("8. Coverage between interactions and metadata")



# ============================================================
# SUMMARY OF WHAT THIS EDA CODE DOES
# ============================================================
#
# 1. Loads the two datasets:
#    - interactions_train.csv
#    - items.csv
#
# 2. Prints a general overview of both files:
#    - shape
#    - column names
#    - first rows
#
# 3. Checks data quality:
#    - missing values
#    - duplicate rows
#    - duplicate user-item pairs
#    - duplicate item IDs
#
# 4. Analyzes the interactions dataset:
#    - total number of interactions
#    - number of users
#    - number of items
#    - matrix density and sparsity
#    - interactions per user
#    - interactions per item
#    - top active users
#    - most popular books
#
# 5. Creates plots for interaction patterns:
#    - histogram of interactions per user
#    - histogram of interactions per item
#    - log-scale version for user activity
#
# 6. Analyzes time information (if timestamp column exists):
#    - converts timestamps to datetime
#    - checks min and max date
#    - plots monthly interaction counts
#
# 7. Checks coverage between datasets:
#    - items in interactions but missing from metadata
#    - items in metadata never interacted with
#
# 8. Analyzes item metadata:
#    - data types
#    - missing value percentages
#    - text length statistics
#    - top authors
#    - top publishers
#    - number of subjects per book
#    - most frequent subjects
#
# 9. Merges interactions with item metadata:
#    - checks if merge is successful
#    - finds most interacted authors
#    - finds most interacted publishers
#
# 10. Studies the long-tail effect:
#     - checks whether a small number of items
#       accounts for a large share of interactions
#     - plots concentration curve
#
# 11. Saves outputs:
#     - figures as .png
#     - summary tables as .csv
#
# Goal:
# Understand the structure, quality, and main patterns
# of the data before building or evaluating the recommender.
#
# ============================================================