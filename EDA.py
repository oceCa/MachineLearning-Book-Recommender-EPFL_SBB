# EDA.py
# Exploratory Data Analysis for:
# - interactions_train.csv
# - items.csv

# Goal of this script:
# This script performs an exploratory data analysis (EDA) on the two main
# datasets used in the recommender project:

# 1. interactions_train.csv:
#    contains user-item interactions, with user IDs, item IDs, and timestamps.

# 2. items.csv:
#    contains metadata about the books, such as title, author, ISBN, publisher,
#    subjects, and item ID.

# The script prints summary statistics, checks data quality, creates plots,
# analyzes sparsity and long-tail effects, and saves all generated figures
# and summary tables in the eda_outputs/ folder.



# os is used to create folders and build file paths.
# Here, it is mainly used to create the output directory and save figures/tables.
import os

# re is imported for regular expressions.
# In this current script, it is not heavily used, but it can be useful for
# text cleaning or parsing if the EDA is extended later.
import re

# numpy is used for numerical operations.
# Here, it is used to create cumulative item shares for the long-tail analysis.
import numpy as np

# pandas is used to load, manipulate, group, merge, and summarize tabular data.
import pandas as pd

# matplotlib is used to generate and save the EDA plots.
import matplotlib.pyplot as plt



# 1. Paths


# Path to the interaction dataset.
# This file contains the user-item interactions with columns such as:
# u = user ID, i = item ID, t = timestamp.
INTERACTIONS_PATH = "kaggle_data/interactions_train.csv"

# Path to the item metadata dataset.
# This file contains descriptive information about books.
ITEMS_PATH = "kaggle_data/items.csv"

# Folder where all EDA outputs will be saved.
# This includes plots as PNG files and summary tables as CSV files.
OUTPUT_DIR = "eda_outputs"

# Create the output folder if it does not already exist.
# exist_ok=True avoids an error if the folder already exists.
os.makedirs(OUTPUT_DIR, exist_ok=True)



# 2. Helper functions


def save_fig(filename):
    """
    Save the current matplotlib figure into the output directory.

    This helper function:
    - adjusts the layout so labels and titles fit properly,
    - saves the figure as a PNG file,
    - displays the figure in the notebook or script output.

    Parameters
    ----------
    filename : str
        Name of the file to save inside the eda_outputs folder.
    """

    # Automatically adjust spacing between plot elements.
    plt.tight_layout()

    # Save the current figure in the output directory.
    # dpi=200 gives a good image resolution.
    # bbox_inches="tight" prevents labels from being cut off.
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches="tight")

    # Display the plot.
    plt.show()


def print_section(title):
    """
    Print a formatted section title in the console.

    This makes the terminal output easier to read by clearly separating
    the different EDA sections.
    """

    # Print a blank line, a separator, the section title, and another separator.
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def count_subjects(subject_string):
    """
    Count how many subjects are associated with one book.

    The Subjects column stores multiple subjects in one string, usually
    separated by semicolons.

    Example:
    "Mangas; Fiction; Japan" gives 3 subjects.

    Parameters
    ----------
    subject_string : str
        The raw subject string from the metadata.

    Returns
    -------
    int
        Number of non-empty subjects.
    """

    # If the subject field is missing, return 0.
    if pd.isna(subject_string):
        return 0

    # Split the string on semicolons, remove surrounding spaces,
    # and keep only non-empty parts.
    parts = [x.strip() for x in str(subject_string).split(";") if x.strip()]

    # Return the number of subjects.
    return len(parts)


def split_subjects(subject_string):
    """
    Split a subject string into a list of individual subjects.

    This is used to explode the Subjects column and count the most frequent
    subjects across the full catalog.

    Parameters
    ----------
    subject_string : str
        The raw subject string.

    Returns
    -------
    list
        List of cleaned subject strings.
    """

    # If the subject field is missing, return an empty list.
    if pd.isna(subject_string):
        return []

    # Split the string on semicolons and remove empty values.
    return [x.strip() for x in str(subject_string).split(";") if x.strip()]



# 3. Load data


# Load the user-item interaction data.
interactions = pd.read_csv(INTERACTIONS_PATH)

# Load the book metadata.
items = pd.read_csv(ITEMS_PATH)



# 4. Basic overview


# Print a formatted section title.
print_section("DATA OVERVIEW")

# Print the shape of both datasets.
# Shape gives the number of rows and columns.
print("Interactions shape:", interactions.shape)
print("Items shape:", items.shape)

# Print the column names of the interactions dataset.
print("\nInteractions columns:")
print(interactions.columns.tolist())

# Print the column names of the item metadata dataset.
print("\nItems columns:")
print(items.columns.tolist())

# Show the first rows of the interactions dataset.
# This helps understand the data format.
print("\nFirst rows of interactions:")
print(interactions.head())

# Show the first rows of the item metadata dataset.
print("\nFirst rows of items:")
print(items.head())



# 5. Data quality checks


# Print a formatted section title.
print_section("DATA QUALITY CHECKS")

# Count missing values in each column of the interactions dataset.
# This checks whether user IDs, item IDs, or timestamps are missing.
print("\nMissing values in interactions:")
print(interactions.isna().sum())

# Count missing values in each column of the item metadata.
# This is important because missing authors or subjects can affect
# content-based recommendation.
print("\nMissing values in items:")
print(items.isna().sum())

# Count fully duplicated rows in the interactions dataset.
print("\nDuplicate rows in interactions:", interactions.duplicated().sum())

# Count fully duplicated rows in the item metadata.
print("Duplicate rows in items:", items.duplicated().sum())

# If the expected interaction columns exist, perform more specific duplicate checks.
if {"u", "i", "t"}.issubset(interactions.columns):

    # Count duplicated rows based only on user, item, and timestamp.
    # This identifies exact repeated interaction events.
    print("\nDuplicate (u, i, t) rows:", interactions.duplicated(subset=["u", "i", "t"]).sum())

    # Count duplicated user-item pairs.
    # This identifies cases where the same user interacted with the same item
    # more than once, possibly at different times.
    print("Duplicate (u, i) pairs:", interactions.duplicated(subset=["u", "i"]).sum())

# If the item ID column exists, check whether item IDs are unique.
if "i" in items.columns:

    # Duplicate item IDs would be problematic because each item should have
    # one unique metadata row.
    print("\nDuplicate item IDs in items:", items.duplicated(subset=["i"]).sum())



# 6. Interactions EDA


# Print a formatted section title.
print_section("INTERACTIONS EDA")

# Total number of observed interactions.
n_interactions = len(interactions)

# Number of unique users in the interaction dataset.
n_users = interactions["u"].nunique()

# Number of unique items that appear at least once in the interactions.
n_items_interacted = interactions["i"].nunique()

# Total number of items in the metadata catalog.
n_items_total = items["i"].nunique()

# Matrix density:
# observed interactions divided by all possible user-item pairs.
# This estimates how filled the user-item matrix is.
density = n_interactions / (n_users * n_items_total)

# Matrix sparsity:
# share of possible user-item pairs that are not observed.
sparsity = 1 - density

# Print key dataset size and sparsity indicators.
print(f"Number of interactions: {n_interactions:,}")
print(f"Number of users: {n_users:,}")
print(f"Number of unique interacted items: {n_items_interacted:,}")
print(f"Number of total items in metadata: {n_items_total:,}")
print(f"Matrix density: {density:.6f}")
print(f"Matrix sparsity: {sparsity:.6f}")

# Interactions per user

# Count how many interactions each user has.
# Sorting in descending order helps identify the most active users.
user_activity = interactions.groupby("u").size().sort_values(ascending=False)

# Print descriptive statistics for user activity:
# count, mean, std, min, quartiles, max.
print("\nInteractions per user:")
print(user_activity.describe())

# Interactions per item

# Count how many interactions each item receives.
# This is also the item popularity distribution.
item_popularity = interactions.groupby("i").size().sort_values(ascending=False)

# Print descriptive statistics for item popularity.
print("\nInteractions per item:")
print(item_popularity.describe())

# Plot: interactions per user

# Create a histogram of the number of interactions per user.
plt.figure(figsize=(8, 5))
plt.hist(user_activity, bins=50, edgecolor="black")
plt.title("Distribution of interactions per user")
plt.xlabel("Number of interactions")
plt.ylabel("Number of users")

# Save the figure.
save_fig("hist_interactions_per_user.png")

# Plot: interactions per user with log y-scale

# Create the same histogram but with logarithmic y-axis.
# This makes the long tail more visible.
plt.figure(figsize=(8, 5))
plt.hist(user_activity, bins=50, edgecolor="black", log=True)
plt.title("Distribution of interactions per user (log y-scale)")
plt.xlabel("Number of interactions")
plt.ylabel("Number of users (log scale)")

# Save the figure.
save_fig("hist_interactions_per_user_log.png")

# Plot: interactions per item

# Create a histogram of the number of interactions per item.
plt.figure(figsize=(8, 5))
plt.hist(item_popularity, bins=50, edgecolor="black")
plt.title("Distribution of interactions per item")
plt.xlabel("Number of interactions")
plt.ylabel("Number of items")

# Save the figure.
save_fig("hist_interactions_per_item.png")

# Top active users

# Print the 10 users with the largest number of interactions.
print("\nTop 10 most active users:")
print(user_activity.head(10))

# Top popular items

# Print the 10 item IDs with the largest number of interactions.
print("\nTop 10 most popular item IDs:")
print(item_popularity.head(10))

# Create a table of the top 10 most popular items and merge it with metadata
# to get readable titles and authors.
top_popular_items = (
    item_popularity.head(10)
    .rename("interaction_count")
    .reset_index()
    .merge(items[["i", "Title", "Author"]], on="i", how="left")
)

# Print the top 10 most popular books with their title and author.
print("\nTop 10 most popular books:")
print(top_popular_items)


# 7. Time analysis

# Print a formatted section title.
print_section("TIME ANALYSIS")

# Check that the timestamp column exists before performing time analysis.
if "t" in interactions.columns:

    # Convert Unix timestamps to pandas datetime.
    # unit="s" means timestamps are expressed in seconds.
    # errors="coerce" converts invalid timestamps to NaT instead of crashing.
    interactions["datetime"] = pd.to_datetime(interactions["t"], unit="s", errors="coerce")

    # Extract only the date part.
    interactions["date"] = interactions["datetime"].dt.date

    # Convert datetime to year-month period, then to string.
    # This is useful for monthly aggregation.
    interactions["year_month"] = interactions["datetime"].dt.to_period("M").astype(str)

    # Print the first and last interaction dates.
    print("Min timestamp:", interactions["datetime"].min())
    print("Max timestamp:", interactions["datetime"].max())

    # Count the number of interactions per month.
    monthly_interactions = interactions.groupby("year_month").size()

    # Print the first monthly counts.
    print("\nMonthly interaction counts:")
    print(monthly_interactions.head())

    # Plot monthly interaction volume as a line chart.
    plt.figure(figsize=(12, 5))
    monthly_interactions.plot(kind="line")
    plt.title("Interactions over time (monthly)")
    plt.xlabel("Month")
    plt.ylabel("Number of interactions")
    plt.xticks(rotation=45)

    # Save the figure.
    save_fig("interactions_over_time_monthly.png")


# 8. Coverage checks

# Print a formatted section title.
print_section("COVERAGE CHECKS")

# Items that appear in interactions but are missing from metadata.
# Ideally this should be zero, because every interacted item should have metadata.
items_in_interactions_not_in_metadata = set(interactions["i"].unique()) - set(items["i"].unique())

# Items that exist in metadata but never appear in the interaction dataset.
# These items are part of the catalog but have no observed user interactions.
items_in_metadata_not_in_interactions = set(items["i"].unique()) - set(interactions["i"].unique())

# Print coverage results.
print("Items appearing in interactions but missing in items metadata:",
      len(items_in_interactions_not_in_metadata))
print("Items appearing in metadata but never interacted with:",
      len(items_in_metadata_not_in_interactions))

# Count how many unique items each user has interacted with.
# This is slightly different from total interactions because repeated
# interactions with the same item are counted only once.
user_item_counts = interactions.groupby("u")["i"].nunique()

# Print descriptive statistics for unique items per user.
print("\nUnique items per user:")
print(user_item_counts.describe())


# 9. Items metadata EDA

# Print a formatted section title.
print_section("ITEMS METADATA EDA")

# Print data types of all metadata columns.
# This helps identify whether columns are numeric, object/string, etc.
print("\nColumn types:")
print(items.dtypes)

# Text length statistics

# For each important text column, compute the character length.
# This helps understand how rich or detailed the text metadata is.
for col in ["Title", "Author", "Publisher", "Subjects"]:
    if col in items.columns:

        # Create a new column containing the text length.
        # Missing values are replaced by empty strings before measuring length.
        items[f"{col}_len"] = items[col].fillna("").astype(str).str.len()

        # Print descriptive statistics of the text lengths.
        print(f"\n{col} length stats:")
        print(items[f"{col}_len"].describe())

# Missing values percentage

# Compute percentage of missing values in each item metadata column.
missing_pct = (items.isna().mean() * 100).sort_values(ascending=False)

# Print missingness percentage.
print("\nMissing values percentage in items:")
print(missing_pct)

# Plot missing value percentage by column.
plt.figure(figsize=(10, 5))
missing_pct.plot(kind="bar")
plt.title("Missing value percentage by column (items)")
plt.ylabel("Percentage")
plt.xlabel("Column")
plt.xticks(rotation=45)

# Save the figure.
save_fig("items_missing_values_percentage.png")

# Top authors

# If the Author column exists, count the most frequent authors.
if "Author" in items.columns:

    # Replace missing authors with "Unknown" so they appear explicitly.
    top_authors = items["Author"].fillna("Unknown").value_counts().head(15)

    # Print the top authors.
    print("\nTop 15 authors:")
    print(top_authors)

    # Plot top authors as a horizontal bar chart.
    plt.figure(figsize=(10, 6))
    top_authors.sort_values().plot(kind="barh")
    plt.title("Top 15 authors by number of books")
    plt.xlabel("Number of books")

    # Save the figure.
    save_fig("top_authors.png")

# Top publishers

# If the Publisher column exists, count the most frequent publishers.
if "Publisher" in items.columns:

    # Replace missing publishers with "Unknown".
    top_publishers = items["Publisher"].fillna("Unknown").value_counts().head(15)

    # Print the top publishers.
    print("\nTop 15 publishers:")
    print(top_publishers)

    # Plot top publishers as a horizontal bar chart.
    plt.figure(figsize=(10, 6))
    top_publishers.sort_values().plot(kind="barh")
    plt.title("Top 15 publishers by number of books")
    plt.xlabel("Number of books")

    # Save the figure.
    save_fig("top_publishers.png")

# Subject count per book

# If the Subjects column exists, analyze subject information.
if "Subjects" in items.columns:

    # Count the number of subjects attached to each book.
    items["n_subjects"] = items["Subjects"].apply(count_subjects)

    # Print descriptive statistics for number of subjects per book.
    print("\nNumber of subjects per book:")
    print(items["n_subjects"].describe())

    # Plot the distribution of number of subjects per book.
    plt.figure(figsize=(8, 5))
    plt.hist(items["n_subjects"], bins=30, edgecolor="black")
    plt.title("Distribution of number of subjects per book")
    plt.xlabel("Number of subjects")
    plt.ylabel("Number of books")

    # Save the figure.
    save_fig("subjects_per_book.png")


    # Explode subjects


    # Split each subject string into a list, then explode the lists so that
    # each subject appears in its own row.
    all_subjects = items["Subjects"].apply(split_subjects).explode()

    # Count the most frequent subjects.
    top_subjects = all_subjects.value_counts().head(20)

    # Print the top 20 subjects.
    print("\nTop 20 subjects:")
    print(top_subjects)

    # Plot the top subjects as a horizontal bar chart.
    plt.figure(figsize=(10, 7))
    top_subjects.sort_values().plot(kind="barh")
    plt.title("Top 20 most frequent subjects")
    plt.xlabel("Count")

    # Save the figure.
    save_fig("top_subjects.png")

# ISBN presence

# If the ISBN Valid column exists, compute the percentage of books
# with a non-missing ISBN.
if "ISBN Valid" in items.columns:
    has_isbn = items["ISBN Valid"].notna().mean() * 100
    print(f"\nPercentage of books with non-missing ISBN Valid: {has_isbn:.2f}%")

# Title length distribution

# If the Title_len column was created earlier, plot the title length distribution.
if "Title_len" in items.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(items["Title_len"], bins=50, edgecolor="black")
    plt.title("Distribution of title length")
    plt.xlabel("Title length (characters)")
    plt.ylabel("Number of books")

    # Save the figure.
    save_fig("title_length_distribution.png")



# 10. Join interactions with metadata


# Print a formatted section title.
print_section("JOINED ANALYSIS: INTERACTIONS + METADATA")

# Merge the interactions with item metadata using item ID i.
# This allows us to analyze interactions by author, publisher, title, etc.
merged = interactions.merge(items, on="i", how="left")

# Print the shape of the merged dataset.
print("Merged shape:", merged.shape)

# Check how many interactions could not be matched with a title.
# Ideally this should be zero if metadata coverage is complete.
print("Missing titles after merge:", merged["Title"].isna().sum())

# Most interacted authors

# If the Author column exists in the merged dataset, analyze which authors
# receive the most interactions.
if "Author" in merged.columns:

    # Count interactions by author.
    # Missing authors are grouped under "Unknown".
    popular_authors = merged["Author"].fillna("Unknown").value_counts().head(15)

    # Print the most interacted authors.
    print("\nMost interacted authors:")
    print(popular_authors)

    # Plot most interacted authors as a horizontal bar chart.
    plt.figure(figsize=(10, 6))
    popular_authors.sort_values().plot(kind="barh")
    plt.title("Most interacted authors")
    plt.xlabel("Number of interactions")

    # Save the figure.
    save_fig("most_interacted_authors.png")

# Most interacted publishers

# If the Publisher column exists, analyze which publishers receive the
# most interactions.
if "Publisher" in merged.columns:

    # Count interactions by publisher.
    popular_publishers = merged["Publisher"].fillna("Unknown").value_counts().head(15)

    # Print the most interacted publishers.
    print("\nMost interacted publishers:")
    print(popular_publishers)

    # Plot most interacted publishers as a horizontal bar chart.
    plt.figure(figsize=(10, 6))
    popular_publishers.sort_values().plot(kind="barh")
    plt.title("Most interacted publishers")
    plt.xlabel("Number of interactions")

    # Save the figure.
    save_fig("most_interacted_publishers.png")



# 11. Long tail analysis


# Print a formatted section title.
print_section("LONG-TAIL ANALYSIS")

# Convert item popularity series into a DataFrame.
item_popularity_df = item_popularity.reset_index()

# Rename columns for clarity.
item_popularity_df.columns = ["i", "interaction_count"]

# Sort items from most popular to least popular.
item_popularity_df = item_popularity_df.sort_values("interaction_count", ascending=False).reset_index(drop=True)

# Compute cumulative share of interactions.
# This shows how much of total interaction volume is captured by the top items.
item_popularity_df["cum_share_interactions"] = item_popularity_df["interaction_count"].cumsum() / item_popularity_df["interaction_count"].sum()

# Compute cumulative share of items.
# This goes from the most popular item to the least popular item.
item_popularity_df["cum_share_items"] = (np.arange(len(item_popularity_df)) + 1) / len(item_popularity_df)

# Plot the long-tail / concentration curve.
plt.figure(figsize=(8, 6))
plt.plot(item_popularity_df["cum_share_items"], item_popularity_df["cum_share_interactions"])

# Add a diagonal reference line.
# If interactions were perfectly evenly distributed, the curve would follow
# this diagonal. A curve above the diagonal indicates concentration.
plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("Long-tail / concentration curve of item popularity")
plt.xlabel("Cumulative share of items")
plt.ylabel("Cumulative share of interactions")

# Save the long-tail curve.
save_fig("long_tail_curve.png")

# Compute how many items correspond to the top 20% most popular items.
top_20pct_items = int(0.2 * len(item_popularity_df))

# Compute the share of all interactions captured by the top 20% items.
share_top_20pct = item_popularity_df.iloc[:top_20pct_items]["interaction_count"].sum() / item_popularity_df["interaction_count"].sum()

# Print the result.
print(f"Top 20% most popular items account for {share_top_20pct:.2%} of all interactions.")



# 12. Save summary tables


# Print a formatted section title.
print_section("SAVING SUMMARY TABLES")

# Save descriptive statistics for user activity.
user_activity.describe().to_csv(os.path.join(OUTPUT_DIR, "user_activity_describe.csv"))

# Save descriptive statistics for item popularity.
item_popularity.describe().to_csv(os.path.join(OUTPUT_DIR, "item_popularity_describe.csv"))

# Save the top popular books table.
top_popular_items.to_csv(os.path.join(OUTPUT_DIR, "top_popular_items.csv"), index=False)

# Save the missing value percentage table.
missing_pct.to_csv(os.path.join(OUTPUT_DIR, "items_missing_percentage.csv"))

# Print the output folder location.
print(f"EDA outputs saved in: {OUTPUT_DIR}")


# 13. Final remarks

# Print a final section title.
print_section("EDA COMPLETED")

# Print a checklist of the main findings to report later in a notebook,
# report, README, or presentation.
print("Main things to report in your notebook/report:")
print("1. Dataset size: users, items, interactions")
print("2. Sparsity of the user-item matrix")
print("3. User activity and item popularity distributions")
print("4. Time evolution of interactions")
print("5. Missing values and duplicates")
print("6. Metadata quality: titles, authors, publishers, subjects")
print("7. Long-tail effect in item popularity")
print("8. Coverage between interactions and metadata")


# SUMMARY OF WHAT THIS EDA CODE DOES

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
