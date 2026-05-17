# UI_ML.py
# Improved Streamlit UI with side background library image

# Goal of this file:
# This script creates the Streamlit web interface for BookMatch AI.

# The interface allows:
# - existing users to receive precomputed recommendations,
# - new users to receive cold-start recommendations based on books they like,
# - users to explore popular books,
# - users to explore books similar to a selected book,
# - books to be displayed with covers, titles, authors, scores, and descriptions.

# The recommender itself is not recomputed from scratch inside the UI.
# Instead, the app loads precomputed files:
# - final recommendations for existing users,
# - top item-item similarities,
# - top content-based similarities,
# - item metadata,
# - book covers,
# - book descriptions.


# os is used to check whether files exist and to work with local paths.
# Here, it is mainly used for the background image and cover images.
import os

# base64 is used to convert local images into base64 strings.
# This allows images to be inserted directly into HTML/CSS in Streamlit.
import base64

# numpy is used for numerical operations.
# Here, it is mainly used for arrays and sorted item IDs.
import numpy as np

# pandas is used to load CSV files and manipulate tabular data.
import pandas as pd

# streamlit is the main framework used to build the interactive web app.
import streamlit as st

# matplotlib is imported but not directly used in the current UI.
# It can be useful if plots are added later.
import matplotlib.pyplot as plt

# cosine_similarity is imported but not directly used in this lightweight UI version.
# Similarities are already precomputed and loaded from CSV files.
from sklearn.metrics.pairwise import cosine_similarity


# Streamlit page configuration

# Configure the Streamlit page:
# - page_title is displayed in the browser tab,
# - page_icon is the small icon in the tab,
# - layout="wide" gives more horizontal space for book cards.
st.set_page_config(
    page_title="BookMatch AI",
    page_icon="🏷️",
    layout="wide"
)


# BACKGROUND IMAGE HELPERS

def get_base64_of_bin_file(bin_file):
    """
    Convert an image file into a base64 string.

    Streamlit can display images directly, but here the background image
    is inserted through custom HTML/CSS. For that, the image is encoded
    into base64 and embedded directly in the page.
    """

    # Open the image in binary mode.
    with open(bin_file, "rb") as f:

        # Read the file content, encode it in base64, and convert it to string.
        return base64.b64encode(f.read()).decode()


# Local path to the decorative library background image.
LIBRARY_BG_PATH = "best-libraries-from-around-the-world-the-admont-1.v1517654478.png"

# Initialize the base64 version of the background image as None.
library_bg_base64 = None

# If the background image exists locally, encode it in base64.
if os.path.exists(LIBRARY_BG_PATH):
    library_bg_base64 = get_base64_of_bin_file(LIBRARY_BG_PATH)


# STYLE

# This block injects custom CSS into the Streamlit app.
# It controls the visual identity of the UI:
# - background gradients,
# - typography,
# - buttons,
# - side panels,
# - book cards,
# - cover placeholders,
# - metrics,
# - tabs,
# - responsive layout.

# unsafe_allow_html=True is required because Streamlit normally escapes HTML.
# Here, custom HTML and CSS are intentionally inserted.

st.markdown("""
<style>
/* Global app */
.stApp {
    overflow-x: hidden;
    background:
        radial-gradient(circle at top left, rgba(193, 154, 107, 0.18), transparent 22%),
        radial-gradient(circle at top right, rgba(133, 164, 184, 0.14), transparent 20%),
        radial-gradient(circle at bottom center, rgba(111, 78, 55, 0.18), transparent 28%),
        linear-gradient(135deg, #2b1f1a 0%, #4a3427 28%, #5e4636 52%, #3d2c24 72%, #243746 100%);
    color: #f5efe3;
}

/* Main container */
.block-container {
    position: relative;
    z-index: 1;
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1120px;
}

/* Fixed side decorative panels */
.side-bg {
    position: fixed;
    top: 0;
    height: 140vh;
    width: 16vw;
    min-width: 110px;
    max-width: 260px;
    pointer-events: none;
    z-index: 0;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    opacity: 0.80;
}

.side-bg-left {
    left: 0;
}

.side-bg-right {
    right: 0;
}

/* Titles */
h1, h2, h3 {
    color: #f8f1e4 !important;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 3rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.5rem !important;
}

h2 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
}

h3 {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
}

/* Text */
p, li, label, .stMarkdown, .stText {
    color: #f1e7d8 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(59, 40, 30, 0.82), rgba(36, 28, 24, 0.92));
    border-right: 1px solid rgba(214, 186, 140, 0.18);
}

/* Metrics */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(87, 62, 47, 0.70), rgba(43, 55, 69, 0.42));
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(214, 186, 140, 0.18);
    box-shadow: 0 8px 30px rgba(20, 12, 8, 0.30);
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 14px;
    padding: 0.7rem 1.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #9c6a43, #cda56a);
    color: #2d1f18;
    border: none;
    box-shadow: 0 8px 18px rgba(0,0,0,0.25);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    transition: 0.2s ease;
}

/* Inputs */
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"],
.stTextInput input {
    border-radius: 14px !important;
    background-color: rgba(255,255,255,0.04) !important;
}

/* Input labels and text */
div[data-testid="stRadio"] label {
    color: #f5efe3 !important;
    font-weight: 600 !important;
}

div[data-testid="stTextInput"] input {
    color: #f5efe3 !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stTextInput"] label {
    color: #f5efe3 !important;
    font-weight: 600 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: rgba(70, 52, 42, 0.30);
    border-radius: 18px;
    border: 1px solid rgba(214, 186, 140, 0.14);
    overflow: hidden;
}

/* Expander */
.streamlit-expanderHeader {
    font-weight: 700;
    color: #f8f1e4 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    color: #f1e7d8 !important;
}

/* Hero / cards */
.custom-card {
    padding: 1.2rem 1.2rem 1rem 1.2rem;
    border-radius: 22px;
    background:
        linear-gradient(135deg, rgba(82, 59, 45, 0.72), rgba(35, 51, 63, 0.30));
    border: 1px solid rgba(219, 191, 144, 0.14);
    backdrop-filter: blur(8px);
    box-shadow: 0 12px 30px rgba(20, 10, 6, 0.22);
    margin-bottom: 1rem;
}

.hero-box {
    padding: 2.2rem;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(122, 82, 60, 0.55), rgba(180, 145, 98, 0.14), rgba(55, 76, 96, 0.18)),
        rgba(255,255,255,0.03);
    border: 1px solid rgba(224, 196, 149, 0.16);
    box-shadow: 0 14px 45px rgba(24, 14, 10, 0.25);
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.6rem;
    color: #fff7eb;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #f2e7d7;
    margin-bottom: 0.7rem;
}

.hero-badge {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    margin-right: 0.5rem;
    margin-top: 0.4rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    background: rgba(231, 205, 160, 0.10);
    border: 1px solid rgba(231, 205, 160, 0.16);
    color: #fff2de;
}

/* Book covers */
.book-cover-box {
    width: 100%;
    height: 260px;
    border-radius: 16px;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    box-sizing: border-box;
}

.book-cover-img {
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
    border-radius: 14px;
    box-shadow: 0 12px 25px rgba(0,0,0,0.35);
    display: block;
}

.book-cover-placeholder {
    position: relative;
    background:
        linear-gradient(180deg, #fbf1d0 0%, #f4e3b7 48%, #ead39e 100%);
    border: 1px solid #3d2418;
    padding: 1.1rem;
    text-align: center;
    box-shadow:
        inset 0 0 0 7px #fbf1d0,
        inset 0 0 0 9px rgba(136, 42, 31, 0.78),
        inset 0 0 0 15px #fbf1d0,
        inset 0 0 0 17px rgba(136, 42, 31, 0.55),
        0 12px 25px rgba(0,0,0,0.35);
}

.book-cover-placeholder::before {
    content: "BOOKMATCH";
    position: absolute;
    top: 25px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    font-weight: 800;
    color: #2f241d;
    opacity: 0.78;
}

.book-cover-placeholder::after {
    content: "recommended edition";
    position: absolute;
    bottom: 25px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.58rem;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #5c4030;
    opacity: 0.75;
}

.book-cover-placeholder-title {
    color: #b42318;
    font-size: 0.75rem;
    font-weight: 900;
    line-height: 1.25;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    max-width: 82%;
    margin: 0 auto;
}
            
.book-info-card {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0.85rem 0.2rem 0.85rem 0.2rem;
    min-height: 190px;
    margin-bottom: 1rem;
    box-sizing: border-box;
    box-shadow: none;
}

.book-rank {
    font-size: 0.85rem;
    color: #fca5a5;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.book-title {
    font-size: 1rem;
    font-weight: 700;
    color: white;
    line-height: 1.3;
    margin-bottom: 0.5rem;
}

.book-author {
    font-size: 0.9rem;
    color: #cbd5e1;
    margin-bottom: 0.45rem;
    line-height: 1.35;
}

.book-score {
    font-size: 0.85rem;
    color: #fde68a;
    margin-top: auto;
}

/* Small screen */
@media (max-width: 1100px) {
    .side-bg {
        display: none;
    }

    .block-container {
        max-width: 95%;
    }
}
</style>
""", unsafe_allow_html=True)


# SIDE BACKGROUNDS

# If the library image exists locally, this block creates two fixed decorative
# side panels: one on the left and one on the right of the interface.
# The image is embedded directly using base64.

if library_bg_base64 is not None:
    side_bg_style = (
        "background-image:"
        "linear-gradient(rgba(60,42,28,0.18), rgba(36,46,58,0.22)),"
        f"url('data:image/png;base64,{library_bg_base64}');"
    )

    st.markdown(
        f"""
        <div class="side-bg side-bg-left" style="{side_bg_style}"></div>
        <div class="side-bg side-bg-right" style="{side_bg_style}"></div>
        """,
        unsafe_allow_html=True,
    )


# DATA + MODEL

@st.cache_data
def load_data(items_path, interactions_path, recommendations_path, clean_items_path, descriptions_path):
    """
    Load all CSV files needed by the Streamlit app.

    The function is cached with st.cache_data so that Streamlit does not reload
    the files every time the user interacts with the app.

    Inputs:
    - items_path: metadata enriched with cover paths,
    - interactions_path: historical user-item interactions,
    - recommendations_path: precomputed recommendations for existing users,
    - clean_items_path: cleaned title/author metadata,
    - descriptions_path: book descriptions.

    Output:
    - five pandas DataFrames.
    """

    # Load the enriched item metadata.
    items = pd.read_csv(items_path)

    # Load historical user-item interactions.
    interactions = pd.read_csv(interactions_path)

    # Load precomputed recommendations.
    recommendations = pd.read_csv(recommendations_path)

    # Load cleaned item metadata.
    clean_items = pd.read_csv(clean_items_path)

    # Try loading book descriptions.
    # If the file does not exist, create an empty description dataframe
    # so that the rest of the app can still run.
    try:
        descriptions = pd.read_csv(descriptions_path)
    except FileNotFoundError:
        descriptions = pd.DataFrame(columns=["i", "description"])

    return items, interactions, recommendations, clean_items, descriptions


@st.cache_data(show_spinner="Building lightweight interaction lookup...")
def build_interaction_lookup(interactions):
    """
    Lightweight replacement for the dense user-item matrix.
    This avoids creating a huge users x items matrix on Streamlit Cloud.

    Instead of storing a full matrix, this function creates:
    - seen_by_user: dictionary mapping each user to the set of books they saw,
    - popularity_df: dataframe ranking books by number of interactions.
    """

    # Build a dictionary where:
    # key = user ID
    # value = set of item IDs already interacted with by that user.
    seen_by_user = (
        interactions
        .groupby("u")["i"]
        .apply(lambda x: set(x.astype(int)))
        .to_dict()
    )

    # Compute item popularity from historical interactions.
    # This counts how many times each item appears in the interactions.
    popularity_df = (
        interactions
        .groupby("i")
        .size()
        .reset_index(name="number_of_interactions")
        .rename(columns={"i": "item_id"})
        .sort_values("number_of_interactions", ascending=False)
        .reset_index(drop=True)
    )

    return seen_by_user, popularity_df


def get_seen_items_light(user_id, seen_by_user, items):
    """
    Retrieve the books already seen by an existing user.

    This function uses the lightweight dictionary created by
    build_interaction_lookup instead of a dense matrix.
    """

    # Get the set of item IDs seen by the user.
    # If the user is not found, return an empty set.
    seen_items = list(seen_by_user.get(int(user_id), set()))

    # Put seen item IDs into a dataframe.
    seen_df = pd.DataFrame({"item_id": seen_items})

    # Merge with item metadata to recover title, author, cover, description, etc.
    seen_df = enrich_with_items(seen_df, items, "item_id")

    return seen_df


@st.cache_data(show_spinner="Loading top similarities...")
def load_top_similarities(item_top_path, content_top_path):
    """
    Load lightweight top-N similarity files.

    Instead of loading the full item-item and content similarity matrices,
    the UI loads CSV files containing only the top similar items for each book.

    Each CSV row contains:
    - item_id,
    - similar_items as a space-separated string,
    - similar_scores as a space-separated string.

    The function converts these CSV files into dictionaries for fast lookup.
    """

    # Load top item-item similarities.
    item_top = pd.read_csv(item_top_path)

    # Load top content-based similarities.
    content_top = pd.read_csv(content_top_path)

    # Convert the item-item top similarities into a dictionary.
    # key = item_id
    # value = tuple(list of similar item IDs, list of scores)
    item_top_dict = {
        int(row["item_id"]): (
            [int(x) for x in str(row["similar_items"]).split()],
            [float(x) for x in str(row["similar_scores"]).split()]
        )
        for _, row in item_top.iterrows()
    }

    # Convert the content-based top similarities into a dictionary.
    content_top_dict = {
        int(row["item_id"]): (
            [int(x) for x in str(row["similar_items"]).split()],
            [float(x) for x in str(row["similar_scores"]).split()]
        )
        for _, row in content_top.iterrows()
    }

    return item_top_dict, content_top_dict


def get_title_column(items):
    """
    Detect which column should be used as the book title.

    The function checks several possible column names because different
    datasets or preprocessing steps may use different naming conventions.
    """

    for col in ["Title_clean", "title_clean", "Title", "title", "Book-Title", "book_title", "name"]:
        if col in items.columns:
            return col

    return None


def get_author_column(items):
    """
    Detect which column should be used as the author field.

    Like titles, authors can appear under different column names depending
    on the dataset.
    """

    for col in ["Author_clean", "author_clean", "Author", "author", "authors", "Book-Author"]:
        if col in items.columns:
            return col

    return None


def merge_clean_metadata(items, clean_items):
    """
    Keep covers and all original columns from items,
    but replace display metadata with clean titles/authors from clean_items.

    This is useful because items_with_covers.csv contains cover paths,
    while clean_items.csv may contain cleaner display versions of titles/authors.
    """

    # Work on copies to avoid modifying the original dataframes directly.
    items = items.copy()
    clean_items = clean_items.copy()

    # If either dataframe has no item ID column, the merge cannot be done.
    if "i" not in items.columns or "i" not in clean_items.columns:
        return items

    # Detect the title column in clean_items.
    clean_title_col = None
    for col in ["Title", "title", "Book-Title", "book_title", "name"]:
        if col in clean_items.columns:
            clean_title_col = col
            break

    # Detect the author column in clean_items.
    clean_author_col = None
    for col in ["Author", "author", "authors", "Book-Author"]:
        if col in clean_items.columns:
            clean_author_col = col
            break

    # Start with item ID as merge key.
    cols_to_merge = ["i"]

    # Add clean title if available.
    if clean_title_col is not None:
        cols_to_merge.append(clean_title_col)

    # Add clean author if available.
    if clean_author_col is not None:
        cols_to_merge.append(clean_author_col)

    # Keep only the selected columns from clean_items.
    clean_subset = clean_items[cols_to_merge].copy()

    # Prepare column renaming so the UI can prioritize clean fields.
    rename_dict = {}

    if clean_title_col is not None:
        rename_dict[clean_title_col] = "Title_clean"

    if clean_author_col is not None:
        rename_dict[clean_author_col] = "Author_clean"

    # Rename clean metadata columns.
    clean_subset = clean_subset.rename(columns=rename_dict)

    # Merge clean metadata into the main items dataframe.
    items = items.merge(clean_subset, on="i", how="left")

    return items


def enrich_with_items(df, items, item_col="item_id"):
    """
    Merge a dataframe containing item IDs with full item metadata.

    This is used for:
    - recommendations,
    - seen items,
    - popular items,
    - similar items.

    The metadata adds title, author, cover_path, description, etc.
    """

    # If the input dataframe is empty, return it directly.
    if df.empty:
        return df

    # Merge using the item ID column.
    if "i" in items.columns and item_col in df.columns:
        return df.merge(items, left_on=item_col, right_on="i", how="left")

    return df


def get_recommendations_from_csv(user_id, recommendations_df, items, top_k):
    """
    Retrieve precomputed recommendations for an existing user.

    The recommendation CSV contains one row per user with a space-separated
    list of recommended item IDs.

    This function:
    1. finds the row corresponding to the selected user,
    2. parses the recommendation string,
    3. creates a ranked dataframe,
    4. merges it with item metadata,
    5. keeps only top_k items.
    """

    # Find the recommendation row for the selected user.
    row = recommendations_df[recommendations_df["user_id"] == int(user_id)]

    # If the user is not found, return an empty dataframe.
    if row.empty:
        return pd.DataFrame()

    # Extract the recommendation string.
    rec_string = row.iloc[0]["recommendation"]

    # Convert the space-separated string into a list of integer item IDs.
    rec_items = [int(x) for x in str(rec_string).split()]

    # Create a dataframe with rank and item ID.
    recs = pd.DataFrame({
        "rank": range(1, len(rec_items) + 1),
        "item_id": rec_items
    })

    # Merge recommendation item IDs with item metadata.
    recs = enrich_with_items(recs, items, "item_id")

    # Keep only the requested number of recommendations.
    recs = recs.head(top_k).reset_index(drop=True)

    # Recompute rank after filtering.
    recs["rank"] = range(1, len(recs) + 1)

    return recs


def get_book_labels(items, title_col, author_col=None):
    """
    Create readable dropdown labels for books.

    Labels have the format:
    item_id - title — author

    This makes search and selection easier for users.
    """

    # If no title column or item ID column exists, labels cannot be created.
    if title_col is None or "i" not in items.columns:
        return None

    # Start with item ID and title.
    cols = ["i", title_col]

    # Add author if available.
    if author_col is not None and author_col in items.columns:
        cols.append(author_col)

    # Keep only rows with valid item ID and title.
    book_labels = items[cols].dropna(subset=["i", title_col]).copy()

    def make_label(row):
        """
        Build one readable label for a book.
        """

        title = str(row[title_col])

        author = (
            str(row[author_col])
            if author_col and author_col in row and pd.notna(row[author_col])
            else "Unknown author"
        )

        return f"{int(row['i'])} - {title} — {author}"

    # Apply the label creation function to every row.
    book_labels["label"] = book_labels.apply(make_label, axis=1)

    return book_labels


def recommend_for_new_user_light(liked_item_ids, item_top_dict, content_top_dict, items, top_k=10, alpha=0.5):
    """
    Lightweight new-user recommender for Streamlit Cloud.

    Instead of loading full dense similarity matrices,
    it uses precomputed top similar items for each selected book.

    The final score combines:
    - item-item collaborative similarity with weight alpha,
    - content-based similarity with weight 1 - alpha.
    """

    # Convert selected liked item IDs to integers.
    liked_item_ids = [int(i) for i in liked_item_ids]

    # Store liked items in a set for fast exclusion.
    liked_set = set(liked_item_ids)

    # Dictionary that accumulates recommendation scores.
    # key = candidate item ID
    # value = accumulated score
    scores = {}

    # Loop over each book selected by the new user.
    for liked_id in liked_item_ids:

        # Item-item collaborative similarity neighbors
        if liked_id in item_top_dict:
            neighbor_ids, neighbor_scores = item_top_dict[liked_id]

            # Add weighted item-item similarity scores.
            for item_id, score in zip(neighbor_ids, neighbor_scores):
                if item_id not in liked_set:
                    scores[item_id] = scores.get(item_id, 0) + alpha * score

        # Content-based similarity neighbors
        if liked_id in content_top_dict:
            neighbor_ids, neighbor_scores = content_top_dict[liked_id]

            # Add weighted content-based similarity scores.
            for item_id, score in zip(neighbor_ids, neighbor_scores):
                if item_id not in liked_set:
                    scores[item_id] = scores.get(item_id, 0) + (1 - alpha) * score

    # If no candidate was found, return an empty dataframe.
    if len(scores) == 0:
        return pd.DataFrame()

    # Sort candidates from highest to lowest score and keep top_k.
    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Convert top items into a dataframe.
    recs = pd.DataFrame({
        "rank": range(1, len(top_items) + 1),
        "item_id": [x[0] for x in top_items],
        "score": [x[1] for x in top_items]
    })

    # Merge with item metadata.
    recs = enrich_with_items(recs, items, "item_id")

    return recs


def display_book_cards(df, title_col=None, author_col=None, score_col=None, max_items=20):
    """
    Display books as visual cards in the Streamlit app.

    Each card can include:
    - cover image or placeholder cover,
    - rank,
    - title,
    - author,
    - score,
    - expandable description.
    """

    # If no books are available, show a warning and stop.
    if df.empty:
        st.warning("No items to display.")
        return

    # Keep only max_items books.
    df = df.head(max_items).reset_index(drop=True)

    # Number of columns used to display book cards.
    n_cols = 5

    def image_to_base64(path):
        """
        Convert a local cover image into base64.

        This allows the cover to be embedded into HTML.
        """

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # Display books row by row, with n_cols cards per row.
    for start in range(0, len(df), n_cols):
        row_df = df.iloc[start:start + n_cols]
        cols = st.columns(n_cols)

        # Loop through the books in the current row.
        for j, (_, row) in enumerate(row_df.iterrows()):
            with cols[j]:

                # Display rank if available.
                rank_text = f"#{int(row['rank'])}" if "rank" in row and pd.notna(row["rank"]) else ""

                # Extract title if available, otherwise fallback to item ID.
                title_text = (
                    str(row[title_col])
                    if title_col and title_col in row and pd.notna(row[title_col])
                    else f"Item {row.get('item_id', 'unknown')}"
                )

                # Extract author if available.
                author_text = (
                    str(row[author_col])
                    if author_col and author_col in row and pd.notna(row[author_col])
                    else "Unknown author"
                )

                # Prepare optional score text.
                score_text = ""

                if score_col and score_col in row and pd.notna(row[score_col]):
                    if score_col == "number_of_interactions":
                        score_text = f"{int(row[score_col])} interactions"
                    elif score_col == "similarity_score":
                        score_text = f"Similarity: {row[score_col]:.4f}"
                    else:
                        score_text = f"Score: {row[score_col]:.4f}"

                # Get cover path if available.
                cover_path = row.get("cover_path", None)

                # If a valid cover exists locally, display it.
                if pd.notna(cover_path) and isinstance(cover_path, str) and os.path.exists(cover_path):
                    ext = cover_path.split(".")[-1].lower()
                    mime = "jpeg" if ext in ["jpg", "jpeg"] else ext
                    img_b64 = image_to_base64(cover_path)

                    cover_html = f"""
                    <div class="book-cover-box">
                        <img src="data:image/{mime};base64,{img_b64}" class="book-cover-img"/>
                    </div>
                    """

                # Otherwise, display a stylized placeholder cover.
                else:
                    cover_html = f"""
                    <div class="book-cover-box book-cover-placeholder">
                        <div class="book-cover-placeholder-title">{title_text}</div>
                    </div>
                    """

                # Render the book cover and metadata using custom HTML.
                st.markdown(f"""
                {cover_html}
                <div class="book-info-card">
                    <div class="book-rank">{rank_text}</div>
                    <div class="book-title">{title_text}</div>
                    <div class="book-author">{author_text}</div>
                    <div class="book-score">{score_text}</div>
                </div>
                """, unsafe_allow_html=True)

                # Prepare the book description.
                # If no description is available, use a default message.
                description_text = (
                    str(row["description"])
                    if "description" in row
                    and pd.notna(row["description"])
                    and str(row["description"]).strip()
                    else "No description available for this book."
                )

                # Display description inside an expandable section.
                with st.expander("View description"):
                    st.write(description_text)


# HEADER

# This section displays the hero banner at the top of the app.
# It gives the app title, a short subtitle, and visual badges.

st.markdown("""
<div class="hero-box">
    <div class="hero-title"> 🏷️ Personalized book recommendations for a life full of adventure</div>
    <div class="hero-subtitle">
        Discover books you may love with a precomputed recommender for existing users.
    </div>
    <span class="hero-badge">Precomputed recommendations</span>
    <span class="hero-badge">Book covers</span>
    <span class="hero-badge">Interactive UI</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-card">
    <p class="muted">
        Existing users use the recommendation list already generated by the hybrid recommender.
        New users use lightweight item-item and content similarity based on selected books.
    </p>
</div>
""", unsafe_allow_html=True)


# SIDEBAR

# The sidebar contains file paths and user-adjustable parameters.
# This makes the app flexible because paths and settings can be changed
# without editing the code.

st.sidebar.markdown("## ⚙️ Settings")
st.sidebar.markdown(
    "<p style='color:#cbd5e1; font-size:0.95rem;'>Adjust the recommender parameters and explore personalized results.</p>",
    unsafe_allow_html=True
)

# Path to items enriched with cover paths.
items_path = st.sidebar.text_input(
    "Items path",
    "kaggle_data/items_with_covers.csv"
)

# Path to item descriptions.
descriptions_path = st.sidebar.text_input(
    "Descriptions path",
    "kaggle_data/item_descriptions.csv"
)

# Path to cleaned item metadata.
clean_items_path = st.sidebar.text_input(
    "Clean items path",
    "clean_items.csv"
)

# Path to interaction data.
interactions_path = st.sidebar.text_input(
    "Interactions path",
    "kaggle_data/interactions_train.csv"
)

# Path to precomputed existing-user recommendations.
recommendations_path = st.sidebar.text_input(
    "Recommendations path",
    "Submission/Hybrid_0.3_0.3_SBB_R08_final.csv"
)

# Path to lightweight top item-item similarities.
item_similarity_path = st.sidebar.text_input(
    "Top item similarities path",
    "kaggle_data/top_item_similarities.csv"
)

# Path to lightweight top content similarities.
content_similarity_path = st.sidebar.text_input(
    "Top content similarities path",
    "kaggle_data/top_content_similarities.csv"
)

# Alpha controls the balance between item-item and content similarity
# for the new-user recommender.
new_user_alpha = st.sidebar.slider(
    "New user alpha: item-item vs content",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Number of recommendations displayed.
# Fixed to 10 because the app should always recommend exactly 10 books.
top_k = 10



# LOAD DATA

# Load all required datasets.
# If any required file is missing, display an error and stop the app.
try:
    items, user_pref, recommendations_df, clean_items, descriptions = load_data(
        items_path,
        interactions_path,
        recommendations_path,
        clean_items_path,
        descriptions_path
    )
except FileNotFoundError:
    st.error(
        "File not found. Check that items, clean_items, interactions, recommendations, and descriptions CSV files exist."
    )
    st.stop()

# Merge clean titles/authors into the main item metadata.
items = merge_clean_metadata(items, clean_items)

# Merge descriptions into item metadata if both files contain item IDs.
if "i" in items.columns and "i" in descriptions.columns:
    description_cols = ["i"]

    # Keep only description-related columns that actually exist.
    for col in ["description", "api_title", "api_authors", "api_source", "api_query"]:
        if col in descriptions.columns:
            description_cols.append(col)

    # Drop old description columns from items if they already exist,
    # to avoid duplicate columns after merging.
    cols_to_drop = [
        col for col in ["description", "api_title", "api_authors", "api_source", "api_query"]
        if col in items.columns
    ]

    if cols_to_drop:
        items = items.drop(columns=cols_to_drop)

    # Merge description information using item ID.
    items = items.merge(
        descriptions[description_cols],
        on="i",
        how="left"
    )

# Detect which columns should be used for display titles and authors.
title_col = get_title_column(items)
author_col = get_author_column(items)

# Sort interactions by user and timestamp.
user_pref = user_pref.sort_values(["u", "t"]).reset_index(drop=True)

# Compute number of users and items based on max IDs.
n_users = int(user_pref["u"].max()) + 1
n_items = int(max(items["i"].max(), user_pref["i"].max())) + 1

# Get all unique user IDs.
users = np.sort(user_pref["u"].unique())

# Build lightweight interaction structures:
# - seen_by_user for seen item lookup,
# - popularity_df for popular items.
seen_by_user, popularity_df = build_interaction_lookup(user_pref)

# Load lightweight top similarity dictionaries.
try:
    item_top_dict, content_top_dict = load_top_similarities(
        item_similarity_path,
        content_similarity_path
    )
except FileNotFoundError:
    st.error(
        "Top similarity file not found. Check that top_item_similarities.csv and top_content_similarities.csv exist in kaggle_data."
    )
    st.stop()


# DATASET OVERVIEW

st.subheader("Dataset overview")

# Display three key dataset metrics.
c1, c2, c3 = st.columns(3)

c1.metric("Users", f"{len(users):,}")
c2.metric("Items", f"{len(items):,}")
c3.metric("Interactions", f"{len(user_pref):,}")

# Optional expandable preview of loaded data.
with st.expander("Preview data"):
    st.write("Items with covers + clean metadata")
    st.dataframe(items.head(20), use_container_width=True)

    st.write("Interactions")
    st.dataframe(user_pref.head(20), use_container_width=True)

    st.write("Precomputed recommendations")
    st.dataframe(recommendations_df.head(20), use_container_width=True)


# PERSONALIZED ENTRY FLOW

st.subheader("Start your personalized experience")

# Explanation card for the user.
st.markdown("""
<div class="custom-card">
    <p class="muted">
        Tell us who you are and whether you already exist in the dataset.
        If you are an existing user, the app uses the recommendation list already generated offline.
        If you are new, select a few books you love and the app will generate recommendations using item-item and content similarity.
    </p>
</div>
""", unsafe_allow_html=True)

# Create readable labels for book search/dropdowns.
book_labels = get_book_labels(items, title_col, author_col)

# Main user flow container.
with st.container():

    # Optional visitor name.
    visitor_name = st.text_input(
        "Your name",
        placeholder="e.g. Michalis"
    )

    # User chooses whether they are an existing or new user.
    user_type = st.radio(
        "Choose your profile type",
        ["Existing user", "New user"],
        horizontal=True
    )



    # EXISTING USER


    if user_type == "Existing user":

        # Existing users select their user ID.
        selected_user_id = st.selectbox(
            "Select your user ID",
            users
        )

        # When clicked, retrieve and display recommendations.
        if st.button("Get my personalized recommendations"):
            display_name = visitor_name.strip() if visitor_name.strip() else "there"

            # Load exactly 10 precomputed recommendations from the CSV.
            # Seen items are NOT removed, because the recommendation list should
            # remain exactly as produced by the recommender.
            recs = get_recommendations_from_csv(
                user_id=selected_user_id,
                recommendations_df=recommendations_df,
                items=items,
                top_k=10
            )

            # Make sure the app always displays exactly 10 recommendations if available.
            recs = recs.head(10).reset_index(drop=True)
            recs["rank"] = range(1, len(recs) + 1)

            # Success message.
            st.success(
                f"Welcome {display_name}! Here are your 10 personalized recommendations."
            )

            # Display summary metrics for this user.
            col1, col2 = st.columns(2)
            col1.metric("Recommendations", len(recs))
            col2.metric("Source", "R08 CSV")

            # Tabs for recommendation cards and table only.
            tab1, tab2 = st.tabs([
                "Book cards",
                "Recommendation table"
            ])

            # Visual card display.
            with tab1:
                display_book_cards(
                    recs,
                    title_col=title_col,
                    author_col=author_col,
                    score_col=None,
                    max_items=10
                )

            # Raw recommendation dataframe.
            with tab2:
                st.dataframe(recs, use_container_width=True)

            # Allow the user to download recommendations as CSV.
            st.download_button(
                "Download my recommendations",
                recs.to_csv(index=False),
                file_name=f"recommendations_user_{selected_user_id}.csv",
                mime="text/csv"
            )



    # NEW USER


    else:

        # Explanation for cold-start users.
        st.markdown("""
        Because you are not in the training data, the app cannot use the precomputed CSV.
        Instead, it uses item-item and content similarity: you choose books you like,
        and the app recommends similar books.
        """)

        # Initialize session state to store selected liked books.
        # This persists across interactions inside the same Streamlit session.
        if "liked_item_ids_new_user" not in st.session_state:
            st.session_state.liked_item_ids_new_user = []

        liked_item_ids = st.session_state.liked_item_ids_new_user

        # If readable book labels are available, use the search-based interface.
        if book_labels is not None:

            # Search input for title, author, or subject/category.
            search_query = st.text_input(
                "Search for a book category you enjoy reading",
                placeholder="Type a title, author, or subject keyword",
                key="new_user_search_query"
            )

            # Build list of searchable metadata columns.
            searchable_cols = []

            if title_col is not None and title_col in items.columns:
                searchable_cols.append(title_col)

            if author_col is not None and author_col in items.columns:
                searchable_cols.append(author_col)

            # Add additional metadata fields if available.
            for possible_col in ["Subjects", "subjects", "concepts", "Title", "Author"]:
                if possible_col in items.columns and possible_col not in searchable_cols:
                    searchable_cols.append(possible_col)

            # Create a search dataframe with item ID and searchable text.
            search_df = items[["i"] + searchable_cols].copy()

            # Combine searchable columns into one lowercase text string per item.
            search_df["search_text"] = (
                search_df[searchable_cols]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
            )

            # Initialize selected search results.
            selected_books_from_search = []
            selected_ids_from_search = []

            # If search box is empty, guide the user.
            if not search_query.strip():
                st.info("Start typing a title, author, or category to find books.")

            else:
                # Split the query into words.
                query_words = search_query.lower().split()

                # Start with all items as potential matches.
                mask = np.ones(len(search_df), dtype=bool)

                # Require every query word to appear in the searchable text.
                for word in query_words:
                    mask &= search_df["search_text"].str.contains(word, case=False, na=False)

                # Extract matching item IDs.
                matching_ids = search_df.loc[mask, "i"].astype(int).tolist()

                # Filter book labels to matching items.
                filtered_books = book_labels[
                    book_labels["i"].astype(int).isin(matching_ids)
                ].copy()

                # Do not show books already saved in the temporary user profile.
                already_saved = set(st.session_state.liked_item_ids_new_user)
                filtered_books = filtered_books[
                    ~filtered_books["i"].astype(int).isin(already_saved)
                ]

                # Limit visible results to keep the UI fast and readable.
                filtered_books = filtered_books.head(100)

                # If no result is found, show a warning.
                if filtered_books.empty:
                    st.warning(
                        "There may be a typo in your search. Please try again. "
                        "If the problem persists, try another category."
                    )

                # Otherwise show matching books in a multiselect widget.
                else:
                    st.caption(f"Showing up to 100 matching books for: '{search_query}'")

                    selected_books_from_search = st.multiselect(
                        "Select books from the category above that you have read and enjoyed",
                        options=filtered_books["label"].tolist(),
                        key="selected_books_from_search"
                    )

                    # Extract item IDs from selected labels.
                    selected_ids_from_search = [
                        int(label.split(" - ")[0]) for label in selected_books_from_search
                    ]

            # Add selected books to the temporary profile.
            if st.button("Add selected books"):
                if len(selected_ids_from_search) == 0:
                    st.warning("Please select at least one book from the search results first.")
                else:
                    for item_id in selected_ids_from_search:
                        if item_id not in st.session_state.liked_item_ids_new_user:
                            st.session_state.liked_item_ids_new_user.append(item_id)

                    st.success(
                        f"{len(selected_ids_from_search)} selected book(s) added to your profile."
                    )

                    # Rerun Streamlit so the selected books appear immediately.
                    st.rerun()

            # Refresh liked item list from session state.
            liked_item_ids = st.session_state.liked_item_ids_new_user

        # Fallback interface if no readable labels can be created.
        else:
            liked_item_ids = st.multiselect(
                "Select item IDs you like",
                options=np.sort(items["i"].unique())
            )

        # Display books currently selected by the new user.
        if len(liked_item_ids) > 0:
            st.markdown("### Books currently saved in your profile")

            saved_books_df = pd.DataFrame({"item_id": liked_item_ids})
            saved_books_df = enrich_with_items(saved_books_df, items, "item_id")

            display_book_cards(
                saved_books_df,
                title_col=title_col,
                author_col=author_col,
                max_items=len(liked_item_ids)
            )

            # Clear all selected books.
            if st.button("Clear selected books"):
                st.session_state.liked_item_ids_new_user = []
                st.rerun()

        # Recommendation button for new user.
        st.markdown("### Generate your recommendations")

        if st.button("Get recommendations for me"):
            display_name = visitor_name.strip() if visitor_name.strip() else "there"

            # A new user needs at least one selected book.
            if len(liked_item_ids) == 0:
                st.warning("Please select and add at least one book first.")
            else:

                # Build dataframe of selected liked books.
                liked_df = pd.DataFrame({"item_id": liked_item_ids})
                liked_df = enrich_with_items(liked_df, items, "item_id")

                st.success(
                    f"Welcome {display_name}! Here are your personalized recommendations."
                )

                # Generate cold-start recommendations using lightweight similarity dictionaries.
                new_user_recs = recommend_for_new_user_light(
                    liked_item_ids=liked_item_ids,
                    item_top_dict=item_top_dict,
                    content_top_dict=content_top_dict,
                    items=items,
                    top_k=10,
                    alpha=new_user_alpha
                )

                new_user_recs = new_user_recs.head(10).reset_index(drop=True)
                new_user_recs["rank"] = range(1, len(new_user_recs) + 1)

                # Display new-user results in tabs.
                tab_new_1, tab_new_2, tab_new_3 = st.tabs([
                    "Book cards",
                    "Recommendation table",
                    "Selected books"
                ])

                # Visual cards for recommendations.
                with tab_new_1:
                    display_book_cards(
                        new_user_recs,
                        title_col=title_col,
                        author_col=author_col,
                        score_col="score",
                        max_items=10
                    )

                # Raw recommendation table.
                with tab_new_2:
                    st.dataframe(new_user_recs, use_container_width=True)

                # Selected books used to build the profile.
                with tab_new_3:
                    st.markdown("These are the books used to build your new-user profile.")
                    display_book_cards(
                        liked_df,
                        title_col=title_col,
                        author_col=author_col,
                        max_items=len(liked_item_ids)
                    )

                # Download new-user recommendations.
                st.download_button(
                    "Download my recommendations",
                    new_user_recs.to_csv(index=False),
                    file_name="recommendations_new_user.csv",
                    mime="text/csv"
                )

    # Closing HTML div.
    st.markdown('</div>', unsafe_allow_html=True)


# POPULAR ITEMS

st.subheader("Popular items")

# Display most interacted books when the button is clicked.
if st.button("Show popular items"):

    # Take the top_k most popular items.
    popular_df = popularity_df.head(top_k).copy()

    # Add rank column.
    popular_df["rank"] = range(1, len(popular_df) + 1)

    # Merge with item metadata.
    popular_df = enrich_with_items(popular_df, items, "item_id")

    # Display popular items as book cards.
    display_book_cards(
        popular_df,
        title_col=title_col,
        author_col=author_col,
        score_col="number_of_interactions",
        max_items=top_k
    )

    # Also provide the raw table.
    with st.expander("Popular items table"):
        st.dataframe(popular_df, use_container_width=True)


# SIMILARITY EXPLORATION

st.subheader("Similarity exploration")

# Create one tab for similar-item exploration.
tab_i = st.tabs(["Similar items"])[0]

with tab_i:

    # Build readable dropdown labels if title and item ID are available.
    if title_col is not None and "i" in items.columns:
        sim_item_labels_df = items[["i", title_col]].copy()

        # Add author to labels if available.
        if author_col is not None and author_col in items.columns:
            sim_item_labels_df[author_col] = items[author_col]
            author_col_for_sim = author_col

        # Otherwise use a fallback author column.
        else:
            sim_item_labels_df["Unknown_author"] = "Unknown author"
            author_col_for_sim = "Unknown_author"

        # Keep only rows with item ID and title.
        sim_item_labels_df = sim_item_labels_df.dropna(subset=["i", title_col])

        def make_sim_label(row):
            """
            Create labels for the similarity dropdown.

            Format:
            item_id - title — author
            """

            item_id = int(row["i"])
            title = str(row[title_col])
            author = (
                str(row[author_col_for_sim])
                if pd.notna(row[author_col_for_sim])
                else "Unknown author"
            )
            return f"{item_id} - {title} — {author}"

        # Apply label creation.
        sim_item_labels_df["label"] = sim_item_labels_df.apply(make_sim_label, axis=1)

        # Dropdown to select the reference item.
        selected_item_label = st.selectbox(
            "Item",
            sim_item_labels_df["label"].tolist(),
            key="sim_i_label"
        )

        # Extract item ID from the label.
        i = int(selected_item_label.split(" - ")[0])

    # Fallback if readable labels cannot be created.
    else:
        item_options = np.sort(items["i"].unique()) if "i" in items.columns else np.arange(n_items)
        i = st.selectbox("Item", item_options, key="sim_i")
        i = int(i)

    # Retrieve the selected item's metadata.
    selected_item_row = items[items["i"].astype(int) == int(i)]

    # Extract selected title and author for the explanatory sentence.
    if not selected_item_row.empty:
        selected_item_row = selected_item_row.iloc[0]

        selected_title = (
            str(selected_item_row[title_col])
            if title_col is not None and title_col in selected_item_row and pd.notna(selected_item_row[title_col])
            else f"Item {i}"
        )

        selected_author = (
            str(selected_item_row[author_col])
            if author_col is not None and author_col in selected_item_row and pd.notna(selected_item_row[author_col])
            else "Unknown author"
        )

    # Fallback if the selected item is not found.
    else:
        selected_title = f"Item {i}"
        selected_author = "Unknown author"

    # Display explanation for the selected item.
    st.markdown(
        f"""
        <div class="custom-card">
            <p>
                Here are the books most similar to 
                <strong>{selected_title}</strong> by <strong>{selected_author}</strong>
                <span style="opacity:0.75;">(item {i} in the database)</span>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Look up top similar items from the precomputed item similarity dictionary.
    if int(i) in item_top_dict:
        top_items, top_scores = item_top_dict[int(i)]

        # Keep only the top 10 similar items for display.
        top_items = top_items[:10]
        top_scores = top_scores[:10]

        # Create dataframe for similar items.
        similar_items_df = pd.DataFrame({
            "rank": range(1, len(top_items) + 1),
            "item_id": top_items,
            "similarity_score": top_scores
        })

    # If no similarity information exists for the item, return an empty dataframe.
    else:
        similar_items_df = pd.DataFrame()

    # Merge similar item IDs with metadata.
    similar_items_df = enrich_with_items(similar_items_df, items, "item_id")

    # Display similar items as book cards.
    display_book_cards(
        similar_items_df,
        title_col=title_col,
        author_col=author_col,
        score_col="similarity_score",
        max_items=10
    )

    # Also provide the raw similar-items table.
    with st.expander("Similar items table"):
        st.dataframe(similar_items_df, use_container_width=True)
        