# ============================================================
# UI_ML.py
# Improved Streamlit UI with side background library image
# ============================================================

import os
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="BookMatch AI",
    page_icon="🏷️",
    layout="wide"
)


# ============================================================
# BACKGROUND IMAGE HELPERS
# ============================================================

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()


LIBRARY_BG_PATH = "best-libraries-from-around-the-world-the-admont-1.v1517654478.png"

library_bg_base64 = None
if os.path.exists(LIBRARY_BG_PATH):
    library_bg_base64 = get_base64_of_bin_file(LIBRARY_BG_PATH)


# ============================================================
# STYLE
# ============================================================

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


# ============================================================
# SIDE BACKGROUNDS
# ============================================================

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


# ============================================================
# DATA + MODEL
# ============================================================

@st.cache_data
def load_data(items_path, interactions_path, recommendations_path, clean_items_path, descriptions_path):
    items = pd.read_csv(items_path)
    interactions = pd.read_csv(interactions_path)
    recommendations = pd.read_csv(recommendations_path)
    clean_items = pd.read_csv(clean_items_path)

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
    """

    seen_by_user = (
        interactions
        .groupby("u")["i"]
        .apply(lambda x: set(x.astype(int)))
        .to_dict()
    )

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
    seen_items = list(seen_by_user.get(int(user_id), set()))
    seen_df = pd.DataFrame({"item_id": seen_items})
    seen_df = enrich_with_items(seen_df, items, "item_id")
    return seen_df


@st.cache_data(show_spinner="Loading top similarities...")
def load_top_similarities(item_top_path, content_top_path):
    item_top = pd.read_csv(item_top_path)
    content_top = pd.read_csv(content_top_path)

    item_top_dict = {
        int(row["item_id"]): (
            [int(x) for x in str(row["similar_items"]).split()],
            [float(x) for x in str(row["similar_scores"]).split()]
        )
        for _, row in item_top.iterrows()
    }

    content_top_dict = {
        int(row["item_id"]): (
            [int(x) for x in str(row["similar_items"]).split()],
            [float(x) for x in str(row["similar_scores"]).split()]
        )
        for _, row in content_top.iterrows()
    }

    return item_top_dict, content_top_dict


def get_title_column(items):
    for col in ["Title_clean", "title_clean", "Title", "title", "Book-Title", "book_title", "name"]:
        if col in items.columns:
            return col
    return None


def get_author_column(items):
    for col in ["Author_clean", "author_clean", "Author", "author", "authors", "Book-Author"]:
        if col in items.columns:
            return col
    return None


def merge_clean_metadata(items, clean_items):
    """
    Keep covers and all original columns from items,
    but replace display metadata with clean titles/authors from clean_items.
    """
    items = items.copy()
    clean_items = clean_items.copy()

    if "i" not in items.columns or "i" not in clean_items.columns:
        return items

    clean_title_col = None
    for col in ["Title", "title", "Book-Title", "book_title", "name"]:
        if col in clean_items.columns:
            clean_title_col = col
            break

    clean_author_col = None
    for col in ["Author", "author", "authors", "Book-Author"]:
        if col in clean_items.columns:
            clean_author_col = col
            break

    cols_to_merge = ["i"]

    if clean_title_col is not None:
        cols_to_merge.append(clean_title_col)

    if clean_author_col is not None:
        cols_to_merge.append(clean_author_col)

    clean_subset = clean_items[cols_to_merge].copy()

    rename_dict = {}

    if clean_title_col is not None:
        rename_dict[clean_title_col] = "Title_clean"

    if clean_author_col is not None:
        rename_dict[clean_author_col] = "Author_clean"

    clean_subset = clean_subset.rename(columns=rename_dict)

    items = items.merge(clean_subset, on="i", how="left")

    return items


def enrich_with_items(df, items, item_col="item_id"):
    if df.empty:
        return df

    if "i" in items.columns and item_col in df.columns:
        return df.merge(items, left_on=item_col, right_on="i", how="left")

    return df


def get_recommendations_from_csv(user_id, recommendations_df, items, top_k):
    row = recommendations_df[recommendations_df["user_id"] == int(user_id)]

    if row.empty:
        return pd.DataFrame()

    rec_string = row.iloc[0]["recommendation"]
    rec_items = [int(x) for x in str(rec_string).split()]

    recs = pd.DataFrame({
        "rank": range(1, len(rec_items) + 1),
        "item_id": rec_items
    })

    recs = enrich_with_items(recs, items, "item_id")
    recs = recs.head(top_k).reset_index(drop=True)
    recs["rank"] = range(1, len(recs) + 1)

    return recs


def get_book_labels(items, title_col, author_col=None):
    if title_col is None or "i" not in items.columns:
        return None

    cols = ["i", title_col]

    if author_col is not None and author_col in items.columns:
        cols.append(author_col)

    book_labels = items[cols].dropna(subset=["i", title_col]).copy()

    def make_label(row):
        title = str(row[title_col])

        author = (
            str(row[author_col])
            if author_col and author_col in row and pd.notna(row[author_col])
            else "Unknown author"
        )

        return f"{int(row['i'])} - {title} — {author}"

    book_labels["label"] = book_labels.apply(make_label, axis=1)

    return book_labels


def recommend_for_new_user_light(liked_item_ids, item_top_dict, content_top_dict, items, top_k=10, alpha=0.5):
    """
    Lightweight new-user recommender for Streamlit Cloud.

    Instead of loading full dense similarity matrices,
    it uses precomputed top similar items for each selected book.
    """

    liked_item_ids = [int(i) for i in liked_item_ids]
    liked_set = set(liked_item_ids)

    scores = {}

    for liked_id in liked_item_ids:

        # Item-item collaborative similarity neighbors
        if liked_id in item_top_dict:
            neighbor_ids, neighbor_scores = item_top_dict[liked_id]

            for item_id, score in zip(neighbor_ids, neighbor_scores):
                if item_id not in liked_set:
                    scores[item_id] = scores.get(item_id, 0) + alpha * score

        # Content-based similarity neighbors
        if liked_id in content_top_dict:
            neighbor_ids, neighbor_scores = content_top_dict[liked_id]

            for item_id, score in zip(neighbor_ids, neighbor_scores):
                if item_id not in liked_set:
                    scores[item_id] = scores.get(item_id, 0) + (1 - alpha) * score

    if len(scores) == 0:
        return pd.DataFrame()

    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    recs = pd.DataFrame({
        "rank": range(1, len(top_items) + 1),
        "item_id": [x[0] for x in top_items],
        "score": [x[1] for x in top_items]
    })

    recs = enrich_with_items(recs, items, "item_id")

    return recs


def display_book_cards(df, title_col=None, author_col=None, score_col=None, max_items=20):
    if df.empty:
        st.warning("No items to display.")
        return

    df = df.head(max_items).reset_index(drop=True)
    n_cols = 5

    def image_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    for start in range(0, len(df), n_cols):
        row_df = df.iloc[start:start + n_cols]
        cols = st.columns(n_cols)

        for j, (_, row) in enumerate(row_df.iterrows()):
            with cols[j]:
                rank_text = f"#{int(row['rank'])}" if "rank" in row and pd.notna(row["rank"]) else ""

                title_text = (
                    str(row[title_col])
                    if title_col and title_col in row and pd.notna(row[title_col])
                    else f"Item {row.get('item_id', 'unknown')}"
                )

                author_text = (
                    str(row[author_col])
                    if author_col and author_col in row and pd.notna(row[author_col])
                    else "Unknown author"
                )

                score_text = ""

                if score_col and score_col in row and pd.notna(row[score_col]):
                    if score_col == "number_of_interactions":
                        score_text = f"{int(row[score_col])} interactions"
                    elif score_col == "similarity_score":
                        score_text = f"Similarity: {row[score_col]:.4f}"
                    else:
                        score_text = f"Score: {row[score_col]:.4f}"

                cover_path = row.get("cover_path", None)

                if pd.notna(cover_path) and isinstance(cover_path, str) and os.path.exists(cover_path):
                    ext = cover_path.split(".")[-1].lower()
                    mime = "jpeg" if ext in ["jpg", "jpeg"] else ext
                    img_b64 = image_to_base64(cover_path)

                    cover_html = f"""
                    <div class="book-cover-box">
                        <img src="data:image/{mime};base64,{img_b64}" class="book-cover-img"/>
                    </div>
                    """
                else:
                    cover_html = f"""
                    <div class="book-cover-box book-cover-placeholder">
                        <div class="book-cover-placeholder-title">{title_text}</div>
                    </div>
                    """

                st.markdown(f"""
                {cover_html}
                <div class="book-info-card">
                    <div class="book-rank">{rank_text}</div>
                    <div class="book-title">{title_text}</div>
                    <div class="book-author">{author_text}</div>
                    <div class="book-score">{score_text}</div>
                </div>
                """, unsafe_allow_html=True)

                description_text = (
                    str(row["description"])
                    if "description" in row
                    and pd.notna(row["description"])
                    and str(row["description"]).strip()
                    else "No description available for this book."
                )

                with st.expander("View description"):
                    st.write(description_text)





# ============================================================
# HEADER
# ============================================================

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


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ⚙️ Settings")
st.sidebar.markdown(
    "<p style='color:#cbd5e1; font-size:0.95rem;'>Adjust the recommender parameters and explore personalized results.</p>",
    unsafe_allow_html=True
)

items_path = st.sidebar.text_input(
    "Items path",
    "kaggle_data/items_with_covers.csv"
)

descriptions_path = st.sidebar.text_input(
    "Descriptions path",
    "kaggle_data/item_descriptions.csv"
)

clean_items_path = st.sidebar.text_input(
    "Clean items path",
    "clean_items.csv"
)

interactions_path = st.sidebar.text_input(
    "Interactions path",
    "kaggle_data/interactions_train.csv"
)

recommendations_path = st.sidebar.text_input(
    "Recommendations path",
    "Submission/Hybrid_0.3_0.3_SBB_R08_final.csv"
)

item_similarity_path = st.sidebar.text_input(
    "Top item similarities path",
    "kaggle_data/top_item_similarities.csv"
)

content_similarity_path = st.sidebar.text_input(
    "Top content similarities path",
    "kaggle_data/top_content_similarities.csv"
)

new_user_alpha = st.sidebar.slider(
    "New user alpha: item-item vs content",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

top_k = st.sidebar.slider(
    "Top K",
    min_value=1,
    max_value=50,
    value=10
)

remove_seen = st.sidebar.checkbox(
    "Remove seen items",
    value=True
)


# ============================================================
# LOAD DATA
# ============================================================

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

items = merge_clean_metadata(items, clean_items)

if "i" in items.columns and "i" in descriptions.columns:
    description_cols = ["i"]

    for col in ["description", "api_title", "api_authors", "api_source", "api_query"]:
        if col in descriptions.columns:
            description_cols.append(col)

    cols_to_drop = [
        col for col in ["description", "api_title", "api_authors", "api_source", "api_query"]
        if col in items.columns
    ]

    if cols_to_drop:
        items = items.drop(columns=cols_to_drop)

    items = items.merge(
        descriptions[description_cols],
        on="i",
        how="left"
    )

title_col = get_title_column(items)
author_col = get_author_column(items)

user_pref = user_pref.sort_values(["u", "t"]).reset_index(drop=True)

n_users = int(user_pref["u"].max()) + 1
n_items = int(max(items["i"].max(), user_pref["i"].max())) + 1
users = np.sort(user_pref["u"].unique())

seen_by_user, popularity_df = build_interaction_lookup(user_pref)

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


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader("Dataset overview")

c1, c2, c3 = st.columns(3)

c1.metric("Users", f"{len(users):,}")
c2.metric("Items", f"{len(items):,}")
c3.metric("Interactions", f"{len(user_pref):,}")

with st.expander("Preview data"):
    st.write("Items with covers + clean metadata")
    st.dataframe(items.head(20), use_container_width=True)

    st.write("Interactions")
    st.dataframe(user_pref.head(20), use_container_width=True)

    st.write("Precomputed recommendations")
    st.dataframe(recommendations_df.head(20), use_container_width=True)


# ============================================================
# PERSONALIZED ENTRY FLOW
# ============================================================

st.subheader("Start your personalized experience")

st.markdown("""
<div class="custom-card">
    <p class="muted">
        Tell us who you are and whether you already exist in the dataset.
        If you are an existing user, the app uses the recommendation list already generated offline.
        If you are new, select a few books you love and the app will generate recommendations using item-item and content similarity.
    </p>
</div>
""", unsafe_allow_html=True)

book_labels = get_book_labels(items, title_col, author_col)

with st.container():
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)

    visitor_name = st.text_input(
        "Your name",
        placeholder="e.g. Michalis"
    )

    user_type = st.radio(
        "Choose your profile type",
        ["Existing user", "New user"],
        horizontal=True
    )

    # ============================================================
    # EXISTING USER
    # ============================================================

    if user_type == "Existing user":
        selected_user_id = st.selectbox(
            "Select your user ID",
            users
        )

        if st.button("Get my personalized recommendations"):
            display_name = visitor_name.strip() if visitor_name.strip() else "there"

            recs = get_recommendations_from_csv(
                user_id=selected_user_id,
                recommendations_df=recommendations_df,
                items=items,
                top_k=max(top_k * 3, top_k)
            )

            seen = seen_by_user.get(int(selected_user_id), set())

            if remove_seen:
                recs = recs[~recs["item_id"].isin(seen)].head(top_k).reset_index(drop=True)
                recs["rank"] = range(1, len(recs) + 1)
            else:
                recs = recs.head(top_k).reset_index(drop=True)
                recs["rank"] = range(1, len(recs) + 1)

            seen_df = get_seen_items_light(selected_user_id, seen_by_user, items)

            st.success(
                f"Welcome {display_name}! Here are your personalized recommendations."
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Seen items", len(seen))
            col2.metric("Recommendations", len(recs))
            col3.metric("Source", "R08 CSV")

            tab1, tab2, tab3 = st.tabs([
                "Book cards",
                "Recommendation table",
                "Seen items"
            ])

            with tab1:
                display_book_cards(
                    recs,
                    title_col=title_col,
                    author_col=author_col,
                    score_col=None,
                    max_items=top_k
                )

            with tab2:
                st.dataframe(recs, use_container_width=True)

            with tab3:
                st.markdown("These are books already interacted with by this user.")
                display_book_cards(
                    seen_df,
                    title_col=title_col,
                    author_col=author_col,
                    max_items=20
                )

                with st.expander("Seen items table"):
                    st.dataframe(seen_df, use_container_width=True)

            st.download_button(
                "Download my recommendations",
                recs.to_csv(index=False),
                file_name=f"recommendations_user_{selected_user_id}.csv",
                mime="text/csv"
            )

    # ============================================================
    # NEW USER
    # ============================================================

    else:
        st.markdown("""
        Because you are not in the training data, the app cannot use the precomputed CSV.
        Instead, it uses item-item and content similarity: you choose books you like,
        and the app recommends similar books.
        """)

        if "liked_item_ids_new_user" not in st.session_state:
            st.session_state.liked_item_ids_new_user = []

        liked_item_ids = st.session_state.liked_item_ids_new_user

        if book_labels is not None:
            search_query = st.text_input(
                "Search for a book category you enjoy reading",
                placeholder="Type a title, author, or subject keyword",
                key="new_user_search_query"
            )

            searchable_cols = []

            if title_col is not None and title_col in items.columns:
                searchable_cols.append(title_col)

            if author_col is not None and author_col in items.columns:
                searchable_cols.append(author_col)

            for possible_col in ["Subjects", "subjects", "concepts", "Title", "Author"]:
                if possible_col in items.columns and possible_col not in searchable_cols:
                    searchable_cols.append(possible_col)

            search_df = items[["i"] + searchable_cols].copy()

            search_df["search_text"] = (
                search_df[searchable_cols]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
            )

            selected_books_from_search = []
            selected_ids_from_search = []

            if not search_query.strip():
                st.info("Start typing a title, author, or category to find books.")

            else:
                query_words = search_query.lower().split()

                mask = np.ones(len(search_df), dtype=bool)

                for word in query_words:
                    mask &= search_df["search_text"].str.contains(word, case=False, na=False)

                matching_ids = search_df.loc[mask, "i"].astype(int).tolist()

                filtered_books = book_labels[
                    book_labels["i"].astype(int).isin(matching_ids)
                ].copy()

                already_saved = set(st.session_state.liked_item_ids_new_user)
                filtered_books = filtered_books[
                    ~filtered_books["i"].astype(int).isin(already_saved)
                ]

                filtered_books = filtered_books.head(100)

                if filtered_books.empty:
                    st.warning(
                        "There may be a typo in your search. Please try again. "
                        "If the problem persists, try another category."
                    )
                else:
                    st.caption(f"Showing up to 100 matching books for: '{search_query}'")

                    selected_books_from_search = st.multiselect(
                        "Select books from the category above that you have read and enjoyed",
                        options=filtered_books["label"].tolist(),
                        key="selected_books_from_search"
                    )

                    selected_ids_from_search = [
                        int(label.split(" - ")[0]) for label in selected_books_from_search
                    ]

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

                    st.rerun()

            liked_item_ids = st.session_state.liked_item_ids_new_user

        else:
            liked_item_ids = st.multiselect(
                "Select item IDs you like",
                options=np.sort(items["i"].unique())
            )

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

            if st.button("Clear selected books"):
                st.session_state.liked_item_ids_new_user = []
                st.rerun()

        st.markdown("### Generate your recommendations")

        if st.button("Get recommendations for me"):
            display_name = visitor_name.strip() if visitor_name.strip() else "there"

            if len(liked_item_ids) == 0:
                st.warning("Please select and add at least one book first.")
            else:
                liked_df = pd.DataFrame({"item_id": liked_item_ids})
                liked_df = enrich_with_items(liked_df, items, "item_id")

                st.success(
                    f"Welcome {display_name}! Here are your personalized recommendations."
                )

                new_user_recs = recommend_for_new_user_light(
                    liked_item_ids=liked_item_ids,
                    item_top_dict=item_top_dict,
                    content_top_dict=content_top_dict,
                    items=items,
                    top_k=top_k,
                    alpha=new_user_alpha
                )

                tab_new_1, tab_new_2, tab_new_3 = st.tabs([
                    "Book cards",
                    "Recommendation table",
                    "Selected books"
                ])

                with tab_new_1:
                    display_book_cards(
                        new_user_recs,
                        title_col=title_col,
                        author_col=author_col,
                        score_col="score",
                        max_items=top_k
                    )

                with tab_new_2:
                    st.dataframe(new_user_recs, use_container_width=True)

                with tab_new_3:
                    st.markdown("These are the books used to build your new-user profile.")
                    display_book_cards(
                        liked_df,
                        title_col=title_col,
                        author_col=author_col,
                        max_items=len(liked_item_ids)
                    )

                st.download_button(
                    "Download my recommendations",
                    new_user_recs.to_csv(index=False),
                    file_name="recommendations_new_user.csv",
                    mime="text/csv"
                )

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# POPULAR ITEMS
# ============================================================

st.subheader("Popular items")

if st.button("Show popular items"):
    popular_df = popularity_df.head(top_k).copy()
    popular_df["rank"] = range(1, len(popular_df) + 1)

    popular_df = enrich_with_items(popular_df, items, "item_id")

    display_book_cards(
        popular_df,
        title_col=title_col,
        author_col=author_col,
        score_col="number_of_interactions",
        max_items=top_k
    )

    with st.expander("Popular items table"):
        st.dataframe(popular_df, use_container_width=True)


# ============================================================
# SIMILARITY EXPLORATION
# ============================================================

st.subheader("Similarity exploration")

tab_i = st.tabs(["Similar items"])[0]

with tab_i:
    if title_col is not None and "i" in items.columns:
        sim_item_labels_df = items[["i", title_col]].copy()

        if author_col is not None and author_col in items.columns:
            sim_item_labels_df[author_col] = items[author_col]
            author_col_for_sim = author_col
        else:
            sim_item_labels_df["Unknown_author"] = "Unknown author"
            author_col_for_sim = "Unknown_author"

        sim_item_labels_df = sim_item_labels_df.dropna(subset=["i", title_col])

        def make_sim_label(row):
            item_id = int(row["i"])
            title = str(row[title_col])
            author = (
                str(row[author_col_for_sim])
                if pd.notna(row[author_col_for_sim])
                else "Unknown author"
            )
            return f"{item_id} - {title} — {author}"

        sim_item_labels_df["label"] = sim_item_labels_df.apply(make_sim_label, axis=1)

        selected_item_label = st.selectbox(
            "Item",
            sim_item_labels_df["label"].tolist(),
            key="sim_i_label"
        )

        i = int(selected_item_label.split(" - ")[0])

    else:
        item_options = np.sort(items["i"].unique()) if "i" in items.columns else np.arange(n_items)
        i = st.selectbox("Item", item_options, key="sim_i")
        i = int(i)

    selected_item_row = items[items["i"].astype(int) == int(i)]

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
    else:
        selected_title = f"Item {i}"
        selected_author = "Unknown author"

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

    if int(i) in item_top_dict:
        top_items, top_scores = item_top_dict[int(i)]

        top_items = top_items[:10]
        top_scores = top_scores[:10]

        similar_items_df = pd.DataFrame({
            "rank": range(1, len(top_items) + 1),
            "item_id": top_items,
            "similarity_score": top_scores
        })
    else:
        similar_items_df = pd.DataFrame()

    similar_items_df = enrich_with_items(similar_items_df, items, "item_id")

    display_book_cards(
        similar_items_df,
        title_col=title_col,
        author_col=author_col,
        score_col="similarity_score",
        max_items=10
    )

    with st.expander("Similar items table"):
        st.dataframe(similar_items_df, use_container_width=True)

