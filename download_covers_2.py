# ============================================================
# download_covers.py
# Download more book covers using ISBN + title/author fallback
# ============================================================

import os
import re
import time
import requests
import pandas as pd
from urllib.parse import quote_plus


ITEMS_PATH = "kaggle_data/items.csv"
OUTPUT_DIR = "kaggle_data/covers"
OUTPUT_CSV = "kaggle_data/items_with_covers.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_isbn(isbn):
    if pd.isna(isbn):
        return None

    isbn = str(isbn).strip()
    isbn = isbn.replace("-", "").replace(" ", "")
    isbn = re.sub(r"[^0-9Xx]", "", isbn)

    if len(isbn) in [10, 13]:
        return isbn

    return None


def extract_best_isbn(isbn_field):
    if pd.isna(isbn_field):
        return None

    raw_isbns = re.split(r"[;,| ]+", str(isbn_field))
    cleaned = [clean_isbn(x) for x in raw_isbns]
    cleaned = [x for x in cleaned if x is not None]

    if not cleaned:
        return None

    isbn13 = [x for x in cleaned if len(x) == 13]
    if isbn13:
        return isbn13[0]

    return cleaned[0]


def get_title_column(items):
    for col in ["Title", "title", "Book-Title", "book_title", "name"]:
        if col in items.columns:
            return col
    return None


def get_author_column(items):
    for col in ["Author", "author", "authors", "Book-Author"]:
        if col in items.columns:
            return col
    return None


def save_image_from_url(url, output_path):
    try:
        r = requests.get(url, timeout=15)

        if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
            if len(r.content) > 1000:
                with open(output_path, "wb") as f:
                    f.write(r.content)
                return output_path

    except requests.RequestException:
        pass

    return None


def download_cover_by_isbn(isbn, item_id, size="L"):
    if isbn is None:
        return None

    output_path = os.path.join(OUTPUT_DIR, f"{item_id}.jpg")

    if os.path.exists(output_path):
        return output_path

    url = f"https://covers.openlibrary.org/b/isbn/{isbn}-{size}.jpg?default=false"
    return save_image_from_url(url, output_path)


def search_cover_id_by_title_author(title, author=None):
    if pd.isna(title):
        return None

    title = str(title).strip()
    if title == "":
        return None

    query = f"title={quote_plus(title)}"

    if author is not None and pd.notna(author):
        author = str(author).strip()
        if author != "":
            query += f"&author={quote_plus(author)}"

    url = f"https://openlibrary.org/search.json?{query}&limit=3"

    try:
        r = requests.get(url, timeout=15)

        if r.status_code != 200:
            return None

        data = r.json()
        docs = data.get("docs", [])

        for doc in docs:
            if "cover_i" in doc:
                return doc["cover_i"]

    except requests.RequestException:
        return None
    except ValueError:
        return None

    return None


def download_cover_by_cover_id(cover_id, item_id, size="L"):
    if cover_id is None:
        return None

    output_path = os.path.join(OUTPUT_DIR, f"{item_id}.jpg")

    if os.path.exists(output_path):
        return output_path

    url = f"https://covers.openlibrary.org/b/id/{cover_id}-{size}.jpg?default=false"
    return save_image_from_url(url, output_path)


items = pd.read_csv(ITEMS_PATH)

title_col = get_title_column(items)
author_col = get_author_column(items)

if title_col is None:
    raise ValueError("No title column found in items.csv")

items["clean_isbn"] = items["ISBN Valid"].apply(extract_best_isbn) if "ISBN Valid" in items.columns else None
items["cover_path"] = None
items["cover_source"] = None


for idx, row in items.iterrows():
    item_id = int(row["i"]) if "i" in items.columns else idx

    output_path = os.path.join(OUTPUT_DIR, f"{item_id}.jpg")

    if os.path.exists(output_path):
        items.at[idx, "cover_path"] = output_path
        items.at[idx, "cover_source"] = "existing"
        continue

    cover_path = download_cover_by_isbn(row.get("clean_isbn"), item_id)

    if cover_path is not None:
        items.at[idx, "cover_path"] = cover_path
        items.at[idx, "cover_source"] = "isbn"
    else:
        title = row[title_col]
        author = row[author_col] if author_col is not None else None

        cover_id = search_cover_id_by_title_author(title, author)
        cover_path = download_cover_by_cover_id(cover_id, item_id)

        if cover_path is not None:
            items.at[idx, "cover_path"] = cover_path
            items.at[idx, "cover_source"] = "title_author"

    if idx % 100 == 0:
        found = items["cover_path"].notna().sum()
        print(f"Processed {idx}/{len(items)} books | covers found so far: {found}")

    time.sleep(0.15)


items.to_csv(OUTPUT_CSV, index=False)

print("Done!")
print(f"Saved enriched file to: {OUTPUT_CSV}")
print(f"Covers saved in: {OUTPUT_DIR}")
print(f"Covers found: {items['cover_path'].notna().sum()} / {len(items)}")