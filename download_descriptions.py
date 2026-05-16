# ============================================================
# download_descriptions_robust.py
# Robust description downloader for BookMatch AI
#
# It does NOT download covers.
# It creates only:
# kaggle_data/item_descriptions.csv
# ============================================================

import os
import re
import time
import html
import requests
import pandas as pd
from urllib.parse import quote_plus


INPUT_PATH = "kaggle_data/items.csv"
OUTPUT_PATH = "kaggle_data/item_descriptions.csv"

SAVE_EVERY = 25
SLEEP_SECONDS = 0.20
MAX_ITEMS = None  # set to 100 for testing, then None for full run


HEADERS = {
    "User-Agent": "BookMatchAI-EPFL-StudentProject/1.0"
}


# ============================================================
# Cleaning helpers
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value)
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_title_for_query(title):
    title = clean_text(title)

    # Dataset titles often contain a trailing slash or catalog punctuation
    title = title.split("/")[0].strip()
    title = title.strip(" .,:;-/")

    # Remove bracketed catalog noise
    title = re.sub(r"\[[^\]]*\]", "", title).strip()

    # Keep a shorter main title when the full title is too long
    if ":" in title and len(title) > 55:
        title = title.split(":")[0].strip()

    return title


def clean_author_for_query(author):
    author = clean_text(author)
    if not author:
        return ""

    # Keep first listed author
    author = author.split(";")[0].strip()
    author = author.split("/")[0].strip()

    # Remove birth years or catalog suffixes if present
    author = re.sub(r"\b\d{4}-?\b", "", author).strip(" ,;-")

    return author


def extract_best_isbn(row):
    if "ISBN Valid" not in row or pd.isna(row["ISBN Valid"]):
        return ""

    text = str(row["ISBN Valid"])
    text = text.replace("-", "").replace(" ", "")
    matches = re.findall(r"(?:97[89])?\d{9}[\dXx]", text)

    if not matches:
        return ""

    isbn13 = [x for x in matches if len(x) == 13]
    if isbn13:
        return isbn13[0]

    return matches[0]


def normalize_description(value):
    """
    Open Library sometimes returns:
    - a string
    - a dict with {"type": "...", "value": "..."}
    """
    if isinstance(value, dict):
        value = value.get("value", "")

    value = clean_text(value)

    # Avoid useless descriptions
    if len(value) < 40:
        return ""

    return value


def make_fallback_description(row):
    """
    Generates a simple fallback description from local metadata.
    This is not an API summary, but it avoids empty cards.
    """
    title = clean_text(row.get("Title", "this book"))
    author = clean_text(row.get("Author", ""))
    publisher = clean_text(row.get("Publisher", ""))
    subjects = clean_text(row.get("Subjects", ""))

    subject_list = []
    if subjects:
        subject_list = [s.strip() for s in subjects.split(";") if s.strip()]
        subject_list = subject_list[:5]

    parts = []

    if title and author:
        parts.append(f"{title} is a book by {author}.")
    elif title:
        parts.append(f"{title} is a book from the catalog.")

    if subject_list:
        parts.append("It is associated with the following themes: " + ", ".join(subject_list) + ".")

    if publisher:
        parts.append(f"The book is published by {publisher}.")

    if not parts:
        return "No external description was found for this book, but it is part of the BookMatch catalog."

    return " ".join(parts)


# ============================================================
# Open Library helpers
# ============================================================

def openlibrary_work_description(work_key):
    if not work_key:
        return ""

    if not work_key.startswith("/works/"):
        work_key = "/works/" + work_key.replace("/works/", "")

    url = f"https://openlibrary.org{work_key}.json"

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return ""

        data = r.json()
        return normalize_description(data.get("description", ""))

    except Exception:
        return ""


def openlibrary_by_isbn(isbn):
    if not isbn:
        return None

    url = f"https://openlibrary.org/isbn/{isbn}.json"

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()

        # Direct edition description
        desc = normalize_description(data.get("description", ""))
        if desc:
            return {
                "description": desc,
                "api_title": clean_text(data.get("title", "")),
                "api_authors": "",
                "api_source": "Open Library ISBN",
                "api_query": isbn
            }

        # Work-level description
        works = data.get("works", [])
        if works:
            work_key = works[0].get("key", "")
            desc = openlibrary_work_description(work_key)
            if desc:
                return {
                    "description": desc,
                    "api_title": clean_text(data.get("title", "")),
                    "api_authors": "",
                    "api_source": "Open Library ISBN -> Work",
                    "api_query": isbn
                }

    except Exception:
        return None

    return None


def openlibrary_by_search(title, author=""):
    title_q = clean_title_for_query(title)
    author_q = clean_author_for_query(author)

    if not title_q:
        return None

    params = {
        "title": title_q,
        "limit": 5
    }

    if author_q:
        params["author"] = author_q

    try:
        r = requests.get(
            "https://openlibrary.org/search.json",
            params=params,
            headers=HEADERS,
            timeout=10
        )

        if r.status_code != 200:
            return None

        data = r.json()
        docs = data.get("docs", [])

        for doc in docs:
            work_key = doc.get("key", "")
            desc = openlibrary_work_description(work_key)

            if desc:
                return {
                    "description": desc,
                    "api_title": clean_text(doc.get("title", "")),
                    "api_authors": ", ".join(doc.get("author_name", [])),
                    "api_source": "Open Library Search -> Work",
                    "api_query": f"title={title_q}; author={author_q}"
                }

    except Exception:
        return None

    return None


# ============================================================
# Google Books helpers
# ============================================================

def google_books_search(title, author="", isbn=""):
    title_q = clean_title_for_query(title)
    author_q = clean_author_for_query(author)

    queries = []

    if isbn:
        queries.append(f"isbn:{isbn}")

    if title_q and author_q:
        queries.append(f"{title_q} {author_q}")

    if title_q:
        queries.append(title_q)

    for q in queries:
        try:
            r = requests.get(
                "https://www.googleapis.com/books/v1/volumes",
                params={
                    "q": q,
                    "maxResults": 5,
                    "printType": "books"
                },
                headers=HEADERS,
                timeout=10
            )

            if r.status_code != 200:
                continue

            data = r.json()
            results = data.get("items", [])

            for result in results:
                info = result.get("volumeInfo", {})
                desc = normalize_description(info.get("description", ""))

                if desc:
                    return {
                        "description": desc,
                        "api_title": clean_text(info.get("title", "")),
                        "api_authors": ", ".join(info.get("authors", [])),
                        "api_source": "Google Books",
                        "api_query": q
                    }

        except Exception:
            continue

    return None


# ============================================================
# Main description retrieval
# ============================================================

def find_description(row):
    title = clean_text(row.get("Title", ""))
    author = clean_text(row.get("Author", ""))
    isbn = extract_best_isbn(row)

    sources = [
        lambda: openlibrary_by_isbn(isbn),
        lambda: openlibrary_by_search(title, author),
        lambda: google_books_search(title, author, isbn),
    ]

    for source in sources:
        result = source()
        if result and result.get("description"):
            return result

        time.sleep(SLEEP_SECONDS)

    return {
        "description": make_fallback_description(row),
        "api_title": "",
        "api_authors": "",
        "api_source": "Local metadata fallback",
        "api_query": ""
    }


def main():
    items = pd.read_csv(INPUT_PATH)

    if MAX_ITEMS is not None:
        items = items.head(MAX_ITEMS)

    rows = []

    for idx, row in items.iterrows():
        item_id = int(row["i"]) if "i" in items.columns else int(idx)

        result = find_description(row)

        rows.append({
            "i": item_id,
            "Title": clean_text(row.get("Title", "")),
            "Author": clean_text(row.get("Author", "")),
            "description": result.get("description", ""),
            "api_title": result.get("api_title", ""),
            "api_authors": result.get("api_authors", ""),
            "api_source": result.get("api_source", ""),
            "api_query": result.get("api_query", "")
        })

        if len(rows) % SAVE_EVERY == 0:
            out = pd.DataFrame(rows)
            out.to_csv(OUTPUT_PATH, index=False)

            n_external = out["api_source"].fillna("").astype(str).str.contains(
                "Open Library|Google Books", regex=True
            ).sum()

            n_fallback = (out["api_source"] == "Local metadata fallback").sum()

            print(
                f"Saved progress: {len(rows)} books processed | "
                f"external descriptions: {n_external} | "
                f"fallback descriptions: {n_fallback}"
            )

        time.sleep(SLEEP_SECONDS)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    n_external = out["api_source"].fillna("").astype(str).str.contains(
        "Open Library|Google Books", regex=True
    ).sum()

    n_fallback = (out["api_source"] == "Local metadata fallback").sum()

    print(f"Done. Saved descriptions to: {OUTPUT_PATH}")
    print(f"External descriptions: {n_external} / {len(out)}")
    print(f"Fallback descriptions: {n_fallback} / {len(out)}")


if __name__ == "__main__":
    main()