# download_descriptions.py
# Robust description downloader for BookMatch AI

# It does NOT download covers.
# It creates only:
# kaggle_data/item_descriptions.csv

# Goal of this script:
# This script enriches the book catalog with textual descriptions.
# It does not download images or covers.
#
# For each book in kaggle_data/items.csv, the script tries to find a description
# using several sources:
#
# 1. Open Library by ISBN
# 2. Open Library by title and author search
# 3. Google Books by ISBN, title, and author
# 4. Local metadata fallback if no external description is found

# The final output is a CSV file:

# kaggle_data/item_descriptions.csv

# This output is later used in the Streamlit UI to display expandable
# descriptions for each book card.



# os is imported for potential path or file operations.
# In this script, it is not heavily used, but it is useful to keep
# if future versions need to check files or folders.
import os

# re is used for regular expressions.
# Here, it helps clean titles, authors, ISBNs, and HTML/text patterns.
import re

# time is used to pause between API requests.
# This avoids sending too many requests too quickly to external APIs.
import time

# html is used to decode HTML entities.
# For example, "&amp;" becomes "&".
import html

# requests is used to send HTTP requests to Open Library and Google Books.
import requests

# pandas is used to load the input CSV, manipulate rows, and save the output CSV.
import pandas as pd

# quote_plus is imported for URL encoding.
# In the current version of the script, it is not directly used,
# but it can be useful if query strings are manually built later.
from urllib.parse import quote_plus



# Paths and configuration


# Path to the original item metadata file.
# This file contains the catalog of books, including title, author, ISBN,
# publisher, subjects, and item ID.
INPUT_PATH = "kaggle_data/items.csv"

# Path where the final description CSV will be saved.
# This is the only output created by this script.
OUTPUT_PATH = "kaggle_data/item_descriptions.csv"

# Save progress every 25 processed books.
# This is useful because the full script can take a long time.
# If the script crashes or is interrupted, at least partial progress is saved.
SAVE_EVERY = 25

# Number of seconds to wait between API calls.
# This reduces the risk of overloading external APIs or being rate-limited.
SLEEP_SECONDS = 0.20

# Optional limit for testing.
# If MAX_ITEMS = 100, only the first 100 books are processed.
# If MAX_ITEMS = None, the full dataset is processed.
MAX_ITEMS = None  # set to 100 for testing, then None for full run


# Custom HTTP headers sent with API requests.
# The User-Agent identifies the script as a student project.
# This is cleaner than sending anonymous/default Python requests.
HEADERS = {
    "User-Agent": "BookMatchAI-EPFL-StudentProject/1.0"
}



# Cleaning helpers


def clean_text(value):
    """
    Clean a text field.

    This function is used for titles, authors, publishers, subjects,
    and descriptions.

    It does several things:
    - converts missing values to an empty string,
    - converts the input to string,
    - decodes HTML entities,
    - removes HTML tags,
    - normalizes repeated spaces,
    - strips spaces at the beginning and end.

    Example:
    "<p>Hello&nbsp;world</p>" becomes "Hello world".
    """

    # If the value is missing, return an empty string.
    if pd.isna(value):
        return ""

    # Convert the value to a string.
    value = str(value)

    # Decode HTML entities.
    # Example: "&amp;" becomes "&".
    value = html.unescape(value)

    # Remove HTML tags.
    # Example: "<p>Description</p>" becomes " Description ".
    value = re.sub(r"<[^>]+>", " ", value)

    # Replace multiple whitespace characters by a single space.
    # This removes messy formatting, line breaks, tabs, etc.
    value = re.sub(r"\s+", " ", value).strip()

    # Return the cleaned text.
    return value


def clean_title_for_query(title):
    """
    Clean a book title before using it in an API search query.

    Catalog titles often contain extra punctuation, slashes, subtitles,
    or bracketed information. These elements can reduce the quality of
    API search results.

    The goal is to keep a simpler and cleaner title that is easier
    for Open Library or Google Books to match.
    """

    # First apply the general text cleaning function.
    title = clean_text(title)

    # Dataset titles often contain a trailing slash or catalog information.
    # Example: "Some title / author name" becomes "Some title".
    title = title.split("/")[0].strip()

    # Remove punctuation characters often found at the end of catalog titles.
    title = title.strip(" .,:;-/")

    # Remove bracketed catalog noise.
    # Example: "Book title [text]" becomes "Book title".
    title = re.sub(r"\[[^\]]*\]", "", title).strip()

    # If the title is very long and contains a colon, keep only the main title.
    # This can improve search results because long subtitles may confuse APIs.
    if ":" in title and len(title) > 55:
        title = title.split(":")[0].strip()

    # Return the cleaned query title.
    return title


def clean_author_for_query(author):
    """
    Clean an author name before using it in an API search query.

    The dataset may contain several authors, birth years, slashes,
    semicolons, or catalog suffixes. This function keeps a simpler author
    name to improve matching with external APIs.
    """

    # Apply general text cleaning.
    author = clean_text(author)

    # If no author is available, return an empty string.
    if not author:
        return ""

    # Keep only the first listed author if several are separated by semicolons.
    author = author.split(";")[0].strip()

    # Remove any part after a slash.
    # Example: "Author Name / editor" becomes "Author Name".
    author = author.split("/")[0].strip()

    # Remove birth years or year-like catalog suffixes.
    # Example: "Cicurel, Francine, 1947-" becomes "Cicurel, Francine".
    author = re.sub(r"\b\d{4}-?\b", "", author).strip(" ,;-")

    # Return the cleaned author.
    return author


def extract_best_isbn(row):
    """
    Extract the best ISBN from a row of the items dataframe.

    The original ISBN field can contain messy values, separators, or several
    ISBN candidates. This function tries to extract a valid ISBN-like string.

    It prefers ISBN-13 when available because it is more standard in modern
    book databases.
    """

    # If the ISBN column does not exist, or the value is missing,
    # return an empty string.
    if "ISBN Valid" not in row or pd.isna(row["ISBN Valid"]):
        return ""

    # Convert the ISBN field to string.
    text = str(row["ISBN Valid"])

    # Remove common separators.
    text = text.replace("-", "").replace(" ", "")

    # Search for ISBN-like patterns.
    # This pattern accepts:
    # - ISBN-10-like strings,
    # - ISBN-13-like strings starting with 978 or 979,
    # - possible final X/x for ISBN-10.
    matches = re.findall(r"(?:97[89])?\d{9}[\dXx]", text)

    # If no ISBN-like pattern is found, return an empty string.
    if not matches:
        return ""

    # Prefer ISBN-13 if available.
    isbn13 = [x for x in matches if len(x) == 13]

    # Return the first ISBN-13 if one exists.
    if isbn13:
        return isbn13[0]

    # Otherwise, return the first ISBN-like match.
    return matches[0]


def normalize_description(value):
    """
    Normalize a description returned by an API.

    Open Library sometimes returns descriptions in different formats:
    - directly as a string,
    - as a dictionary like {"type": "...", "value": "..."}.

    This function extracts the actual text, cleans it, and removes
    descriptions that are too short to be useful.
    """

    # If the API returned a dictionary, extract the "value" field.
    if isinstance(value, dict):
        value = value.get("value", "")

    # Clean the description text.
    value = clean_text(value)

    # Avoid useless descriptions.
    # Very short descriptions are often not meaningful.
    if len(value) < 40:
        return ""

    # Return the normalized description.
    return value


def make_fallback_description(row):
    """
    Generate a fallback description from local metadata.

    This is used when no external description is found from Open Library
    or Google Books.

    It is not a true API book summary. Instead, it builds a simple description
    using the local metadata already available in items.csv:
    - title,
    - author,
    - publisher,
    - subjects.

    This avoids empty description cards in the UI.
    """

    # Extract and clean local metadata fields.
    title = clean_text(row.get("Title", "this book"))
    author = clean_text(row.get("Author", ""))
    publisher = clean_text(row.get("Publisher", ""))
    subjects = clean_text(row.get("Subjects", ""))

    # Initialize an empty subject list.
    subject_list = []

    # If subjects exist, split them on semicolons.
    # Keep only the first five subjects to avoid overly long fallback text.
    if subjects:
        subject_list = [s.strip() for s in subjects.split(";") if s.strip()]
        subject_list = subject_list[:5]

    # The fallback description is built progressively in this list.
    parts = []

    # Add title and author information when available.
    if title and author:
        parts.append(f"{title} is a book by {author}.")
    elif title:
        parts.append(f"{title} is a book from the catalog.")

    # Add subject/theme information when available.
    if subject_list:
        parts.append("It is associated with the following themes: " + ", ".join(subject_list) + ".")

    # Add publisher information when available.
    if publisher:
        parts.append(f"The book is published by {publisher}.")

    # If no useful local metadata exists, return a generic fallback.
    if not parts:
        return "No external description was found for this book, but it is part of the BookMatch catalog."

    # Join all parts into one fallback description.
    return " ".join(parts)



# Open Library helpers


def openlibrary_work_description(work_key):
    """
    Retrieve a description from an Open Library work page.

    Open Library distinguishes between:
    - editions: specific published versions of a book,
    - works: the abstract intellectual work behind editions.

    Sometimes the edition page does not contain a description, but the work
    page does. This function queries the work-level JSON endpoint.
    """

    # If no work key exists, no request can be made.
    if not work_key:
        return ""

    # Ensure the work key has the correct Open Library format.
    # Example: "OL123W" becomes "/works/OL123W".
    if not work_key.startswith("/works/"):
        work_key = "/works/" + work_key.replace("/works/", "")

    # Build the Open Library work URL.
    url = f"https://openlibrary.org{work_key}.json"

    try:
        # Send the request to Open Library.
        r = requests.get(url, headers=HEADERS, timeout=10)

        # If the request failed, return an empty string.
        if r.status_code != 200:
            return ""

        # Parse the JSON response.
        data = r.json()

        # Extract and normalize the description field.
        return normalize_description(data.get("description", ""))

    # If anything goes wrong, return an empty string.
    # This keeps the script robust and prevents one failed API call
    # from stopping the full process.
    except Exception:
        return ""


def openlibrary_by_isbn(isbn):
    """
    Try to retrieve a book description from Open Library using ISBN.

    This is the first source used because ISBN lookup is usually more precise
    than title/author search.

    The function first tries to find a direct edition description.
    If none exists, it checks whether the edition is linked to a work page
    and then tries to retrieve the work-level description.
    """

    # If no ISBN is available, this method cannot be used.
    if not isbn:
        return None

    # Build the Open Library ISBN endpoint.
    url = f"https://openlibrary.org/isbn/{isbn}.json"

    try:
        # Query the ISBN endpoint.
        r = requests.get(url, headers=HEADERS, timeout=10)

        # If the request failed, return None.
        if r.status_code != 200:
            return None

        # Parse the JSON response.
        data = r.json()

        # First attempt: direct edition description

        # Some Open Library edition pages contain a direct description.
        desc = normalize_description(data.get("description", ""))

        # If a valid description is found, return it with metadata.
        if desc:
            return {
                "description": desc,
                "api_title": clean_text(data.get("title", "")),
                "api_authors": "",
                "api_source": "Open Library ISBN",
                "api_query": isbn
            }

        # Second attempt: work-level description

        # Some editions are linked to a work entry.
        works = data.get("works", [])

        # If a work link exists, retrieve the description from that work.
        if works:
            work_key = works[0].get("key", "")
            desc = openlibrary_work_description(work_key)

            # If a valid work-level description is found, return it.
            if desc:
                return {
                    "description": desc,
                    "api_title": clean_text(data.get("title", "")),
                    "api_authors": "",
                    "api_source": "Open Library ISBN -> Work",
                    "api_query": isbn
                }

    # If any error occurs, return None.
    except Exception:
        return None

    # Return None if no description was found.
    return None


def openlibrary_by_search(title, author=""):
    """
    Try to retrieve a description from Open Library using title and author.

    This is used when ISBN lookup fails or when no ISBN is available.

    The function:
    1. cleans the title and author,
    2. searches Open Library,
    3. checks the returned works,
    4. retrieves the first valid work-level description.
    """

    # Clean title and author for search.
    title_q = clean_title_for_query(title)
    author_q = clean_author_for_query(author)

    # If there is no usable title, the search cannot be performed.
    if not title_q:
        return None

    # Build search parameters.
    params = {
        "title": title_q,
        "limit": 5
    }

    # Add author to the search parameters if available.
    if author_q:
        params["author"] = author_q

    try:
        # Query Open Library search endpoint.
        r = requests.get(
            "https://openlibrary.org/search.json",
            params=params,
            headers=HEADERS,
            timeout=10
        )

        # If the request failed, return None.
        if r.status_code != 200:
            return None

        # Parse the JSON response.
        data = r.json()

        # Extract search results.
        docs = data.get("docs", [])

        # Loop through the first search results.
        for doc in docs:

            # Each result can have a work key.
            work_key = doc.get("key", "")

            # Try to retrieve the work-level description.
            desc = openlibrary_work_description(work_key)

            # If a valid description is found, return it.
            if desc:
                return {
                    "description": desc,
                    "api_title": clean_text(doc.get("title", "")),
                    "api_authors": ", ".join(doc.get("author_name", [])),
                    "api_source": "Open Library Search -> Work",
                    "api_query": f"title={title_q}; author={author_q}"
                }

    # If anything fails, return None.
    except Exception:
        return None

    # Return None if no description was found.
    return None



# Google Books helpers


def google_books_search(title, author="", isbn=""):
    """
    Try to retrieve a book description from the Google Books API.

    This is used after Open Library attempts.

    The function builds several possible queries:
    1. ISBN-based query, if ISBN exists,
    2. title + author query,
    3. title-only query.

    It returns the first valid description found.
    """

    # Clean title and author before querying.
    title_q = clean_title_for_query(title)
    author_q = clean_author_for_query(author)

    # Store candidate queries.
    queries = []

    # First try ISBN because it is usually the most precise.
    if isbn:
        queries.append(f"isbn:{isbn}")

    # Then try title + author if both are available.
    if title_q and author_q:
        queries.append(f"{title_q} {author_q}")

    # Finally try title only.
    if title_q:
        queries.append(title_q)

    # Try each query in order.
    for q in queries:
        try:
            # Query the Google Books volumes endpoint.
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

            # If the request failed, try the next query.
            if r.status_code != 200:
                continue

            # Parse the JSON response.
            data = r.json()

            # Extract result list.
            results = data.get("items", [])

            # Loop through returned books.
            for result in results:

                # Google Books stores metadata inside "volumeInfo".
                info = result.get("volumeInfo", {})

                # Extract and normalize the description.
                desc = normalize_description(info.get("description", ""))

                # If a valid description is found, return it.
                if desc:
                    return {
                        "description": desc,
                        "api_title": clean_text(info.get("title", "")),
                        "api_authors": ", ".join(info.get("authors", [])),
                        "api_source": "Google Books",
                        "api_query": q
                    }

        # If one query fails, continue to the next query.
        except Exception:
            continue

    # Return None if Google Books did not provide a usable description.
    return None



# Main description retrieval


def find_description(row):
    """
    Find the best available description for one book.

    This function coordinates all retrieval methods.

    Search order:
    1. Open Library by ISBN
    2. Open Library by title/author search
    3. Google Books
    4. Local metadata fallback

    The fallback ensures that every book gets at least some description text,
    even if no external API has a real summary.
    """

    # Extract basic metadata from the row.
    title = clean_text(row.get("Title", ""))
    author = clean_text(row.get("Author", ""))

    # Extract best available ISBN.
    isbn = extract_best_isbn(row)

    # Define the external sources to try, in order.
    # lambdas are used so that the functions are called one by one inside
    # the loop, instead of all being called immediately.
    sources = [
        lambda: openlibrary_by_isbn(isbn),
        lambda: openlibrary_by_search(title, author),
        lambda: google_books_search(title, author, isbn),
    ]

    # Try each external source.
    for source in sources:

        # Run the current source function.
        result = source()

        # If the source returns a non-empty description, keep it.
        if result and result.get("description"):
            return result

        # Pause between attempts.
        time.sleep(SLEEP_SECONDS)

    # If no external source worked, create a fallback description
    # from local metadata.
    return {
        "description": make_fallback_description(row),
        "api_title": "",
        "api_authors": "",
        "api_source": "Local metadata fallback",
        "api_query": ""
    }


def main():
    """
    Main execution function.

    It:
    1. loads the item catalog,
    2. optionally limits the number of rows for testing,
    3. loops over all books,
    4. retrieves or generates descriptions,
    5. saves progress every SAVE_EVERY rows,
    6. saves the final CSV.
    """

    # Load the original items.csv file.
    items = pd.read_csv(INPUT_PATH)

    # If MAX_ITEMS is set, keep only the first MAX_ITEMS rows.
    # Useful for testing before running the full dataset.
    if MAX_ITEMS is not None:
        items = items.head(MAX_ITEMS)

    # This list will store the output rows before saving them as a CSV.
    rows = []

    # Loop over all books in the item catalog.
    for idx, row in items.iterrows():

        # Use the item ID from the "i" column if available.
        # Otherwise, fall back to the row index.
        item_id = int(row["i"]) if "i" in items.columns else int(idx)

        # Retrieve the best available description for this book.
        result = find_description(row)

        # Store the result in the output format.
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

        # Save progress regularly

        # Every SAVE_EVERY rows, save a temporary version of the output.
        # This prevents losing all progress if the script stops unexpectedly.
        if len(rows) % SAVE_EVERY == 0:

            # Convert current rows into a DataFrame.
            out = pd.DataFrame(rows)

            # Save current progress to CSV.
            out.to_csv(OUTPUT_PATH, index=False)

            # Count how many descriptions came from external APIs.
            n_external = out["api_source"].fillna("").astype(str).str.contains(
                "Open Library|Google Books", regex=True
            ).sum()

            # Count how many descriptions came from local fallback.
            n_fallback = (out["api_source"] == "Local metadata fallback").sum()

            # Print progress information.
            print(
                f"Saved progress: {len(rows)} books processed | "
                f"external descriptions: {n_external} | "
                f"fallback descriptions: {n_fallback}"
            )

        # Pause between books to avoid calling APIs too aggressively.
        time.sleep(SLEEP_SECONDS)

    
    # Final save
    

    # Convert all rows to a DataFrame.
    out = pd.DataFrame(rows)

    # Save the final complete output CSV.
    out.to_csv(OUTPUT_PATH, index=False)

    # Count final number of descriptions from external APIs.
    n_external = out["api_source"].fillna("").astype(str).str.contains(
        "Open Library|Google Books", regex=True
    ).sum()

    # Count final number of fallback descriptions.
    n_fallback = (out["api_source"] == "Local metadata fallback").sum()

    # Print final summary.
    print(f"Done. Saved descriptions to: {OUTPUT_PATH}")
    print(f"External descriptions: {n_external} / {len(out)}")
    print(f"Fallback descriptions: {n_fallback} / {len(out)}")



# Script entry point


# This ensures that main() runs only when this file is executed directly.
# It would not run automatically if the file were imported as a module.
if __name__ == "__main__":
    main()