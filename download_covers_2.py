# download_covers_2.py
# Download more book covers using ISBN + title/author fallback

# Goal of this script:
# This script enriches the original items.csv file by adding book cover images.
# It first tries to download covers using ISBN numbers, which is usually the
# most precise method. If no cover is found with the ISBN, it then falls back
# to a title/author search using the Open Library API.

# Main outputs:
# - downloaded cover images saved in kaggle_data/covers/
# - enriched CSV saved as kaggle_data/items_with_covers.csv

# The enriched CSV contains the original item metadata plus:
# - clean_isbn: cleaned ISBN used for cover search
# - cover_path: local path to the downloaded cover image
# - cover_source: method used to find the cover


# os is used to work with folders and file paths.
# For example, it checks whether a cover image already exists locally.
import os

# re is used for regular expressions.
# Here, it is mainly used to clean ISBN strings and split ISBN fields.
import re

# time is used to add a small pause between API requests.
# This avoids sending too many requests too quickly to Open Library.
import time

# requests is used to make HTTP requests to Open Library.
# It allows the script to download cover images and query the search API.
import requests

# pandas is used to load, manipulate, and save the CSV files.
import pandas as pd

# quote_plus is used to safely encode titles and authors into URL query format.
# For example, "Harry Potter" becomes "Harry+Potter".
from urllib.parse import quote_plus


# File paths

# Path to the original item metadata file.
# This file contains the catalog of books, including title, author, ISBN, etc.
ITEMS_PATH = "kaggle_data/items.csv"

# Folder where downloaded cover images will be saved.
OUTPUT_DIR = "kaggle_data/covers"

# Path of the enriched CSV file that will be created by this script.
# This file will contain all original book metadata plus cover information.
OUTPUT_CSV = "kaggle_data/items_with_covers.csv"


# Create the output directory if it does not already exist.
# exist_ok=True means that no error is raised if the folder already exists.
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ISBN cleaning function

def clean_isbn(isbn):
    """
    Clean a single ISBN string.

    The goal is to transform a messy ISBN value into a standardized ISBN-10
    or ISBN-13 format.

    Example:
    "978-2-1234-5678-9" becomes "9782123456789".

    If the value cannot be converted into a valid ISBN length, the function
    returns None.
    """

    # If the ISBN value is missing, return None immediately.
    if pd.isna(isbn):
        return None

    # Convert the ISBN to string and remove spaces at the beginning/end.
    isbn = str(isbn).strip()

    # Remove common separators.
    # ISBNs often contain hyphens or spaces, but they are not needed for lookup.
    isbn = isbn.replace("-", "").replace(" ", "")

    # Remove every character that is not a digit or X/x.
    # X can appear as the last character of some ISBN-10 numbers.
    isbn = re.sub(r"[^0-9Xx]", "", isbn)

    # A valid ISBN should have either 10 or 13 characters.
    # If the cleaned string has the correct length, return it.
    if len(isbn) in [10, 13]:
        return isbn

    # Otherwise, the ISBN is considered invalid or unusable.
    return None


# Best ISBN extraction function

def extract_best_isbn(isbn_field):
    """
    Extract the best usable ISBN from a field that may contain several ISBNs.

    Some books may have several ISBNs in the same cell, separated by spaces,
    semicolons, commas, or other separators.

    This function:
    1. splits the field into possible ISBN candidates,
    2. cleans each candidate,
    3. keeps valid ISBNs only,
    4. prefers ISBN-13 over ISBN-10 when available.

    ISBN-13 is preferred because it is usually more standard for modern book
    databases and often gives better lookup results.
    """

    # If the full ISBN field is missing, no ISBN can be extracted.
    if pd.isna(isbn_field):
        return None

    # Split the ISBN field into possible individual ISBN values.
    # The field may contain separators such as ; , | or spaces.
    raw_isbns = re.split(r"[;,| ]+", str(isbn_field))

    # Clean each candidate ISBN.
    cleaned = [clean_isbn(x) for x in raw_isbns]

    # Keep only valid cleaned ISBNs.
    cleaned = [x for x in cleaned if x is not None]

    # If no valid ISBN remains, return None.
    if not cleaned:
        return None

    # Prefer ISBN-13 if available.
    isbn13 = [x for x in cleaned if len(x) == 13]

    # If at least one ISBN-13 exists, return the first one.
    if isbn13:
        return isbn13[0]

    # Otherwise, return the first valid ISBN, likely ISBN-10.
    return cleaned[0]


# Metadata column detection

def get_title_column(items):
    """
    Detect the column containing book titles.

    Different datasets can use different names for the title column.
    This function checks several common alternatives and returns the first
    one found.

    If no title column is found, it returns None.
    """

    # Try different possible title column names.
    for col in ["Title", "title", "Book-Title", "book_title", "name"]:
        if col in items.columns:
            return col

    # If none of the expected columns exists, return None.
    return None


def get_author_column(items):
    """
    Detect the column containing book authors.

    The author column can also have different names depending on the dataset.
    This function checks common alternatives and returns the first one found.

    If no author column is found, it returns None.
    """

    # Try different possible author column names.
    for col in ["Author", "author", "authors", "Book-Author"]:
        if col in items.columns:
            return col

    # If no author column exists, return None.
    return None


# Image download helper

def save_image_from_url(url, output_path):
    """
    Download an image from a URL and save it locally.

    This function is used by both cover download methods:
    - ISBN-based cover download
    - cover-ID-based cover download

    The function checks:
    1. whether the HTTP request is successful,
    2. whether the returned content is actually an image,
    3. whether the file is large enough to be a real cover image.

    If everything is valid, the image is saved and the local path is returned.
    Otherwise, None is returned.
    """

    try:
        # Send a GET request to the provided URL.
        # timeout=15 prevents the script from waiting forever if the server
        # does not respond.
        r = requests.get(url, timeout=15)

        # Check that:
        # - status_code == 200 means the request succeeded,
        # - Content-Type contains "image", meaning the response is an image.
        if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):

            # Ignore very small files.
            # Open Library may sometimes return tiny placeholder or invalid files.
            # A real cover should generally be larger than 1000 bytes.
            if len(r.content) > 1000:

                # Save the image in binary mode.
                with open(output_path, "wb") as f:
                    f.write(r.content)

                # Return the saved image path.
                return output_path

    # If there is a network error, timeout, or connection problem,
    # silently ignore it and return None.
    except requests.RequestException:
        pass

    # Return None if the image could not be downloaded or was invalid.
    return None


# ISBN-based cover download

def download_cover_by_isbn(isbn, item_id, size="L"):
    """
    Try to download a book cover using its ISBN.

    Open Library provides a cover API where covers can be retrieved directly
    from ISBN numbers.

    Example URL:
    https://covers.openlibrary.org/b/isbn/978XXXXXXXXXX-L.jpg?default=false

    size="L" requests a large cover image.
    """

    # If no valid ISBN is available, this method cannot be used.
    if isbn is None:
        return None

    # Define the local output path for this book cover.
    # Each cover is saved as item_id.jpg.
    output_path = os.path.join(OUTPUT_DIR, f"{item_id}.jpg")

    # If the image already exists locally, do not download it again.
    # This makes the script resumable.
    if os.path.exists(output_path):
        return output_path

    # Build the Open Library cover URL using the ISBN.
    # default=false prevents Open Library from returning a generic placeholder
    # when no cover exists.
    url = f"https://covers.openlibrary.org/b/isbn/{isbn}-{size}.jpg?default=false"

    # Download and save the image.
    return save_image_from_url(url, output_path)


# Title/author fallback search

def search_cover_id_by_title_author(title, author=None):
    """
    Search Open Library for a book using its title and optionally its author.

    This is used as a fallback when the ISBN method fails.

    Instead of directly downloading the cover, this function searches the
    Open Library catalog and tries to find a document containing a cover ID
    called "cover_i".

    That cover ID can then be used to download the image.
    """

    # If title is missing, the search cannot be performed.
    if pd.isna(title):
        return None

    # Convert title to string and remove surrounding spaces.
    title = str(title).strip()

    # If the title is empty, return None.
    if title == "":
        return None

    # Start building the Open Library search query using the title.
    # quote_plus makes the title safe for use inside a URL.
    query = f"title={quote_plus(title)}"

    # If an author is available, add it to the search query.
    # This usually improves the accuracy of the result.
    if author is not None and pd.notna(author):
        author = str(author).strip()
        if author != "":
            query += f"&author={quote_plus(author)}"

    # Build the full Open Library search URL.
    # limit=3 means we only check the first three results to keep the script fast.
    url = f"https://openlibrary.org/search.json?{query}&limit=3"

    try:
        # Send the search request to Open Library.
        r = requests.get(url, timeout=15)

        # If the request failed, return None.
        if r.status_code != 200:
            return None

        # Convert the JSON response into a Python dictionary.
        data = r.json()

        # Extract the list of matching documents.
        docs = data.get("docs", [])

        # Look through the first returned documents.
        for doc in docs:

            # If the document has a cover_i field, return it.
            # cover_i is the Open Library internal cover ID.
            if "cover_i" in doc:
                return doc["cover_i"]

    # If there is a network-related error, return None.
    except requests.RequestException:
        return None

    # If the response cannot be decoded as JSON, return None.
    except ValueError:
        return None

    # If no cover ID was found, return None.
    return None


# Cover-ID-based cover download

def download_cover_by_cover_id(cover_id, item_id, size="L"):
    """
    Download a cover using an Open Library cover ID.

    This is used after the title/author search finds a valid cover_i value.

    Example URL:
    https://covers.openlibrary.org/b/id/123456-L.jpg?default=false
    """

    # If no cover ID was found, this method cannot be used.
    if cover_id is None:
        return None

    # Define the local output path for this cover.
    output_path = os.path.join(OUTPUT_DIR, f"{item_id}.jpg")

    # If the image already exists, do not redownload it.
    if os.path.exists(output_path):
        return output_path

    # Build the Open Library cover URL using the internal cover ID.
    url = f"https://covers.openlibrary.org/b/id/{cover_id}-{size}.jpg?default=false"

    # Download and save the image.
    return save_image_from_url(url, output_path)


# Load item metadata

# Load the original items.csv file into a pandas DataFrame.
items = pd.read_csv(ITEMS_PATH)

# Automatically detect the title column.
title_col = get_title_column(items)

# Automatically detect the author column.
author_col = get_author_column(items)

# A title column is required for the fallback method.
# If there is no title column, the script cannot work properly.
if title_col is None:
    raise ValueError("No title column found in items.csv")


# Add columns for enriched metadata

# Create a cleaned ISBN column if the original ISBN Valid column exists.
# If ISBN Valid does not exist, the clean_isbn column is set to None.
items["clean_isbn"] = items["ISBN Valid"].apply(extract_best_isbn) if "ISBN Valid" in items.columns else None

# Initialize a column that will store the local path of each downloaded cover.
items["cover_path"] = None

# Initialize a column that will store how the cover was found:
# - existing: already downloaded before
# - isbn: downloaded using ISBN
# - title_author: found using title/author fallback
items["cover_source"] = None


# Main loop over all books

# Iterate over every row of the item metadata.
for idx, row in items.iterrows():

    # Use the item ID from column "i" if it exists.
    # Otherwise, use the row index as item ID.
    item_id = int(row["i"]) if "i" in items.columns else idx

    # Define the expected local path for this book cover.
    output_path = os.path.join(OUTPUT_DIR, f"{item_id}.jpg")

    
    # Step 1: Check if the cover already exists locally
    

    # If the image file already exists, there is no need to download it again.
    # This is useful if the script was interrupted and restarted later.
    if os.path.exists(output_path):
        items.at[idx, "cover_path"] = output_path
        items.at[idx, "cover_source"] = "existing"
        continue

    
    # Step 2: Try downloading the cover using ISBN
    

    # The ISBN method is tried first because it is usually more precise than
    # searching by title and author.
    cover_path = download_cover_by_isbn(row.get("clean_isbn"), item_id)

    # If a cover was found using ISBN, save the result in the DataFrame.
    if cover_path is not None:
        items.at[idx, "cover_path"] = cover_path
        items.at[idx, "cover_source"] = "isbn"

    
    # Step 3: Fallback to title/author search
    

    # If no cover was found using ISBN, try searching Open Library by title
    # and author.
    else:

        # Get the title from the detected title column.
        title = row[title_col]

        # Get the author if an author column exists.
        # If not, author is set to None.
        author = row[author_col] if author_col is not None else None

        # Search Open Library for a cover ID using title and author.
        cover_id = search_cover_id_by_title_author(title, author)

        # If a cover ID is found, try downloading the cover image.
        cover_path = download_cover_by_cover_id(cover_id, item_id)

        # If the fallback download succeeds, save the result in the DataFrame.
        if cover_path is not None:
            items.at[idx, "cover_path"] = cover_path
            items.at[idx, "cover_source"] = "title_author"

    
    # Progress display
    

    # Every 100 rows, print the current progress.
    # This is useful because downloading covers can take a long time.
    if idx % 100 == 0:
        found = items["cover_path"].notna().sum()
        print(f"Processed {idx}/{len(items)} books | covers found so far: {found}")

    # Add a small pause between requests.
    # This avoids overloading the Open Library API.
    time.sleep(0.15)


# Save enriched dataset

# Save the enriched DataFrame to a new CSV file.
# This CSV contains all original metadata plus:
# - clean_isbn
# - cover_path
# - cover_source
items.to_csv(OUTPUT_CSV, index=False)


# Final summary

# Print a final message once the script is complete.
print("Done!")

# Print where the enriched CSV was saved.
print(f"Saved enriched file to: {OUTPUT_CSV}")

# Print where the cover images were saved.
print(f"Covers saved in: {OUTPUT_DIR}")

# Print the total number of covers found.
print(f"Covers found: {items['cover_path'].notna().sum()} / {len(items)}")