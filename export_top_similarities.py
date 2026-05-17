# Export top similarities

# Goal of this script:
# This script converts large full similarity matrices stored as .npy files
# into much lighter CSV files containing only the top N most similar items
# for each book.

# This is useful because the full similarity matrices can be very large,
# especially for deployment on Streamlit Cloud. Instead of loading the full
# item-item and content similarity matrices, the UI can load smaller CSV files
# that contain only the most relevant neighbors for each item.

# Inputs:
# - NPYs/UI-item_similarity.npy
# - NPYs/UI-content_similarity.npy

# Outputs:
# - kaggle_data/top_item_similarities.csv
# - kaggle_data/top_content_similarities.csv

# Each output CSV contains:
# - item_id: the reference book/item
# - similar_items: the top N most similar item IDs
# - similar_scores: the corresponding similarity scores


# os is used to build file paths in a clean and portable way.
import os

# numpy is used to load and process the similarity matrices.
# The similarity matrices are stored as .npy files.
import numpy as np

# pandas is used to create and save the final CSV files.
import pandas as pd


# Input paths

# Path to the full item-item collaborative filtering similarity matrix.
# This matrix was computed offline and saved as a .npy file.

# Shape:
# number_of_items x number_of_items

# Each value sim[i, j] represents how similar item i is to item j
# based on collaborative filtering signals.
ITEM_SIM_PATH = "NPYs/UI-item_similarity.npy"

# Path to the full content-based similarity matrix.
# This matrix was computed offline from item metadata, such as titles,
# authors, and subjects.

# Shape:
# number_of_items x number_of_items

# Each value sim[i, j] represents how similar item i is to item j
# based on content/metadata similarity.
CONTENT_SIM_PATH = "NPYs/UI-content_similarity.npy"


# Output paths

# Folder where the lightweight CSV files will be saved.
OUTPUT_DIR = "kaggle_data"

# Output CSV for the top item-item similarities.
ITEM_OUTPUT = os.path.join(OUTPUT_DIR, "top_item_similarities.csv")

# Output CSV for the top content-based similarities.
CONTENT_OUTPUT = os.path.join(OUTPUT_DIR, "top_content_similarities.csv")


# Number of neighbors to keep

# TOP_N controls how many similar books are kept for each item.
#
# Example:
# If TOP_N = 10, then for every book, the script saves only its 10 most
# similar books instead of saving similarities to all 15,000+ books.
#
# This massively reduces file size and memory usage.
TOP_N = 10


# Function to export top similarities

def export_top_similarities(sim_path, output_path, top_n=10):
    """
    Convert a full similarity matrix into a lightweight top-N similarity CSV.

    Parameters
    ----------
    sim_path : str
        Path to the full similarity matrix stored as a .npy file.

    output_path : str
        Path where the output CSV file should be saved.

    top_n : int
        Number of most similar items to keep for each item.

    What this function does
    -----------------------
    For each item:
    1. Load its similarity scores to all other items.
    2. Exclude the item itself, because an item is always maximally similar
       to itself and should not be recommended as its own neighbor.
    3. Select the top_n most similar item IDs.
    4. Store their IDs and scores as space-separated strings.
    5. Save everything into a CSV file.

    This makes the Streamlit app lighter because it no longer needs to load
    the full dense similarity matrix.
    """

    # Load the similarity matrix from the .npy file.
    
    # mmap_mode="r" means memory-mapped read-only mode.
    # This avoids loading the entire matrix fully into RAM at once.
    # It is useful when the matrix is large.
    sim = np.load(sim_path, mmap_mode="r")

    # Get the number of items from the first dimension of the matrix.
    # Since the matrix is square, sim.shape[0] = sim.shape[1].
    n_items = sim.shape[0]

    # This list will store one dictionary per item.
    # Each dictionary will later become one row in the output CSV.
    rows = []

    # Loop over every item in the similarity matrix.
    for item_id in range(n_items):

        # Extract all similarity scores for the current item.
        
        # sim[item_id] gives the similarity between the current item and
        # every other item.
        
        # np.array(..., dtype=np.float32) creates a writable copy and stores
        # values in float32 format to reduce memory usage.
        scores = np.array(sim[item_id], dtype=np.float32)

        # Exclude the item itself from its list of similar items.
        
        # The similarity of an item with itself is usually the highest score,
        # but recommending the same item would not be useful.
        
        # Setting it to -infinity guarantees it will not appear in the top N.
        scores[item_id] = -np.inf

        # Get the indices of the top_n highest similarity scores.
        
        # np.argsort(scores) sorts indices from lowest to highest score.
        # [-top_n:] keeps the indices of the top_n largest scores.
        # [::-1] reverses them so they are ordered from highest to lowest.
        top_ids = np.argsort(scores)[-top_n:][::-1]

        # Retrieve the similarity scores corresponding to the selected top IDs.
        top_scores = scores[top_ids]

        # Add one row to the output list.
        
        # similar_items is stored as a space-separated string of item IDs.
        # Example: "12 45 982 301"
        
        # similar_scores is stored as a space-separated string of scores.
        # Example: "0.842100 0.790233 0.755421"
        
        # This format is compact and easy to parse later in the Streamlit UI.
        rows.append({
            "item_id": item_id,
            "similar_items": " ".join(map(str, top_ids)),
            "similar_scores": " ".join(f"{s:.6f}" for s in top_scores)
        })

        # Print progress every 500 items.
        #
        # This is helpful because processing all items can take some time,
        # especially when the similarity matrix is large.
        if item_id % 500 == 0:
            print(f"Processed {item_id}/{n_items}")

    # Convert the list of dictionaries into a pandas DataFrame
    # and save it as a CSV file.
    pd.DataFrame(rows).to_csv(output_path, index=False)

    # Print a confirmation message when the file is saved.
    print(f"Saved: {output_path}")


# Export item-item top similarities

# This creates:

# kaggle_data/top_item_similarities.csv

# It contains, for each book, the top TOP_N books that are most similar
# according to the item-item collaborative filtering similarity matrix.
export_top_similarities(ITEM_SIM_PATH, ITEM_OUTPUT, TOP_N)


# Export content-based top similarities

# This creates:

# kaggle_data/top_content_similarities.csv

# It contains, for each book, the top TOP_N books that are most similar
# according to the content-based similarity matrix.
export_top_similarities(CONTENT_SIM_PATH, CONTENT_OUTPUT, TOP_N)