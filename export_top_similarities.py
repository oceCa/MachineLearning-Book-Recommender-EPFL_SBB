import os
import numpy as np
import pandas as pd

ITEM_SIM_PATH = "NPYs/UI-item_similarity.npy"
CONTENT_SIM_PATH = "NPYs/UI-content_similarity.npy"

OUTPUT_DIR = "kaggle_data"
ITEM_OUTPUT = os.path.join(OUTPUT_DIR, "top_item_similarities.csv")
CONTENT_OUTPUT = os.path.join(OUTPUT_DIR, "top_content_similarities.csv")

TOP_N = 50


def export_top_similarities(sim_path, output_path, top_n=50):
    sim = np.load(sim_path, mmap_mode="r")
    n_items = sim.shape[0]

    rows = []

    for item_id in range(n_items):
        scores = np.array(sim[item_id], dtype=np.float32)
        scores[item_id] = -np.inf

        top_ids = np.argsort(scores)[-top_n:][::-1]
        top_scores = scores[top_ids]

        rows.append({
            "item_id": item_id,
            "similar_items": " ".join(map(str, top_ids)),
            "similar_scores": " ".join(f"{s:.6f}" for s in top_scores)
        })

        if item_id % 500 == 0:
            print(f"Processed {item_id}/{n_items}")

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


export_top_similarities(ITEM_SIM_PATH, ITEM_OUTPUT, TOP_N)
export_top_similarities(CONTENT_SIM_PATH, CONTENT_OUTPUT, TOP_N)