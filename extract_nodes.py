import os
import pickle
import pandas as pd
import networkx as nx

# Folder with .pkl files
FOLDER = "data/by_year"

# Store all node data here
rows = []

# Loop through all .pkl files in folder
for filename in os.listdir(FOLDER):
    if filename.endswith(".pkl"):
        filepath = os.path.join(FOLDER, filename)
        with open(filepath, "rb") as f:
            G = pickle.load(f)
            for node_id, attrs in G.nodes(data=True):
                row = {"id": node_id, "file": filename}
                row.update(attrs)
                rows.append(row)

# Convert to DataFrame and save as CSV
df = pd.DataFrame(rows)
df = df.drop_duplicates(subset="id", keep="first")
df.to_csv("all_nodes.csv", index=False)
