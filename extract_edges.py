import os
import pickle
import pandas as pd

FOLDER = "data/by_year"

rows = []
for filename in os.listdir(FOLDER):
    if filename.endswith(".pkl"):
        path = os.path.join(FOLDER, filename)
        with open(path, "rb") as f:
            G = pickle.load(f)
            edge = G[73553]
            print(edge)
        for u, v in G.edges():
            rows.append({"id1": u, "id2": v})

df_edges = pd.DataFrame(rows)
df_edges.to_csv("all_edges.csv", index=False)
