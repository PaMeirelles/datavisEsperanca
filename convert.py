import pandas as pd

# Load the CSV
df = pd.read_csv("all_nodes.csv")

# Merge rare parties
party_counts = df["siglaPartido"].value_counts()
rare_parties = party_counts[party_counts <= 50].index
df["siglaPartido"] = df["siglaPartido"].replace(rare_parties, "others")

# Merge rare occupations
occupation_counts = df["occupation"].value_counts()
rare_occupations = occupation_counts[occupation_counts <= 50].index
df["occupation"] = df["occupation"].replace(rare_occupations, "others")

# Save the cleaned version
df.to_csv("all_nodes_merged.csv", index=False)
