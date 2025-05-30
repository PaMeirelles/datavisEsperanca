import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import comb

# 1) styling to match your previous plots
sns.set_theme(
    style="whitegrid",
    font="Palatino Linotype",
    rc={
        "axes.facecolor": "#f0eade",
        "figure.facecolor": "#f0eade",
        "font.family": "Palatino Linotype",
    }
)

# 2) load data
nodes = pd.read_csv("all_nodes.csv")
edges = pd.read_csv("all_edges.csv")

# identify categorical features (drop id and file)
features = [c for c in nodes.columns if c not in ["id", "file"]]

# storage for bar heights and colors
bottom_pcts = []
delta_pcts  = []
top_colors  = []

E_tot = len(edges)  # total number of edges

for col in features:
    # drop nodes missing this feature
    dfc = nodes.dropna(subset=[col])
    N   = len(dfc)

    # expected fraction of same‐category edges under random mixing
    counts        = dfc[col].value_counts()
    same_pairs    = (counts*(counts-1)/2).sum()
    total_pairs   = comb(N, 2)
    p_same        = same_pairs/total_pairs if total_pairs>0 else 0
    expected_pct  = p_same * 100

    # actual fraction of same‐category edges
    e = (edges
         .merge(dfc[["id", col]], left_on="id1", right_on="id")
         .rename(columns={col: f"{col}_1"}).drop(columns="id")
         .merge(dfc[["id", col]], left_on="id2", right_on="id")
         .rename(columns={col: f"{col}_2"}).drop(columns="id")
        )
    actual = len(e[e[f"{col}_1"] == e[f"{col}_2"]])
    actual_pct = actual / E_tot * 100

    # split into common + extra/missing
    base       = min(actual_pct, expected_pct)
    diff       = abs(actual_pct - expected_pct)
    bottom_pcts.append(base)
    delta_pcts.append(diff)
    top_colors.append("green" if actual_pct > expected_pct else "red")

# 3) Plot
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(features))

# bottom (common) bars
ax.bar(x, bottom_pcts, color="blue", label="common")

# top (extra or missing) bars
for xi, (b, d, c) in enumerate(zip(bottom_pcts, delta_pcts, top_colors)):
    ax.bar(xi, d, bottom=b, color=c)

ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha="right")
ax.set_ylabel("Percentage of total edges (%)")
ax.set_title("Actual vs. Expected % of same-category edges per feature")

# legend
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="blue", label="common"),
    Patch(facecolor="green", label="extra"),
    Patch(facecolor="red", label="missing")
]
ax.legend(handles=legend_elems, loc="upper right")

plt.tight_layout()
plt.show()
