import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import comb
from matplotlib.patches import Patch
from pandas.api.types import CategoricalDtype

# 1) global styling
sns.set_theme(
    style="whitegrid",
    font="Palatino Linotype",
    rc={
        "axes.facecolor": "#f0eade",
        "figure.facecolor": "#f0eade",
        "font.family": "Palatino Linotype",
        # Adjust base font size slightly for better scaling with larger figure
        "font.size": 12,
        "axes.titlesize": 18,  # Base title size
        "axes.labelsize": 16,  # Base label size
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14
    }
)

# 2) load data

nodes = pd.read_csv("all_nodes.csv")
edges = pd.read_csv("all_edges.csv")
# 3) preprocess special features
# 3a) merge small parties → 'others'
pc = nodes["siglaPartido"].value_counts()
nodes["siglaPartido"] = nodes["siglaPartido"].replace(pc[pc < 20].index, "others")
# 3b) education order
edu_order = ["other", "undergraduate", "high school", "elementary school", "no education"]
nodes["education"] = nodes["education"].astype(CategoricalDtype(edu_order, ordered=True))
# 3c) age_group order
# Ensure age_group has valid string format for sorting key
nodes["age_group"] = nodes["age_group"].astype(str)  # Convert to string if not already
valid_age_groups = nodes["age_group"].dropna().unique()
# Filter out any potential non-string or malformed entries if necessary before sorting
age_order = sorted([ag for ag in valid_age_groups if isinstance(ag, str) and "-" in ag],
                   key=lambda x: int(x.split("-")[0]))
nodes["age_group"] = nodes["age_group"].astype(CategoricalDtype(age_order, ordered=True))

# 4) define the eight features and their grid positions
features = [
    "education", "gender", "siglaUf", "siglaPartido",
    "region", "occupation", "ethnicity", "age_group"
]
positions = [
    (0, 0), (0, 1), (0, 2),
    (1, 2),
    (2, 2), (2, 1), (2, 0),
    (1, 0)
]

# 5) compute per-feature expected & actual counts for edges
per_feat = {}
for feat in features:
    dfc = nodes.dropna(subset=[feat])
    ids = set(dfc["id"])
    N = len(dfc)
    # Filter edges where both nodes are in the current feature's valid ID set
    ef = edges[edges["id1"].isin(ids) & edges["id2"].isin(ids)]
    E = len(ef)

    # prepare category order
    if feat == "education":
        cats = edu_order
    elif feat == "age_group":
        cats = age_order
    else:
        # Ensure all categories present in dfc[feat] are considered, even if count is 0 in some cases
        # For value_counts(), ensure it's on the filtered dfc
        cats = dfc[feat].value_counts().index.tolist()
        # If a category might exist in nodes but not in dfc after dropna, this is fine.

    exp_list = []
    act_list = []
    for cat in cats:
        ids_cat = set(dfc[dfc[feat] == cat]["id"])
        # Expected edges involving this category (k_i * E / N)
        # This formula is for edges connected to nodes in category i, not same-category edges.
        # The problem seems to be about same-category edges.
        # For homophily, expected same-category edges for category i: E * (n_i/N) * ((n_i-1)/(N-1)) or approx E * (n_i/N)^2
        # Let's stick to the user's original calculation for exp_k if it was intended for a different purpose.
        # The user's exp_k = (len(ids_cat)/N)*E is the expected number of edge *endpoints* in category k.
        # If an edge has two endpoints in category k, it's counted twice. If one, once.
        # This seems to align with how act_k is calculated (both + 0.5*one where 'one' means one endpoint in cat)

        exp_k = (len(ids_cat) / N) * E if N > 0 else 0
        exp_list.append(exp_k)

        m1 = ef["id1"].isin(ids_cat)
        m2 = ef["id2"].isin(ids_cat)
        both = (m1 & m2).sum()  # Edges with both nodes in category
        one = (m1 ^ m2).sum()  # Edges with exactly one node in category

        # act_k: if an edge is fully within ids_cat, it contributes 1 to the "actual" count for that category.
        # If an edge has one node in ids_cat and one outside, it contributes 0.5.
        # This means act_k is the sum of "within-category" edges + 0.5 * "boundary-crossing" edges for that category.
        act_k = both + 0.5 * one
        act_list.append(act_k)

    per_feat[feat] = {
        "cats": cats,
        "expected": exp_list,
        "actual": act_list
    }

# 6) compute the central summary (actual% vs expected%)
records = []
# Define label_map earlier to be used for titles as well
label_map = {"siglaUf": "State (UF)", "siglaPartido": "Party", "occupation": "Occupation",
             "education": "Education", "age_group": "Age Group", "ethnicity": "Ethnicity",
             "region": "Region", "gender": "Gender"}

for feat in features:
    cats = per_feat[feat]["cats"]
    # exp = per_feat[feat]["expected"] # This 'expected' is per category, not total for same-class
    # act = per_feat[feat]["actual"]   # This 'actual' is per category

    dfc = nodes.dropna(subset=[feat])  # Nodes relevant to this feature
    ids = set(dfc["id"])
    ef = edges[edges["id1"].isin(ids) & edges["id2"].isin(ids)]  # Edges relevant to this feature

    if len(ef) == 0:
        act_pct = 0
    else:
        # Actual same-class edges percentage
        m = (
            ef
            .merge(dfc[["id", feat]], left_on="id1", right_on="id")
            .rename(columns={feat: feat + "_1"}).drop(columns="id")
            .merge(dfc[["id", feat]], left_on="id2", right_on="id")
            .rename(columns={feat: feat + "_2"}).drop(columns="id")
        )
        same_class_edges_count = len(m[m[f"{feat}_1"] == m[f"{feat}_2"]])
        act_pct = (same_class_edges_count / len(ef) * 100) if len(ef) > 0 else 0

    # Expected % same-class edges under random mixing
    if len(dfc) < 2:
        exp_pct = 0
    else:
        # Sum of n_i * (n_i - 1) / 2 for all categories i
        # This is the number of possible same-category pairs
        category_counts = dfc[feat].value_counts()
        expected_same_class_pairs = (category_counts * (category_counts - 1) / 2).sum()
        # Total possible pairs in the network for this feature
        total_possible_pairs = comb(len(dfc), 2, exact=True)  # Using scipy.special.comb

        exp_frac = expected_same_class_pairs / total_possible_pairs if total_possible_pairs > 0 else 0
        exp_pct = exp_frac * 100

    records.append((feat, act_pct, exp_pct))

dfc_summary = pd.DataFrame(records, columns=["feature", "actual_pct", "expected_pct"])
dfc_summary["common"] = dfc_summary[["actual_pct", "expected_pct"]].min(axis=1)
dfc_summary["diff"] = (dfc_summary["actual_pct"] - dfc_summary["expected_pct"]).abs()  # abs is correct
# Color based on actual > expected (homophily) or actual < expected (heterophily)
dfc_summary["color"] = dfc_summary.apply(lambda r: "forestgreen" if r.actual_pct > r.expected_pct else "crimson",
                                         axis=1)
dfc_summary["label"] = dfc_summary["feature"].map(lambda f: label_map.get(f, f))

# 7) plot 3×3 grid
# Significantly increase figure size for projectors
fig, axes = plt.subplots(3, 3, figsize=(30, 24))  # Increased from (15,12)

# --- Configure Center Cell ---
# Set the entire background of the center cell (axes[1,1]) to yellow
center_ax_row, center_ax_col = 1, 1
axes[center_ax_row, center_ax_col].set_facecolor('#FFFFE0')  # LightYellow, more distinct
axes[center_ax_row, center_ax_col].axis("on")  # Ensure it's on if we are plotting on it

# Peripheral plots
title_fontsize = 22
label_fontsize = 16
tick_fontsize = 14
xtick_rotation = 45

for feat, (r, c) in zip(features, positions):
    # Skip if this is the center plot's designated position, as it's handled separately
    if r == center_ax_row and c == center_ax_col:
        continue

    ax = axes[r, c]
    data = per_feat[feat]
    plot_cats = data["cats"]  # Categories for plotting

    # Ensure actual and expected lists are of the same length as plot_cats
    # This can happen if some categories have 0 nodes after filtering or were not in value_counts()
    # For simplicity, we assume data["actual"] and data["expected"] align with data["cats"]
    # If not, more complex alignment would be needed.

    actual_values = data["actual"]
    expected_values = data["expected"]

    # Defensive check for length mismatch (should ideally not happen with current logic)
    if len(actual_values) != len(plot_cats) or len(expected_values) != len(plot_cats):
        print(f"Warning: Mismatch in lengths for feature '{feat}'. Skipping its peripheral plot.")
        ax.set_title(f"{label_map.get(feat, feat)}\n(Data Error)", fontsize=title_fontsize, color='red')
        ax.axis("off")
        continue

    bottom = [min(a, e) for a, e in zip(actual_values, expected_values)]
    diff_values = [abs(a - e) for a, e in zip(actual_values, expected_values)]
    colors = ["green" if a > e else "red" for a, e in zip(actual_values, expected_values)]

    ax.bar(plot_cats, bottom, color="blue")  # Common part
    for i, cat_label in enumerate(plot_cats):
        ax.bar(cat_label, diff_values[i], bottom=bottom[i], color=colors[i])  # Difference part

    ax.set_title(label_map.get(feat, feat), fontsize=title_fontsize, wrap=True)
    ax.set_xticklabels(plot_cats, rotation=xtick_rotation, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_ylabel("Edge Count (or related metric)", fontsize=label_fontsize)  # Clarify Y-axis

# --- Center Summary Plot ---
cax = axes[center_ax_row, center_ax_col]  # This is already axes[1,1]
# cax.set_facecolor('#FFFFE0') # Already set above for the entire cell

x_indices = range(len(dfc_summary))
cax.bar(x_indices, dfc_summary["common"], color="blue", label="Overlap")  # Changed label from "common"
for i, row in dfc_summary.iterrows():
    cax.bar(i, row["diff"], bottom=row["common"], color=row["color"])

cax.set_xticks(x_indices)
cax.set_xticklabels(dfc_summary["label"], rotation=xtick_rotation, ha="right",
                    fontsize=tick_fontsize + 2)  # Slightly larger for center
cax.set_ylabel("% Same-Attribute Edges", fontsize=label_fontsize + 2)
cax.set_title("Homophily: Actual vs. Expected Same-Attribute Edges", fontsize=title_fontsize + 2, wrap=True)
cax.legend(handles=[
    Patch(facecolor="blue", label="Min(Actual, Expected) %"),
    Patch(facecolor="green", label="Actual > Expected % (Homophily)"),
    Patch(facecolor="red", label="Actual < Expected % (Heterophily)")
], loc="upper right", fontsize=tick_fontsize)  # Adjusted legend labels for clarity

# Remove any unused subplots (if any, though 3x3 grid is full if center is used)
all_subplot_indices = set((r, c) for r in range(3) for c in range(3))
used_subplot_indices = set(positions)
used_subplot_indices.add((center_ax_row, center_ax_col))  # Add center plot explicitly

for r_idx in range(3):
    for c_idx in range(3):
        if (r_idx, c_idx) not in used_subplot_indices:
            axes[r_idx, c_idx].axis("off")

plt.tight_layout(pad=2.0)  # Add some padding
plt.tight_layout(pad=2.0)
plt.savefig("dataVisEsperanca.png", dpi=300, bbox_inches='tight')
plt.savefig("dataVisEsperanca.svg")  # or .pdf, .eps
plt.show()