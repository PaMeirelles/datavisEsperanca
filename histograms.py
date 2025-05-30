import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

sns.set_theme(
    style="whitegrid",
    font="Palatino Linotype",        # pick one
    rc={
        "axes.facecolor": "#f0eade",
        "figure.facecolor": "#f0eade",
        "font.family": "Palatino Linotype",
    }
)

df = pd.read_csv("all_nodes.csv")
df = df.drop(columns=["id", "file"])

edu_order = ["other", "undergraduate", "high school", "elementary school", "no education"]
df["education"] = df["education"].astype(CategoricalDtype(edu_order, ordered=True))

age_order = sorted(df["age_group"].dropna().unique(), key=lambda x: int(x.split("-")[0]))
df["age_group"] = df["age_group"].astype(CategoricalDtype(age_order, ordered=True))

for col in df.columns:
    data = df[col].dropna()
    plt.figure(figsize=(8, 6))
    if col == "education":
        order = edu_order
    elif col in ["siglaUf", "siglaPartido", "occupation"]:
        order = data.value_counts().index
    elif col == "age_group":
        order = age_order
    else:
        order = None

    sns.countplot(
        x=data,
        order=order,
        color="purple",
        edgecolor="black",
        linewidth=1
    )

    if col in ["siglaPartido", "occupation"]:
        plt.xticks(rotation=90)

    plt.title(col)
    plt.tight_layout()
    plt.show()
