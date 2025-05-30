import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# apply same aesthetic theme
sns.set_theme(
    style="whitegrid",
    font="Palatino Linotype",
    rc={
        "axes.facecolor": "#f0eade",
        "figure.facecolor": "#f0eade",
        "font.family": "Palatino Linotype",
    }
)

# load and drop same columns
df = pd.read_csv("all_nodes.csv").drop(columns=["id", "file"])

# calculate missing and present percentages
miss_pct = df.isnull().sum() / len(df) * 100
present_pct = 100 - miss_pct

# assemble for plotting
plot_df = pd.DataFrame({
    "Present": present_pct,
    "Missing": miss_pct
})

# sort by missing percentage descending
plot_df = plot_df.sort_values("Missing", ascending=False)

# plot stacked 100% bar chart
ax = plot_df.plot(
    kind="bar",
    stacked=True,
    color=["green", "red"],  # grey for present, red for missing
    figsize=(10, 6)
)
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Column")
ax.set_title("Missing Values Percentage by Column")
plt.xticks(rotation=45, ha="right")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
