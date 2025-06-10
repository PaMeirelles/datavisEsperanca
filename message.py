import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from io import StringIO
import math

# --- Configuration for Category Selection ---
# Choose the column you want to use for clustering nodes.
# Possible options from your CSV: 'education', 'gender', 'marital_status', 'ethnicity', 'occupation', 'siglaPartido', 'ufNascimento', 'sexo'
# Make sure the chosen_category_column exists in your CSV.
chosen_category_column = 'ethnicity' # <--- CHANGE THIS LINE TO SELECT YOUR CATEGORY

# Load data from CSV. Assuming 'all_nodes.csv' is in the same directory.
try:
    df = pd.read_csv("./all_nodes.csv")
except FileNotFoundError:
    print("Error: 'all_nodes.csv' not found. Please make sure the file is in the correct directory.")
    exit() # Exit if the file isn't found, as we can't proceed without data

print("Education", df["education"].dropna().unique())

# --- Data Preprocessing ---
# Group by 'id' to get unique politicians and extract relevant attributes.
# .first() takes the first non-NA value in each group for each column.
node_data = df.groupby('id').first().reset_index()

# Select only the 'id' and the chosen category column
# Check if the chosen_category_column exists in the DataFrame
if chosen_category_column not in node_data.columns:
    print(f"Error: The chosen category column '{chosen_category_column}' does not exist in the CSV data.")
    print(f"Available columns are: {node_data.columns.tolist()}")
    exit()

node_data = node_data[['id', chosen_category_column]].copy()
node_data.rename(columns={chosen_category_column: 'category'}, inplace=True) # Rename for generic use

# Handle potential NaN values in the chosen category
node_data['category'].fillna('UNDEFINED', inplace=True)

# Create a mapping from 'id' to category for quick lookup
id_to_category = node_data.set_index('id')['category'].to_dict()

# --- 1. Create a graph ---
G = nx.Graph()

# Add nodes with their categorical attributes
for index, row in node_data.iterrows():
    node_id = row['id']
    category = row['category']
    G.add_node(node_id, category=category)

# --- Define Edges ---
# Create edges: connect politicians who served in the same 'election_year'
# This creates a co-occurrence network based on shared legislative years.
unique_election_years = df['election_year'].unique()

for year in unique_election_years:
    # Get all unique politician IDs for the current election year
    politicians_in_year = df[df['election_year'] == year]['id'].unique()

    # Create connections between all pairs of politicians in this year
    for i, p1_id in enumerate(politicians_in_year):
        for p2_id in politicians_in_year[i+1:]: # Only iterate over subsequent politicians to avoid duplicates
            if p1_id != p2_id: # Ensure no self-loops
                # Add edge. Using 'add_edge' handles cases where the edge already exists without error
                G.add_edge(p1_id, p2_id)

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Check if there are any nodes or edges
if G.number_of_nodes() == 0:
    print("No nodes found in the graph. Check your data and node creation logic.")
    exit()
if G.number_of_edges() == 0:
    print("No edges found in the graph. The edge creation logic might need adjustment for your data.")


# --- 2. Define centroids on a circle based on the chosen category ---
unique_categories = sorted(list(set(id_to_category.values()))) # Get unique categories
num_categories = len(unique_categories)

if num_categories == 0:
    raise ValueError(f"No categories found in the data for column '{chosen_category_column}'. Check your data or the preprocessing.")

circle_radius = 5.0 # Radius of the circle on which centroids are placed
centroid_coords = {}
# Generate equally spaced angles for centroids. endpoint=False avoids duplicating 0 and 2*pi.
angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)

for i, category in enumerate(unique_categories):
    x_centroid = circle_radius * np.cos(angles[i])
    y_centroid = circle_radius * np.sin(angles[i])
    centroid_coords[category] = (x_centroid, y_centroid)

# --- 3. Position nodes around their category centroids ---
node_positions = {}
perturb_scale = 0.8 # Small perturbation factor to spread nodes around their centroid

lognormal_mu = 0.5   # Mean of the logarithm of the radius (controls average distance)
lognormal_sigma = 0.5 # Standard deviation of the logarithm of the radius (controls spread)
max_radius_cutoff = 1.5 # An upper bound to prevent extremely large radii, crucial with log-normal

for node_id in G.nodes():
    category = G.nodes[node_id]['category']
    cx, cy = centroid_coords[category]

    initial_r = np.random.lognormal(lognormal_mu, lognormal_sigma)
    if initial_r > max_radius_cutoff:
        initial_r = max_radius_cutoff # Fallback if initial is too large
    initial_angle = random.uniform(0, 2 * math.pi)
    initial_px = cx + initial_r * math.cos(initial_angle)
    initial_py = cy + initial_r * math.sin(initial_angle)
    node_positions[node_id] = (initial_px, initial_py)

# --- 4. Plot the graph with translucent edges and categorized nodes ---
plt.figure(figsize=(15, 12)) # Larger figure size for better visibility

# Draw edges with low alpha for overlap effect
# Using a slightly darker gray for better contrast on possible white background
nx.draw_networkx_edges(G, node_positions, edge_color='#666666', alpha=0.01, width=0.1) # Adjust alpha and width

# Get unique categories and assign colors for nodes using a colormap
cmap = plt.cm.get_cmap('tab20', num_categories)

# Draw nodes, iterating by category to add labels for the legend
for i, category in enumerate(unique_categories):
    # Filter nodes belonging to the current category
    nodes_in_category = [node for node in G.nodes() if G.nodes[node]['category'] == category]

    # Get positions for these nodes
    pos_in_category = {node: node_positions[node] for node in nodes_in_category}

    # Get color for this category
    color = cmap(i) # Use the index 'i' to get the color from the colormap

    nx.draw_networkx_nodes(G, pos_in_category,
                           nodelist=nodes_in_category, # Specify which nodes to draw
                           node_color=[color] * len(nodes_in_category), # Assign uniform color to all nodes in this category
                           node_size=20, alpha=0.9, linewidths=0.5, edgecolors='black',
                           label=category) # Add label for the legend

# --- 5. Plot the centroids and their labels ---
centroid_x = [p[0] for p in centroid_coords.values()]
centroid_y = [p[1] for p in centroid_coords.values()]
centroid_labels = list(centroid_coords.keys())


plt.title(f"Brazilian Politicians Clustered by {chosen_category_column.replace('_', ' ').title()}")
plt.axis('equal') # Ensures the circle looks like a circle
plt.axis('off') # Hide axes for a cleaner graph
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1)) # Adjust legend position to not overlap graph
plt.show()

