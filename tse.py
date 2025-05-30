import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors


def plot_dimensionality_reduction_tsne(csv_path, highlight_feature=None, weight_multiplier=5, tsne_perplexity=30.0):
    """
    Reads a CSV, performs dimensionality reduction using t-SNE with optional
    feature weighting, and plots the results, coloring by the highlighted feature.

    Args:
        csv_path (str): The path to the CSV file.
        highlight_feature (str, optional): The name of the column to highlight
                                           and give more weight. Defaults to None.
        weight_multiplier (float, optional): How much more weight to give the
                                             highlight_feature. Defaults to 5.
                                             For categorical, it's the number of times
                                             the feature effectively appears.
        tsne_perplexity (float, optional): The perplexity value for t-SNE.
                                           Related to the number of nearest neighbors.
                                           Usually between 5 and 50. Defaults to 30.0.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # --- Feature Selection and Initial DataFrame Preparation ---
    # Store the original 'highlight_feature' column for coloring before any modification
    coloring_data_original = None
    if highlight_feature and highlight_feature in df.columns:
        coloring_data_original = df[highlight_feature].copy()
    elif highlight_feature:
        print(
            f"Warning: Highlight feature '{highlight_feature}' not found in DataFrame columns. No coloring will be applied for it.")
        highlight_feature = None  # Disable if not found

    # Prepare features for processing by dropping 'id' and 'file'
    if 'id' in df.columns and 'file' in df.columns:
        features_to_process = df.drop(columns=['id', 'file'])
    else:
        print("Warning: 'id' or 'file' column not found. Proceeding with available columns, excluding them if present.")
        temp_cols_to_drop = [col for col in ['id', 'file'] if col in df.columns]
        features_to_process = df.drop(columns=temp_cols_to_drop)

    # Ensure highlight_feature still exists after dropping id/file
    if highlight_feature and highlight_feature not in features_to_process.columns:
        print(
            f"Warning: Highlight feature '{highlight_feature}' was among 'id' or 'file' and has been dropped. No specific weighting/coloring.")
        highlight_feature = None  # Disable if it was id or file
        coloring_data_original = None

    features_for_weighting = features_to_process.copy()  # This df will be modified for weighting

    if highlight_feature:
        print(f"Highlighting and weighting feature: '{highlight_feature}' with multiplier: {weight_multiplier}")

    # --- Identify Feature Types (from the non-weighted DataFrame) ---
    original_categorical_cols = features_to_process.select_dtypes(include=['object', 'category']).columns.tolist()
    original_numerical_cols = features_to_process.select_dtypes(include=[np.number]).columns.tolist()

    # --- Apply Weighting to 'features_for_weighting' ---
    # For categorical features, duplicate them
    if highlight_feature and highlight_feature in original_categorical_cols:
        if weight_multiplier > 1:
            print(
                f"Duplicating categorical feature '{highlight_feature}' {int(weight_multiplier) - 1} times for weighting.")
            for i in range(1, int(weight_multiplier)):
                new_col_name = f"{highlight_feature}_weighted_copy{i}"
                # Ensure new column name is unique
                k = 1
                while new_col_name in features_for_weighting.columns:
                    new_col_name = f"{highlight_feature}_weighted_copy{i}_{k}"
                    k += 1
                features_for_weighting[new_col_name] = features_to_process[highlight_feature]
        else:
            print("Weight multiplier <= 1 for categorical, no duplication for weighting.")

    # Identify features for preprocessing from the (potentially modified) features_for_weighting
    categorical_features_for_processing = features_for_weighting.select_dtypes(
        include=['object', 'category']).columns.tolist()
    numerical_features_for_processing = features_for_weighting.select_dtypes(include=[np.number]).columns.tolist()

    if not numerical_features_for_processing and not categorical_features_for_processing:
        print("Error: No numerical or categorical features found to process for t-SNE.")
        return

    # --- Preprocessing ---
    numerical_transformer = StandardScaler()
    # For OneHotEncoder, sparse_output=False can sometimes simplify things if memory allows,
    # but default (True) is usually fine and more memory-efficient.
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_for_processing),
            ('cat', categorical_transformer, categorical_features_for_processing)
        ],
        remainder='passthrough'  # In case some columns were missed, though ideally all are covered
    )

    try:
        # Fit the preprocessor and transform the data
        processed_data = preprocessor.fit_transform(features_for_weighting)
        # Get transformed feature names for potential debugging or numerical weighting
        transformed_feature_names = preprocessor.get_feature_names_out()
    except ValueError as ve:
        print(f"ValueError during preprocessing: {ve}. This might happen if a category list is empty.")
        print("Numerical features for processing:", numerical_features_for_processing)
        print("Categorical features for processing:", categorical_features_for_processing)
        # Check if any categorical feature has only one unique value after potential NaN drop by OHE
        for col in categorical_features_for_processing:
            if features_for_weighting[col].nunique(dropna=True) <= 1 and preprocessor.transformers_[1][
                1].drop == 'first':  # cat transformer
                print(
                    f"Warning: Categorical feature '{col}' has <=1 unique value and 'drop=first' is used, which might lead to empty columns.")
        return
    except Exception as e:
        print(f"Error during ColumnTransformer fitting/transforming: {e}")
        return

    # Convert to dense array if it's sparse (common after OneHotEncoding)
    if hasattr(processed_data, 'toarray'):
        processed_data_dense = processed_data.toarray()
    else:
        processed_data_dense = processed_data

    # --- Apply Weighting for NUMERICAL Highlighted Feature (Post-Scaling) ---
    if highlight_feature and highlight_feature in original_numerical_cols:  # Check against original_numerical_cols
        # The name in transformed_feature_names will be like 'num__original_feature_name'
        expected_transformed_name = f"num__{highlight_feature}"
        try:
            # Find the index of the column in the processed_data_dense array
            col_idx = list(transformed_feature_names).index(expected_transformed_name)
            print(
                f"Applying weight multiplier ({weight_multiplier}) to transformed numerical column: '{transformed_feature_names[col_idx]}'")
            processed_data_dense[:, col_idx] = processed_data_dense[:, col_idx] * float(weight_multiplier)
        except ValueError:
            print(
                f"Warning: Could not find transformed numerical feature '{expected_transformed_name}' for weighting. It might have been dropped or named differently by the preprocessor.")
            # print("Available transformed features:", transformed_feature_names)
        except Exception as e:
            print(f"Error applying numerical weight: {e}")

    if processed_data_dense.shape[1] == 0:
        print("Error: No features remaining after preprocessing. Cannot perform t-SNE.")
        return
    # t-SNE can work with 1 feature, but results in a line. For 2D plot, we need n_components=2.
    # The input data itself can have any number of features >= 1.

    # --- t-SNE Dimensionality Reduction ---
    print(f"Performing t-SNE with perplexity={tsne_perplexity}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, n_iter=300, init='pca',
                learning_rate='auto')
    try:
        reduced_data = tsne.fit_transform(processed_data_dense)
    except Exception as e:
        print(f"Error during t-SNE: {e}")
        # print(f"Shape of data fed to t-SNE: {processed_data_dense.shape}")
        # print(f"Sample of data fed to t-SNE:\n {processed_data_dense[:5]}")
        # print(f"NaNs in data fed to t-SNE: {np.isnan(processed_data_dense).sum()}")
        # print(f"Infs in data fed to t-SNE: {np.isinf(processed_data_dense).sum()}")
        return

    # --- Plotting ---
    plt.figure(figsize=(13, 9))  # Adjusted size for potentially better legend display

    plot_title = 'Dimensionality Reduction Scatter Plot (t-SNE)'
    if highlight_feature:
        plot_title += f'\nHighlighted & Weighted: {highlight_feature} (Perplexity: {tsne_perplexity})'
    else:
        plot_title += f' (Perplexity: {tsne_perplexity})'

    if coloring_data_original is not None:
        # Ensure coloring_data_original has the same length as reduced_data
        if len(coloring_data_original) != reduced_data.shape[0]:
            print("Error: Length mismatch between coloring data and reduced data. This shouldn't happen.")
            # Fallback to no specific coloring
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)
        elif pd.api.types.is_numeric_dtype(coloring_data_original) and not pd.api.types.is_bool_dtype(
                coloring_data_original):
            # Numerical feature for coloring
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                                  c=coloring_data_original, cmap='viridis', alpha=0.7)
            cbar = plt.colorbar(scatter)
            cbar.set_label(highlight_feature)
        else:
            # Categorical or boolean feature for coloring
            # Use pd.factorize to get integer codes and unique category names
            # Important: factorize on the *original* coloring data
            codes, unique_cats = pd.factorize(
                coloring_data_original.astype(str).fillna('Unknown'))  # Ensure string type and handle NaNs

            num_unique = len(unique_cats)

            # Choose a colormap
            if num_unique <= 10:
                cmap_name = 'tab10'
            elif num_unique <= 20:
                cmap_name = 'tab20'
            else:
                cmap_name = 'viridis'  # Fallback for many categories, though legend might be too long

            # Create a ListedColormap for precise color control if using tab10/tab20
            if cmap_name in ['tab10', 'tab20']:
                cmap = plt.get_cmap(cmap_name, num_unique)
            else:  # For continuous-like colormaps like viridis, generate discrete colors
                cmap = plt.get_cmap(cmap_name)  # Get the base cmap
                # Generate num_unique distinct colors from this cmap
                colors_for_scatter = [cmap(i / num_unique) for i in range(num_unique)]
                cmap = matplotlib.colors.ListedColormap(colors_for_scatter)

            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                                  c=codes, cmap=cmap, alpha=0.7)

            # Create legend
            if num_unique <= 20:  # Only create legend if manageable number of categories
                handles = []
                for i, category_name in enumerate(unique_cats):
                    # The color for the legend marker should be cmap(i) if using ListedColormap directly
                    # or cmap(i / (num_unique -1 if num_unique > 1 else 1)) if normalizing for a continuous cmap
                    color_for_legend = cmap(
                        i / (num_unique - 1 if num_unique > 1 else 1)) if cmap_name == 'viridis' else cmap(i)

                    handles.append(plt.Line2D([0], [0], marker='o', color='w', label=str(category_name),
                                              markerfacecolor=color_for_legend, markersize=8))

                plt.legend(title=highlight_feature, handles=handles, bbox_to_anchor=(1.02, 1), loc='upper left',
                           borderaxespad=0.)
                plt.subplots_adjust(right=0.78)  # Adjust plot to make space for legend
            else:  # Too many categories for a clean legend, use a colorbar with category names
                # This is trickier for categorical data with a colormap like tab10.
                # A simpler approach for many categories might be to not show a legend or colorbar.
                # Or, if essential, create a custom colorbar.
                print(
                    f"Warning: Too many unique categories ({num_unique}) in '{highlight_feature}' for a clear legend. Consider a different feature or filtering.")
                # Basic colorbar for codes (less interpretable for categorical)
                cbar = plt.colorbar(scatter, ticks=np.arange(num_unique))
                try:
                    cbar.ax.set_yticklabels(unique_cats)  # Try to set category names
                except Exception as e_cbar:
                    print(f"Could not set category names on colorbar: {e_cbar}")
                cbar.set_label(f'{highlight_feature} (Codes)')

    else:  # No highlight_feature specified
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)

    plt.title(plot_title, fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    # plt.tight_layout() # Often helps, but can conflict with bbox_to_anchor for legend
    plt.show()
    print("Plot generated successfully!")


# --- How to use ---
if __name__ == '__main__':
    # To use with your actual file (e.g., 'all_nodes_merged.csv'):
    # print("\n--- Plotting 'all_nodes_merged.csv' with 'your_feature_to_highlight' ---")
    plot_dimensionality_reduction_tsne('all_nodes_merged.csv', highlight_feature='siglaPartido', weight_multiplier=3, tsne_perplexity=30)
