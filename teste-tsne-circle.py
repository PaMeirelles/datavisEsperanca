import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize
import time
import os


# --- Helper Functions from the original script (unchanged) ---

def preprocess_and_calculate_distances(df, weighted_feature=None, weight_value=1.0):
    """
    Preprocesses a dataframe and calculates the pairwise distance matrix,
    applying a specified weight to a single feature.
    """
    df_features = df.drop(columns=[col for col in ['id', 'file'] if col in df.columns], errors='ignore')

    for col in df_features.select_dtypes(include=['object']).columns:
        df_features[col] = df_features[col].fillna('Unknown')

    if 'age_group' in df_features.columns:
        def age_group_to_numeric(age_group):
            if isinstance(age_group, str) and '-' in age_group:
                low, high = map(int, age_group.split('-'))
                return (low + high) / 2
            return 0

        df_features['age_group'] = df_features['age_group'].apply(age_group_to_numeric)

    categorical_cols = df_features.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df_features, columns=categorical_cols)

    if weighted_feature:
        # A small print statement here is useful for monitoring
        # print(f"Applying weight {weight_value} to the '{weighted_feature}' feature...")
        for col in df_encoded.columns:
            # Apply weight to all one-hot encoded columns derived from the feature
            if col.startswith(weighted_feature):
                df_encoded[col] *= weight_value

    distance_matrix = pairwise_distances(df_encoded.to_numpy(), metric='euclidean')
    return distance_matrix


def create_warm_start_angles(df_processed, warm_start_category, spread_radians=0.2):
    """
    Creates a 'warm start' initial solution based on a chosen category.
    """
    unique_cats = sorted(df_processed[warm_start_category].astype(str).unique())
    n_cats = len(unique_cats)
    centroid_angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False)
    centroid_map = {category: angle for category, angle in zip(unique_cats, centroid_angles)}

    base_angles = df_processed[warm_start_category].astype(str).map(centroid_map)
    n_points = len(df_processed)
    jitter = (np.random.rand(n_points) - 0.5) * spread_radians
    initial_angles = np.mod(base_angles + jitter, 2 * np.pi)

    return initial_angles


def circular_objective_function(angles, distance_matrix):
    """
    Objective function: Minimize the difference between circular distance and feature distance.
    """
    max_dist = np.max(distance_matrix)
    norm_dist_atr = distance_matrix / max_dist if max_dist > 0 else distance_matrix
    angle_diff = angles[:, np.newaxis] - angles
    dist_circ = np.minimum(np.abs(angle_diff), 2 * np.pi - np.abs(angle_diff))
    norm_dist_circ = dist_circ / np.pi
    squared_diff = (norm_dist_circ - norm_dist_atr) ** 2
    return np.sum(np.triu(squared_diff, 1))


# --- Main Pre-calculation Logic ---

def run_precalculation():
    """
    Main function to run the analysis for all features and weights.
    Saves results incrementally to allow for interruption and resumption.
    """
    # --- Configuration ---
    SAMPLE_FRACTION = .05
    WEIGHT_VALUES = [100.0, 200.0, 400.0]
    INPUT_FILENAME = 'all_nodes.csv'
    OUTPUT_FILENAME = 'precalculated_coordinates.csv'

    print("--- Starting Pre-calculation ---")

    # --- Load and Sample Data ---
    try:
        df_full = pd.read_csv(INPUT_FILENAME)
        print(f"Loaded {len(df_full)} total nodes from '{INPUT_FILENAME}'.")
    except FileNotFoundError:
        print(f"\nError: Could not find '{INPUT_FILENAME}'. Please ensure it's in the correct directory.")
        return

    df_sample = df_full.sample(frac=SAMPLE_FRACTION, random_state=42).copy()
    print(f"Sampled {len(df_sample)} nodes for analysis ({SAMPLE_FRACTION * 100}% of total).")

    feature_columns = [col for col in df_sample.columns if col not in ['id', 'file']]
    print(f"Identified features for analysis: {feature_columns}\n")

    # --- MODIFICATION 1: Check for existing results to avoid re-calculating ---
    completed_runs = set()
    if os.path.exists(OUTPUT_FILENAME):
        print(f"Found existing results file: '{OUTPUT_FILENAME}'. Reading to prevent re-work.")
        df_existing = pd.read_csv(OUTPUT_FILENAME)
        for _, row in df_existing.iterrows():
            completed_runs.add((row['weighted_feature'], row['weight']))
        print(f"Loaded {len(completed_runs)} previously completed runs.")

    # --- Main Loop ---
    total_runs = len(feature_columns) * len(WEIGHT_VALUES)
    current_run_number = 0

    for feature in feature_columns:
        for weight in WEIGHT_VALUES:
            current_run_number += 1
            print(f"--- Preparing Analysis {current_run_number}/{total_runs} ---")

            # --- MODIFICATION 2: Skip this loop iteration if the result already exists ---
            if (feature, weight) in completed_runs:
                print(f"Skipping: Feature '{feature}', Weight {weight} already calculated.\n")
                continue

            print(f"Running for: Weighted Feature '{feature}', Weight: {weight}")
            start_time = time.time()

            distance_matrix = preprocess_and_calculate_distances(
                df=df_sample,
                weighted_feature=feature,
                weight_value=weight
            )
            initial_angles = create_warm_start_angles(
                df_processed=df_sample,
                warm_start_category=feature,
                spread_radians=0.1
            )
            result = minimize(
                fun=circular_objective_function,
                x0=initial_angles,
                args=(distance_matrix,),
                method='L-BFGS-B',
                bounds=[(0, 2 * np.pi)] * len(df_sample),
                options={'disp': False, 'maxiter': 5000}
            )

            if result.success:
                final_angles = result.x
                radius = 1.0
                x_coords = radius * np.cos(final_angles)
                y_coords = radius * np.sin(final_angles)

                run_results = pd.DataFrame({
                    'id': df_sample['id'],
                    'weighted_feature': feature,
                    'weight': weight,
                    'x': x_coords,
                    'y': y_coords
                })

                # --- MODIFICATION 3: Append results immediately to the CSV file ---
                # Determine if the header should be written (only for the very first time)
                write_header = not os.path.exists(OUTPUT_FILENAME)

                run_results.to_csv(OUTPUT_FILENAME, mode='a', header=write_header, index=False)

                print(f"SUCCESS: Results saved to '{OUTPUT_FILENAME}'.")
                print(f"Completed in {time.time() - start_time:.2f} seconds.\n")

            else:
                print(f"Optimization failed for feature '{feature}' with weight {weight}.\n")

    # --- MODIFICATION 4: Final message is updated as saving is now incremental ---
    print(f"--- Pre-calculation process finished. ---")
    if os.path.exists(OUTPUT_FILENAME):
        print(f"All results are stored in '{OUTPUT_FILENAME}'.")
    else:
        print("No successful runs were completed.")


if __name__ == '__main__':
    run_precalculation()