import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize
import time
import os

# --- Helper Functions from the original script ---

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
        print(f"Applying weight {weight_value} to the '{weighted_feature}' feature...")
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
    """
    # --- Configuration ---
    SAMPLE_FRACTION = 0.05
    WEIGHT_VALUES = [10.0, 100.0, 1000.0]
    OUTPUT_FILENAME = 'precalculated_coordinates.csv'

    print("--- Starting Pre-calculation ---")

    # --- Load and Sample Data ---
    try:
        df_full = pd.read_csv('all_nodes.csv')
        print(f"Loaded {len(df_full)} total nodes.")
    except FileNotFoundError:
        print("\nError: Could not find 'all_nodes.csv'. Please ensure it's in the correct directory.")
        return

    # Sample the nodes once to ensure consistency across all runs
    df_sample = df_full.sample(frac=SAMPLE_FRACTION, random_state=42).copy()
    print(f"Sampled {len(df_sample)} nodes for analysis ({SAMPLE_FRACTION * 100}% of total).")

    # Identify feature columns to iterate over
    feature_columns = [col for col in df_sample.columns if col not in ['id', 'file']]
    print(f"Identified features for analysis: {feature_columns}\n")

    # --- Main Loop ---
    all_results = []
    total_runs = len(feature_columns) * len(WEIGHT_VALUES)
    current_run = 0

    for feature in feature_columns:
        for weight in WEIGHT_VALUES:
            current_run += 1
            print(f"--- Running Analysis {current_run}/{total_runs} ---")
            print(f"Weighted Feature: '{feature}', Weight: {weight}")
            start_time = time.time()

            # 1. Preprocess data and calculate weighted distance matrix
            distance_matrix = preprocess_and_calculate_distances(
                df=df_sample,
                weighted_feature=feature,
                weight_value=weight
            )

            # 2. Create a warm start solution based on the feature being weighted
            initial_angles = create_warm_start_angles(
                df_processed=df_sample,
                warm_start_category=feature,
                spread_radians=0.4
            )

            # 3. Run the optimization
            result = minimize(
                fun=circular_objective_function,
                x0=initial_angles,
                args=(distance_matrix,),
                method='L-BFGS-B',
                bounds=[(0, 2 * np.pi)] * len(df_sample),
                options={'disp': False, 'maxiter': 5000} # disp=False for cleaner output
            )

            if result.success:
                # 4. Calculate Cartesian coordinates from optimized angles
                final_angles = result.x
                radius = 1.0
                x_coords = radius * np.cos(final_angles)
                y_coords = radius * np.sin(final_angles)

                # 5. Store results
                run_results = pd.DataFrame({
                    'id': df_sample['id'],
                    'weighted_feature': feature,
                    'weight': weight,
                    'x': x_coords,
                    'y': y_coords
                })
                all_results.append(run_results)
                print(f"Completed in {time.time() - start_time:.2f} seconds.\n")
            else:
                print(f"Optimization failed for feature '{feature}' with weight {weight}.\n")


    # --- Save Final Results ---
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"--- Pre-calculation complete. ---")
        print(f"Results for {len(final_df)} data points saved to '{OUTPUT_FILENAME}'.")
    else:
        print("--- Pre-calculation finished with no successful runs. ---")


if __name__ == '__main__':
    run_precalculation()