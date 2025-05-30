import base64
import io
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
DEFAULT_CSV_PATH = 'all_nodes.csv'  # Dataset to be pre-loaded
# EDGES_CSV_PATH = 'all_edges.csv' # Edges file -- REMOVED
FIXED_TSNE_PERPLEXITY = 30.0
MAX_CATEGORIES_TO_HIGHLIGHT = 3
HIGHLIGHT_COLORS = px.colors.qualitative.Plotly[:MAX_CATEGORIES_TO_HIGHLIGHT]
DEFAULT_POINT_COLOR = 'lightgrey'
HIGHLIGHT_POINT_SIZE = 6  # Adjusted size
DEFAULT_POINT_SIZE = 3  # Adjusted size
# EDGE_COLOR = "rgba(100,100,100,0.15)" # REMOVED
# EDGE_WIDTH = 0.5 # REMOVED

# Predefined order for 'age_group' or similar ordinal categories
AGE_GROUP_ORDER = [
    '0-30',
    '30-40',
    '40-50',
    '50-60',
    '60-70',
    '70-80',
    '80-90',
]

# --- Load Initial Data ---
GLOBAL_DF = pd.DataFrame()
# EDGES_DF = pd.DataFrame() # REMOVED
INITIAL_STATUS_MESSAGE = ""
HIGHLIGHT_FEATURE_DROPDOWN_OPTIONS = []
ORIGINAL_COLUMNS_FOR_HOVER = []

try:
    GLOBAL_DF = pd.read_csv(DEFAULT_CSV_PATH)
    POTENTIAL_COLS_TO_DROP_FROM_SELECTION = ['id', 'file']
    AVAILABLE_COLS_FOR_HIGHLIGHT_FEATURE = [col for col in GLOBAL_DF.columns if
                                            col not in POTENTIAL_COLS_TO_DROP_FROM_SELECTION]
    HIGHLIGHT_FEATURE_DROPDOWN_OPTIONS = [{'label': col, 'value': col} for col in AVAILABLE_COLS_FOR_HIGHLIGHT_FEATURE]
    # Ensure 'point_size_col' (or similar if named differently in plot_df) is not in hover data
    ORIGINAL_COLUMNS_FOR_HOVER = [col for col in GLOBAL_DF.columns if
                                  col.lower() not in ['id', 'file', 'point_size_col', 'point_size',
                                                      'display_color_group']]


except FileNotFoundError:
    INITIAL_STATUS_MESSAGE = f"Error: Default node file '{DEFAULT_CSV_PATH}' not found. Please place it in the same directory as the app."
except Exception as e:
    INITIAL_STATUS_MESSAGE = f"Error loading '{DEFAULT_CSV_PATH}': {e}"

# Load Edges Data -- ENTIRE BLOCK REMOVED


# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Interactive t-SNE Explorer"

# --- App Layout ---
app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1300px', 'margin': 'auto', 'padding': '20px'}, children=[
        html.H1("Interactive t-SNE Dimensionality Reduction Explorer", style={'textAlign': 'center', 'color': '#333'}),

        html.Div(id='initial-load-status', children=INITIAL_STATUS_MESSAGE,
                 style={'display': 'block' if INITIAL_STATUS_MESSAGE.strip() else 'none',
                        # Only show if there's a non-empty message
                        'marginTop': '10px', 'fontWeight': 'bold',
                        'color': 'red' if 'Error' in INITIAL_STATUS_MESSAGE else (
                            'orange' if 'Warning' in INITIAL_STATUS_MESSAGE or 'Info' in INITIAL_STATUS_MESSAGE else 'green'),
                        'textAlign': 'center', 'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px',
                        'marginBottom': '20px'}),

        html.Div(className="controls-section",
                 style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '20px', 'marginBottom': '20px',
                        'backgroundColor': '#f9f9f9'}, children=[
                html.H3("Controls",
                        style={'marginTop': '0', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                html.Div(className="control-row",
                         style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '15px',
                                'alignItems': 'flex-start'}, children=[
                        html.Div(style={'width': '48%'}, children=[
                            html.Label("Feature for Weighting & Category Selection:",
                                       style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                            dcc.Dropdown(
                                id='highlight-feature-dropdown',
                                options=HIGHLIGHT_FEATURE_DROPDOWN_OPTIONS,
                                placeholder="Select a feature...",
                                clearable=True,
                                value=None
                            )
                        ]),
                        html.Div(style={'width': '48%'}, children=[
                            html.Label(id='weight-multiplier-label', children=f"Weight Multiplier (1-20): 5",
                                       style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                            dcc.Slider(id='weight-multiplier-slider', min=1, max=20, step=1, value=5,
                                       marks={i: str(i) for i in range(1, 21, 2)})
                        ])
                    ]),
                html.Div(className="control-row", style={'marginBottom': '15px'}, children=[
                    html.Div(id='category-selector-div', style={'display': 'none', 'width': '100%'}, children=[
                        html.Label(id='category-select-label', children="Select Categories to Highlight (up to 3):",
                                   style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                        dcc.Dropdown(
                            id='category-select-dropdown',
                            placeholder="Select categories...",
                            multi=True,
                            clearable=True
                        )
                    ])
                ]),
                html.Div(className="control-row", style={'textAlign': 'right', 'paddingTop': '10px'}, children=[
                    html.Button('Run t-SNE Visualization', id='run-button', n_clicks=0,
                                style={'padding': '10px 15px', 'backgroundColor': '#007bff', 'color': 'white',
                                       'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
                ]),
            ]),

        dcc.Loading(id="loading-spinner", type="circle", children=[
            html.Div(id='tsne-plot-div', children=[
                dcc.Graph(id='tsne-scatter-plot', figure=go.Figure())  # Initialize with an empty figure
            ]),
        ]),

        html.Div(id='error-message-div', style={'color': 'red', 'marginTop': '10px', 'fontWeight': 'bold'}),

        dcc.Store(id='stored-dataframe-json',
                  data=GLOBAL_DF.to_json(date_format='iso', orient='split') if not GLOBAL_DF.empty else None),
        dcc.Store(id='stored-original-columns-for-hover', data=ORIGINAL_COLUMNS_FOR_HOVER)
    ])


# --- Callback to update category selector dropdown ---
@app.callback(
    [Output('category-selector-div', 'style'),
     Output('category-select-dropdown', 'options'),
     Output('category-select-dropdown', 'value'),
     Output('category-select-label', 'children')],
    [Input('highlight-feature-dropdown', 'value')],
    [State('stored-dataframe-json', 'data')]
)
def update_category_selector(selected_feature, df_json):
    if not selected_feature or df_json is None:
        return {'display': 'none'}, [], [], "Select Categories to Highlight (up to 3):"

    try:
        df = pd.read_json(df_json, orient='split')
    except ValueError:
        return {'display': 'none'}, [], [], "Select Categories to Highlight (up to 3):"

    if selected_feature not in df.columns:
        return {'display': 'none'}, [], [], "Select Categories to Highlight (up to 3):"

    if pd.api.types.is_object_dtype(df[selected_feature]) or pd.api.types.is_categorical_dtype(df[selected_feature]):
        unique_categories = df[selected_feature].astype(str).fillna('Unknown').unique()

        column_name_normalized = df[selected_feature].name.lower().replace("_", " ")
        if 'age group' in column_name_normalized:
            temp_df = pd.DataFrame({'category': unique_categories})
            temp_df['ordered_category'] = pd.Categorical(temp_df['category'], categories=AGE_GROUP_ORDER, ordered=True)
            unique_categories = temp_df.sort_values('ordered_category')['category'].tolist()
        else:
            unique_categories = sorted(list(unique_categories))

        options = [{'label': cat, 'value': cat} for cat in unique_categories]
        label_text = f"Select Categories from '{selected_feature}' to Highlight (up to {MAX_CATEGORIES_TO_HIGHLIGHT}):"
        return {'display': 'block', 'width': '100%', 'marginBottom': '15px'}, options, [], label_text
    else:
        label_text = f"'{selected_feature}' is numerical. Category selection not applicable."
        return {'display': 'block', 'width': '100%', 'marginBottom': '15px'}, [], [], label_text


# --- Callback to update weight slider label ---
@app.callback(
    Output('weight-multiplier-label', 'children'),
    [Input('weight-multiplier-slider', 'value')]
)
def update_weight_label(value):
    return f"Weight Multiplier (1-20): {value}"


# --- Main callback to perform t-SNE and generate plot ---
@app.callback(
    [Output('tsne-scatter-plot', 'figure'),
     Output('error-message-div', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('stored-dataframe-json', 'data'),
     State('highlight-feature-dropdown', 'value'),
     State('category-select-dropdown', 'value'),
     State('weight-multiplier-slider', 'value'),
     State('stored-original-columns-for-hover', 'data')]
)
def run_tsne_and_plot(n_clicks, df_json, highlight_feature_for_weighting, selected_categories_to_highlight,
                      weight_multiplier, original_columns_for_hover):
    if n_clicks == 0:
        return go.Figure(), ""  # Return empty figure, no message

    if GLOBAL_DF.empty:
        return go.Figure(), f"Error: Could not load node data from '{DEFAULT_CSV_PATH}'. Plotting is not possible."
    if df_json is None:
        return go.Figure(), "Error: Stored data is missing."

    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'run-button':
        raise PreventUpdate

    try:
        df_full = pd.read_json(df_json, orient='split')
    except Exception as e:
        return go.Figure(), f"Error loading stored data: {e}"

    df = df_full.copy()

    cols_to_drop_for_processing = [col for col in ['id', 'file'] if col in df.columns]
    features_to_process = df.drop(columns=cols_to_drop_for_processing, errors='ignore')
    features_for_weighting = features_to_process.copy()

    original_categorical_cols = features_to_process.select_dtypes(include=['object', 'category']).columns.tolist()
    original_numerical_cols = features_to_process.select_dtypes(include=[np.number]).columns.tolist()

    if highlight_feature_for_weighting and highlight_feature_for_weighting in original_categorical_cols and weight_multiplier > 1:
        for i in range(1, int(weight_multiplier)):
            new_col_name = f"{highlight_feature_for_weighting}_weighted_copy{i}"
            k = 1
            while new_col_name in features_for_weighting.columns:
                new_col_name = f"{highlight_feature_for_weighting}_weighted_copy{i}_{k}";
                k += 1
            features_for_weighting[new_col_name] = features_to_process[highlight_feature_for_weighting]

    cat_features_proc = features_for_weighting.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features_proc = features_for_weighting.select_dtypes(include=[np.number]).columns.tolist()

    if not num_features_proc and not cat_features_proc:
        return go.Figure(), "Error: No features found to process for t-SNE."

    transformers_list = []
    if num_features_proc: transformers_list.append(('num', StandardScaler(), num_features_proc))
    if cat_features_proc: transformers_list.append(
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=True), cat_features_proc))

    if not transformers_list: return go.Figure(), "Error: No transformations to apply."

    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop')

    try:
        processed_data = preprocessor.fit_transform(features_for_weighting)
        transformed_feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        return go.Figure(), f"Error during preprocessing: {e}"

    processed_data_dense = processed_data.toarray() if hasattr(processed_data, 'toarray') else processed_data

    if highlight_feature_for_weighting and highlight_feature_for_weighting in original_numerical_cols and weight_multiplier > 1:
        expected_transformed_name = f"num__{highlight_feature_for_weighting}"
        try:
            col_idx = list(transformed_feature_names).index(expected_transformed_name)
            processed_data_dense[:, col_idx] *= float(weight_multiplier)
        except ValueError:
            print(f"Warning: Could not find transformed numerical feature '{expected_transformed_name}' for weighting.")
        except Exception as e:
            return go.Figure(), f"Error applying numerical weight: {e}"

    if processed_data_dense.shape[0] == 0 or processed_data_dense.shape[1] == 0:
        return go.Figure(), "Error: Processed data is empty after preprocessing."

    tsne = TSNE(n_components=2, random_state=42, perplexity=FIXED_TSNE_PERPLEXITY, n_iter=300, init='pca',
                learning_rate='auto')
    try:
        reduced_data = tsne.fit_transform(processed_data_dense)
    except Exception as e:
        return go.Figure(), f"Error during t-SNE: {e}"

    plot_df = pd.DataFrame(reduced_data, columns=['t-SNE_1', 't-SNE_2'])

    # Add original columns for hover data, using the index from features_to_process to align
    temp_hover_df = df_full.loc[features_to_process.index].reset_index(drop=True)
    for col_for_hover in original_columns_for_hover:  # These are pre-filtered
        if col_for_hover in temp_hover_df.columns:
            plot_df[col_for_hover] = temp_hover_df[col_for_hover]

    fig_title = f't-SNE Dimensionality Reduction (Perplexity: {FIXED_TSNE_PERPLEXITY})'
    color_discrete_map = {}
    plot_df['display_color_group'] = 'Other Points'
    plot_df['point_size_col'] = DEFAULT_POINT_SIZE  # This column controls point size

    actual_categories_to_highlight = []
    category_order_for_legend = ['Other Points']

    if highlight_feature_for_weighting and selected_categories_to_highlight and \
            highlight_feature_for_weighting in temp_hover_df.columns and \
            (pd.api.types.is_object_dtype(temp_hover_df[highlight_feature_for_weighting]) or \
             pd.api.types.is_categorical_dtype(temp_hover_df[highlight_feature_for_weighting])):

        actual_categories_to_highlight = selected_categories_to_highlight[:MAX_CATEGORIES_TO_HIGHLIGHT]
        feature_series_for_mask = temp_hover_df[highlight_feature_for_weighting].astype(str)

        # Determine legend order for highlighted categories
        temp_highlight_feature_name_norm = highlight_feature_for_weighting.lower().replace("_", " ")
        if 'age group' in temp_highlight_feature_name_norm:
            # Filter AGE_GROUP_ORDER to only include the selected highlighted categories
            ordered_selection = [cat for cat in AGE_GROUP_ORDER if cat in actual_categories_to_highlight]
            category_order_for_legend.extend(ordered_selection)
            # Add any selected categories not in AGE_GROUP_ORDER (shouldn't happen if dropdown is sourced correctly)
            # and ensure they are unique before adding
            remaining_selected = sorted(list(set(actual_categories_to_highlight) - set(ordered_selection)))
            category_order_for_legend.extend(remaining_selected)

        else:
            category_order_for_legend.extend(sorted(actual_categories_to_highlight))

        for i, category_val in enumerate(actual_categories_to_highlight):
            original_index_mask = (feature_series_for_mask == str(category_val))
            aligned_mask = original_index_mask.values
            plot_df.loc[aligned_mask, 'display_color_group'] = str(category_val)
            plot_df.loc[aligned_mask, 'point_size_col'] = HIGHLIGHT_POINT_SIZE
            try:
                legend_pos = category_order_for_legend.index(str(category_val)) - 1
                if legend_pos >= 0:
                    color_discrete_map[str(category_val)] = HIGHLIGHT_COLORS[legend_pos % len(HIGHLIGHT_COLORS)]
                else:
                    color_discrete_map[str(category_val)] = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
            except ValueError:
                color_discrete_map[str(category_val)] = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]

        fig_title += f'<br>Highlighting: {", ".join(actual_categories_to_highlight)} from {highlight_feature_for_weighting}'
        if highlight_feature_for_weighting in original_categorical_cols or highlight_feature_for_weighting in original_numerical_cols:
            fig_title += f' (Weighted by {weight_multiplier}x)'

    color_discrete_map['Other Points'] = DEFAULT_POINT_COLOR

    # Ensure hover_data list only contains columns that actually exist in plot_df
    valid_hover_cols = [col for col in original_columns_for_hover if col in plot_df.columns]

    # Prepare hover_data dictionary to explicitly exclude aesthetics columns
    hover_data_config = {col: True for col in valid_hover_cols}
    # Explicitly disable hover for columns used purely for aesthetics if they are not in valid_hover_cols
    # or if we want to be absolutely sure they don't appear.
    if 'point_size_col' in plot_df.columns:
        hover_data_config['point_size_col'] = False
    if 'display_color_group' in plot_df.columns:
        hover_data_config['display_color_group'] = False

    fig = px.scatter(
        plot_df, x='t-SNE_1', y='t-SNE_2',
        color='display_color_group',
        size='point_size_col',
        title=fig_title,
        hover_data=hover_data_config,  # Use the configured dictionary
        color_discrete_map=color_discrete_map,
        category_orders={'display_color_group': category_order_for_legend}
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=100, b=20),
        height=700,
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        plot_bgcolor='white', paper_bgcolor='white', font_color='#333',
        legend_title_text='Highlighted Categories',
        xaxis=dict(showgrid=True, gridcolor='#e5e5e5', zeroline=True, zerolinecolor='#ccc', zerolinewidth=1),
        yaxis=dict(showgrid=True, gridcolor='#e5e5e5', zeroline=True, zerolinecolor='#ccc', zerolinewidth=1)
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))

    return fig, ""


# --- Run the app ---
if __name__ == '__main__':
    app.run_server(debug=True)
