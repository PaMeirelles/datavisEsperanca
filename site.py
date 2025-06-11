import pandas as pd
import numpy as np  # Import numpy for faster operations
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
NODES_CSV_PATH = 'all_nodes.csv'
COORDS_CSV_PATH = 'precalculated_coordinates_2.csv'
EDGES_CSV_PATH = 'all_edges.csv'
WEIGHT_OPTIONS = [100, 200, 400]

# Highlighting and plot aesthetics
MAX_CATEGORIES_TO_HIGHLIGHT = 5
HIGHLIGHT_COLORS = px.colors.qualitative.Plotly[:MAX_CATEGORIES_TO_HIGHLIGHT]
DEFAULT_POINT_COLOR = 'lightgrey'
HIGHLIGHT_POINT_SIZE = 8  # Slightly larger for WebGL
DEFAULT_POINT_SIZE = 5  # Slightly larger for WebGL
EDGE_COLOR = "rgba(100, 100, 100, 0.2)"
EDGE_WIDTH = 0.5

# Predefined order for 'age_group' or similar ordinal categories
AGE_GROUP_ORDER = ['0-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']

# --- Load Initial Data ---
GLOBAL_DF = pd.DataFrame()
PRECALCULATED_DF = pd.DataFrame()
EDGES_DF = pd.DataFrame()
INITIAL_STATUS_MESSAGE = ""
HIGHLIGHT_FEATURE_DROPDOWN_OPTIONS = []
ORIGINAL_COLUMNS_FOR_HOVER = []

try:
    GLOBAL_DF = pd.read_csv(NODES_CSV_PATH)
    PRECALCULATED_DF = pd.read_csv(COORDS_CSV_PATH)
    EDGES_DF = pd.read_csv(EDGES_CSV_PATH)

    POTENTIAL_COLS_TO_DROP_FROM_SELECTION = ['id', 'file']
    AVAILABLE_COLS_FOR_HIGHLIGHT_FEATURE = [col for col in GLOBAL_DF.columns if
                                            col not in POTENTIAL_COLS_TO_DROP_FROM_SELECTION]
    HIGHLIGHT_FEATURE_DROPDOWN_OPTIONS = [{'label': col, 'value': col} for col in AVAILABLE_COLS_FOR_HIGHLIGHT_FEATURE]
    ORIGINAL_COLUMNS_FOR_HOVER = [col for col in GLOBAL_DF.columns if col.lower() not in ['id', 'file']]

except FileNotFoundError as e:
    INITIAL_STATUS_MESSAGE = (f"Error: A required data file was not found. Please ensure "
                              f"'{NODES_CSV_PATH}', '{COORDS_CSV_PATH}', and '{EDGES_CSV_PATH}' "
                              f"are in the same directory. Details: {e}")
except Exception as e:
    INITIAL_STATUS_MESSAGE = f"Error loading initial data files: {e}"

# --- Initialize the Dash App ---
app = dash.Dash(__name__)
app.title = "Interactive t-SNE Explorer"

# --- App Layout ---
app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1300px', 'margin': 'auto', 'padding': '20px'},
    children=[
        html.H1("Interactive t-SNE Pre-calculated Explorer", style={'textAlign': 'center', 'color': '#333'}),

        html.Div(id='initial-load-status', children=INITIAL_STATUS_MESSAGE,
                 style={'display': 'block' if INITIAL_STATUS_MESSAGE.strip() else 'none', 'marginTop': '10px',
                        'fontWeight': 'bold', 'color': 'red', 'textAlign': 'center', 'padding': '10px',
                        'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px'}),

        # --- Controls Section (Unchanged) ---
        html.Div(className="controls-section",
                 style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '20px', 'marginBottom': '20px',
                        'backgroundColor': '#f9f9f9'},
                 children=[
                     html.H3("Controls",
                             style={'marginTop': '0', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                     html.Div(className="control-row",
                              style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '15px',
                                     'alignItems': 'center'},
                              children=[
                                  html.Div(style={'width': '48%'}, children=[
                                      html.Label("Feature for Weighting & Category Selection:",
                                                 style={'fontWeight': 'bold', 'display': 'block',
                                                        'marginBottom': '5px'}),
                                      dcc.Dropdown(id='highlight-feature-dropdown',
                                                   options=HIGHLIGHT_FEATURE_DROPDOWN_OPTIONS,
                                                   placeholder="Select a feature...", clearable=True, value=None)
                                  ]),
                                  html.Div(style={'width': '48%'}, children=[
                                      html.Label("Select Weight:", style={'fontWeight': 'bold', 'display': 'block',
                                                                          'marginBottom': '5px'}),
                                      dcc.RadioItems(id='weight-selector',
                                                     options=[{'label': f'{w}x', 'value': w} for w in WEIGHT_OPTIONS],
                                                     value=WEIGHT_OPTIONS[0],
                                                     labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                                                     style={'paddingTop': '5px'})
                                  ])
                              ]),
                     html.Div(className="control-row", style={'marginBottom': '15px'}, children=[
                         html.Div(id='category-selector-div', style={'display': 'none', 'width': '100%'}, children=[
                             html.Label(id='category-select-label',
                                        children=f"Select Categories to Highlight (up to {MAX_CATEGORIES_TO_HIGHLIGHT}):",
                                        style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                             dcc.Dropdown(id='category-select-dropdown', placeholder="Select categories...", multi=True,
                                          clearable=True)
                         ])
                     ]),
                     html.Div(className="control-row", style={'textAlign': 'right', 'paddingTop': '10px'}, children=[
                         html.Button('Run Visualization', id='run-button', n_clicks=0,
                                     style={'padding': '10px 15px', 'backgroundColor': '#007bff', 'color': 'white',
                                            'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
                     ]),
                 ]),

        dcc.Loading(id="loading-spinner", type="circle", children=[
            html.Div(id='tsne-plot-div', children=[dcc.Graph(id='tsne-scatter-plot', figure=go.Figure())]),
        ]),

        html.Div(id='error-message-div', style={'color': 'red', 'marginTop': '10px', 'fontWeight': 'bold'}),

        # REMOVED: No longer storing the large dataframe in the browser. It's already in server memory.
        # dcc.Store(id='stored-dataframe-json', data=GLOBAL_DF.to_json(...)),
        dcc.Store(id='stored-original-columns-for-hover', data=ORIGINAL_COLUMNS_FOR_HOVER)
    ])


# --- Callback to update category selector (Unchanged, but now reads from global df) ---
@app.callback(
    [Output('category-selector-div', 'style'), Output('category-select-dropdown', 'options'),
     Output('category-select-dropdown', 'value'), Output('category-select-label', 'children')],
    [Input('highlight-feature-dropdown', 'value')]
)
def update_category_selector(selected_feature):
    if not selected_feature:
        raise PreventUpdate

    df = GLOBAL_DF  # Read directly from global dataframe

    if selected_feature not in df.columns:
        return {'display': 'none'}, [], [], f"Select Categories to Highlight (up to {MAX_CATEGORIES_TO_HIGHLIGHT}):"

    if pd.api.types.is_object_dtype(df[selected_feature]) or pd.api.types.is_categorical_dtype(df[selected_feature]):
        unique_categories = df[selected_feature].astype(str).fillna('Unknown').unique()
        if 'age_group' in selected_feature.lower():
            temp_df = pd.DataFrame({'category': unique_categories})
            temp_df['ordered_category'] = pd.Categorical(temp_df['category'], categories=AGE_GROUP_ORDER, ordered=True)
            unique_categories = temp_df.sort_values('ordered_category')['category'].tolist()
        else:
            unique_categories = sorted(list(unique_categories))
        options = [{'label': cat, 'value': cat} for cat in unique_categories]
        label_text = f"Select Categories from '{selected_feature}' to Highlight (up to {MAX_CATEGORIES_TO_HIGHLIGHT}):"
        return {'display': 'block', 'width': '100%', 'marginBottom': '15px'}, options, [], label_text
    else:
        label_text = f"'{selected_feature}' is numerical. Highlighting specific categories is not applicable."
        return {'display': 'block', 'width': '100%', 'marginBottom': '15px'}, [], [], label_text


# --- Main callback to generate plot (HEAVILY REFACTORED FOR PERFORMANCE) ---
@app.callback(
    [Output('tsne-scatter-plot', 'figure'), Output('error-message-div', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('highlight-feature-dropdown', 'value'),
     State('category-select-dropdown', 'value'), State('weight-selector', 'value'),
     State('stored-original-columns-for-hover', 'data')]
)
def generate_plot_from_precalculated(n_clicks, highlight_feature, selected_categories_to_highlight,
                                     weight, original_columns_for_hover):
    if n_clicks == 0: raise PreventUpdate
    if any(df.empty for df in [GLOBAL_DF, PRECALCULATED_DF, EDGES_DF]):
        return go.Figure(), "Error: Data files are not loaded. Cannot generate plot."
    if not highlight_feature: return go.Figure(), "Error: Please select a feature to weight before running."

    coords_df = PRECALCULATED_DF[
        (PRECALCULATED_DF['weighted_feature'] == highlight_feature) & (PRECALCULATED_DF['weight'] == weight)]
    if coords_df.empty: return go.Figure(), f"Error: No pre-calculated coordinates found for '{highlight_feature}' with weight '{weight}x'."

    # Merge node features with the correct coordinates
    plot_df = pd.merge(GLOBAL_DF, coords_df[['id', 'x', 'y']], on='id', how='inner')
    if plot_df.empty: return go.Figure(), "Error: Could not map coordinates to node data."
    plot_df.rename(columns={'x': 'tSNE_1', 'y': 'tSNE_2'}, inplace=True)

    # --- EFFICIENT Edge Trace Generation ---
    # 1. Create a dictionary for fast coordinate lookup
    id_to_coords = plot_df.set_index('id')[['tSNE_1', 'tSNE_2']].to_dict('index')

    # 2. Map start and end coordinates for all edges at once
    # Ensure your edge file has columns like 'id1' and 'id2'
    edges_subset = EDGES_DF[EDGES_DF['id1'].isin(id_to_coords) & EDGES_DF['id2'].isin(id_to_coords)].copy()
    edges_subset['start_coords'] = edges_subset['id1'].map(id_to_coords)
    edges_subset['end_coords'] = edges_subset['id2'].map(id_to_coords)
    edges_subset.dropna(subset=['start_coords', 'end_coords'], inplace=True)

    # 3. Build the coordinate arrays for a single line trace
    edge_x, edge_y = [], []
    for index, row in edges_subset.iterrows():
        x0, y0 = row['start_coords']['tSNE_1'], row['start_coords']['tSNE_2']
        x1, y1 = row['end_coords']['tSNE_1'], row['end_coords']['tSNE_2']
        edge_x.extend([x0, x1, None])  # Use None to break the line between edges
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scattergl(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color=EDGE_COLOR, width=EDGE_WIDTH),
        hoverinfo='none',
        showlegend=False
    )

    # --- Prepare Node Highlighting ---
    fig_title = f't-SNE Visualization with weight on <b>{highlight_feature}</b> (Weight: {weight}x)'
    plot_df['display_color_group'] = 'Other Points'
    plot_df['point_size_col'] = DEFAULT_POINT_SIZE
    color_discrete_map = {'Other Points': DEFAULT_POINT_COLOR}
    category_order_for_legend = ['Other Points']

    if selected_categories_to_highlight and highlight_feature in plot_df.columns:
        actual_categories_to_highlight = selected_categories_to_highlight[:MAX_CATEGORIES_TO_HIGHLIGHT]
        feature_series_for_mask = plot_df[highlight_feature].astype(str)

        # Sort selection for consistent legend ordering
        if 'age_group' in highlight_feature.lower():
            ordered_selection = [cat for cat in AGE_GROUP_ORDER if cat in actual_categories_to_highlight]
        else:
            ordered_selection = sorted(actual_categories_to_highlight)

        category_order_for_legend.extend(ordered_selection)

        for i, category_val in enumerate(ordered_selection):
            mask = (feature_series_for_mask == str(category_val))
            plot_df.loc[mask, 'display_color_group'] = str(category_val)
            plot_df.loc[mask, 'point_size_col'] = HIGHLIGHT_POINT_SIZE
            color_discrete_map[str(category_val)] = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
        fig_title += f'<br>Highlighting: {", ".join(actual_categories_to_highlight)}'

    # --- Create the Main Scatter Plot using Plotly Express ---
    fig = px.scatter(
        plot_df, x='tSNE_1', y='tSNE_2',
        color='display_color_group',
        size='point_size_col',
        title=fig_title,
        hover_data={col: True for col in original_columns_for_hover},
        color_discrete_map=color_discrete_map,
        category_orders={'display_color_group': category_order_for_legend},
        render_mode='webgl'  # Use WebGL for faster rendering of points
    )

    # --- Combine the Edge Trace and the Scatter Plot ---
    fig.add_trace(edge_trace)

    fig.update_layout(
        margin=dict(l=20, r=20, t=100, b=20),
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#333',
        legend_title_text='Highlighted Categories',
        xaxis=dict(visible=False, showgrid=False),
        yaxis=dict(visible=False, showgrid=False, scaleanchor="x", scaleratio=1),
        showlegend=True
    )

    return fig, ""


# --- Run the app ---
if __name__ == '__main__':
    app.run_server(debug=True)