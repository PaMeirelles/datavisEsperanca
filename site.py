import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
NODES_CSV_PATH = 'all_nodes.csv'  # Original features for nodes
COORDS_CSV_PATH = 'precalculated_coordinates.csv'  # Pre-calculated coordinates
EDGES_CSV_PATH = 'all_edges.csv'  # Edges between nodes (source, target)
WEIGHT_OPTIONS = [10, 100, 1000]  # Fixed weight options

# Highlighting and plot aesthetics
MAX_CATEGORIES_TO_HIGHLIGHT = 3
HIGHLIGHT_COLORS = px.colors.qualitative.Plotly[:MAX_CATEGORIES_TO_HIGHLIGHT]
DEFAULT_POINT_COLOR = 'lightgrey'
HIGHLIGHT_POINT_SIZE = 6
DEFAULT_POINT_SIZE = 3
EDGE_COLOR = "rgba(100, 100, 100, 0.2)"  # Low alpha for edges
EDGE_WIDTH = 0.5

# Predefined order for 'age_group' or similar ordinal categories
AGE_GROUP_ORDER = [
    '0-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
]

# --- Load Initial Data ---
GLOBAL_DF = pd.DataFrame()
PRECALCULATED_DF = pd.DataFrame()
EDGES_DF = pd.DataFrame()
INITIAL_STATUS_MESSAGE = ""
HIGHLIGHT_FEATURE_DROPDOWN_OPTIONS = []
ORIGINAL_COLUMNS_FOR_HOVER = []

try:
    # Load node features, coordinates, and edges
    GLOBAL_DF = pd.read_csv(NODES_CSV_PATH)
    PRECALCULATED_DF = pd.read_csv(COORDS_CSV_PATH)
    EDGES_DF = pd.read_csv(EDGES_CSV_PATH)
    # Prepare dropdown options for feature selection
    POTENTIAL_COLS_TO_DROP_FROM_SELECTION = ['id', 'file']
    AVAILABLE_COLS_FOR_HIGHLIGHT_FEATURE = [col for col in GLOBAL_DF.columns if
                                            col not in POTENTIAL_COLS_TO_DROP_FROM_SELECTION]
    HIGHLIGHT_FEATURE_DROPDOWN_OPTIONS = [{'label': col, 'value': col} for col in AVAILABLE_COLS_FOR_HIGHLIGHT_FEATURE]

    # Prepare columns for the hover template
    ORIGINAL_COLUMNS_FOR_HOVER = [col for col in GLOBAL_DF.columns if
                                  col.lower() not in ['id', 'file']]

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
                                        children="Select Categories to Highlight (up to 3):",
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

        dcc.Store(id='stored-dataframe-json',
                  data=GLOBAL_DF.to_json(date_format='iso', orient='split') if not GLOBAL_DF.empty else None),
        dcc.Store(id='stored-original-columns-for-hover', data=ORIGINAL_COLUMNS_FOR_HOVER)
    ])


# --- Callback to update category selector based on chosen feature ---
@app.callback(
    [Output('category-selector-div', 'style'), Output('category-select-dropdown', 'options'),
     Output('category-select-dropdown', 'value'), Output('category-select-label', 'children')],
    [Input('highlight-feature-dropdown', 'value')],
    [State('stored-dataframe-json', 'data')]
)
def update_category_selector(selected_feature, df_json):
    if not selected_feature or df_json is None:
        return {'display': 'none'}, [], [], "Select Categories to Highlight (up to 3):"
    df = pd.read_json(df_json, orient='split')
    if selected_feature not in df.columns:
        return {'display': 'none'}, [], [], "Select Categories to Highlight (up to 3):"
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


# --- Main callback to look up coordinates, add edges, and generate plot ---
@app.callback(
    [Output('tsne-scatter-plot', 'figure'), Output('error-message-div', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('stored-dataframe-json', 'data'), State('highlight-feature-dropdown', 'value'),
     State('category-select-dropdown', 'value'), State('weight-selector', 'value'),
     State('stored-original-columns-for-hover', 'data')]
)
def generate_plot_from_precalculated(n_clicks, df_json, highlight_feature, selected_categories_to_highlight,
                                     weight, original_columns_for_hover):
    if n_clicks == 0: return go.Figure(), ""
    if any(df.empty for df in [GLOBAL_DF, PRECALCULATED_DF, EDGES_DF]):
        return go.Figure(), "Error: Data files are not loaded. Cannot generate plot."
    if df_json is None: return go.Figure(), "Error: Stored data is missing."
    if not highlight_feature: return go.Figure(), "Error: Please select a feature to weight before running."

    coords_df = PRECALCULATED_DF[
        (PRECALCULATED_DF['weighted_feature'] == highlight_feature) & (PRECALCULATED_DF['weight'] == weight)]
    if coords_df.empty: return go.Figure(), f"Error: No pre-calculated coordinates found for '{highlight_feature}' with weight '{weight}x'."

    df_full = pd.read_json(df_json, orient='split')
    plot_df = pd.merge(df_full, coords_df[['id', 'x', 'y']], on='id', how='inner')
    if plot_df.empty: return go.Figure(), "Error: Could not map coordinates to node data."
    plot_df.rename(columns={'x': 't-SNE_1', 'y': 't-SNE_2'}, inplace=True)

    # --- Generate Edge Shapes ---
    edge_shapes = []
    # Use 'source' and 'target' as column names, adjust if your file uses different names
    if 'id1' in EDGES_DF.columns and 'id2' in EDGES_DF.columns:
        coords_for_edges = plot_df[['id', 't-SNE_1', 't-SNE_2']]
        edges_temp = pd.merge(EDGES_DF, coords_for_edges, left_on='id1', right_on='id', how='inner')
        edges_temp.rename(columns={'t-SNE_1': 'x0', 't-SNE_2': 'y0'}, inplace=True)
        edges_with_coords = pd.merge(edges_temp.drop(columns=['id']), coords_for_edges, left_on='id2', right_on='id',
                                     how='inner')
        edges_with_coords.rename(columns={'t-SNE_1': 'x1', 't-SNE_2': 'y1'}, inplace=True)

        edge_shapes = edges_with_coords.apply(
            lambda row: go.layout.Shape(type="line", layer="below", x0=row.x0, y0=row.y0, x1=row.x1, y1=row.y1,
                                        line=dict(color=EDGE_COLOR, width=EDGE_WIDTH)),
            axis=1
        ).tolist()
    print(edge_shapes)
    # --- Prepare for plotting (coloring, sizing, etc.) ---
    fig_title = f't-SNE Visualization with weight on <b>{highlight_feature}</b> (Weight: {weight}x)'
    plot_df['display_color_group'] = 'Other Points'
    plot_df['point_size_col'] = DEFAULT_POINT_SIZE
    color_discrete_map = {'Other Points': DEFAULT_POINT_COLOR}
    category_order_for_legend = ['Other Points']

    if selected_categories_to_highlight and highlight_feature in plot_df.columns:
        actual_categories_to_highlight = selected_categories_to_highlight[:MAX_CATEGORIES_TO_HIGHLIGHT]
        feature_series_for_mask = plot_df[highlight_feature].astype(str)
        if 'age_group' in highlight_feature.lower():
            ordered_selection = [cat for cat in AGE_GROUP_ORDER if cat in actual_categories_to_highlight]
            category_order_for_legend.extend(ordered_selection)
        else:
            category_order_for_legend.extend(sorted(actual_categories_to_highlight))
        for i, category_val in enumerate(actual_categories_to_highlight):
            mask = (feature_series_for_mask == str(category_val))
            plot_df.loc[mask, 'display_color_group'] = str(category_val)
            plot_df.loc[mask, 'point_size_col'] = HIGHLIGHT_POINT_SIZE
            color_discrete_map[str(category_val)] = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
        fig_title += f'<br>Highlighting: {", ".join(actual_categories_to_highlight)}'

    hover_data_config = {col: True for col in original_columns_for_hover if col in plot_df.columns}
    hover_data_config['point_size_col'] = False
    hover_data_config['display_color_group'] = False

    # --- Create the scatter plot ---
    fig = px.scatter(
        plot_df, x='t-SNE_1', y='t-SNE_2', color='display_color_group', size='point_size_col',
        title=fig_title, hover_data=hover_data_config, color_discrete_map=color_discrete_map,
        category_orders={'display_color_group': category_order_for_legend})

    # --- Update Layout with Edges and Square Aspect Ratio ---
    fig.update_layout(
        shapes=edge_shapes,  # Add the lines for the edges here
        margin=dict(l=20, r=20, t=100, b=20),
        height=700,
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        plot_bgcolor='white', paper_bgcolor='white', font_color='#333',
        legend_title_text='Highlighted Categories',
        xaxis=dict(showgrid=True, gridcolor='#e5e5e5'),
        yaxis=dict(showgrid=True, gridcolor='#e5e5e5', scaleanchor="x", scaleratio=1)
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))

    return fig, ""


# --- Run the app ---
if __name__ == '__main__':
    app.run_server(debug=True)