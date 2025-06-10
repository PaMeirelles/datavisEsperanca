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
WEIGHT_OPTIONS = [10, 100, 1000]  # Fixed weight options

# Highlighting and plot aesthetics
MAX_CATEGORIES_TO_HIGHLIGHT = 3
HIGHLIGHT_COLORS = px.colors.qualitative.Plotly[:MAX_CATEGORIES_TO_HIGHLIGHT]
DEFAULT_POINT_COLOR = 'lightgrey'
HIGHLIGHT_POINT_SIZE = 6
DEFAULT_POINT_SIZE = 3

# Predefined order for 'age_group' or similar ordinal categories
AGE_GROUP_ORDER = [
    '0-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
]

# --- Load Initial Data ---
GLOBAL_DF = pd.DataFrame()
PRECALCULATED_DF = pd.DataFrame()
INITIAL_STATUS_MESSAGE = ""
HIGHLIGHT_FEATURE_DROPDOWN_OPTIONS = []
ORIGINAL_COLUMNS_FOR_HOVER = []

try:
    # Load node features
    GLOBAL_DF = pd.read_csv(NODES_CSV_PATH)
    # Load pre-calculated t-SNE coordinates
    PRECALCULATED_DF = pd.read_csv(COORDS_CSV_PATH)

    # Prepare dropdown options for feature selection
    POTENTIAL_COLS_TO_DROP_FROM_SELECTION = ['id', 'file']
    AVAILABLE_COLS_FOR_HIGHLIGHT_FEATURE = [col for col in GLOBAL_DF.columns if
                                            col not in POTENTIAL_COLS_TO_DROP_FROM_SELECTION]
    HIGHLIGHT_FEATURE_DROPDOWN_OPTIONS = [{'label': col, 'value': col} for col in AVAILABLE_COLS_FOR_HIGHLIGHT_FEATURE]

    # Prepare columns for the hover template
    ORIGINAL_COLUMNS_FOR_HOVER = [col for col in GLOBAL_DF.columns if
                                  col.lower() not in ['id', 'file']]

except FileNotFoundError as e:
    INITIAL_STATUS_MESSAGE = (f"Error: A required data file was not found. Please ensure both "
                              f"'{NODES_CSV_PATH}' and '{COORDS_CSV_PATH}' are in the same directory as the app. Details: {e}")
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

        # Display initial loading status or errors
        html.Div(id='initial-load-status', children=INITIAL_STATUS_MESSAGE,
                 style={'display': 'block' if INITIAL_STATUS_MESSAGE.strip() else 'none',
                        'marginTop': '10px', 'fontWeight': 'bold', 'color': 'red', 'textAlign': 'center',
                        'padding': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px'}),

        # --- Controls Section ---
        html.Div(className="controls-section",
                 style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '20px', 'marginBottom': '20px',
                        'backgroundColor': '#f9f9f9'},
                 children=[
                     html.H3("Controls", style={'marginTop': '0', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                     html.Div(className="control-row",
                              style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '15px', 'alignItems': 'center'},
                              children=[
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
                                      html.Label("Select Weight:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px'}),
                                      dcc.RadioItems(
                                          id='weight-selector',
                                          options=[{'label': f'{w}x', 'value': w} for w in WEIGHT_OPTIONS],
                                          value=WEIGHT_OPTIONS[0],
                                          labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                                          style={'paddingTop': '5px'}
                                      )
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
                         html.Button('Run Visualization', id='run-button', n_clicks=0,
                                     style={'padding': '10px 15px', 'backgroundColor': '#007bff', 'color': 'white',
                                            'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
                     ]),
                 ]),

        dcc.Loading(id="loading-spinner", type="circle", children=[
            html.Div(id='tsne-plot-div', children=[
                dcc.Graph(id='tsne-scatter-plot', figure=go.Figure())
            ]),
        ]),

        html.Div(id='error-message-div', style={'color': 'red', 'marginTop': '10px', 'fontWeight': 'bold'}),

        # Store data in the browser for performance
        dcc.Store(id='stored-dataframe-json', data=GLOBAL_DF.to_json(date_format='iso', orient='split') if not GLOBAL_DF.empty else None),
        dcc.Store(id='stored-original-columns-for-hover', data=ORIGINAL_COLUMNS_FOR_HOVER)
    ])

# --- Callback to update category selector based on chosen feature ---
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

    df = pd.read_json(df_json, orient='split')
    if selected_feature not in df.columns:
        return {'display': 'none'}, [], [], "Select Categories to Highlight (up to 3):"

    # Show category dropdown only if the feature is categorical/object type
    if pd.api.types.is_object_dtype(df[selected_feature]) or pd.api.types.is_categorical_dtype(df[selected_feature]):
        unique_categories = df[selected_feature].astype(str).fillna('Unknown').unique()

        # Handle special sorting for age groups
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

# --- Main callback to look up coordinates and generate plot ---
@app.callback(
    [Output('tsne-scatter-plot', 'figure'),
     Output('error-message-div', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('stored-dataframe-json', 'data'),
     State('highlight-feature-dropdown', 'value'),
     State('category-select-dropdown', 'value'),
     State('weight-selector', 'value'),
     State('stored-original-columns-for-hover', 'data')]
)
def generate_plot_from_precalculated(n_clicks, df_json, highlight_feature, selected_categories_to_highlight,
                                     weight, original_columns_for_hover):
    if n_clicks == 0:
        return go.Figure(), ""  # Return empty figure on initial load

    # Validate that data is loaded and a feature is selected
    if GLOBAL_DF.empty or PRECALCULATED_DF.empty:
        return go.Figure(), "Error: Data files are not loaded. Cannot generate plot."
    if df_json is None:
        return go.Figure(), "Error: Stored data is missing."
    if not highlight_feature:
        return go.Figure(), "Error: Please select a feature to weight before running the visualization."

    # Filter pre-calculated coordinates based on user's selection
    coords_df = PRECALCULATED_DF[
        (PRECALCULATED_DF['weighted_feature'] == highlight_feature) &
        (PRECALCULATED_DF['weight'] == weight)
    ]

    if coords_df.empty:
        return go.Figure(), f"Error: No pre-calculated coordinates found for '{highlight_feature}' with weight '{weight}x'."

    # Merge the coordinates with the main feature dataframe
    df_full = pd.read_json(df_json, orient='split')
    plot_df = pd.merge(df_full, coords_df[['id', 'x', 'y']], on='id', how='inner')

    if plot_df.empty:
        return go.Figure(), "Error: Could not map coordinates to node data. Check 'id' columns match in your CSV files."

    # Rename coordinate columns for Plotly Express
    plot_df.rename(columns={'x': 't-SNE_1', 'y': 't-SNE_2'}, inplace=True)

    # --- Prepare data for plotting (coloring, sizing, hover text) ---
    fig_title = f't-SNE Visualization with weight on <b>{highlight_feature}</b> (Weight: {weight}x)'
    plot_df['display_color_group'] = 'Other Points'
    plot_df['point_size_col'] = DEFAULT_POINT_SIZE  # Column to control point size

    color_discrete_map = {'Other Points': DEFAULT_POINT_COLOR}
    category_order_for_legend = ['Other Points']
    actual_categories_to_highlight = []

    # Apply highlighting if categories are selected
    if selected_categories_to_highlight and highlight_feature in plot_df.columns:
        actual_categories_to_highlight = selected_categories_to_highlight[:MAX_CATEGORIES_TO_HIGHLIGHT]
        feature_series_for_mask = plot_df[highlight_feature].astype(str)

        # Determine legend order
        if 'age_group' in highlight_feature.lower():
            ordered_selection = [cat for cat in AGE_GROUP_ORDER if cat in actual_categories_to_highlight]
            category_order_for_legend.extend(ordered_selection)
        else:
            category_order_for_legend.extend(sorted(actual_categories_to_highlight))

        # Apply colors and sizes to highlighted points
        for i, category_val in enumerate(actual_categories_to_highlight):
            mask = (feature_series_for_mask == str(category_val))
            plot_df.loc[mask, 'display_color_group'] = str(category_val)
            plot_df.loc[mask, 'point_size_col'] = HIGHLIGHT_POINT_SIZE
            color_discrete_map[str(category_val)] = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]

        fig_title += f'<br>Highlighting: {", ".join(actual_categories_to_highlight)}'

    # Configure hover data to show all original features but hide internal ones
    hover_data_config = {col: True for col in original_columns_for_hover if col in plot_df.columns}
    hover_data_config['point_size_col'] = False
    hover_data_config['display_color_group'] = False

    # --- Create the scatter plot ---
    fig = px.scatter(
        plot_df,
        x='t-SNE_1',
        y='t-SNE_2',
        color='display_color_group',
        size='point_size_col',
        title=fig_title,
        hover_data=hover_data_config,
        color_discrete_map=color_discrete_map,
        category_orders={'display_color_group': category_order_for_legend}
    )

    # --- Update Layout ---
    fig.update_layout(
        margin=dict(l=20, r=20, t=100, b=20),
        height=700,
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#333',
        legend_title_text='Highlighted Categories',
        xaxis=dict(showgrid=True, gridcolor='#e5e5e5'),
        # vvvvvvvvvvvv THIS IS THE MODIFICATION vvvvvvvvvvvv
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e5e5',
            scaleanchor="x",  # This forces a square aspect ratio
            scaleratio=1,
        )
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))

    return fig, ""

# --- Run the app ---
if __name__ == '__main__':
    app.run_server(debug=True)