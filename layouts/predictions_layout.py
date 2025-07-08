from dash import dcc, html

def load_predictions_layout(make_field_row, LABELS, STEPS, ALL_FIELDS):
    return html.Div([
            # Left-side: Controls (Model, Prediction Bar, County Selector, and Editing UI)
            html.Div([
                html.Div([
                    html.Label('Select AI Model:', style={'fontSize': '12px'}),
                    html.Span(
                        "i",
                        title="Choose a tract to edit and apply the new values, then reset to change another one. Update one tract at a time.",
                        style={
                            'display': 'inline-block',
                            'backgroundColor': '#ccc',
                            'border': '1px solid #999',
                            'borderRadius': '50%',
                            'width': '16px',
                            'height': '16px',
                            'textAlign': 'center',
                            'lineHeight': '16px',
                            'cursor': 'help',
                            'marginLeft': '5px',
                            'fontSize': '12px'
                        }
                    )
                ], style={'display': 'flex', 'alignItems': 'center'}),
                dcc.Dropdown(
                    id='model_selector_tab4',
                    options=[
                        {'label': 'MGWR Model',   'value': 'mgwr_predict.py'},
                    ],
                    value='mgwr_predict.py',
                    clearable=False,
                    style={'width': '100%', 'fontSize': '12px'}
                ),
                html.Div(
                    id='prediction_bar',
                    style={
                        'width': '100%',
                        'padding': '10px',
                        'backgroundColor': '#eee',
                        'textAlign': 'center',
                        'fontWeight': 'bold'
                    }
                ),
                dcc.Store(id='original_prediction'),
                html.Br(),
                html.Label('Select County:', style={'fontSize': '12px'}),
                dcc.Dropdown(
                    id='county_selector_tab4',
                    options=[],  # to be updated via callback
                    multi=True,
                    placeholder='Select county by CNTY_NAME',
                    style={'width': '100%', 'fontSize': '12px'}
                ),
                html.Br(),
                html.Label('Prediction Data Controls', style={'fontSize': '12px'}),
                html.Button('Refresh Predictions', id='refresh_predictions_tab4', n_clicks=0, style={'fontSize': '12px'}),
                html.Hr(),
                html.Div(
                    [ make_field_row(var, LABELS[var], STEPS[var]) for var,_,_ in ALL_FIELDS ],
                    id="modal_fields_container"
                ),
                html.Div([
                html.Button("Apply Updated Data",
                    id="apply_updated_data",
                    n_clicks=0,
                    style={'marginRight':'10px','fontSize':'12px'}),
                html.Button("Reset Predictions",
                    id="reset_predictions",
                    n_clicks=0,
                    style={'fontSize':'12px'})
                    ], style={'marginTop':'10px','textAlign':'center'}),
            ], className='responsive-controls'),
            # Right-side: Predictions Map
            html.Div([
                # Predictions map
                dcc.Graph(
                    id='predictions_map',
                    className='responsive-graph',
                    figure={
                        'data': [],
                        'layout': {
                            'mapbox': {
                                'style': "open-street-map",
                                'center': {'lat': 40.7128, 'lon': -74.0060},
                                'zoom': 10
                            },
                            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0}
                        }
                    },
                    config={
                        'modeBarButtonsToRemove': ['lasso2d'],
                        'displayModeBar': True,
                        'scrollZoom': True
                    }
                ),

                # Comparison scatter plot
                dcc.Graph(
                    id='comparison_graph',
                    className='responsive-graph'
                )
            ], className='responsive-graph'),
        ], className='desktop-layout')