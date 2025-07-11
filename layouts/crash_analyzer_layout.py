from dash import dcc, html

def load_crash_analyzer_layout(available_counties, 
                               unique_weather, 
                               unique_light, 
                               unique_road, 
                               unique_crash_types,
                               county_coordinates,
                               common_controls):
    return html.Div(
            children=[
                # Header Section
                html.Div([
                    html.Div([
                        html.Img(src='/assets/Poly.svg', style={
                            'height': '128px', 'float': 'left', 'margin-right': '40px', 
                            'margin-left': '-20px', 'margin-top': '-8px'
                        }),
                        html.H1('Crash Analyzer', className='app-title'),
                        html.Img(src='/assets/NY.svg', className='ny-logo')
                    ],style={
                        'backgroundColor': '#18468B', 'padding': '7.5px', 'position': 'fixed', 
                        'top': '50px', 'left': '0', 'width': '100%', 'zIndex': '999', 'height': '90px'
                    }),
                ]),
                html.Div(
                    children=[
                        html.Div(
                            common_controls(
                                'tab5',
                                show_buttons=True,
                                available_counties=available_counties,
                                unique_weather=unique_weather,
                                unique_light=unique_light,
                                unique_road=unique_road,
                                unique_crash_types=unique_crash_types
                            ),
                            className='responsive-controls'
                        ),
                        html.Div([
                            dcc.Graph(
                                id='scatter_map_tab5',
                                className='responsive-graph',
                                figure={
                                    'data': [],
                                    'layout': {
                                        'mapbox': {
                                            'style': "open-street-map",
                                            'center': {
                                                'lat': county_coordinates[available_counties[0]]['lat'],
                                                'lon': county_coordinates[available_counties[0]]['lon']
                                            },
                                            'zoom': 10
                                        },
                                        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0}
                                    }
                                },
                                config={'modeBarButtonsToRemove': ['lasso2d'], 'displayModeBar': True, 'scrollZoom': True}
                            ),
                            # Hidden button for closing the popup (must exist in initial layout)
                            html.Button(html.I(className="fa-window-close"),id="close-popup-tab5",className="close-popup-button", n_clicks=0, style={'display': 'none'}),
                            
                            dcc.Loading(
                                children=[
                                    html.Div(id='image-popup-tab5', style={
                                        'position': 'fixed', # This popup will now be relative to the viewport
                                        'left': '50%',
                                        'top': '50%',
                                        'transform': 'translate(-50%, -50%)',
                                        'zIndex': '1000',
                                        'backgroundColor': 'white',
                                        'border': '1px solid black',
                                        'padding': '10px',
                                        'display': 'none', # Start hidden
                                        'maxWidth': '320px',
                                        'boxShadow': '0px 0px 10px rgba(0,0,0,0.5)'
                                    }),
                                ]
                            ),
                        ], className='responsive-graph'),    
                    ],
                    className='desktop-layout'
                ),
                html.Div(
                    id='warning_message_tab5',
                    style={'color': 'red', 'textAlign': 'center', 'margin': '10px'}
                )
            ]
        )