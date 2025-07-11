from dash import dcc, html

def load_heatmap_layout(available_counties, 
                        unique_weather, 
                        unique_light, 
                        unique_road, 
                        unique_crash_types,
                        county_coordinates,
                        common_controls):
    return html.Div([
            # Header Section
            html.Div([
                html.Div([
                    html.Img(src='/assets/Poly.svg', style={
                        'height': '128px', 'float': 'left', 'margin-right': '40px', 
                        'margin-left': '-20px', 'margin-top': '-8px'
                    }),
                    html.H1('Heatmap', className='app-title'),
                    html.Img(src='/assets/NY.svg', className='ny-logo')
                ],style={
                    'backgroundColor': '#18468B', 'padding': '7.5px', 'position': 'fixed', 
                    'top': '50px', 'left': '0', 'width': '100%', 'zIndex': '999', 'height': '90px'
                }),
            ]),
            html.Div([
                html.Div([
                    html.Label('Adjust Heatmap Radius:', style={'font-weight': 'bold'}),
                    html.Div(
                        "i",
                        title="This slider adjusts the radius of influence for the heatmap.",
                        style={
                            'display': 'inline-block',
                            'background-color': '#ccc',
                            'border': '1px solid #999',
                            'border-radius': '3px',
                            'width': '20px',
                            'height': '20px',
                            'text-align': 'center',
                            'line-height': '20px',
                            'cursor': 'help',
                            'margin-left': '5px'
                        }
                    ),
                    dcc.Slider(
                        id='radius_slider_tab2',
                        min=0.1,
                        max=10,
                        step=0.1,
                        value=1,
                        marks={
                            0.1: '0.1 mi',
                            1: '1 mi',
                            2: '2 mi',
                            3: '3 mi',
                            4: '4 mi',
                            5: '5 mi',
                            6: '6 mi',
                            7: '7 mi',
                            8: '8 mi',
                            9: '9 mi',
                            10: '10 mi'
                        },
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ], style={'margin-bottom': '20px'}),
                
                common_controls(
                    'tab2',
                    show_buttons=True,
                    available_counties=available_counties,
                    unique_weather=unique_weather,
                    unique_light=unique_light,
                    unique_road=unique_road,
                    unique_crash_types=unique_crash_types)
            ], className='responsive-controls'),
            
            # Right-side: The Heatmap Graph container
            html.Div([
                dcc.Graph(
                    id='heatmap_graph',
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
                html.Button(html.I(className="fa-window-close"),id="close-popup-tab2",className="close-popup-button", n_clicks=0, style={'display': 'none'}),

                dcc.Loading(
                    html.Div(id='image-popup-tab2', style={
                        'position': 'fixed',
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
                ),
                html.Button(
                    id='insight-button',n_clicks=0, 
                    children=[
                        html.I(className="fas fa-lightbulb", title="Get AI Powered Insights"),
                    ]),
        ], className='responsive-graph'),
    
    ], className='desktop-layout')