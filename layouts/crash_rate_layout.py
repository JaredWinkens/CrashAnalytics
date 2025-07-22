from dash import dcc, html

def load_crash_rate_layout(available_counties,
                           available_func_classes, 
                           county_coordinates,
                           unique_weather, 
                           unique_light, 
                           unique_road,
                           unique_crash_types,
                           min_date,
                           max_date,
                           common_controls
                           ):
    return html.Div(
                children=[
                    # Header Section
                    html.Div([
                        html.Div([
                            html.Img(src='/assets/Poly.svg', style={
                                'height': '128px', 'float': 'left', 'margin-right': '40px', 
                                'margin-left': '-20px', 'margin-top': '-8px'
                            }),
                            html.H1('Crash Rate', className='app-title'),
                            html.Img(src='/assets/NY.svg', className='ny-logo')
                        ],style={
                            'backgroundColor': '#18468B', 'padding': '7.5px', 'position': 'fixed', 
                            'top': '50px', 'left': '0', 'width': '100%', 'zIndex': '999', 'height': '90px'
                        }),
                    ]),
                    html.Div(
                        children=[
                            html.Div([
                                html.Div(children=[
                                    html.Label('Select Analysis Type:', style={'margin-top': '10px'}),
                                    html.Div(id='analysis_type-div', children=[
                                        dcc.RadioItems(
                                            id='analysis_selector_tab7',
                                            options=[
                                                {'label': 'Segment', 'value': 'Segment'},
                                                {'label': 'Intersection', 'value': 'Intersection'}
                                            ],
                                            value='Segment',
                                            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                                        ),
                                        html.Button(
                                            id='formula-button',n_clicks=0, 
                                            children=[
                                                html.I(id='formula-button-icon',className="fa fa-info", title="Get AI Powered Insights"),
                                        ]),   
                                    ]),
                                    html.Label('Functional Class:', style={'margin-top': '10px'}),
                                    dcc.Dropdown(
                                        id='select_functional_class_tab7',
                                        options=[{'label': 'All', 'value': 'All'}] +
                                                [{'label': f_class, 'value': f_class} for f_class in available_func_classes],
                                        value=[available_func_classes[0]],  # default county
                                        multi=True,
                                        placeholder='Select one or more classes or All',
                                        style={'width': '100%'}  # let the container control the overall width
                                    ),
                                ], className='responsive-controls'),
                                common_controls(
                                        'tab7',
                                        show_buttons=True,
                                        available_counties=available_counties,
                                        unique_weather=unique_weather,
                                        unique_light=unique_light,
                                        unique_road=unique_road,
                                        unique_crash_types=unique_crash_types,
                                        min_date=min_date,
                                        max_date=max_date,
                                ),
                            ], className='responsive-controls'),
                            
                            html.Div([
                                dcc.Graph(
                                    id='scatter_map_tab7',
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
                            ],className='responsive-graph'),
                        ],
                        className='desktop-layout'
                    ),
                    html.Div(
                        id='warning_message_tab7',
                        style={'color': 'red', 'textAlign': 'center', 'margin': '10px'}
                    )
                ]
            )