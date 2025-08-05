from dash import dcc, html

def load_data_download_layout(available_counties, 
                               unique_weather, 
                               unique_light, 
                               unique_road, 
                               unique_crash_types,
                               county_coordinates,
                               common_controls):
    return html.Div(
                children=[
                    # Header Section
                    # html.Div([
                    #     html.Div([
                    #         html.Img(src='/assets/Poly.svg', style={
                    #             'height': '128px', 'float': 'left', 'margin-right': '40px', 
                    #             'margin-left': '-20px', 'margin-top': '-8px'
                    #         }),
                    #         html.H1('Data Downloader', className='app-title'),
                    #         html.Img(src='/assets/NY.svg', className='ny-logo')
                    #     ],style={
                    #         'backgroundColor': '#18468B', 'padding': '7.5px', 'position': 'fixed', 
                    #         'top': '50px', 'left': '0', 'width': '100%', 'zIndex': '999', 'height': '90px'
                    #     }),
                    # ]),
                    html.Div(
                        children=[
                            html.Div(
                                common_controls(
                                    'tab1',
                                    show_buttons=True,
                                    available_counties=available_counties,
                                    unique_weather=unique_weather,
                                    unique_light=unique_light,
                                    unique_road=unique_road,
                                    unique_crash_types=unique_crash_types
                                ),
                                className='responsive-controls'
                            ),
                            dcc.Graph(
                                id='scatter_map',
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
                            )
                        ],
                        className='desktop-layout'
                    ),
                    html.Div(
                        id='warning_message_tab1',
                        style={'color': 'red', 'textAlign': 'center', 'margin': '10px'}
                    )
                ]
            )