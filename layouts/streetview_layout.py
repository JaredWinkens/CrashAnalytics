from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_pannellum

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
                # html.Div([
                #     html.Div([
                #         html.Img(src='/assets/Poly.svg', style={
                #             'height': '128px', 'float': 'left', 'margin-right': '40px', 
                #             'margin-left': '-20px', 'margin-top': '-8px'
                #         }),
                #         html.H1('Crash Analyzer', className='app-title'),
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
                            
                            dcc.Loading(
                                children=[
                                    dbc.Modal(
                                        id="image-popup-tab5", 
                                        children=[],
                                        size="lg",
                                        is_open=False,
                                    ),
                                ]
                            ),
                        ], className='responsive-graph'),    
                    ],
                    className='desktop-layout'
                ),
                html.Div(
                    id='warning_message_tab5',
                    style={'color': 'red', 'textAlign': 'center', 'margin': '10px'}
                ),
                dcc.Store(id='filtered_data_tab5', data=None)
            ]
        )