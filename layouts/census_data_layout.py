from dash import dcc, html

def load_census_data_layout(census_controls, county_coordinates, available_counties):
    return html.Div([
        # Header Section
        # html.Div([
        #     html.Div([
        #         html.Img(src='/assets/Poly.svg', style={
        #             'height': '128px', 'float': 'left', 'margin-right': '40px', 
        #             'margin-left': '-20px', 'margin-top': '-8px'
        #         }),
        #         html.H1('Census Data', className='app-title'),
        #         html.Img(src='/assets/NY.svg', className='ny-logo')
        #     ],style={
        #         'backgroundColor': '#18468B', 'padding': '7.5px', 'position': 'fixed', 
        #         'top': '50px', 'left': '0', 'width': '100%', 'zIndex': '999', 'height': '90px'
        #     }),
        # ]),
        html.Div([
            census_controls(),
            html.Div(id='warning_message_tab3', style={'color': 'red'})
        ], className='responsive-controls'),
        html.Div(
            dcc.Graph(
                id='scatter_map_tab3',
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
                config={
                    'modeBarButtonsToRemove': ['lasso2d'], 
                    'displayModeBar': True, 
                    'scrollZoom': True
                }
            ),
            className='responsive-graph'
        )
    ], className='desktop-layout')