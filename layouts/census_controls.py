from dash import dcc, html
from app_instance import census_polygons_by_county

def census_controls():
    counties = sorted(census_polygons_by_county.keys())

    controls = [
        html.Div([
            html.Label('County:'),
            html.Div(
                "i",
                title='Select one or more counties. Use "All" to include every county.',
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
            )
        ]),
        dcc.Dropdown(
            id='county_selector_tab3',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': c, 'value': c} for c in counties],
            value=['Albany'],  # default selection
            multi=True,
            placeholder='Select one or more counties or All',
            style={'width': '100%'}
        ),
        html.Label("Select Census Attribute for Opacity:", style={'margin-top': '20px'}),
        dcc.Dropdown(
            id='census_attribute_selector',
            options=[
                {'label': 'Demographic Index',               'value': 'DEMOGIDX_5'},
                {'label': 'People of Color %',               'value': 'PEOPCOLORPCT'},
                {'label': 'Unemployment %',                  'value': 'UNEMPPCT'},
                {'label': 'Residential %',                   'value': 'pct_residential'},
                {'label': 'Industrial %',                    'value': 'pct_industrial'},
                {'label': 'Retail %',                        'value': 'pct_retail'},
                {'label': 'Commercial %',                    'value': 'pct_commercial'},
                {'label': 'AADT Crash Rate',                 'value': 'AADT Crash Rate'},
                {'label': 'VRU Crash Rate',                  'value': 'VRU Crash Rate'},
                {'label': 'AADT',                            'value': 'AADT'},
                {'label': 'Commute TripMiles Start Avg',     'value': 'Commute_TripMiles_TripStart_avg'},
                {'label': 'Commute TripMiles End Avg',       'value': 'Commute_TripMiles_TripEnd_avg'},
                {'label': 'Commute Biking and Walking Mile', 'value': 'Commute_BIKING_and_WALKING_Mile'},
                {'label': 'Commute Biking and Walking Mi 1', 'value': 'Commute_BIKING_and_WALKING_Mi_1'},
            ],
            value='DEMOGIDX_5',
            style={'width': '100%'}
        ),
        html.Div(id='census_color_legend', style={'margin-top': '20px'}),
        # contextual notes
        html.Div([
            html.P("Crash data 2020-2023", style={'margin': '0'}),
            html.P("Environmental justice data 2023", style={'margin': '0'}),
            html.P("Census tract data 2023", style={'margin': '0'}),
            html.P("Poverty data 2022", style={'margin': '0'}),
            html.P("Disability data 2022", style={'margin': '0'}),
            html.P("Race/Pop data 2022", style={'margin': '0'})
        ], style={
            'margin-top': '10px',
            'font-size': '12px',
            'color': '#333',
            'border': '1px solid #ccc',
            'padding': '5px'
        }),
        html.Div([
            html.Button('Apply Filter', id='apply_filter_tab3', n_clicks=0, style={'margin-top': '30px'})
        ], style={'margin-bottom': '20px', 'margin-top': '20px'})
    ]

    return html.Div(controls, className='responsive-controls')