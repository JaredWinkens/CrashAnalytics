from dash import dcc, html
import dash_bootstrap_components as dbc


def common_controls(prefix, show_buttons, available_counties, unique_weather, unique_light, unique_road , unique_crash_types, min_date=None, max_date=None):
    controls = [
        html.Div([
            html.Label('County:'),
            html.Div(
                "i",
                title='Select one or more counties. Selecting multiple counties can degrade performance.',
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
            id=f'county_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': county, 'value': county} for county in available_counties],
            value=['Albany'],  # default county
            multi=True,
            placeholder='Select one or more counties or All',
            style={'width': '100%'}  # let the container control the overall width
        ),
        html.Label('Data Type:', style={'margin-top': '10px'}),
        dcc.RadioItems(
            id=f'data_type_selector_main_{prefix}',
            options=[
                {'label': 'All', 'value': 'All'},
                {'label': 'VRU', 'value': 'VRU'}
            ],
            value='All',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        ),
        # This div is hidden unless "VRU" is selected above.
        html.Div(
            id=f'data_type_vru_options_{prefix}',
            children=[
                dcc.RadioItems(
                    id=f'data_type_selector_vru_{prefix}',
                    options=[
                        {'label': 'All', 'value': 'ALL_VRU'},
                        {'label': 'Bicycle', 'value': 'BICYCLE'},
                        {'label': 'Pedestrian', 'value': 'PEDESTRIAN'}
                    ],
                    value='ALL_VRU',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                )
            ],
            style={'display': 'none'}
        ),
        html.Label('Date Range:', style={'margin-top': '10px'}),
        dcc.DatePickerRange(
            id=f'date_picker_{prefix}',
            start_date='2020-01-01',
            end_date='2023-12-31',
            min_date_allowed=min_date,
            max_date_allowed=max_date,
            display_format='YYYY-MM-DD',
            start_date_placeholder_text='Select a start date',
            style={
                'transform': 'scale(0.8)',
                'transform-origin': 'top left',
                'font-size': '12px',
                'padding': '2px',
                'display': 'inline-block',
                'margin-top': '5px',
                'margin-bottom': '10px',
                'zIndex': '100'
            }
        ),
        html.Label('Time Range (Hour):', style={'margin-top': '20px'}),
        dcc.RangeSlider(
            id=f'time_slider_{prefix}',
            min=0,
            max=23,
            step=1,
            value=[0, 6],
            marks={
                0: '12am', 3: '3am', 6: '6am', 9: '9am', 12: '12pm', 
                15: '3pm', 18: '6pm', 21: '9pm', 23: '11pm'
            },
            tooltip={'always_visible': True}
        ),
        html.Label('Days of the Week:', style={'margin-top': '20px'}),
        dcc.Checklist(
            id=f'day_of_week_checklist_{prefix}',
            options=[
                {'label': 'Mon', 'value': 'Monday'},
                {'label': 'Tue', 'value': 'Tuesday'},
                {'label': 'Wed', 'value': 'Wednesday'},
                {'label': 'Thu', 'value': 'Thursday'},
                {'label': 'Fri', 'value': 'Friday'},
                {'label': 'Sat', 'value': 'Saturday'},
                {'label': 'Sun', 'value': 'Sunday'}
            ],
            value=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            labelStyle={'display': 'inline-block', 'margin-right': '5px', 'font-size': '12px'},
            style={'margin-top': '10px'}
        ),
        html.Label('Select Weather Condition:', style={'margin-top': '20px'}),
        dcc.Dropdown(
            id=f'weather_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': w, 'value': w} for w in unique_weather],
            value='All',
            placeholder='Select weather condition',
            style={'width': '100%'}
        ),
        html.Label('Select Road Surface Condition:', style={'margin-top': '20px'}),
        dcc.Dropdown(
            id=f'road_surface_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': road, 'value': road} for road in unique_road],
            value='All',
            placeholder='Select road surface condition',
            style={'width': '100%'}
        ),
        html.Label('Select Light Condition:', style={'margin-top': '20px'}),
        dcc.Dropdown(
            id=f'light_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': l, 'value': l} for l in unique_light],
            value='All',
            placeholder='Select light condition',
            style={'width': '100%'}
        ),
        
        html.Label('Select Crash Type:', style={'margin-top': '20px'}),
        dcc.Dropdown(
            id=f'crash_type_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': ct,    'value': ct} for ct in unique_crash_types],
            value='All',
            placeholder='Select crash type',
            style={'width': '100%'}
        ),

        html.Label('Select Severity Category:', style={'margin-top': '20px'}),
        dcc.Dropdown(
            id=f'severity_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': v, 'value': v} for v in ['Fatal', 'Non-Fatal']],
            value='All',
            placeholder='Select severity category',
            style={'width': '100%'}
        ),
    ]
    
    if show_buttons:
        if prefix == 'tab1':
            controls += [
                html.Div([
                    dbc.Button('Apply Filter', id=f'apply_filter_{prefix}', n_clicks=0, style={'margin-top': '10px'}),
                    dbc.Button('Clear Drawing', id=f'clear_drawing_{prefix}', n_clicks=0, style={'margin-top': '10px', 'margin-left': '10px'}),
                    dbc.Button('Download Filtered Data', id='download_button_tab1', n_clicks=0, style={'margin-top': '10px', 'display': 'block'})
                ])
            ]
        elif prefix == 'tab5':
            controls += [
                html.Div([
                    dbc.Button('Apply Filter', id=f'apply_filter_{prefix}', n_clicks=0, style={'margin-top': '10px'}),
                    dbc.Button('Clear Drawing', id=f'clear_drawing_{prefix}', n_clicks=0, style={'margin-top': '10px', 'margin-left': '10px'}),
                ])
            ]
        elif prefix == 'tab7':
            controls += [
                html.Div([
                    dbc.Button('Apply Filter', id=f'apply_filter_{prefix}', n_clicks=0, style={'margin-top': '10px'}),
                    dbc.Button('Export to Shapefile', id=f'export_shapefile_{prefix}', n_clicks=0, style={'margin-top': '10px', 'margin-left': '10px'}),
                ])
            ]
        else:
            controls += [
                html.Div([
                    dbc.Button('Apply Filter', id=f'apply_filter_{prefix}', n_clicks=0, style={'margin-top': '10px'})
                ])
            ]
    
    # Remove the inline width, float, and margin settings and use a CSS class instead.
    return html.Div(controls, className='responsive-controls')