from app_instance import app, make_field_row, ALL_FIELDS, LABELS, STEPS, county_coordinates, available_road_classes, data_final_df, unique_counties, unique_crash_types, unique_light, unique_road, unique_weather
from dash import dcc, html, Output, Input
import dash_bootstrap_components as dbc
from layouts import census_controls, common_controls, census_data_layout, chatbot_layout, crash_analyzer_layout, crash_rate_layout, data_download_layout, heatmap_layout, predictions_layout
from callbacks import census_data_callbacks, chatbot_callbacks, crash_rate_callbacks, data_download_callbacks, heatmap_callbacks, predictions_callbacks, streetview_callbacks

available_counties = list(county_coordinates.keys())
available_func_classes = available_road_classes
min_date = data_final_df['Crash_Date'].min()
max_date = data_final_df['Crash_Date'].max()
initial_bot_message = "Hello! I am an interactive safety chatbot designed to provide you with real-time, data-driven insights on roadway safety. Whether you seek information about high-risk areas, traffic incident trends, or general road safety guidance, I will offer reliable and context-aware responses.\n\n" \
        "**Example Prompts**\n\n" \
        "- What are the top 5 cities with the most crashes in 2021, showing counts?\n\n" \
        "- Describe a typical crash at an intersection.\n\n" \
        "- Plot all pedestrian-related crashes in Buffalo. \n\n"

tab1 = data_download_layout.load_data_download_layout(available_counties, unique_weather, unique_light, unique_road, unique_crash_types, county_coordinates, common_controls.common_controls)
tab2 = heatmap_layout.load_heatmap_layout(available_counties, unique_weather, unique_light, unique_road, unique_crash_types, county_coordinates, common_controls.common_controls)
tab3 = census_data_layout.load_census_data_layout(census_controls.census_controls, county_coordinates, available_counties)
tab4 = predictions_layout.load_predictions_layout(make_field_row, LABELS, STEPS, ALL_FIELDS)
tab5 = crash_analyzer_layout.load_crash_analyzer_layout(available_counties, unique_weather, unique_light, unique_road, unique_crash_types, county_coordinates, common_controls.common_controls)
tab6 = chatbot_layout.load_chatbot_layout([{"sender": "bot", "message": initial_bot_message, "map": None, "loading": False}])
tab7 = crash_rate_layout.load_crash_rate_layout(available_counties, available_func_classes, county_coordinates, unique_weather, unique_light, unique_road, unique_crash_types, min_date, max_date, common_controls.common_controls)

app.layout = html.Div([
    # Navigation Tabs
    dbc.Tabs(id='tabs', children=[
        dbc.Tab(tab1, label='Data Downloader'),
        dbc.Tab(tab2, label='Heatmap'),
        dbc.Tab(tab3, label='Census Data'),
        dbc.Tab(tab4, label='Predictions'),
        dbc.Tab(tab5, label='Street View Analyzer'),
        dbc.Tab(tab6, label='Safety ChatBot'),
        dbc.Tab(tab7, label='Crash Rate Analysis')
    ], style={'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'zIndex': '1000'}),

    #Dynamic Content Based on Selected Tab
    html.Div([
        html.Div(id='tabs-content', style={'margin-top': '160px'})
    ]),

    # Download Components
    dcc.Download(id='download_data'),
    dcc.Download(id='download_data_tab3'),
    dcc.Download(id='download_data_tab7'),   
    
    dcc.Store(id='editable_gpkg_path'),
    dcc.Store(id='selected_census_tract'),
    dcc.Store(id='predictions_refresh', data=0),
    html.Button(
        id='refresh_predictions_tab4',
        n_clicks=0,
        style={'display': 'none'}
    ),
    
    html.Div(
    html.Button('Edit Selected Census Tract', id='open_edit_modal', n_clicks=0),
        style={'display': 'none'}
    ),

    # Modal for editing county data:
    html.Div(
        id='county_edit_modal',
        children=[
            html.H3("Edit County Data"),
            html.Div(
                id="modal_fields_container",
                children=[
                    make_field_row(var, LABELS[var], STEPS[var])
                    for var, _, _ in ALL_FIELDS
                ]
            ),
            html.Div([
                html.Button(
                    "Apply Updated Data",
                    id="apply_updated_data",
                    n_clicks=0,
                    style={'marginRight': '10px'}
                ),
                #unused but do not remove or else everything breaks (i keep removing it)
                html.Button(
                    "Reset Predictions",
                    id="reset_predictions",
                    n_clicks=0,
                    style={'marginRight': '10px'}
                ),
                html.Button(
                    "Close",
                    id="close_modal",
                    n_clicks=0
                ),
            ], style={'marginBottom': '10px', 'textAlign': 'center'})
        ],
        style={
            'display': 'none',
            'position': 'fixed',
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)',
            'padding': '20px',
            'backgroundColor': 'white',
            'border': '2px solid black',
            'zIndex': 1000
        }
    )
])

# Callback to render content based on selected tab
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    available_counties = list(county_coordinates.keys())
    available_func_classes = available_road_classes
    min_date = data_final_df['Crash_Date'].min()
    max_date = data_final_df['Crash_Date'].max()
    initial_bot_message = "Hello! I am an interactive safety chatbot designed to provide you with real-time, data-driven insights on roadway safety. Whether you seek information about high-risk areas, traffic incident trends, or general road safety guidance, I will offer reliable and context-aware responses.\n\n" \
            "**Example Prompts**\n\n" \
            "- What are the top 5 cities with the most crashes in 2021, showing counts?\n\n" \
            "- Describe a typical crash at an intersection.\n\n" \
            "- Plot all pedestrian-related crashes in Buffalo. \n\n"
    # Ensure that the unique lists are available (they must be defined globally after loading the data)
    global unique_weather, unique_light, unique_road

    if tab == 'tab-1': # Data Downloader Tab
        return data_download_layout.load_data_download_layout(available_counties, unique_weather, unique_light, unique_road, unique_crash_types, county_coordinates, common_controls.common_controls)

    elif tab == 'tab-2': # Heatmap Tab
        return heatmap_layout.load_heatmap_layout(available_counties, unique_weather, unique_light, unique_road, unique_crash_types, county_coordinates, common_controls.common_controls)

    elif tab == 'tab-3': # Census Data Tab
        return census_data_layout.load_census_data_layout(census_controls.census_controls, county_coordinates, available_counties)

    elif tab == 'tab-4':  # Predictions Tab
        return predictions_layout.load_predictions_layout(make_field_row, LABELS, STEPS, ALL_FIELDS)
    
    elif tab == 'tab-5': # Crash Analyzer
        return crash_analyzer_layout.load_crash_analyzer_layout(available_counties, unique_weather, unique_light, unique_road, unique_crash_types, county_coordinates, common_controls.common_controls)
    
    elif tab == 'tab-6': # Chatbot Tab
        return chatbot_layout.load_chatbot_layout([{"sender": "bot", "message": initial_bot_message, "map": None, "loading": False}])
    
    elif tab == 'tab-7': # Chatbot Tab
        return crash_rate_layout.load_crash_rate_layout(available_counties, available_func_classes, county_coordinates, unique_weather, unique_light, unique_road, unique_crash_types, min_date, max_date, common_controls.common_controls)

# Run the app
if __name__ == '__main__':
    app.run(port="8050", debug=True)