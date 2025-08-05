from app_instance import app, data_final_df, logger, get_county_data, county_coordinates
from dash import ctx, html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from utils import helper_functions
import streetview_analyzer.streetview_analyzer as streetview
import dash_pannellum
import dash
import pandas as pd
import plotly.express as px

@app.callback(
    Output('image-popup-tab5', 'children'),
    Output('image-popup-tab5', 'is_open'),
    Output('scatter_map_tab5', 'clickData'),
    Input('scatter_map_tab5', 'clickData'),
    State('filtered_data_tab5', 'data'),     
    prevent_initial_call=True
)
def display_streetview_popup(clickData, filtered_data):
    triggered_id = ctx.triggered_id
    
    if triggered_id == 'scatter_map_tab5':
        
        lon = clickData['points'][0]['lon']
        lat = clickData['points'][0]['lat']
        location_name = streetview.get_location_name(lat, lon)
        
        image_meta = streetview.get_street_view_image_metadata(lat, lon)

        image_bytes = streetview.stitch_street_view_images_from_lat_lon(lat, lon, [0, 45, 90, 135, 180, 225, 270, 315], "640x640", 90,".png")
        data_uri = streetview.encode_image_to_base64_data_uri(image_bytes, format="png")

        caseNumber = clickData['points'][0]['customdata'][0]
        crash = data_final_df.query(f"Case_Number == '{caseNumber}'")

        historical_data = streetview.get_historical_crash_data(77, lat, lon, pd.DataFrame(filtered_data))
        
        analysis = streetview.analyze_image_ai(image_bytes, image_meta, crash.to_string(), historical_data.to_string())

        image_element = [
            dbc.ModalHeader(dbc.ModalTitle(location_name)),
            dbc.ModalBody(html.Div([
                dcc.Markdown(analysis),
                dash_pannellum.DashPannellum(
                    id='partial-panorama-component',
                    tour={
                        "default": {
                            "firstScene": "scene1",
                            "sceneFadeDuration": 1000
                        },
                        "scenes": {
                            "scene1": {
                                "hfov": 110,
                                "pitch": -3,
                                "yaw": 117,
                                "type": "equirectangular",
                                "panorama": data_uri
                            }
                        }
                    },
                width='100%',
                height='400px',)
            ], style={'width': '100%', 'margin': '0 auto', 'padding': '5px'}))
        ]
        print(f"Displaying popup for {location_name}.")
        return image_element, True, None
    
    # Fallback or initial state
    print("No valid trigger for display/hide, returning no_update.")
    return dash.no_update, False, None

@app.callback(
    Output('data_type_vru_options_tab5', 'style'),
    Input('data_type_selector_main_tab5', 'value')
)
def toggle_vru_options_tab5(main_value):
    return {'display': 'block'} if main_value == 'VRU' else {'display': 'none'}

@app.callback(
    [
        Output('scatter_map_tab5', 'figure'),
        Output('scatter_map_tab5', 'selectedData'),
        Output('filtered_data_tab5', 'data')
    ],
    [
        Input('apply_filter_tab5', 'n_clicks'),
        Input('clear_drawing_tab5', 'n_clicks'),
    ],
    [
        State('county_selector_tab5', 'value'),
        State('scatter_map_tab5', 'selectedData'),
        State('date_picker_tab5', 'start_date'),
        State('date_picker_tab5', 'end_date'),
        State('time_slider_tab5', 'value'),
        State('day_of_week_checklist_tab5', 'value'),
        State('weather_selector_tab5', 'value'),
        State('light_selector_tab5', 'value'),
        State('road_surface_selector_tab5', 'value'),
        State('severity_selector_tab5','value'),
        State('data_type_selector_main_tab5', 'value'),
        State('data_type_selector_vru_tab5', 'value'),
        State('crash_type_selector_tab5','value')
    ]
)
def map_tab5(apply_n_clicks, clear_n_clicks, counties_selected, selected_data,
             start_date, end_date, time_range, days_of_week,
            weather, light, road_surface, severity_category, main_data_type, vru_data_type, crash_type):
    ctx = callback_context
    triggered = (
        ctx.triggered[0]['prop_id'].split('.')[0]
        if ctx.triggered else 'initial_load'
    )
    logger.debug(f"Triggered Input: {triggered}")

    # load data
    df = get_county_data(counties_selected)

    # compute default center
    if isinstance(counties_selected, list) and counties_selected and 'All' not in counties_selected:
        lat_center = sum(county_coordinates[c]['lat'] for c in counties_selected) / len(counties_selected)
        lon_center = sum(county_coordinates[c]['lon'] for c in counties_selected) / len(counties_selected)
    else:
        lat_center, lon_center = 40.7128, -74.0060

    # helper to set uirevision key
    key = 'tab5-' + '-'.join(sorted(counties_selected or []))

    # empty‐data fallback
    if df.empty:
        fig = px.scatter_map(
            pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
            lat='Latitude', lon='Longitude', zoom=10, map_style="open-street-map"
        )
        fig.update_traces(marker=dict(opacity=0))
        fig.update_layout(uirevision=key)
        return fig, None, None

    # apply filters
    if triggered == 'apply_filter_tab5':
        filtered = helper_functions.filter_data_tab1(
            df, start_date, end_date, time_range,
            days_of_week, weather, light, road_surface,  severity_category, crash_type,
            main_data_type, vru_data_type, 
        )
        if selected_data and 'points' in selected_data:
            keep = [pt['customdata'][0] for pt in selected_data['points']]
            filtered = filtered[filtered['Case_Number'].isin(keep)]
        df_to_plot = filtered
        out_selected = selected_data
        if crash_type and crash_type != 'All':
            df_to_plot = df_to_plot[df_to_plot['Crash_Type'] == crash_type]

    elif triggered == 'clear_drawing_tab5':
        # reapply filters but drop box selection
        df_to_plot = helper_functions.filter_data_tab1(
            df, start_date, end_date, time_range,
            days_of_week, weather, light, road_surface, severity_category,
            main_data_type, vru_data_type, crash_type
        )
        out_selected = None

    else:  # initial_load
        df_to_plot = helper_functions.filter_data_tab1(
            df, '1900-01-01', '1901-01-01', [0, 23],
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
            'All','All','All', 'All', 'All',
            main_data_type, vru_data_type
        )
        out_selected = None

    # build the figure
    if df_to_plot.empty:
        fig = px.scatter_map(
            pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
            lat='Latitude', lon='Longitude', zoom=10, map_style="open-street-map"
        )
        fig.update_traces(marker=dict(opacity=0))
    else:
        fig = px.scatter_map(
            df_to_plot,
            lat='Y_Coord', lon='X_Coord', zoom=10, map_style="open-street-map",
            hover_name='Case_Number',
            hover_data={
                'Crash_Date': True, 'Crash_Time': True,
                'WeatherCon': True, 'LightCon': True,
                'RoadSurfac': True
            },
            custom_data=['Case_Number']
        )
        fig.update_layout(map_center={'lat': lat_center, 'lon': lon_center})

    fig.update_layout(uirevision=key)
    return fig, out_selected, df_to_plot.to_dict()