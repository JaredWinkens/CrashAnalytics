from dash import dcc, html, Input, Output, State, callback_context
from app_instance import app, logger, county_coordinates, get_county_data
import plotly.express as px
import pandas as pd
from dash.exceptions import PreventUpdate
from utils import helper_functions


@app.callback(
    Output('data_type_vru_options_tab1', 'style'),
    Input('data_type_selector_main_tab1', 'value')
)
def toggle_vru_options_tab1(main_value):
    return {'display': 'block'} if main_value == 'VRU' else {'display': 'none'}

@app.callback(
    [
        Output('scatter_map', 'figure'),
        Output('scatter_map', 'selectedData')
    ],
    [
        Input('apply_filter_tab1', 'n_clicks'),
        Input('clear_drawing_tab1', 'n_clicks'),
    ],
    [
        State('county_selector_tab1', 'value'),
        State('scatter_map', 'selectedData'),
        State('date_picker_tab1', 'start_date'),
        State('date_picker_tab1', 'end_date'),
        State('time_slider_tab1', 'value'),
        State('day_of_week_checklist_tab1', 'value'),
        State('weather_selector_tab1', 'value'),
        State('light_selector_tab1', 'value'),
        State('road_surface_selector_tab1', 'value'),
        State('severity_selector_tab1','value'),
        State('data_type_selector_main_tab1', 'value'),
        State('data_type_selector_vru_tab1', 'value'),
        State('crash_type_selector_tab1','value')
    ]
)
def map_tab1(apply_n_clicks, clear_n_clicks, counties_selected, selected_data,
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
    key = 'tab1-' + '-'.join(sorted(counties_selected or []))

    # empty‐data fallback
    if df.empty:
        fig = px.scatter_map(
            pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
            lat='Latitude', lon='Longitude', zoom=10, map_style="open-street-map"
        )
        fig.update_traces(marker=dict(opacity=0))
        fig.update_layout(uirevision=key)
        return fig, None

    # apply filters
    if triggered == 'apply_filter_tab1':
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

    elif triggered == 'clear_drawing_tab1':
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
    return fig, out_selected


# Callback to Download Filtered Data in Data Downloader Tab (tab1)
@app.callback(
    Output('download_data', 'data'),
    [Input('download_button_tab1', 'n_clicks')],
    [
        State('county_selector_tab1', 'value'),
        State('date_picker_tab1', 'start_date'),
        State('date_picker_tab1', 'end_date'),
        State('time_slider_tab1', 'value'),
        State('day_of_week_checklist_tab1', 'value'),
        State('weather_selector_tab1', 'value'),
        State('light_selector_tab1', 'value'),
        State('road_surface_selector_tab1', 'value'),
        State('severity_selector_tab1','value'),
        State('data_type_selector_main_tab1', 'value'),
        State('data_type_selector_vru_tab1', 'value'),
        State('crash_type_selector_tab1','value'),
        State('scatter_map', 'selectedData')
    ]
)
def download_filtered_data_tab1(n_clicks, counties_selected, start_date, end_date, time_range, days_of_week,
                                weather, light, road_surface, severity_category, main_data_type, vru_data_type, crash_type, selected_data):
    if n_clicks > 0:
        try:
            # Load full data for the selected counties.
            df = get_county_data(counties_selected)
            if df.empty:
                logger.warning(f"No data available to download for counties: {counties_selected}")
                raise PreventUpdate

            # Apply the same filters as in update_map_tab1.
            filtered_df = helper_functions.filter_data_tab1(
                df, start_date, end_date, time_range,
                days_of_week, weather, light, road_surface, severity_category, crash_type,
                main_data_type, vru_data_type
            )

            # If a box selection exists, further filter by the selected points.
            if selected_data and 'points' in selected_data and selected_data['points']:
                selected_case_numbers = [point['customdata'][0] for point in selected_data['points']]
                filtered_df = filtered_df[filtered_df['Case_Number'].isin(selected_case_numbers)]
                logger.debug(f"Downloading {len(filtered_df)} records after box selection filtering.")
            else:
                logger.debug(f"Downloading all {len(filtered_df)} records after applying filters.")

            return dcc.send_data_frame(filtered_df.to_csv, filename="filtered_data.csv")
        except Exception as e:
            logger.error(f"Error in download_filtered_data_tab1: {e}")
            raise PreventUpdate
    return None