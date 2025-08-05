from dash import dcc, html, Input, Output, State, callback_context, dash_table
from dash.exceptions import PreventUpdate
from app_instance import app, logger, available_intersection_classes, available_road_classes, county_coordinates, get_county_data
from utils import helper_functions
from crash_rate_analysis import main
import plotly.graph_objects as go  
import datetime
import geopandas as gpd
import pandas as pd
import tempfile
import os
import shutil
import dash_bootstrap_components as dbc

@app.callback(
    Output('data_type_vru_options_tab7', 'style'),
    Input('data_type_selector_main_tab7', 'value')
)
def toggle_vru_options_tab7(main_value):
    return {'display': 'block'} if main_value == 'VRU' else {'display': 'none'}

@app.callback(
    [
        Output('formula-button', 'title'),
        Output('select_functional_class_tab7', 'options'),
        Output('select_functional_class_tab7', 'value')
    ],
    Input('analysis_selector_tab7', 'value')
)
def update_formula_btn_info(analysis_type):
    if analysis_type == 'Segment':
        options = [{'label': 'All', 'value': 'All'}] + [{'label': f_class, 'value': f_class} for f_class in available_road_classes]
        value = [available_road_classes[0]]
        return """Segment Crash Rate Formula:

        Rseg = C x 10^6 / AADT x 365 x T x L

    Where:
        Rseg = Segment Crash Rate per million vehicle miles of travel
        C = Crashes during analysis period
        AADT = Annual Average Daily Traffic (veh./day)
        T = Study Period (yrs.)
        L = Length of the Segment (mi)
        """, options, value
    elif analysis_type == 'Intersection':
        options = [{'label': 'All', 'value': 'All'}] + [{'label': f_class, 'value': f_class} for f_class in available_intersection_classes]
        value = [available_intersection_classes[0]]
        return """Intersection Crash Rate Formula:

        Rint = C x 10^6 / DEV x 365 x T

    Where:
        Rint = Intersection crash rate (per million entering vehicles)
        C = Crashes during analysis period
        DEV = Daily Entering Volume (veh./day)
        T = Study Period (yrs.)
        """, options, value
    else:
        return "", []

@app.callback(
    [
        Output('scatter_map_tab7', 'figure'),
        Output('scatter_map_tab7', 'selectedData'),
        Output('crash-rate-content', 'children'),
        Output('filtered_data_tab7', 'data')
    ],
    [
        Input('apply_filter_tab7', 'n_clicks'),
    ],
    [
        State('scatter_map_tab7', 'selectedData'),
        State('select_functional_class_tab7', 'value'),
        State('county_selector_tab7', 'value'),
        State('date_picker_tab7', 'start_date'),
        State('date_picker_tab7', 'end_date'),
        State('time_slider_tab7', 'value'),
        State('day_of_week_checklist_tab7', 'value'),
        State('weather_selector_tab7', 'value'),
        State('light_selector_tab7', 'value'),
        State('road_surface_selector_tab7', 'value'),
        State('severity_selector_tab7','value'),
        State('data_type_selector_main_tab7', 'value'),
        State('data_type_selector_vru_tab7', 'value'),
        State('crash_type_selector_tab7','value'),
        State('analysis_selector_tab7', 'value')
    ]
)
def map_tab7(apply_n_clicks, selected_data, func_class_selected, counties_selected,
            start_date, end_date, time_range, days_of_week, weather, light, road_surface, severity_category, 
            main_data_type, vru_data_type, crash_type, analysis_type):
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
    key = 'tab7-' + '-'.join(sorted(counties_selected or []))

    table = None

    # empty‐data fallback
    if df.empty:
        fig = go.Figure()

        fig.add_trace(go.Scattermap(
            mode="markers",
            lon=[lon_center],
            lat=[lat_center],
            marker=dict(
                size=10,
                color='red', # Highlight crashes on lines in red
                opacity=0.9,
                symbol='circle'
            ),
            name="GeoDataFrame Lines"
        ))
        fig.update_layout(
            map=dict(
                style="open-street-map",
                center={'lat': lat_center, 'lon': lon_center},
                zoom=10
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
        )
        fig.update_layout(uirevision=key)
        return fig, None, None, None
    
    # apply filters
    if triggered == 'apply_filter_tab7':
        filtered_crashes = helper_functions.filter_data_tab1(
            df, start_date, end_date, time_range,
            days_of_week, weather, light, road_surface,  severity_category, crash_type,
            main_data_type, vru_data_type, 
        )
        
        format_string = "%Y-%m-%d"
        min_year = datetime.datetime.strptime(start_date, format_string).year
        max_year = datetime.datetime.strptime(end_date, format_string).year
        selected_years = list(range(min_year, max_year + 1))
        study_period = len(selected_years)

        if analysis_type == "Segment":
            fig, data = main.do_segment_analysis(counties_selected, func_class_selected, study_period, filtered_crashes)
            top_segments = gpd.read_file(data['road_segments'])
            top_segments = top_segments.nlargest(n=10, columns='crash_rate')
            top_segments = top_segments.rename(columns={'AADT_Stats_2023_Table_Description': 'description', 'AADT_Stats_2023_Table_AADT': 'AADT'})
            selected_cols = top_segments[['unique_id', 'description', 'AADT', 'road_length_mi', 'crash_count', 'crash_rate']]
            table = html.Div([
                html.H1('Top 10 Segments', style={'margin-top': '10px'}),
                dbc.Table.from_dataframe(selected_cols, striped=True, bordered=True, hover=True)
            ])

        elif analysis_type == "Intersection":
            fig, data = main.do_intersection_analysis(counties_selected, func_class_selected, study_period, filtered_crashes)
            top_intersections = gpd.read_file(data['intersections'])
            top_intersections = top_intersections.nlargest(n=10, columns='crash_rate')
            top_intersections = top_intersections.rename(columns={'node_id': 'intersection_id'})
            selected_cols = top_intersections[['intersection_id', 'DEV', 'crash_count', 'crash_rate']]
            table = html.Div([
                html.H1('Top 10 Intersections', style={'margin-top': '10px'}),
                dbc.Table.from_dataframe(selected_cols, striped=True, bordered=True, hover=True)
            ])
        out_selected = selected_data

    else:  # initial_load
        fig = go.Figure()

        fig.add_trace(go.Scattermap(
            mode="markers",
            lon=[lon_center],
            lat=[lat_center],
            marker=dict(
                size=10,
                color='red', # Highlight crashes on lines in red
                opacity=0.9,
                symbol='circle'
            ),
            name="GeoDataFrame Lines"
        ))
        fig.update_layout(
            map=dict(
                style="open-street-map",
                center={'lat': lat_center, 'lon': lon_center},
                zoom=10
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            title="MultiLineStrings from GeoDataFrame"
        )
        fig.update_layout(uirevision=key)
        return fig, None, None, None   

    fig.update_layout(uirevision=key)
    return fig, out_selected, table, data

# Callback to Download Shapefile in Crash Rate Tab (tab7)
@app.callback(
    Output('download_data_tab7', 'data'),
    [
        Input('export_shapefile_tab7', 'n_clicks')
    ],
    [
        State('analysis_selector_tab7', 'value'),
        State('filtered_data_tab7', 'data'),
    ],
    prevent_initial_call=True
)

def download_filtered_data_tab7(n_clicks, analysis_type, shapefile_data):
    if n_clicks > 0 and shapefile_data != None:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:

                for key, val in shapefile_data.items():
                    filename = key
                    filepath = os.path.join(temp_dir, filename)
                    gdf = gpd.read_file(val)
                    gdf.to_file(filepath, driver='ESRI Shapefile')

                zip_file_name = 'crash_rate_analysis.zip'
                zip = shutil.make_archive(zip_file_name.split('.')[0], 'zip', temp_dir)
                
                return dcc.send_file(zip)
                
            # # If a box selection exists, further filter by the selected points.
            # if selected_data and 'points' in selected_data and selected_data['points']:
            #     selected_case_numbers = [point['customdata'][0] for point in selected_data['points']]
            #     filtered_df = filtered_df[filtered_df['Case_Number'].isin(selected_case_numbers)]
            #     logger.debug(f"Downloading {len(filtered_df)} records after box selection filtering.")
            # else:
            #     logger.debug(f"Downloading all {len(filtered_df)} records after applying filters.")
        except Exception as e:
            logger.error(f"Error in download_filtered_data_tab7: {e}")
            raise PreventUpdate
    return None