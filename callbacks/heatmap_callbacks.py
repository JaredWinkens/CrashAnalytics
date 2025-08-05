from app_instance import app, get_county_data, county_coordinates
from dash import ctx, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from heatmap_analyzer import map_analyzer
import dash
import base64
import plotly.io as pio
import plotly.express as px
import pandas as pd
from utils import helper_functions

@app.callback(
    Output('image-popup-tab2', 'children'),
    Output('image-popup-tab2', 'is_open'),
    Input('insight-button', 'n_clicks'), 
    State('heatmap_graph', 'figure'),      
    prevent_initial_call=True
)
def display_insight_popup(insight_button_n_clicks, fig_snapshot):
    triggered_id = ctx.triggered_id

    if triggered_id == 'insight-button' and insight_button_n_clicks and insight_button_n_clicks > 0:
        fig_width = 1280
        fig_height = 720
        #pio.write_image(fig_snapshot, "density_map.png", scale=1, width=fig_width, height=fig_height)
        image_bytes = pio.to_image(fig=fig_snapshot, format='png', scale=1, width=fig_width, height=fig_height)
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        data_url = f"data:image/png;base64,{encoded_image}"
        insights = map_analyzer.generate_response(image_bytes)
        
        image_element = [
            dbc.ModalHeader(dbc.ModalTitle("Image Insights")),
            dbc.ModalBody(html.Div([
                dcc.Markdown(insights),
                html.Img(src=data_url, style={'width': '100%', 'height': '400px'})
            ],style={'width': '100%', 'margin': '0 auto', 'padding': '5px'})),
        ]
        return image_element, True
    
    # Fallback or initial state
    print("No valid trigger for display/hide, returning no_update.")
    return dash.no_update, False

@app.callback(
    Output('data_type_vru_options_tab2', 'style'),
    Input('data_type_selector_main_tab2', 'value')
)
def toggle_vru_options_tab2(main_value):
    return {'display': 'block'} if main_value == 'VRU' else {'display': 'none'}

def filter_data_tab2(df, data_type):
    """
    Filter data based on data type for Heatmap Tab.

    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        data_type (str): 'All' or 'VRU'.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if data_type != 'All':
        df = df[df['Data_Type'] == data_type]
    return df

@app.callback(
    Output('heatmap_graph', 'figure'),
    Input('apply_filter_tab2', 'n_clicks'),
    State('radius_slider_tab2', 'value'),
    State('county_selector_tab2', 'value'),
    State('data_type_selector_main_tab2', 'value'),
    State('data_type_selector_vru_tab2', 'value'),
    State('severity_selector_tab2', 'value'),
    State('date_picker_tab2', 'start_date'),
    State('date_picker_tab2', 'end_date'),
    State('time_slider_tab2', 'value'),
    State('day_of_week_checklist_tab2', 'value'),
    State('weather_selector_tab2', 'value'),
    State('light_selector_tab2', 'value'),
    State('road_surface_selector_tab2', 'value'),
    State('crash_type_selector_tab2','value')
)
def update_heatmap_tab2(n_clicks, radius_miles, counties_selected,
                        main_data_type, vru_data_type,
                        severity_category,
                        start_date, end_date, time_range,
                        days_of_week, weather, light, road_surface, crash_type):
    zoom = 10
    default_center = {'lat': 40.7128, 'lon': -74.0060}
    key = 'tab2-' + '-'.join(sorted(counties_selected or []))

    # BEFORE FIRST CLICK: show a blank‐centered “dot”
    if not n_clicks:
        radius_px = helper_functions.convert_miles_to_pixels(radius_miles, zoom, default_center['lat'])
        fig = px.density_map(
            pd.DataFrame(default_center, index=[0]),
            lat='lat', lon='lon',
            radius=radius_px,
            center=default_center, zoom=zoom,
            map_style="open-street-map", opacity=0.7
        )
        fig.update_layout(uirevision=key)
        return fig

    # LOAD YOUR DATA
    df = get_county_data(counties_selected)
    if df.empty:
        # same blank‐dot fallback
        radius_px = helper_functions.convert_miles_to_pixels(radius_miles, zoom, default_center['lat'])
        fig = px.density_map(
            pd.DataFrame(default_center, index=[0]),
            lat='lat', lon='lon',
            radius=radius_px,
            center=default_center, zoom=zoom,
            map_style="open-street-map", opacity=0.7
        )
        fig.update_layout(uirevision=key)
        return fig

    # APPLY *exactly* the same filters as Tab1, *including* VRU sub‐type
    filtered = helper_functions.filter_data_tab1(
        df,
        start_date, end_date, time_range,
        days_of_week, weather, light, road_surface,
        severity_category, crash_type,
        main_data_type, vru_data_type, 
    )

    # COMPUTE CENTER
    if filtered.empty:
        center_lat, center_lon = default_center.values()
    elif 'All' in counties_selected:
        center_lat, center_lon = df['Y_Coord'].mean(), df['X_Coord'].mean()
    else:
        pts = [county_coordinates[c] for c in counties_selected]
        center_lat = sum(p['lat'] for p in pts) / len(pts)
        center_lon = sum(p['lon'] for p in pts) / len(pts)


    # RE‐COMPUTE PIXEL RADIUS AT NEW CENTER
    radius_px = helper_functions.convert_miles_to_pixels(radius_miles, zoom, center_lat)

    # BUILD THE DENSITY MAP
    fig = px.density_map(
        filtered,
        lat='Y_Coord', lon='X_Coord',
        radius=radius_px,
        center={'lat': center_lat, 'lon': center_lon},
        zoom=zoom,
        map_style="open-street-map",
        opacity=0.5,
        hover_data={
            'Case_Number': True,
            'Crash_Date': True,
            'Crash_Time': True,
            'WeatherCon': True,
            'LightCon': True,
            'RoadSurfac': True
        }
    )
    fig.update_layout(uirevision=key)
    return fig