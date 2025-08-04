from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
from app_instance import app, logger, census_polygons_by_county, county_coordinates, get_county_data
import plotly.graph_objects as go  


@app.callback(
    Output('data_type_vru_options_tab3', 'style'),
    Input('data_type_selector_main_tab3', 'value')
)
def toggle_vru_options_tab3(main_value):
    return {'display': 'block'} if main_value == 'VRU' else {'display': 'none'}

@app.callback(
    Output('scatter_map_tab3', 'figure'),
    Input('apply_filter_tab3', 'n_clicks'),
    State('county_selector_tab3', 'value'),
    State('census_attribute_selector', 'value')
)
def update_map_tab3(apply_n_clicks, counties_selected, selected_attribute):
    # Build a uirevision key based on the selected counties
    key = 'tab3-' + '-'.join(sorted(counties_selected or []))

    fig = go.Figure()

    # Determine which counties to draw
    if isinstance(counties_selected, list) and 'All' in counties_selected:
        selected_counties = list(census_polygons_by_county.keys())
    else:
        selected_counties = counties_selected or []

    # Gather values for normalization
    vals = []
    for county in selected_counties:
        for poly in census_polygons_by_county.get(county, []):
            try:
                vals.append(float(poly["properties"].get(selected_attribute, 0)))
            except:
                pass
    min_val, max_val = (min(vals), max(vals)) if vals else (0, 1)

    # Plot each polygon with opacity based on its value
    for county in selected_counties:
        for poly in census_polygons_by_county.get(county, []):
            coords_list = []
            geoms = [poly.get('coordinates')] if poly.get('type') == 'Polygon' else poly.get('coordinates', [])
            for coords in geoms:
                ring = coords[0] if poly.get('type') == 'Polygon' else coords[0]
                if ring[0] != ring[-1]:
                    ring = ring + [ring[0]]
                if len(ring) < 3:
                    continue
                lons, lats = zip(*ring)
                try:
                    norm = (float(poly["properties"].get(selected_attribute, 0)) - min_val) / (max_val - min_val)
                except:
                    norm = 0.5
                opacity = 0.1 + 0.9 * norm
                fig.add_trace(go.Scattermap(
                    lat=list(lats),
                    lon=list(lons),
                    mode='lines',
                    fill='toself',
                    fillcolor=f'rgba(0,255,0,{opacity})',
                    line=dict(color='green', width=2),
                    hoverinfo='skip',
                    showlegend=False
                ))

    # Compute center
    all_lats = [pt for trace in fig.data for pt in trace.lat]
    all_lons = [pt for trace in fig.data for pt in trace.lon]
    if all_lats and all_lons:
        center_lat, center_lon = sum(all_lats)/len(all_lats), sum(all_lons)/len(all_lons)
    else:
        # fallback to first county or NYC
        fallback = county_coordinates.get(selected_counties[0], {'lat': 40.7128, 'lon': -74.0060})
        center_lat, center_lon = fallback['lat'], fallback['lon']

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center={'lat': center_lat, 'lon': center_lon},
            zoom=10
        ),
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        uirevision=key
    )

    return fig


# Callback to Download Filtered Data in Census Data Tab (tab3)
@app.callback(
    Output('download_data_tab3', 'data'),
    [Input('download_button_tab3', 'n_clicks')],
    [
        State('scatter_map_tab3', 'figure'),
        State('county_selector_tab3', 'value')
    ]
)
def download_filtered_data_tab3(n_clicks, figure, counties_selected):
    """
    Download filtered data from Census Data Tab.
    """
    if n_clicks > 0:
        try:
            df = get_county_data(counties_selected)
            if df.empty:
                logger.warning(f"No data available to download for counties: {counties_selected}")
                raise PreventUpdate

            if figure and 'data' in figure and figure['data']:
                selected_case_numbers = []
                for trace in figure['data']:
                    if trace.name.endswith(" Census Polygon"):
                        continue
                    if 'customdata' in trace:
                        for point in trace.customdata:
                            selected_case_numbers.append(point[0])
                if selected_case_numbers:
                    filtered_df = df[df['Case_Number'].isin(selected_case_numbers)]
                    logger.debug(f"Downloading {len(filtered_df)} selected records for Census Data in counties: {counties_selected}")
                else:
                    filtered_df = df
                    logger.debug(f"Downloading all {len(filtered_df)} records for Census Data in counties: {counties_selected}")
            else:
                filtered_df = df

            logger.debug(f"Downloading {len(filtered_df)} records for Census Data in counties: {counties_selected}")
            return dcc.send_data_frame(filtered_df.to_csv, filename="census_filtered_data.csv")
        except Exception as e:
            logger.error(f"Error in download_filtered_data_tab3: {e}")
            raise PreventUpdate
    return None

@app.callback(
    Output('census_color_legend', 'children'),
    [Input('apply_filter_tab3', 'n_clicks')],
    [
        State('county_selector_tab3', 'value'),
        State('census_attribute_selector', 'value')
    ]
)
def update_color_legend(n_clicks, counties_selected, selected_attribute):
    # Determine selected counties.
    if isinstance(counties_selected, list):
        if 'All' in counties_selected:
            selected_counties = list(census_polygons_by_county.keys())
        else:
            selected_counties = counties_selected
    else:
        selected_counties = [counties_selected]

    # Gather all attribute values.
    all_attr_values = []
    for county in selected_counties:
        polygons = census_polygons_by_county.get(county, [])
        if not polygons:
            alternate_key = county + " County"
            polygons = census_polygons_by_county.get(alternate_key, [])
        for poly in polygons:
            try:
                val = float(poly["properties"].get(selected_attribute, 0))
                all_attr_values.append(val)
            except (TypeError, ValueError):
                pass

    if all_attr_values:
        min_val = min(all_attr_values)
        max_val = max(all_attr_values)
    else:
        min_val, max_val = 0, 1

    # Create a gradient legend using a CSS linear-gradient.
    legend = html.Div([
        html.Div("Low: {:.2f}".format(min_val), style={'float': 'left', 'fontSize': '12px'}),
        html.Div("High: {:.2f}".format(max_val), style={'float': 'right', 'fontSize': '12px'}),
        html.Div(style={
            'clear': 'both',
            'height': '20px',
            'background': 'linear-gradient(to right, rgba(0,255,0,0.1), rgba(0,255,0,1))',
            'margin-top': '5px'
        })
    ], style={'margin-top': '10px', 'border': '1px solid #ccc', 'padding': '5px'})
    return legend