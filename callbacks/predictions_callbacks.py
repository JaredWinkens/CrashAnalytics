from dash import dcc, html, Input, Output, State, callback_context, ctx
from dash.exceptions import PreventUpdate
from app_instance import app, logger, DEFAULT_PRED_FILES, ALL_FIELDS, STEPS, REGION_FIELDS
from utils import data_processing
import dash
import os
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go  
import json
import subprocess
import sys

@app.callback(
    Output('comparison_graph', 'figure'),
    Input('predictions_refresh',  'data'),
    State('model_selector_tab4',  'value'),
    State('county_selector_tab4', 'value'),
    State('editable_gpkg_path',   'data'),
)
def update_comparison_graph(refresh, model_file, selected_counties, editable_gpkg_path):
    # choose file‐suffix + default global file
    if model_file == "AI2.py":
        suffix, default_file = "_with_gwr_predictions", DEFAULT_PRED_FILES['AI2.py']
    elif model_file == "mgwr_predict.py":
        suffix, default_file = "_with_mgwr_predictions", DEFAULT_PRED_FILES['mgwr_predict.py']
    else:
        suffix, default_file = "_with_predictions",     DEFAULT_PRED_FILES['AI.py']

    # only splitext when editable_gpkg_path is actually a string
    if isinstance(editable_gpkg_path, str) and editable_gpkg_path:
        base, ext = os.path.splitext(editable_gpkg_path)
        candidate = f"{base}{suffix}{ext}"
        gpkg_file = candidate if os.path.exists(candidate) else default_file
    else:
        gpkg_file = default_file

    # load the GeoPackage
    gdf = gpd.read_file(gpkg_file)

    # apply county filter if requested
    if selected_counties:
        if 'CNTY_NAME' in gdf.columns:
            gdf['CNTY_NAME'] = (
                gdf['CNTY_NAME']
                   .str.replace(" County", "", regex=False)
                   .str.strip()
                   .str.title()
            )
            gdf = gdf[gdf['CNTY_NAME'].isin(selected_counties)]
        elif 'CountyName' in gdf.columns:
            gdf['CountyName'] = (
                gdf['CountyName']
                   .str.strip()
                   .str.title()
            )
            gdf = gdf[gdf['CountyName'].isin(selected_counties)]

    # ensure we have the “Prediction” column
    if 'Prediction' not in gdf.columns:
        return go.Figure()

    # build comparison scatter
    fig = go.Figure()
    if 'AADT Crash Rate' in gdf.columns:
        fig.add_trace(go.Scatter(
            x=gdf['Prediction'],
            y=gdf['AADT Crash Rate'],
            mode='markers',
            name='AADT Crash Rate',
            hovertemplate=(
                "<b>AADT Crash Rate</b><br>"
                "Prediction: %{x:.2f}<br>"
                "Observed: %{y:.2f}<extra></extra>"
            )
        ))
    if 'VRU Crash Rate' in gdf.columns:
        fig.add_trace(go.Scatter(
            x=gdf['Prediction'],
            y=gdf['VRU Crash Rate'],
            mode='markers',
            name='VRU Crash Rate',
            hovertemplate=(
                "<b>VRU Crash Rate</b><br>"
                "Prediction: %{x:.2f}<br>"
                "Observed: %{y:.2f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Model Prediction vs. Observed Crash Rates",
        xaxis_title="Prediction",
        yaxis_title="Crash Rate",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

# Global variable to store mapping from polygon id to county name.
county_mapping = {}
@app.callback(
    Output('predictions_map', 'figure'),
    [
        Input('refresh_predictions_tab4', 'n_clicks'),
        Input('county_selector_tab4', 'value'),
        Input('predictions_refresh', 'data'),
        Input('model_selector_tab4', 'value')
    ],
    State('editable_gpkg_path', 'data')
)
def update_predictions_map(n_clicks, selected_counties, refresh_trigger, model_file, editable_gpkg_path):
    # build a uirevision key so that changing only predictions doesn't move the map
    counties_key = '-'.join(sorted(selected_counties or []))
    key = f"tab4-{model_file}-{counties_key}"

    # pick which gpkg & column
    if model_file == "AI2.py":
        suffix, pred_col, default_file = "_with_gwr_predictions",  "GWR_Prediction", DEFAULT_PRED_FILES['AI2.py']
    elif model_file == "mgwr_predict.py":
        suffix, pred_col, default_file = "_with_mgwr_predictions", "MGWR_Prediction", DEFAULT_PRED_FILES['mgwr_predict.py']
    else:
        suffix, pred_col, default_file = "_with_predictions",     "Prediction",      DEFAULT_PRED_FILES['AI.py']

    # decide which file to load (only split if we actually have a string path)
    if isinstance(editable_gpkg_path, str) and editable_gpkg_path:
        base, ext = os.path.splitext(editable_gpkg_path)
        candidate = f"{base}{suffix}{ext}"
        gpkg_file = candidate if os.path.exists(candidate) else default_file
    else:
        gpkg_file = default_file


    try:
        gdf = gpd.read_file(gpkg_file)
        if pred_col not in gdf.columns:
            raise KeyError(f"Missing '{pred_col}' in {gpkg_file}")
        gdf['Prediction'] = gdf[pred_col]

        # normalize county names & filter
        # ——— filter by selected_counties, using whichever county field exists ———
        if selected_counties:
            # AI/GWR outputs
            if 'CNTY_NAME' in gdf.columns:
                gdf['CNTY_NAME'] = (
                    gdf['CNTY_NAME']
                       .str.replace(" County", "", regex=False)
                       .str.strip()
                       .str.title()
                )
                gdf = gdf[gdf['CNTY_NAME'].isin(selected_counties)]

            # MGWR outputs use CountyName
            elif 'CountyName' in gdf.columns:
                gdf['CountyName'] = (
                    gdf['CountyName']
                       .str.replace(" County", "", regex=False)
                       .str.strip()
                       .str.title()
                )
                gdf = gdf[gdf['CountyName'].isin(selected_counties)]

            # else: no county column, so skip filtering entirely

        valid   = gdf[~gdf['Prediction'].isna()]
        missing = gdf[ gdf['Prediction'].isna()]

        fig = go.Figure([
            go.Choroplethmap(
                geojson=json.loads(missing.to_json()),
                locations=missing['id'],
                z=[0]*len(missing),
                colorscale=[[0,"black"],[1,"black"]],
                marker_opacity=0.9,
                marker_line_width=1,
                showscale=False,
                featureidkey="properties.id",
                name="Missing",
                hovertemplate="<extra></extra>" 
            ),
                go.Choroplethmap(
                geojson=json.loads(valid.to_json()),
                locations=valid['id'],
                z=valid['Prediction'],
                colorscale='YlGnBu',
                marker_opacity=0.6,
                marker_line_width=1,
                colorbar=dict(title="Prediction"),
                featureidkey="properties.id",
                name="Prediction",
                hovertemplate=
                "<b>Tract %{location}</b><br>" +
                "Predicted Crash Rate: %{z:.2f}<extra></extra>"
            )
        ])

        # compute center only if there's data
        if not gdf.empty:
            ctr = gdf.geometry.centroid
            center = {'lat': ctr.y.mean(), 'lon': ctr.x.mean()}
        else:
            center = {'lat': 40.7128, 'lon': -74.0060}

        fig.update_layout(
            map=dict(
                style="open-street-map",
                center=center,
                zoom=10
            ),
            margin={'l':0,'r':0,'t':0,'b':0},
            legend=dict(x=0, y=1),
            uirevision=key
        )
        return fig

    except Exception as e:
        logger.error(f"Error in update_predictions_map: {e}", exc_info=True)
        # fallback blank map, also preserving camera if possible
        fig = go.Figure(go.Scattermap(
            lat=[40.7128], lon=[-74.0060],
            mode='markers', marker=dict(opacity=0)
        ))
        fig.update_layout(
            map=dict(
                style="open-street-map",
                center={'lat':40.7128,'lon':-74.0060},
                zoom=10
            ),
            margin={'l':0,'r':0,'t':0,'b':0},
            annotations=[dict(
                text="Error loading predictions map",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False
            )],
            uirevision=key
        )
        return fig


# -----------------------------------------------------------------------------
# 1) update_modal_values: reload on tract/model/refresh/gpkg change, else bump values
# -----------------------------------------------------------------------------
# 1) Grab your field IDs from ALL_FIELDS:
FIELD_IDS = [ var for var, _, _ in ALL_FIELDS ]

# 2) Build the decorator args:
modal_value_outputs = [
    Output(f"input_{var}", "value")
    for var in FIELD_IDS
]

modal_value_inputs = (
    [ Input("selected_census_tract", "data") ]
  + sum([[ 
       Input(f"plus_input_{var}",  "n_clicks"),
       Input(f"minus_input_{var}", "n_clicks")
    ] for var in FIELD_IDS], [])
)

modal_value_states = (
    [ State("editable_gpkg_path", "data") ]
  + [ State(f"input_{var}", "value") for var in FIELD_IDS ]
)

@app.callback(
    *modal_value_outputs,
    *modal_value_inputs,
    *modal_value_states,
    prevent_initial_call=True
)
def update_modal_values(*all_args):
    """
    On tract‐click: load every field from the editable GPKG.
    On plus/minus: bump only that one field by its step, leave the rest unchanged.
    """
    num = len(FIELD_IDS)
    # Inputs are: 1 selected_tract + 2*num clicks
    trigger_args = all_args[: 1 + 2 * num]
    # States are: 1 gpkg_path + num current values
    state_args   = all_args[1 + 2 * num :]

    selected_tract = trigger_args[0]
    gpkg_path      = state_args[0]
    current_vals   = list(state_args[1:])  # length == num

    trig = callback_context.triggered[0]["prop_id"]

    def clean(v):
        return None if pd.isna(v) else round(v, 2)

    # --- Case 1: new tract selected → load all fields from the GPKG
    if trig.startswith("selected_census_tract"):
        if not (selected_tract and gpkg_path and os.path.exists(gpkg_path)):
            raise PreventUpdate
        gdf = gpd.read_file(gpkg_path)
        rename_map = {
            v.replace('(','.').replace(')','.'): v
            for v in FIELD_IDS
        }
        gdf = gdf.rename(columns=rename_map)
        row = gdf[gdf["id"].astype(str) == str(selected_tract)]
        if row.empty:
            raise PreventUpdate
        row = row.iloc[0]
        return [ clean(row.get(var)) for var in FIELD_IDS ]

    # --- Case 2: plus/minus clicked → find exactly which var to adjust
    # Map each plus/minus input to its index and sign:
    deltas = {}
    for idx, var in enumerate(FIELD_IDS):
        if trig.startswith(f"plus_input_{var}.n_clicks"):
            deltas[idx] = STEPS[var]
        elif trig.startswith(f"minus_input_{var}.n_clicks"):
            deltas[idx] = -STEPS[var]

    if not deltas:
        # no recognized trigger → do nothing
        raise PreventUpdate

    # apply the single delta and return all values
    new_vals = current_vals.copy()
    for idx, delta in deltas.items():
        base = new_vals[idx] or 0
        new_vals[idx] = round(base + delta, 2)

    return new_vals

@app.callback(
    Output('county_selector_tab4', 'options'),
    [
        Input('refresh_predictions_tab4', 'n_clicks'),
        Input('model_selector_tab4',    'value')
    ]
)
def update_county_options(n_clicks, model_file):
    try:
        if model_file == "AI2.py":
            gpkg_file = DEFAULT_PRED_FILES['AI2.py']
        elif model_file == "MGWR.py":
            gpkg_file = DEFAULT_PRED_FILES['MGWR.py']
        else:
            gpkg_file = DEFAULT_PRED_FILES['AI.py']
        gdf = gpd.read_file(gpkg_file)
        # Log the original column values for debugging
        logger.debug("Original CNTY_NAME values: " + str(gdf['CNTY_NAME'].unique()))
        # Remove the " County" suffix and standardize
        gdf['CNTY_NAME'] = gdf['CNTY_NAME'].str.replace(" County", "", regex=False).str.strip().str.title()
        unique_counties = sorted(gdf['CNTY_NAME'].dropna().unique().tolist())
        logger.debug("Unique counties after formatting: " + str(unique_counties))
        options = [{'label': county, 'value': county} for county in unique_counties]
        return options
    except Exception as e:
        logger.error(f"Error updating county selector options: {e}")
        return []



@app.callback(
    Output('county_edit_modal', 'style'),
    [Input('open_edit_modal', 'n_clicks'),
     Input('close_modal', 'n_clicks'),
     Input('apply_updated_data', 'n_clicks')],
    State('county_edit_modal', 'style')
)
def toggle_modal(open_clicks, close_clicks, apply_clicks, current_style):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'open_edit_modal':
        # Show the modal with styling consistent with your project.
        return {
            'display': 'block',
            'position': 'fixed',
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)',
            'padding': '20px',
            'backgroundColor': 'white',
            'border': '2px solid black',
            'zIndex': 1000
        }
    else:
        # Hide the modal on Close or after applying changes.
        return {'display': 'none'}


@app.callback(
    Output('selected_census_tract', 'data'),
    Output('editable_gpkg_path',    'data'),
    Output('predictions_refresh',   'data'),
    Output('county_selector_tab4',  'value'),
    Input('predictions_map',        'clickData'),
    Input('county_selector_tab4',   'value'),
    Input('apply_updated_data',     'n_clicks'),
    Input('reset_predictions',      'n_clicks'),
    Input('model_selector_tab4',    'value'),
    State('selected_census_tract',  'data'),
    State('editable_gpkg_path',     'data'),
    # all 24 field‐States in the right order
    State('input_DEMOGIDX_5',                        'value'),
    State('input_PEOPCOLORPCT',                      'value'),
    State('input_UNEMPPCT',                          'value'),
    State('input_pct_residential',                   'value'),
    State('input_pct_industrial',                    'value'),
    State('input_pct_retail',                        'value'),
    State('input_pct_commercial',                    'value'),
    State('input_AADT',                              'value'),
    State('input_Commute_TripMiles_TripStart_avg',   'value'),
    State('input_Commute_TripMiles_TripEnd_avg',     'value'),
    State('input_ACSTOTPOP',                         'value'),
    State('input_DEMOGIDX_2',                        'value'),
    State('input_PovertyPop',                        'value'),
    State('input_DISABILITYPCT',                     'value'),
    State('input_BikingTrips(Start)',                'value'),
    State('input_BikingTrips(End)',                  'value'),
    State('input_CarpoolTrips(Start)',               'value'),
    State('input_CarpoolTrips(End)',                 'value'),
    State('input_CommercialFreightTrips(Start)',     'value'),
    State('input_CommercialFreightTrips(End)',       'value'),
    State('input_WalkingTrips(Start)',               'value'),
    State('input_WalkingTrips(End)',                 'value'),
    State('input_PublicTransitTrips(Start)',         'value'),
    State('input_PublicTransitTrips(End)',           'value'),
    State('predictions_refresh',                     'data'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def manage_editable_and_predictions(
    clickData,
    county_val,
    apply_n, reset_n, model_file,
    selected_tract, gpkg_path,
    demogidx_5, peopcolorpct, unemppct,
    pct_residential, pct_industrial, pct_retail, pct_commercial,
    aadt, commute_start, commute_end,
    acstotpop, demogidx_2, poverty_pop, disabilitypct,
    biking_start, biking_end,
    carpool_start, carpool_end,
    freight_start, freight_end,
    walking_start, walking_end,
    transit_start, transit_end,
    current_refresh
):



    trig = ctx.triggered_id

    # 1) Map click just update selected_census_tract
    if trig == 'predictions_map':
        if clickData and clickData.get('points'):
            tract = clickData['points'][0]['location']
            return tract, dash.no_update, dash.no_update, dash.no_update
        raise PreventUpdate


    # Model swapclear everything so user must re-choose county
    if trig == 'model_selector_tab4':
        # delete any existing editable & pred files but only if gpkg_path is really a string
        if isinstance(gpkg_path, str) and os.path.exists(gpkg_path):
            os.remove(gpkg_path)
            base, ext = os.path.splitext(gpkg_path)
            suffix = (
                '_with_gwr_predictions'   if model_file == 'AI2.py' else
                '_with_mgwr_predictions'  if model_file == 'mgwr_predict.py' else
                '_with_predictions'
            )
            pred_p = f"{base}{suffix}{ext}"
            if os.path.exists(pred_p):
                os.remove(pred_p)
        # clear out both the county dropdown and the tract store
        return None, None, (current_refresh or 0) + 1, []

    # 3) New county selected  copy per county file
    if trig == 'county_selector_tab4':
        if county_val and len(county_val)==1:
            cnty = county_val[0]
            if model_file=='AI2.py':
                src, dst = DEFAULT_PRED_FILES['AI2.py'],    './AI/'
            elif model_file=='mgwr_predict.py':
                src, dst = DEFAULT_PRED_FILES['mgwr_predict.py'],'./MGWR/'
            else:
                src, dst = DEFAULT_PRED_FILES['AI.py'],     './AI/'
            new_path = data_processing.copy_county_gpkg(cnty, src, dst)
            return (
          dash.no_update,     # selected_tract
          new_path,           # editable_gpkg_path
          dash.no_update,     # predictions_refresh
          dash.no_update      # county_selector itself
        )
        raise PreventUpdate

    # 4) Apply edits → write GPKG, rerun model, bump
    if trig == 'apply_updated_data':
        if apply_n and selected_tract and gpkg_path:
            gdf = gpd.read_file(gpkg_path)
            idx = gdf[gdf['id']==selected_tract].index
            if idx.empty:
                raise PreventUpdate

            # — your 24 field‐writes —
            gdf.loc[idx, 'DEMOGIDX_5']                      = demogidx_5
            gdf.loc[idx, 'PEOPCOLORPCT']                    = peopcolorpct
            gdf.loc[idx, 'UNEMPPCT']                        = unemppct
            gdf.loc[idx, 'pct_residential']                 = pct_residential
            gdf.loc[idx, 'pct_industrial']                  = pct_industrial
            gdf.loc[idx, 'pct_retail']                      = pct_retail
            gdf.loc[idx, 'pct_commercial']                  = pct_commercial
            gdf.loc[idx, 'AADT']                            = aadt
            gdf.loc[idx, 'Commute_TripMiles_TripStart_avg'] = commute_start
            gdf.loc[idx, 'Commute_TripMiles_TripEnd_avg']   = commute_end
            gdf.loc[idx, 'ACSTOTPOP']                       = acstotpop
            gdf.loc[idx, 'DEMOGIDX_2']                      = demogidx_2
            gdf.loc[idx, 'PovertyPop']                      = poverty_pop
            gdf.loc[idx, 'DISABILITYPCT']                   = disabilitypct
            gdf.loc[idx, 'BikingTrips(Start)']              = biking_start
            gdf.loc[idx, 'BikingTrips(End)']                = biking_end
            gdf.loc[idx, 'CarpoolTrips(Start)']             = carpool_start
            gdf.loc[idx, 'CarpoolTrips(End)']               = carpool_end
            gdf.loc[idx, 'CommercialFreightTrips(Start)']   = freight_start
            gdf.loc[idx, 'CommercialFreightTrips(End)']     = freight_end
            gdf.loc[idx, 'WalkingTrips(Start)']             = walking_start
            gdf.loc[idx, 'WalkingTrips(End)']               = walking_end
            gdf.loc[idx, 'PublicTransitTrips(Start)']       = transit_start
            gdf.loc[idx, 'PublicTransitTrips(End)']         = transit_end

            # overwrite and re‐run
            gdf.to_file(gpkg_path, driver="GPKG")
            base, ext = os.path.splitext(gpkg_path)
            suffix   = '_with_mgwr_predictions' if model_file=='mgwr_predict.py' else '_with_predictions'
            out_file = f"{base}{suffix}{ext}"
            subprocess.run([sys.executable, model_file, gpkg_path, out_file],
                           check=True, capture_output=True, text=True)

            return dash.no_update, dash.no_update, (current_refresh or 0) + 1, dash.no_update

        raise PreventUpdate

    # Reset button (delete & recopy or it breaks)
    if trig == 'reset_predictions' and reset_n:
        # delete editable + pred only if gpkg_path is a valid path
        if isinstance(gpkg_path, str) and os.path.exists(gpkg_path):
            os.remove(gpkg_path)
            base, ext = os.path.splitext(gpkg_path)
            suffix = '_with_mgwr_predictions' if model_file == 'mgwr_predict.py' else '_with_predictions'
            pred_p = f"{base}{suffix}{ext}"
            if os.path.exists(pred_p):
                os.remove(pred_p)

        # immediately recopy so user can click again
        new_path = None
        if county_val and len(county_val) == 1:
            cnty = county_val[0]
            if model_file == 'AI2.py':
                src, dst = DEFAULT_PRED_FILES['AI2.py'],    './AI/'
            elif model_file == 'mgwr_predict.py':
                src, dst = DEFAULT_PRED_FILES['mgwr_predict.py'], './MGWR/'
            else:
                src, dst = DEFAULT_PRED_FILES['AI.py'],     './AI/'
            new_path = data_processing.copy_county_gpkg(cnty, src, dst)

            return (dash.no_update,  new_path, (current_refresh or 0) + 1, dash.no_update)

    # fallback if it all breaks
    raise PreventUpdate



@app.callback(
    [Output(f"container_{var}", "style") for var, _, _ in ALL_FIELDS],
    [
        Input("model_selector_tab4",       "value"),
        Input("selected_census_tract",     "data"),
        Input("editable_gpkg_path",        "data"),
    ],
    prevent_initial_call=True,
)
def toggle_field_styles(model_file, selected_tract, gpkg_path):
    # AI and GWR always show all fields (can add more here later)
    if model_file in ("AI.py", "AI2.py"):
        return [{'marginBottom': '10px'}] * len(ALL_FIELDS)

    # MGWR hide everything until a tract is selected and GPkg exists
    if not (selected_tract and gpkg_path and os.path.exists(gpkg_path)):
        return [{'display': 'none'}] * len(ALL_FIELDS)

    tract_id = str(selected_tract)

    # Load the per-county GPKG and get its Region column
    gdf = gpd.read_file(gpkg_path)
    if "Region" not in gdf.columns:
        return [{'display': 'none'}] * len(ALL_FIELDS)

    match = gdf.loc[gdf['id'].astype(str) == tract_id, "Region"]
    if match.empty:
        return [{'display': 'none'}] * len(ALL_FIELDS)

    region = match.iat[0]
    allowed = set(REGION_FIELDS.get(region, []))

    # Build and return the style list based on region
    return [
        {'marginBottom': '10px'} if var in allowed else {'display': 'none'}
        for var, _, _ in ALL_FIELDS
    ]



# --- store_original_prediction ---
@app.callback(
    Output('original_prediction', 'data'),
    [
        Input('selected_census_tract', 'data'),
        Input('model_selector_tab4',  'value')
    ],
    State('editable_gpkg_path', 'data')
)
def store_original_prediction(tract_id, model_file, gpkg_path):
    if not (tract_id and gpkg_path):
        raise PreventUpdate

    # pick suffix + column
    if model_file == "AI2.py":
        suffix, col = "_with_gwr_predictions",  "GWR_Prediction"
    elif model_file == "mgwr_predict.py":
        suffix, col = "_with_mgwr_predictions", "MGWR_Prediction"
    else:
        suffix, col = "_with_predictions",     "Prediction"

    base, ext = os.path.splitext(gpkg_path)
    county_pred = f"{base}{suffix}{ext}"

    # load per‐county if it exists, otherwise global
    if os.path.exists(county_pred):
        gdf = gpd.read_file(county_pred)
    else:
        gdf = gpd.read_file(DEFAULT_PRED_FILES[model_file])
    # if the prediction column itself is missing, bail out
    if col not in gdf.columns:
        return None

    # in store_original_prediction
    subset = gdf.loc[gdf['id'].astype(str) == str(tract_id), col]
    if subset.empty:
        return None
    return subset.iloc[0]

# Single callback to drive the prediction bar for both ForestISO and GWR
@app.callback(
    Output('prediction_bar', 'children'),
    [
        Input('original_prediction',    'data'),
        Input('predictions_refresh',    'data'),
        Input('model_selector_tab4',    'value')
    ],
    [
        State('editable_gpkg_path',     'data'),
        State('selected_census_tract',  'data')
    ]
)
def update_prediction_bar(original_prediction, refresh_val, model_file, gpkg_path, selected_tract):
    # if nothing selected yet
    if not selected_tract or not gpkg_path:
        return "No census tract selected."

    # choose file suffix and column name based on model
    if model_file == "AI2.py":
        suffix, col = "_with_gwr_predictions",  "GWR_Prediction"
    elif model_file == "mgwr_predict.py":
        suffix, col = "_with_mgwr_predictions", "MGWR_Prediction"
    else:
        suffix, col = "_with_predictions",     "Prediction"

    base, ext = os.path.splitext(gpkg_path)
    county_pred_file = f"{base}{suffix}{ext}"

    # load the correct GeoPackage
    if os.path.exists(county_pred_file):
        gdf = gpd.read_file(county_pred_file)
    else:
        # fall back to the global default
        gdf = gpd.read_file(DEFAULT_PRED_FILES[model_file])

    # if column missing
    if col not in gdf.columns:
        return f"Model '{model_file}' has no '{col}' column."

    # grab the current value for this tract
    try:
        current_val = gdf.loc[gdf['id'] == selected_tract, col].iloc[0]
    except Exception:
        return "Selected tract not found."

    # helper for formatting
    def fmt(x):
        try:
            return f"{float(x):.2f}"
        except:
            return str(x)

    return html.Div([
        html.Div(f"Original Prediction: {fmt(original_prediction)}"),
        html.Div(f"Current Prediction:  {fmt(current_val)}"),
    ])