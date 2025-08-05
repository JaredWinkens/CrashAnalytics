import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # For more control over layers

# 1. Create a Sample GeoDataFrame (replace with your actual data)
data = {
    'name': ['Point A', 'Point B', 'Point C', 'Point D', 'Point E', 'Point F'],
    'latitude': [40.7128, 40.7200, 40.7150, 40.6900, 40.7300, 40.7129],
    'longitude': [-74.0060, -73.9900, -74.0100, -74.0200, -73.9800, -74.0061]
}
gdf = gpd.GeoDataFrame(
    data,
    geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
    crs="EPSG:4326"  # WGS84 Lat/Lon
)

print("Original GeoDataFrame:")
print(gdf)

# 2. Define the given point
given_point_lat = 40.7128
given_point_lon = -74.0060
given_point = Point(given_point_lon, given_point_lat)

# Convert the given point to a GeoSeries with the correct CRS
given_point_gdf = gpd.GeoDataFrame(geometry=[given_point], crs="EPSG:4326")

# 3. Define the radius (in meters)
radius_meters = 76  # 500 meters

# 4. Reproject to a suitable CRS for buffering (e.g., Web Mercator EPSG:3857)
# Plotly's `px.scatter_mapbox` and `px.choropleth_mapbox` work best with WGS84 (EPSG:4326)
# for displaying, but for accurate buffering, we still need a projected CRS.
# So we'll do the buffering in EPSG:3857 and then convert the buffer back to EPSG:4326
# for Plotly's mapbox.

gdf_projected_for_buffer = gdf.to_crs("EPSG:3857")
given_point_projected_for_buffer = given_point_gdf.to_crs("EPSG:3857")

# 5. Create a buffer around the given point in the projected CRS
buffer_polygon_projected = given_point_projected_for_buffer.geometry.buffer(radius_meters).unary_union

# Convert the buffer polygon back to WGS84 (EPSG:4326) for Plotly
buffer_polygon_wgs84 = gpd.GeoSeries([buffer_polygon_projected], crs="EPSG:3857").to_crs("EPSG:4326").geometry.iloc[0]


# 6. Perform a spatial query to find points within the buffer
points_within_radius = gdf_projected_for_buffer[gdf_projected_for_buffer.intersects(buffer_polygon_projected)]

# Convert the result back to the original CRS (WGS84) for Plotly
points_within_radius_wgs84 = points_within_radius.to_crs("EPSG:4326")

print(f"\nPoints within {radius_meters} meters of ({given_point_lat}, {given_point_lon}):")
print(points_within_radius_wgs84)


# --- Plotting with Plotly ---

# Convert GeoDataFrame to GeoJSON format for Plotly Express to plot polygons
# For the buffer, we need to convert it to a GeoDataFrame first
buffer_gdf_wgs84 = gpd.GeoDataFrame(geometry=[buffer_polygon_wgs84], crs="EPSG:4326")


# Create a base map using Plotly Express
fig = px.scatter_mapbox(
    gdf,
    lat=gdf.geometry.y,
    lon=gdf.geometry.x,
    hover_name="name",
    color_discrete_sequence=["blue"],
    zoom=12,
    height=600,
    title=f'Points within {radius_meters}m Radius (Interactive)'
)

# Add the given point
fig.add_trace(go.Scattermapbox(
    lat=[given_point_lat],
    lon=[given_point_lon],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=15,
        color='red',
        symbol='star'
    ),
    name='Given Point'
))

# Add the buffer polygon as a GeoJSON layer
fig.update_layout(
    mapbox_style="open-street-map", # You can try "carto-positron", "stamen-terrain", etc.
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "geojson",
            "source": buffer_gdf_wgs84.__geo_interface__, # Convert GeoDataFrame to GeoJSON dict
            "type": "fill",
            "color": "rgba(0,100,0,0.3)" # Green with some transparency
        }
    ]
)

# Add the points within the radius
fig.add_trace(go.Scattermapbox(
    lat=points_within_radius_wgs84.geometry.y,
    lon=points_within_radius_wgs84.geometry.x,
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=10,
        color='orange'
    ),
    #hover_name=points_within_radius_wgs84['name'],
    name='Points within Radius'
))

# Update layout for better appearance (optional)
fig.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},
    legend_title_text="Legend"
)

fig.show()