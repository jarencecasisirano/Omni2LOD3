import os
import requests
import osmnx as ox

def get_road_nodes_around_building(lat, lng, radius_meters=30):
    """
    Uses OpenStreetMap to find all drivable road nodes within a specific 
    radius of a target coordinate.
    """
    print(f"Fetching drivable roads within {radius_meters}m of ({lat}, {lng})...")
    try:
        # Fetch the street network graph for vehicles within the radius
        # network_type='drive' ensures we get roads Street View cars likely visited
        G = ox.graph_from_point((lat, lng), dist=radius_meters, network_type='drive')
        
        # Extract the nodes (points/intersections) from the graph
        nodes = ox.graph_to_gdfs(G, edges=False)
        
        # Convert the GeoDataFrame rows into a list of (latitude, longitude) tuples
        # Note: OSM uses (x, y) for (longitude, latitude)
        road_coordinates = [(row.y, row.x) for idx, row in nodes.iterrows()]
        
        print(f"  -> Found {len(road_coordinates)} road nodes.")
        return road_coordinates

    except Exception as e:
        print(f"  -> No drivable roads found within {radius_meters}m, or an error occurred: {e}")
        return []

def download_surrounding_street_view(api_key, coordinates, save_dir):
    """
    Downloads Street View images for a list of coordinates at 4 different headings.
    """
    if not coordinates:
        print("No coordinates provided to download. Exiting.")
        return

    os.makedirs(save_dir, exist_ok=True)
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    headings = [0, 90, 180, 270]

    for index, (lat, lng) in enumerate(coordinates):
        print(f"Processing node {index + 1}/{len(coordinates)}: ({lat:.5f}, {lng:.5f})")
        
        for heading in headings:
            params = {
                "size": "640x640",
                "location": f"{lat},{lng}",
                "fov": 120,
                "heading": heading,
                "pitch": 0,
                "key": api_key,
                "return_error_code": "true"
            }

            response = requests.get(base_url, params=params)

            if response.status_code == 200:
                filename = f"sv_{lat:.5f}_{lng:.5f}_h{heading}.jpg"
                filepath = os.path.join(save_dir, filename)

                with open(filepath, 'wb') as file:
                    file.write(response.content)
                print(f"  -> Saved: {filename}")
            elif response.status_code == 404:
                print(f"  -> No imagery available for this specific heading/location.")
            else:
                print(f"  -> Failed. HTTP Status Code: {response.status_code}")


# --- Configuration & Execution ---

if __name__ == "__main__":
    # 1. Your Google API Key
    API_KEY = "AIzaSyBQNmEhhsUlTaveWeICQQ0XWb038AisdQQ"
    
    # 2. Your Saving Directory
    SAVE_DIRECTORY = "data/google-street-view/"
    
    # 3. Target Building Coordinates
    # Replace these with the center coordinate of your target building
    BUILDING_LAT = 14.65088087648159  # Example: Latitude
    BUILDING_LNG = 121.07311957218278 # Example: Longitude
    SEARCH_RADIUS = 30      # Radius in meters

    # Step 1: Automatically find road coordinates around the building
    road_coords = get_road_nodes_around_building(BUILDING_LAT, BUILDING_LNG, radius_meters=SEARCH_RADIUS)

    # Step 2: Download the Street View images for those coordinates
    download_surrounding_street_view(API_KEY, road_coords, SAVE_DIRECTORY)