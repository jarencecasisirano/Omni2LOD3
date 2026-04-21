import os
import requests
import math
import argparse
from PIL import Image
from io import BytesIO

# --- CONFIGURATION ---
API_KEY = "AIzaSyBQNmEhhsUlTaveWeICQQ0XWb038AisdQQ"
SAVE_DIR = "data/google-street-view4/"

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_grid_coords(coords, spacing_meters):
    """Generates a grid of coordinates within the bounding box defined by coords."""
    lats = [c[0] for c in coords]
    lngs = [c[1] for c in coords]
    min_lat, max_lat = min(lats), max(lats)
    min_lng, max_lng = min(lngs), max(lngs)
    
    # 1 degree of latitude is approximately 111,139 meters
    lat_step = spacing_meters / 111139.0
    
    # 1 degree of longitude is approximately 111,139 * cos(latitude) meters
    avg_lat = (min_lat + max_lat) / 2.0
    lng_step = spacing_meters / (111139.0 * math.cos(math.radians(avg_lat)))
    
    grid_coords = []
    current_lat = min_lat
    while current_lat <= max_lat + (lat_step * 0.1): # add small epsilon for float precision
        current_lng = min_lng
        while current_lng <= max_lng + (lng_step * 0.1):
            grid_coords.append((current_lat, current_lng))
            current_lng += lng_step
        current_lat += lat_step
        
    return grid_coords

def download_gsv_tiles(lat, lng, index):
    """Downloads 6 directional images (N, E, S, W, Zenith, Nadir) as individual tiles."""
    directions = [
        (0, 0, 'front'),
        (90, 0, 'right'),
        (180, 0, 'back'),
        (270, 0, 'left'),
        (0, 90, 'zenith'),
        (0, -90, 'nadir')
    ]
    
    print(f"Fetching tiles for location {index} ({lat:.5f}, {lng:.5f})...")
    
    saved_count = 0
    for heading, pitch, name in directions:
        filename = f"tile_loc_{index}_{lat:.5f}_{lng:.5f}_{name}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        if os.path.exists(filepath):
            print(f"  -> Skipping existing tile: {filename}")
            saved_count += 1
            continue

        # return_error_code=true ensures we get a 404 if no image exists
        url = (f"https://maps.googleapis.com/maps/api/streetview"
               f"?size=640x640&location={lat},{lng}&heading={heading}"
               f"&pitch={pitch}&fov=90&source=outdoor&return_error_code=true&key={API_KEY}")
        
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            saved_count += 1
        else:
            print(f"  -> No data found for {name} tile (heading {heading}°, pitch {pitch}°).")
            if name == 'front':
                # If we fail on the first direction, assume location has no GSV at all
                return False

    if saved_count > 0:
        print(f"  -> Successfully saved {saved_count} tiles.")
        return True
    return False

def parse_coordinate(coord_str):
    """Parses a 'lat,lng' string into a tuple of floats."""
    try:
        lat, lng = map(float, coord_str.split(','))
        return (lat, lng)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid coordinate format '{coord_str}'. Expected 'lat,lng'.")

def main():
    parser = argparse.ArgumentParser(description="Extract GSV images within a bounding box.")
    # Require 4 coordinate pairs to form a bounding box/polygon
    parser.add_argument('--bbox', type=parse_coordinate, nargs=4, required=True,
                        help="Four coordinate pairs (lat,lng) defining the bounding box, separated by spaces. Example: --bbox 14.65,121.07 14.66,121.07 14.66,121.08 14.65,121.08")
    parser.add_argument('--spacing', type=float, default=20.0,
                        help="Distance in meters between sampled points inside the bounding box. Default is 20m.")
    
    args = parser.parse_args()

    #ICHEM
    # --bbox 14.6512979774915,121.07263700126747 14.651324132684486,121.07391102928595 14.650470794112406,121.0738299207335 14.650467525315392,121.07256603630232
    
    # 1. Generate grid points inside the bounding box
    print(f"Generating coordinates within the bounding box with a spacing of {args.spacing} meters...")
    grid_points = generate_grid_coords(args.bbox, args.spacing)
    print(f"Total points to check: {len(grid_points)}")
    
    # 2. Fetch images for each point as individual tiles
    success_count = 0
    for idx, (lat, lng) in enumerate(grid_points):
        success = download_gsv_tiles(lat, lng, idx)
        if success:
            success_count += 1
            
    print(f"\nFinished! Successfully downloaded tiles for {success_count} locations to {SAVE_DIR}.")

if __name__ == "__main__":
    main()