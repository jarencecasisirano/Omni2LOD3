import os
import requests
import math
from PIL import Image
from io import BytesIO

# --- CONFIGURATION ---
API_KEY = "AIzaSyBQNmEhhsUlTaveWeICQQ0XWb038AisdQQ"
SAVE_DIR = "data/google-street-view/"
# Central coordinate of the building
BUILDING_LAT = 14.6508720400293
BUILDING_LNG = 121.07311040797163
RADIUS_METERS = 50 # Distance from the building to look for roads

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_surrounding_coords(center_lat, center_lng, radius, steps=4):
    """Generates coordinates in a circle around the center point."""
    coords = []
    earth_radius = 6378137.0 # Earth's radius in meters
    
    for i in range(steps):
        angle = math.pi * 2 * i / steps
        # Calculate coordinate offsets
        d_lat = radius * math.cos(angle) / earth_radius
        d_lng = radius * math.sin(angle) / (earth_radius * math.cos(math.pi * center_lat / 180))
        
        new_lat = center_lat + (d_lat * 180 / math.pi)
        new_lng = center_lng + (d_lng * 180 / math.pi)
        coords.append((new_lat, new_lng))
        
    return coords

def download_360_panorama(lat, lng, index):
    """Downloads 4 directional images and stitches them into a 360 panorama."""
    headings = [0, 90, 180, 270]
    images = []
    
    print(f"Fetching panorama for location {index} ({lat}, {lng})...")
    
    for heading in headings:
        # return_error_code=true ensures we get a 404 if no image exists, rather than a blank grey image
        url = (f"https://maps.googleapis.com/maps/api/streetview"
               f"?size=640x640&location={lat},{lng}&heading={heading}"
               f"&pitch=0&fov=90&source=outdoor&return_error_code=true&key={API_KEY}")
        
        response = requests.get(url)
        
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            images.append(img)
        else:
            print(f"  -> No Street View data found at heading {heading}°.")
            return False

    if len(images) == 4:
        # Stitch images horizontally
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        
        stitched_img = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        
        for img in images:
            stitched_img.paste(img, (x_offset, 0))
            x_offset += img.width
            
        filename = f"pano_loc_{index}_{lat:.5f}_{lng:.5f}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        stitched_img.save(filepath)
        print(f"  -> Successfully saved 360 view: {filename}")
        return True
    return False

def main():
    # 1. Generate points around the building
    print("Generating coordinates around the building...")
    surrounding_points = generate_surrounding_coords(BUILDING_LAT, BUILDING_LNG, RADIUS_METERS, steps=6)
    
    # 2. Add the building's exact coordinate just in case the road is right next to it
    surrounding_points.insert(0, (BUILDING_LAT, BUILDING_LNG))
    
    # 3. Fetch and stitch images for each point
    success_count = 0
    for idx, (lat, lng) in enumerate(surrounding_points):
        success = download_360_panorama(lat, lng, idx)
        if success:
            success_count += 1
            
    print(f"\nFinished! Successfully downloaded {success_count} 360-degree panoramas to {SAVE_DIR}.")

if __name__ == "__main__":
    main()