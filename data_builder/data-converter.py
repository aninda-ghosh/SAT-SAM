import cv2
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

root_path = "./dataset/"

parcel_data = gpd.read_file("parcel_data.geojson")

def convert_polygons_to_pixels(parcel_geometry, polygon_labels, size):
        # Create a black background
        background = np.zeros(size, dtype=np.uint8)
         
        # Get the polygon bounds
        min_lon, min_lat, max_lon, max_lat = parcel_geometry.bounds

        # Get the width and height of the parcel
        height = max_lat - min_lat
        width = max_lon - min_lon

        # Get the pixel width and height
        pixel_height = size[0]/height
        pixel_width = size[1]/width

        pixel_masks = []
        for label in polygon_labels.geometry:
            pixel_mask = background.copy()
            # Iterate over the polygons in the multipolygon
            for polygon in label.geoms:
                coords = polygon.exterior.coords.xy

                # Get the pixel coords
                pixel_coords = []
                for i in range(len(coords[0])):
                    x = coords[0][i]
                    y = coords[1][i]

                    # Get the pixel x and y
                    pixel_x = (x - min_lon) * pixel_width
                    pixel_y = (max_lat - y) * pixel_height

                    pixel_coords.append([pixel_x, pixel_y])

                # Convert to int to make it work with cv2
                pixel_coords = np.array(pixel_coords, dtype=np.int32)
                # Fill the polygons and append to the pixel mask
                cv2.fillPoly(pixel_mask, pts=[pixel_coords], color=255)
            pixel_masks.append(pixel_mask)
        
        # Return the pixel masks as a numpy array
        return pixel_masks


# create directory for the images, masks and geojson files if they don't exist
import os
if not os.path.exists('images'):
    os.makedirs('images')
if not os.path.exists('masks'):
    os.makedirs('masks')
if not os.path.exists('geojsons'):
    os.makedirs('geojsons')


for _, data in parcel_data.iterrows():
    parcel_id = data['parcel_id']
    geometry = data['geometry']

    print('Processing:' + parcel_id)

    polygon_mask_list = gpd.read_file(parcel_id + '.geojson')

    # If the parcel has no labels, let's keep it with no labels and no prompts
    if len(polygon_mask_list) == 0:
        pass
        # compressed_mask = np.zeros((448, 448), dtype=np.uint8)
    else:
        # Get the pixel masks for the parcel
        pixel_masks = convert_polygons_to_pixels(geometry, polygon_mask_list, (448, 448))
        
        #Compress the pixel masks into a single mask by combining with instance ids as value
        compressed_mask = np.zeros((448, 448), dtype=np.uint8)
        for i, mask in enumerate(pixel_masks):
            compressed_mask[mask > 0] = i + 1
    
        # Move the image to the images folder
        os.rename(parcel_id + '.png', 'images/' + parcel_id + '.png')
        # Save the compressed mask
        cv2.imwrite('masks/' + parcel_id + '.png', compressed_mask)
        # Save the geojson
        os.rename(parcel_id + '.geojson', 'geojsons/' + parcel_id + '.geojson')



# Read the masks folder and get the file names
import glob
mask_files = glob.glob('masks/*.png')

# Create a new text file to store the file names of the images and masks and geojsons
with open('data.txt', 'w') as f:
    for mask_file in mask_files:
        # Get the image file name
        image_file = mask_file.replace('masks', 'images').replace('.png', '.png')
        # Get the geojson file name
        geojson_file = mask_file.replace('masks', 'geojsons').replace('.png', '.geojson')
        # Write the image and mask file names to the text file
        f.write(image_file + ',' + mask_file + ',' + geojson_file + '\n')
