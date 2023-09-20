import os
from PIL import Image
import fnmatch

start_dir = './datasets/PetImages/'

# Specify the patterns to match image files (e.g., '*.jpg' and '*.png' for JPEG and PNG images)
patterns = ['*.jpg', '*.png']

# Initialize a list to store the paths of corrupted image files
corrupted_images = []

# Recursively search for image files with multiple patterns
for root, _, files in os.walk(start_dir):
    for pattern in patterns:
        for filename in fnmatch.filter(files, pattern):
            # Construct the full path to the image file
            image_path = os.path.join(root, filename)

            try:
                print(image_path)
                with Image.open(image_path):
                    pass
            except Exception as e:
                # An error occurred while opening the image, indicating it's corrupted
                print(f"Corrupted image: {image_path}")
                corrupted_images.append(image_path)

# Print a summary of corrupted images
if corrupted_images:
    print(f"Total corrupted images found: {len(corrupted_images)}")
else:
    print("No corrupted images found.")