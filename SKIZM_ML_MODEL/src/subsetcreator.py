import os
import shutil

# Configuration
source_folder_path = '/home/shki/Documents/CoCoDataset/train2017'  # Replace with the path to your source folder
images_per_subfolder = 500  # Number of images per subfolder


def split_folder_into_subfolders(source_folder, images_per_folder):
    # Supported image formats
    image_formats = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    # Create a list of all image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(image_formats)]

    # Calculate the number of subfolders needed
    total_images = len(image_files)
    num_subfolders = (total_images + images_per_folder - 1) // images_per_folder

    # Create and fill subfolders with images
    for i in range(num_subfolders):
        # Define the subfolder name
        subfolder_name = os.path.join(source_folder, f'Batch_{i + 1}')
        os.makedirs(subfolder_name, exist_ok=True)

        # Move the images to the subfolder
        for j in range(images_per_folder):
            if image_files:
                current_image = image_files.pop(0)
                shutil.move(os.path.join(source_folder, current_image), subfolder_name)
            else:
                break  # Break if there are no more images to move

    print(f"Successfully split {total_images} images into {num_subfolders} subfolders.")


# Run the script
split_folder_into_subfolders(source_folder_path, images_per_subfolder)
