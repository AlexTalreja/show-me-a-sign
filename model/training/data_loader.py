import os
import shutil
from PIL import Image

SOURCE_DIR = 'src/aslwireframedataset'
DEST_DIR = 'src/aslwireframemodified'
MAX_INDEX = 1000

# Create destination folder if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

for label_folder in os.listdir(SOURCE_DIR):
    label_path = os.path.join(SOURCE_DIR, label_folder)
    if not os.path.isdir(label_path):
        continue

    label = label_folder.upper()
    dest_label_path = os.path.join(DEST_DIR, label)
    dest_flipped_path = os.path.join(DEST_DIR, f"{label}_flipped")

    os.makedirs(dest_label_path, exist_ok=True)
    os.makedirs(dest_flipped_path, exist_ok=True)

    for i in range(MAX_INDEX + 1):
        for ext in ['.png', '.jpg', '.jpeg']:
            filename = f"{label}{i}{ext}"
            src_file = os.path.join(label_path, filename)
            if os.path.exists(src_file):
                # Copy original image
                dst_file = os.path.join(dest_label_path, filename)
                shutil.copy2(src_file, dst_file)

                # Mirror image across Y-axis and save in flipped folder
                with Image.open(src_file) as img:
                    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                    flipped_name = f"{label}{i}{ext}"
                    flipped.save(os.path.join(dest_flipped_path, flipped_name))
