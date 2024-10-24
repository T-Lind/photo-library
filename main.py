import os
import lancedb
import pyarrow as pa
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from exif import Image as ExifImage
from datetime import datetime
from pillow_heif import register_heif_opener
import io

# Register HEIF opener with Pillow
register_heif_opener()

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dims = 512

uri = "data/photos-1"
db = lancedb.connect(uri)

# Constructs tables
people_schema = pa.schema([
    pa.field("people_id", pa.int32()),
    pa.field("name", pa.string()),
])
people_tbl = db.create_table("people", schema=people_schema)

imgs_schema = pa.schema([
    pa.field("image_id", pa.int32()),
    pa.field("vector", pa.list_(pa.float32(), list_size=dims)),
    pa.field("image_path", pa.string()),
    pa.field("people_ids", pa.list_(pa.int32())),
    pa.field("date", pa.timestamp('ms')),
    pa.field("location", pa.string()),
])
imgs_tbl = db.create_table("images", schema=imgs_schema)


def get_exif_data(image_path):
    try:
        # For HEIF/HEIC files, convert to JPEG in memory first
        if image_path.lower().endswith(('.heic', '.heif')):
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Save as JPEG to bytes buffer
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                buffer.seek(0)
                exif_image = ExifImage(buffer)
        else:
            # For regular images, read directly
            with open(image_path, 'rb') as image_file:
                exif_image = ExifImage(image_file)

        # Extract date
        date = None
        if hasattr(exif_image, 'datetime_original'):
            date = exif_image.datetime_original
        elif hasattr(exif_image, 'datetime'):
            date = exif_image.datetime

        # Extract location
        location = None
        if hasattr(exif_image, 'gps_latitude') and hasattr(exif_image, 'gps_longitude'):
            location = f"{exif_image.gps_latitude}, {exif_image.gps_longitude}"

        return date, location
    except Exception as e:
        print(f"Warning: Could not extract EXIF data from {image_path}: {str(e)}")
        return None, None


def process_images(folder_path, batch_size=100):
    image_id = 0
    batch = []

    for image_name in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_name)
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.heic', '.heif')):
            continue

        try:
            # Load and process the image
            with Image.open(image_path) as image:
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Process with CLIP
                inputs = processor(images=image, return_tensors="pt", padding=True)
                image_features = model.get_image_features(**inputs).detach().numpy().flatten().tolist()

            # Get EXIF data
            date, location = get_exif_data(image_path)
            if date:
                try:
                    date = datetime.strptime(date, '%Y:%m:%d %H:%M:%S')
                except ValueError:
                    print(f"Warning: Could not parse date {date} for {image_path}")
                    date = None

            batch.append({
                "image_id": image_id,
                "vector": image_features,
                "image_path": image_path,
                "people_ids": [],
                "date": date,
                "location": location
            })
            image_id += 1

            # If batch is full, yield it and reset
            if len(batch) >= batch_size:
                yield batch
                batch = []

        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue

    # Yield any remaining images in the last batch
    if batch:
        yield batch


# Add images to the database in batches
for batch in process_images("ex-images"):
    imgs_tbl.add(batch)