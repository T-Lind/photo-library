import lancedb
import pyarrow as pa
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
from pillow_heif import register_heif_opener
from get_emb import get_embedding
from get_exif import get_exif_data
import os


# Register HEIF opener with Pillow
register_heif_opener()

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dims = 2048

uri = "data/photos-2"
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



def process_images(folder_path, batch_size=100):
    image_id = 0
    batch = []

    for image_name in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_name)
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.heic', '.heif')):
            continue

        try:

            # Get EXIF data
            date, location = get_exif_data(image_path)
            if date:
                try:
                    date = datetime.strptime(date, '%Y:%m:%d %H:%M:%S')
                except ValueError:
                    print(f"Warning: Could not parse date {date} for {image_path}")
                    date = None

            vec = get_embedding(image_path).tolist()[0]

            batch.append({
                "image_id": image_id,
                "vector": vec,
                "image_path": image_path,
                "people_ids": [],  # TODO: implement this
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

if __name__ == "__main__":
    # Add images to the database in batches
    for batch in process_images("ex-images"):
        imgs_tbl.add(batch)