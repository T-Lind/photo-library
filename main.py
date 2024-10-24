import lancedb
import pyarrow as pa
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
from get_emb import get_image_embedding
from get_exif import get_exif_data
from proc_imgs import process_faces
import os

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# LITERALS
DIMS = 512  # TODO: MAKE sure this matches the embedding dimension.
NUM_PARTITIONS = 32
NUM_SUB_VECTORS = 10
BATCH_SIZE = 100


def setup_database(uri):
    """Set up the LanceDB database and tables"""
    db = lancedb.connect(uri)

    # Create people table
    people_schema = pa.schema([
        pa.field("people_id", pa.int32()),
        pa.field("name", pa.string()),
    ])
    people_tbl = db.create_table("people", schema=people_schema)

    # Create images table
    imgs_schema = pa.schema([
        pa.field("image_id", pa.int32()),
        pa.field("vector", pa.list_(pa.float32(), list_size=DIMS)),
        pa.field("image_path", pa.string()),
        pa.field("people_ids", pa.list_(pa.int32())),
        pa.field("date", pa.timestamp('ms')),
        pa.field("location", pa.string()),
    ])
    imgs_tbl = db.create_table("images", schema=imgs_schema)
    return db, people_tbl, imgs_tbl


def process_images(folder_path, image_to_people, batch_size=100):
    """Process images and yield batches for database insertion"""
    image_id = 0
    batch = []

    for image_name in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_name)
        if not image_path.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.heic', '.heif')) or 'cropped_faces' in image_path:
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

            # Get image embedding
            vec = get_image_embedding(image_path)

            # Get people IDs for this image
            people_ids = image_to_people.get(image_name, [])
            # remove duplicates
            people_ids = list(set(people_ids))

            batch.append({
                "image_id": image_id,
                "vector": vec,
                "image_path": image_path,
                "people_ids": people_ids,
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


def main(folder_path, db_uri):
    """Main function to process images and populate the database"""
    print("Step 1: Setting up database...")
    db, people_tbl, imgs_tbl = setup_database(db_uri)

    print("\nStep 2: Processing faces and clustering...")
    image_to_people, label_to_person_id = process_faces(folder_path)

    print(f"Found {len(label_to_person_id)} unique people across all images")

    print("Step 3: Populating people table...")
    # Create entries for all identified people
    people_entries = [
        {"people_id": person_id, "name": ""}
        for person_id in range(len(label_to_person_id))
    ]
    people_tbl.add(people_entries)

    print("Step 4: Processing images and populating images table...")
    # Add images to the database in batches
    for batch in process_images(folder_path, image_to_people, batch_size=BATCH_SIZE):
        imgs_tbl.add(batch)

    imgs_tbl.create_index(num_partitions=NUM_PARTITIONS, num_sub_vectors=NUM_SUB_VECTORS)

    print("Processing complete!")
    return db


if __name__ == "__main__":
    folder_path = "ex-images"
    db_uri = "data/photos-3"
    main(folder_path, db_uri)
