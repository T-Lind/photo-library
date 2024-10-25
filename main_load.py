import lancedb
import pyarrow as pa
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
from get_emb import get_image_embedding
from get_exif import get_exif_data
from proc_imgs import process_faces
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# LITERALS
DIMS = 512
NUM_PARTITIONS = 16
NUM_SUB_VECTORS = 8
BATCH_SIZE = 100
SAVES_DIR = "saves"  # Directory for saved face processing results

SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.heic', '.heif'}


def is_supported_image(filename):
    """Check if the file is a supported image format"""
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS)


def setup_database(uri):
    """Set up the LanceDB database and tables"""
    db = lancedb.connect(uri)

    # Create people table
    people_schema = pa.schema([
        pa.field("people_id", pa.int32()),
        pa.field("name", pa.string()),
    ])

    # Drop existing tables if they exist
    if "people" in db.table_names():
        db.drop_table("people")
    if "images" in db.table_names():
        db.drop_table("images")

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
    failed_images = []

    # Get list of all image files first
    image_files = [f for f in os.listdir(folder_path) if is_supported_image(f) and 'cropped_faces' not in f]

    for image_name in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(folder_path, image_name)
        try:
            # Get EXIF data
            date, location = get_exif_data(image_path)
            if date:
                try:
                    date = datetime.strptime(date, '%Y:%m:%d %H:%M:%S')
                except ValueError:
                    logging.warning(f"Could not parse date {date} for {image_path}")
                    date = None

            # Get image embedding
            try:
                vec = get_image_embedding(image_path)
            except Exception as e:
                logging.error(f"Failed to get embedding for {image_path}: {str(e)}")
                failed_images.append((image_path, "embedding_failed"))
                continue

            # Get people IDs for this image
            people_ids = image_to_people.get(image_name, [])
            people_ids = list(set(people_ids))  # remove duplicates

            batch.append({
                "image_id": image_id,
                "vector": vec,
                "image_path": image_path,
                "people_ids": people_ids,
                "date": date,
                "location": str(location) if location else ""
            })
            image_id += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []

        except Exception as e:
            logging.error(f"Error processing {image_name}: {str(e)}")
            failed_images.append((image_path, str(e)))
            continue

    # Yield any remaining images in the last batch
    if batch:
        yield batch

    # Report failed images
    if failed_images:
        logging.warning(f"\nFailed to process {len(failed_images)} images:")
        for path, error in failed_images:
            logging.warning(f"- {path}: {error}")


def main(folder_path, faces_dir, db_uri):
    """Main function to process images and populate the database"""
    os.makedirs(SAVES_DIR, exist_ok=True)

    logging.info("Step 1: Setting up database...")
    db, people_tbl, imgs_tbl = setup_database(db_uri)

    logging.info("\nStep 2: Processing faces and clustering...")
    image_to_people, label_to_person_id = process_faces(folder_path, faces_dir, SAVES_DIR)

    num_people = len(label_to_person_id)
    logging.info(f"Found {num_people} unique people across all images")

    logging.info("Step 3: Populating people table...")
    people_entries = [
        {"people_id": person_id, "name": ""}
        for person_id in range(num_people)
    ]
    if people_entries:
        people_tbl.add(people_entries)

    logging.info("Step 4: Processing images and populating images table...")
    total_processed = 0
    for batch in process_images(folder_path, image_to_people, batch_size=BATCH_SIZE):
        imgs_tbl.add(batch)
        total_processed += len(batch)

    logging.info(f"Successfully processed {total_processed} images")

    # Create index for vector similarity search
    logging.info("Creating vector similarity search index...")
    imgs_tbl.create_index(num_partitions=NUM_PARTITIONS, num_sub_vectors=NUM_SUB_VECTORS)

    logging.info("Processing complete!")
    return db


if __name__ == "__main__":
    folder_path = "256-images"
    db_uri = "data/photos-256"
    faces_dir = "cropped_faces_256"
    main(folder_path, faces_dir, db_uri)