import argparse
import lancedb
import pyarrow as pa
import pyarrow.compute as pc
from tqdm import tqdm
from datetime import datetime
from get_emb import get_image_embeddings
from get_exif import get_exif_data
from proc_imgs import process_faces, process_new_faces
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# LITERALS
DIMS = 512
NUM_PARTITIONS = 16
NUM_SUB_VECTORS = 8
BATCH_SIZE = 100  # Rows per database insert
EMBED_BATCH_SIZE = 16  # Images per CLIP forward pass
MIN_ROWS_FOR_INDEX = 256  # Below this, brute-force search is fine (and PQ training fails)
SAVES_DIR = "saves"  # Directory for saved face processing results

SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.heic', '.heif'}


def is_supported_image(filename):
    """Check if the file is a supported image format"""
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS)


def list_image_files(folder_path):
    return [f for f in os.listdir(folder_path)
            if is_supported_image(f) and 'cropped_faces' not in f]


def setup_database(db):
    """Create (or recreate) the people and images tables"""
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
    return people_tbl, imgs_tbl


def get_existing_state(imgs_tbl):
    """Return (already-indexed image paths, next free image_id).

    Reads only the two small columns, never the embedding vectors.
    """
    table = imgs_tbl.to_lance().to_table(columns=["image_path", "image_id"])
    paths = set(table["image_path"].to_pylist())
    max_id = pc.max(table["image_id"]).as_py() if table.num_rows else None
    return paths, (max_id + 1 if max_id is not None else 0)


def read_image_metadata(image_path):
    """EXIF date/location for one image; failures degrade to (None, "")."""
    try:
        date, location = get_exif_data(image_path)
    except Exception as e:
        logging.warning(f"Could not read EXIF from {image_path}: {str(e)}")
        return None, ""

    if date:
        try:
            date = datetime.strptime(str(date), '%Y:%m:%d %H:%M:%S')
        except ValueError:
            logging.warning(f"Could not parse date {date} for {image_path}")
            date = None
    else:
        date = None

    return date, location


def process_images(folder_path, image_to_people, image_files, start_id=0,
                   batch_size=BATCH_SIZE, embed_batch_size=EMBED_BATCH_SIZE):
    """Process images (embedding in batches) and yield row batches for insertion"""
    image_id = start_id
    batch = []
    failed_images = []

    progress = tqdm(total=len(image_files), desc="Processing images")

    for chunk_start in range(0, len(image_files), embed_batch_size):
        chunk = image_files[chunk_start:chunk_start + embed_batch_size]

        # Gather per-image metadata first (cheap, failures don't skip the image)
        metas = []
        for image_name in chunk:
            image_path = os.path.join(folder_path, image_name)
            date, location = read_image_metadata(image_path)
            metas.append((image_name, image_path, date, location))

        # Embed the whole chunk in one forward pass; on failure fall back
        # to per-image embedding so one bad file doesn't sink the chunk.
        try:
            vectors = get_image_embeddings([m[1] for m in metas])
        except Exception:
            vectors = []
            for _, image_path, _, _ in metas:
                try:
                    vectors.append(get_image_embeddings([image_path])[0])
                except Exception as e:
                    logging.error(f"Failed to get embedding for {image_path}: {str(e)}")
                    vectors.append(None)

        for (image_name, image_path, date, location), vec in zip(metas, vectors):
            progress.update(1)
            if vec is None:
                failed_images.append((image_path, "embedding_failed"))
                continue

            people_ids = sorted(set(image_to_people.get(image_name, [])))

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

    progress.close()

    # Yield any remaining images in the last batch
    if batch:
        yield batch

    # Report failed images
    if failed_images:
        logging.warning(f"\nFailed to process {len(failed_images)} images:")
        for path, error in failed_images:
            logging.warning(f"- {path}: {error}")


def build_index(imgs_tbl):
    """Create the ANN index; fall back to brute-force search on failure"""
    row_count = imgs_tbl.count_rows(None)
    if row_count < MIN_ROWS_FOR_INDEX:
        logging.info(
            f"Only {row_count} rows; skipping ANN index (brute-force search is fast enough)")
        return

    logging.info("Creating vector similarity search index...")
    try:
        imgs_tbl.create_index(num_partitions=NUM_PARTITIONS, num_sub_vectors=NUM_SUB_VECTORS)
    except Exception as e:
        logging.warning(f"Index creation failed (search falls back to brute force): {str(e)}")


def full_build(db, folder_path, faces_dir):
    """Drop everything and reprocess the whole folder"""
    logging.info("Step 1: Setting up database...")
    people_tbl, imgs_tbl = setup_database(db)

    # A full rebuild must not reuse stale cached face clusters, or images
    # added since the cache was written would get no face assignments.
    cache_path = os.path.join(SAVES_DIR, f"{os.path.basename(folder_path)}_face_data.json")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        logging.info("Cleared cached face data for full rebuild")

    logging.info("Step 2: Processing faces and clustering...")
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
    image_files = list_image_files(folder_path)
    for batch in process_images(folder_path, image_to_people, image_files):
        imgs_tbl.add(batch)
        total_processed += len(batch)

    logging.info(f"Successfully processed {total_processed} images")
    build_index(imgs_tbl)
    logging.info("Processing complete!")


def incremental_update(db, folder_path, faces_dir):
    """Add only images that aren't in the database yet"""
    imgs_tbl = db.open_table("images")
    people_tbl = db.open_table("people")

    existing_paths, next_id = get_existing_state(imgs_tbl)
    new_files = [f for f in list_image_files(folder_path)
                 if os.path.join(folder_path, f) not in existing_paths]

    if not new_files:
        logging.info("No new images found; database is up to date.")
        return

    logging.info(f"Found {len(new_files)} new images (of "
                 f"{len(existing_paths)} already indexed)")

    logging.info("Step 1: Matching faces in new images against known people...")
    try:
        image_to_people, new_person_ids = process_new_faces(
            new_files, folder_path, faces_dir, SAVES_DIR)
    except RuntimeError as e:
        logging.error(str(e))
        return

    if new_person_ids:
        people_tbl.add([{"people_id": pid, "name": ""} for pid in new_person_ids])
        logging.info(f"Added {len(new_person_ids)} new people")

    logging.info("Step 2: Embedding and inserting new images...")
    total_processed = 0
    for batch in process_images(folder_path, image_to_people, new_files, start_id=next_id):
        imgs_tbl.add(batch)
        total_processed += len(batch)

    logging.info(f"Added {total_processed} new images")
    build_index(imgs_tbl)
    logging.info("Incremental update complete!")


def main(folder_path, faces_dir, db_uri, rebuild=False):
    """Main entry point: full rebuild or incremental update"""
    os.makedirs(SAVES_DIR, exist_ok=True)
    db = lancedb.connect(db_uri)

    has_tables = "images" in db.table_names() and "people" in db.table_names()
    if rebuild or not has_tables:
        if not has_tables and not rebuild:
            logging.info("No existing database found; running a full build.")
        full_build(db, folder_path, faces_dir)
    else:
        incremental_update(db, folder_path, faces_dir)

    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a photo folder into LanceDB")
    parser.add_argument("--images-dir", default="256-images",
                        help="Folder containing the photos to index")
    parser.add_argument("--db-uri", default="data/photos-256",
                        help="LanceDB database location")
    parser.add_argument("--faces-dir", default="cropped_faces_256",
                        help="Directory for cropped face images")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop the database and reprocess everything "
                             "(default is incremental: only new images are added)")
    args = parser.parse_args()

    main(args.images_dir, args.faces_dir, args.db_uri, rebuild=args.rebuild)
