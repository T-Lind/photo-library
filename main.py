import lancedb
import pyarrow as pa
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
from pillow_heif import register_heif_opener
from get_emb import get_embedding
from get_exif import get_exif_data
import face_recognition
import os
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from PIL import Image

# Register HEIF opener with Pillow
register_heif_opener()

# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dims = 2048


def convert_heic_to_jpg(heic_path):
    """Convert HEIC/HEIF to JPEG using pillow-heif"""
    with Image.open(heic_path) as image:
        jpeg_path = heic_path.replace('.heic', '.jpg').replace('.HEIC', '.jpg')
        image.save(jpeg_path, "JPEG")
        return jpeg_path


def process_faces(folder_path):
    """Process all images in folder and return face clustering results"""
    image_encodings = []
    faces_dir = "cropped_faces"
    os.makedirs(faces_dir, exist_ok=True)

    print("Processing faces in images...")
    for image_name in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_name)

        # Skip non-image files and the faces directory
        if not image_path.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.heic', '.heif')) or 'cropped_faces' in image_path:
            continue

        try:
            # Convert HEIC images to JPG temporarily
            original_path = image_path
            if image_path.lower().endswith(('.heic', '.heif')):
                image_path = convert_heic_to_jpg(image_path)

            # Process image for face recognition
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            # Store encodings and metadata
            for encoding, location in zip(face_encodings, face_locations):
                image_encodings.append({
                    "encoding": encoding,
                    "image": image_name,
                    "location": location,
                    "image_path": original_path
                })

            # Cleanup temporary files
            if image_path != original_path:
                os.remove(image_path)

        except Exception as e:
            print(f"Error processing faces in {image_name}: {str(e)}")
            continue

    # Perform clustering if faces were found
    if not image_encodings:
        return {}, {}

    # Cluster faces
    encodings = [entry['encoding'] for entry in image_encodings]
    X = np.array(encodings)

    dbscan = DBSCAN(metric='euclidean', eps=0.6, min_samples=1)
    labels = dbscan.fit_predict(X)

    # Create mappings
    image_to_people = defaultdict(list)
    label_to_person_id = {label: idx for idx, label in enumerate(np.unique(labels))}

    # Process clustering results
    for entry, label in zip(image_encodings, labels):
        person_id = label_to_person_id[label]
        image_name = entry['image']
        image_to_people[image_name].append(person_id)

        # Save the face image (only save one per person if it hasn't been saved yet)
        face_filename = os.path.join(faces_dir, f"person_{person_id}.jpg")
        if not os.path.exists(face_filename):
            top, right, bottom, left = entry['location']
            with Image.open(entry['image_path']) as img:
                # Add 20% padding
                height = bottom - top
                width = right - left
                padding_v = int(height * 0.2)
                padding_h = int(width * 0.2)

                top = max(0, top - padding_v)
                bottom = bottom + padding_v
                left = max(0, left - padding_h)
                right = right + padding_h

                face_img = img.crop((left, top, right, bottom))
                face_img.save(face_filename, "JPEG")

    return dict(image_to_people), label_to_person_id


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
        pa.field("vector", pa.list_(pa.float32(), list_size=dims)),
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
            vec = get_embedding(image_path).tolist()[0]

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
    print("Step 1: Processing faces and clustering...")
    image_to_people, label_to_person_id = process_faces(folder_path)

    print(f"Found {len(label_to_person_id)} unique people across all images")

    print("Step 2: Setting up database...")
    db, people_tbl, imgs_tbl = setup_database(db_uri)

    print("Step 3: Populating people table...")
    # Create entries for all identified people
    people_entries = [
        {"people_id": person_id, "name": ""}
        for person_id in range(len(label_to_person_id))
    ]
    people_tbl.add(people_entries)

    print("Step 4: Processing images and populating images table...")
    # Add images to the database in batches
    for batch in process_images(folder_path, image_to_people):
        imgs_tbl.add(batch)

    print("Processing complete!")
    return db


if __name__ == "__main__":
    folder_path = "ex-images"
    db_uri = "data/photos-2"
    main(folder_path, db_uri)