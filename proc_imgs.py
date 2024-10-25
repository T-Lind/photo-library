import os
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm
import json
import logging

# Register HEIF opener with Pillow
register_heif_opener()


def save_face_data(image_to_people, label_to_person_id, save_dir, folder_name):
    """Save face processing results to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{folder_name}_face_data.json")

    data = {
        "image_to_people": {k: list(v) if isinstance(v, set) else v
                            for k, v in image_to_people.items()},
        "label_to_person_id": {str(k): v for k, v in label_to_person_id.items()}
    }

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Face data saved to {save_path}")


def load_face_data(save_dir, folder_name):
    """Load face processing results from JSON file"""
    save_path = os.path.join(save_dir, f"{folder_name}_face_data.json")

    if not os.path.exists(save_path):
        return None, None

    with open(save_path, 'r') as f:
        data = json.load(f)

    # Convert label_to_person_id keys back to integers
    label_to_person_id = {int(k): v for k, v in data["label_to_person_id"].items()}

    logging.info(f"Loaded face data from {save_path}")
    return data["image_to_people"], label_to_person_id


def convert_heic_to_jpg(heic_path):
    """Convert HEIC/HEIF to JPEG using pillow-heif"""
    with Image.open(heic_path) as image:
        jpeg_path = heic_path.replace('.heic', '.jpg').replace('.HEIC', '.jpg')
        image.save(jpeg_path, "JPEG")
        return jpeg_path


def cluster_faces(encodings, threshold=0.5):
    """
    Cluster face encodings using DBSCAN with optimized parameters

    Args:
        encodings: List of face encodings
        threshold: Distance threshold for face similarity (lower = more strict)

    Returns:
        numpy array of cluster labels
    """
    if not encodings:
        return np.array([])

    # Convert encodings to numpy array
    X = np.array(encodings)

    # Parameters explanation:
    # - eps: Maximum distance between two samples for them to be considered in the same cluster
    #        (0.5 is stricter than 0.6, based on face_recognition's own threshold)
    # - min_samples: Minimum number of samples in a cluster (2 means each person should appear at least twice)
    # - metric: Using 'euclidean' as it works well with face_recognition's encodings
    dbscan = DBSCAN(
        eps=threshold,
        min_samples=2,  # Require at least 2 similar faces to form a cluster
        metric='euclidean',
        n_jobs=-1  # Use all CPU cores
    )

    # Fit DBSCAN
    cluster_labels = dbscan.fit_predict(X)

    # Handle outliers (label -1) by assigning them to new unique clusters
    next_label = cluster_labels.max() + 1
    for idx, label in enumerate(cluster_labels):
        if label == -1:
            cluster_labels[idx] = next_label
            next_label += 1

    return cluster_labels


def process_faces(folder_path, faces_dir="cropped_faces", save_dir="saves"):
    """Process all images in folder and return face clustering results"""
    # Try to load existing face data
    folder_name = os.path.basename(folder_path)
    image_to_people, label_to_person_id = load_face_data(save_dir, folder_name)

    if image_to_people is not None and label_to_person_id is not None:
        logging.info("Using cached face processing results")
        return image_to_people, label_to_person_id

    logging.info("No cached data found. Processing faces...")
    image_encodings = []

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
            logging.error(f"Error processing faces in {image_name}: {str(e)}")
            continue

    # Perform clustering if faces were found
    if not image_encodings:
        logging.warning("No faces found in any images")
        return {}, {}

    # Extract encodings for clustering
    encodings = [entry['encoding'] for entry in image_encodings]

    # Perform clustering with optimized parameters
    labels = cluster_faces(encodings)

    # Create mappings
    image_to_people = defaultdict(set)  # Using set to avoid duplicates
    label_to_person_id = {label: idx for idx, label in enumerate(np.unique(labels))}

    # Process clustering results
    for entry, label in zip(image_encodings, labels):
        person_id = label_to_person_id[label]
        image_name = entry['image']
        image_to_people[image_name].add(person_id)  # Using set to avoid duplicates

        # Save the face image
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

    # Convert defaultdict and sets to regular dict and lists before saving
    image_to_people = {k: list(v) for k, v in image_to_people.items()}

    # Save the results
    save_face_data(image_to_people, label_to_person_id, save_dir, folder_name)

    logging.info(f"Found {len(label_to_person_id)} unique people across all images")
    return image_to_people, label_to_person_id