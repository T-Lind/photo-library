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

SUPPORTED_FACE_FORMATS = ('.png', '.jpg', '.jpeg', '.heic', '.heif')

# Maximum face-encoding distance for a new face to be assigned to an
# existing person during incremental indexing (same scale as DBSCAN eps).
MATCH_THRESHOLD = 0.5


def save_face_data(image_to_people, label_to_person_id, save_dir, folder_name,
                   person_centroids=None):
    """Save face processing results to JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{folder_name}_face_data.json")

    data = {
        "image_to_people": {k: list(v) if isinstance(v, set) else v
                            for k, v in image_to_people.items()},
        "label_to_person_id": {str(k): v for k, v in label_to_person_id.items()},
        "person_centroids": {str(k): list(v) for k, v in (person_centroids or {}).items()},
    }

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Face data saved to {save_path}")


def load_face_data(save_dir, folder_name):
    """Load face processing results from JSON file"""
    save_path = os.path.join(save_dir, f"{folder_name}_face_data.json")

    if not os.path.exists(save_path):
        return None, None, None

    with open(save_path, 'r') as f:
        data = json.load(f)

    # Convert label_to_person_id keys back to integers
    label_to_person_id = {int(k): v for k, v in data["label_to_person_id"].items()}
    person_centroids = {int(k): np.array(v) for k, v in data.get("person_centroids", {}).items()}

    logging.info(f"Loaded face data from {save_path}")
    return data["image_to_people"], label_to_person_id, person_centroids


def convert_heic_to_jpg(heic_path):
    """Convert HEIC/HEIF to JPEG using pillow-heif"""
    with Image.open(heic_path) as image:
        jpeg_path = os.path.splitext(heic_path)[0] + '.jpg'
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


def encode_faces_in_image(image_path):
    """Detect and encode all faces in one image.

    Returns a list of (encoding, location) tuples. HEIC/HEIF files are
    converted to a temporary JPEG for face_recognition, then cleaned up.
    """
    work_path = image_path
    if image_path.lower().endswith(('.heic', '.heif')):
        work_path = convert_heic_to_jpg(image_path)

    try:
        image = face_recognition.load_image_file(work_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        return list(zip(face_encodings, face_locations))
    finally:
        if work_path != image_path:
            os.remove(work_path)


def save_face_crop(image_path, location, face_filename):
    """Save a cropped (padded) face image if it doesn't already exist."""
    if os.path.exists(face_filename):
        return

    top, right, bottom, left = location
    with Image.open(image_path) as img:
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
        if face_img.mode != 'RGB':
            face_img = face_img.convert('RGB')
        face_img.save(face_filename, "JPEG")


def process_faces(folder_path, faces_dir="cropped_faces", save_dir="saves"):
    """Process all images in folder and return face clustering results"""
    # Try to load existing face data
    folder_name = os.path.basename(folder_path)
    image_to_people, label_to_person_id, _ = load_face_data(save_dir, folder_name)

    if image_to_people is not None and label_to_person_id is not None:
        logging.info("Using cached face processing results")
        return image_to_people, label_to_person_id

    logging.info("No cached data found. Processing faces...")
    image_encodings = []

    os.makedirs(faces_dir, exist_ok=True)

    for image_name in tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_name)

        # Skip non-image files and the faces directory
        if not image_path.lower().endswith(SUPPORTED_FACE_FORMATS) or 'cropped_faces' in image_path:
            continue

        try:
            for encoding, location in encode_faces_in_image(image_path):
                image_encodings.append({
                    "encoding": encoding,
                    "image": image_name,
                    "location": location,
                    "image_path": image_path
                })
        except Exception as e:
            logging.error(f"Error processing faces in {image_name}: {str(e)}")
            continue

    # Perform clustering if faces were found
    if not image_encodings:
        logging.warning("No faces found in any images")
        save_face_data({}, {}, save_dir, folder_name, {})
        return {}, {}

    # Extract encodings for clustering
    encodings = [entry['encoding'] for entry in image_encodings]

    # Perform clustering with optimized parameters
    labels = cluster_faces(encodings)

    # Create mappings
    image_to_people = defaultdict(set)  # Using set to avoid duplicates
    label_to_person_id = {label: idx for idx, label in enumerate(np.unique(labels))}

    # Accumulate encodings per person so we can store a mean ("centroid")
    # encoding, used later to match new faces during incremental indexing.
    person_encodings = defaultdict(list)

    # Process clustering results
    for entry, label in zip(image_encodings, labels):
        person_id = label_to_person_id[label]
        image_name = entry['image']
        image_to_people[image_name].add(person_id)  # Using set to avoid duplicates
        person_encodings[person_id].append(entry['encoding'])

        # Save the face image
        face_filename = os.path.join(faces_dir, f"person_{person_id}.jpg")
        try:
            save_face_crop(entry['image_path'], entry['location'], face_filename)
        except Exception as e:
            logging.error(f"Failed to save face crop for person {person_id}: {str(e)}")

    person_centroids = {pid: np.mean(encs, axis=0)
                        for pid, encs in person_encodings.items()}

    # Convert defaultdict and sets to regular dict and lists before saving
    image_to_people = {k: list(v) for k, v in image_to_people.items()}

    # Save the results
    save_face_data(image_to_people, label_to_person_id, save_dir, folder_name,
                   person_centroids)

    logging.info(f"Found {len(label_to_person_id)} unique people across all images")
    return image_to_people, label_to_person_id


def process_new_faces(new_image_names, folder_path, faces_dir="cropped_faces",
                      save_dir="saves"):
    """Incrementally process faces for new images only.

    Each new face is assigned to the nearest existing person (by centroid
    distance) if within MATCH_THRESHOLD; otherwise a new person is created.
    Returns (image_to_people for the new images, list of newly created
    person ids). Requires centroid data saved by a previous full run.
    """
    folder_name = os.path.basename(folder_path)
    image_to_people, label_to_person_id, person_centroids = load_face_data(save_dir, folder_name)

    if image_to_people is None:
        raise RuntimeError(
            "No cached face data found; run a full (re)build before indexing incrementally.")

    if not person_centroids and label_to_person_id:
        logging.warning(
            "Cached face data has no centroid encodings (created by an older "
            "version). New faces cannot be matched to existing people; "
            "re-run with --rebuild to regenerate. Skipping face assignment.")
        return {}, []

    os.makedirs(faces_dir, exist_ok=True)

    next_person_id = max(list(person_centroids.keys()) + [-1]) + 1
    new_person_ids = []
    new_image_to_people = {}

    for image_name in tqdm(new_image_names, desc="Processing new faces"):
        image_path = os.path.join(folder_path, image_name)
        try:
            faces = encode_faces_in_image(image_path)
        except Exception as e:
            logging.error(f"Error processing faces in {image_name}: {str(e)}")
            continue

        people = set()
        for encoding, location in faces:
            person_id = None
            if person_centroids:
                ids = list(person_centroids.keys())
                dists = [np.linalg.norm(encoding - person_centroids[pid]) for pid in ids]
                best = int(np.argmin(dists))
                if dists[best] <= MATCH_THRESHOLD:
                    person_id = ids[best]

            if person_id is None:
                person_id = next_person_id
                next_person_id += 1
                new_person_ids.append(person_id)
                person_centroids[person_id] = np.array(encoding)
                face_filename = os.path.join(faces_dir, f"person_{person_id}.jpg")
                try:
                    save_face_crop(image_path, location, face_filename)
                except Exception as e:
                    logging.error(f"Failed to save face crop for person {person_id}: {str(e)}")

            people.add(person_id)

        if people:
            new_image_to_people[image_name] = sorted(people)

    # Persist updated mappings so subsequent incremental runs see them
    image_to_people.update(new_image_to_people)
    save_face_data(image_to_people, label_to_person_id, save_dir, folder_name,
                   person_centroids)

    if new_person_ids:
        logging.info(f"Created {len(new_person_ids)} new people from new images")
    return new_image_to_people, new_person_ids
