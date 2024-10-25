import os
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

# Register HEIF opener with Pillow
register_heif_opener()


def convert_heic_to_jpg(heic_path):
    """Convert HEIC/HEIF to JPEG using pillow-heif"""
    with Image.open(heic_path) as image:
        jpeg_path = heic_path.replace('.heic', '.jpg').replace('.HEIC', '.jpg')
        image.save(jpeg_path, "JPEG")
        return jpeg_path


def process_faces(folder_path, faces_dir="cropped_faces"):
    """Process all images in folder and return face clustering results"""
    image_encodings = []

    os.makedirs(faces_dir, exist_ok=True)

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
            return {}, {}

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
