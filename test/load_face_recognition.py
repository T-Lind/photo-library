import face_recognition
import os
import numpy as np
from sklearn.cluster import DBSCAN
import json
from PIL import Image
from pillow_heif import register_heif_opener
from collections import defaultdict

# Register HEIF opener with Pillow
register_heif_opener()


def convert_heic_to_jpg(heic_path):
    """Convert HEIC/HEIF to JPEG using pillow-heif"""
    with Image.open(heic_path) as image:
        jpeg_path = heic_path.replace('.heic', '.jpg').replace('.HEIC', '.jpg')
        image.save(jpeg_path, "JPEG")
        return jpeg_path


def crop_face(image_path, face_location):
    """Crop a face from an image using its location"""
    # face_location format is (top, right, bottom, left)
    top, right, bottom, left = face_location

    # Add some padding around the face (20%)
    height = bottom - top
    width = right - left
    padding_v = int(height * 0.2)
    padding_h = int(width * 0.2)

    # Adjust coordinates with padding
    top = max(0, top - padding_v)
    bottom = bottom + padding_v
    left = max(0, left - padding_h)
    right = right + padding_h

    # Open and crop the image
    with Image.open(image_path) as img:
        face_img = img.crop((left, top, right, bottom))
        return face_img


def ensure_faces_directory(base_dir):
    """Create a directory to store cropped faces if it doesn't exist"""
    faces_dir = os.path.join(base_dir, "cropped_faces")
    os.makedirs(faces_dir, exist_ok=True)
    return faces_dir


def process_images_from_folder(folder_path):
    image_encodings = []
    faces_dir = ensure_faces_directory(folder_path)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Only process supported image files
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.heic', '.heif')):
            continue

        try:
            # Convert HEIC images to JPG
            original_path = image_path
            if image_path.lower().endswith(('.heic', '.heif')):
                print(f"Converting HEIC image: {image_name}")
                image_path = convert_heic_to_jpg(image_path)

            print(f"Processing image: {image_name}")
            image = face_recognition.load_image_file(image_path)

            # Detect face locations and face encodings
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            # Store the encodings, image paths, and face locations
            for encoding, location in zip(face_encodings, face_locations):
                image_encodings.append({
                    "encoding": encoding,
                    "image": image_name,
                    "location": location,
                    "image_path": original_path
                })

            # Clean up temporary JPG if it was converted
            if image_path.endswith('.jpg') and image_name.lower().endswith(('.heic', '.heif')):
                os.remove(image_path)

        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue

    return image_encodings


def cluster_faces(face_encodings):
    encodings = [entry['encoding'] for entry in face_encodings]
    X = np.array(encodings)

    dbscan = DBSCAN(metric='euclidean', eps=0.6, min_samples=1)
    labels = dbscan.fit_predict(X)

    for i, entry in enumerate(face_encodings):
        entry['label'] = labels[i]

    return face_encodings


def assign_person_ids(clustered_faces, folder_path, output_file):
    unique_labels = np.unique([entry['label'] for entry in clustered_faces])
    label_to_person_id = {label: idx for idx, label in enumerate(unique_labels)}
    faces_dir = ensure_faces_directory(folder_path)

    # Use defaultdict to group person_ids by image
    image_to_persons = defaultdict(set)

    # Create a dictionary to store the best quality face for each person
    best_faces = {}  # person_id -> (face_size, image_entry)

    # Find the best quality face for each person
    for entry in clustered_faces:
        person_id = label_to_person_id[entry['label']]
        top, right, bottom, left = entry['location']
        face_size = (bottom - top) * (right - left)  # Use face area as quality metric

        if person_id not in best_faces or face_size > best_faces[person_id][0]:
            best_faces[person_id] = (face_size, entry)

        image_to_persons[entry['image']].add(person_id)

    # Save the best quality face for each person
    for person_id, (_, entry) in best_faces.items():
        face_img = crop_face(entry['image_path'], entry['location'])
        face_filename = f"person_{person_id}.jpg"
        face_path = os.path.join(faces_dir, face_filename)
        face_img.save(face_path, "JPEG")
        print(f"Saved face for person {person_id} from {entry['image']}")

    # Convert sets to lists for JSON serialization
    output_data = {
        "total_unique_people": len(unique_labels),
        "images": [
            {
                "image": image_name,
                "person_ids": sorted(list(person_ids))
            }
            for image_name, person_ids in image_to_persons.items()
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Clustered data saved to {output_file}")
    print(f"Total number of unique people detected: {len(unique_labels)}")
    print(f"Cropped faces saved in: {faces_dir}")


def main(folder_path, output_file):
    all_face_encodings = process_images_from_folder(folder_path)

    if not all_face_encodings:
        print("No faces found in the images.")
        return

    clustered_faces = cluster_faces(all_face_encodings)
    assign_person_ids(clustered_faces, folder_path, output_file)


# Folder containing images and JSON output file
folder_path = "../spec"
output_file = "clustered_faces.json"

if __name__ == "__main__":
    main(folder_path, output_file)