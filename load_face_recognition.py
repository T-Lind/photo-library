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
    # pillow_heif automatically registers with PIL, so we can use Image.open directly
    with Image.open(heic_path) as image:
        # Convert HEIC to JPEG and save temporarily
        jpeg_path = heic_path.replace('.heic', '.jpg').replace('.HEIC', '.jpg')
        image.save(jpeg_path, "JPEG")
        return jpeg_path


def process_images_from_folder(folder_path):
    image_encodings = []

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        # Only process supported image files
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.heic', '.heif')):
            continue

        try:
            # Convert HEIC images to JPG
            if image_path.lower().endswith(('.heic', '.heif')):
                print(f"Converting HEIC image: {image_name}")
                image_path = convert_heic_to_jpg(image_path)

            print(f"Processing image: {image_name}")
            image = face_recognition.load_image_file(image_path)

            # Detect face locations and face encodings
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            # Store the encodings and image paths
            for encoding in face_encodings:
                image_encodings.append({
                    "encoding": encoding,
                    "image": image_name
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


def assign_person_ids(clustered_faces, output_file):
    unique_labels = np.unique([entry['label'] for entry in clustered_faces])
    label_to_person_id = {label: idx for idx, label in enumerate(unique_labels)}

    # Use defaultdict to group person_ids by image
    image_to_persons = defaultdict(set)

    # Group person IDs by image
    for entry in clustered_faces:
        person_id = label_to_person_id[entry['label']]
        image_to_persons[entry['image']].add(person_id)

    # Convert sets to lists for JSON serialization
    output_data = {
        "total_unique_people": len(unique_labels),  # Add total number of unique people
        "images": [
            {
                "image": image_name,
                "person_ids": sorted(list(person_ids))  # Sort the IDs for consistency
            }
            for image_name, person_ids in image_to_persons.items()
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Clustered data saved to {output_file}")
    print(f"Total number of unique people detected: {len(unique_labels)}")


def main(folder_path, output_file):
    all_face_encodings = process_images_from_folder(folder_path)

    if not all_face_encodings:
        print("No faces found in the images.")
        return

    clustered_faces = cluster_faces(all_face_encodings)
    assign_person_ids(clustered_faces, output_file)


# Folder containing images and JSON output file
folder_path = "ex-images"
output_file = "clustered_faces.json"

if __name__ == "__main__":
    main(folder_path, output_file)