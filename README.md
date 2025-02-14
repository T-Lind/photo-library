# Photo Management System

A sophisticated photo management system that enables semantic search, facial recognition, and temporal organization of
your photo collection. The system processes images to extract embeddings using CLIP, detects and clusters faces using
face_recognition, and stores everything in a LanceDB database for efficient retrieval.

With this system, you can store and search through your family photos locally! There are NO API calls and NO internet
connectivity is required. Your personal photos stay private.

## Redacted Example
![Photo Example](https://github.com/T-Lind/photo-library/blob/master/photos-example.png)

## Features

- **Semantic Image Search**: Using CLIP embeddings for natural language photo search
- **Face Detection & Recognition**: Automatic face detection and clustering of people across photos
- **EXIF Data Processing**: Extraction of timestamp and location data from images
- **Multi-format Support**: Handles various image formats including JPEG, PNG, HEIC, and RAW files
- **Vector Search**: Efficient text-to-image similarity search using LanceDB and the open-source OpenAI CLIP model

## System Components

- `main.py`: Core orchestration script for processing images and managing the database
- `get_emb.py`: CLIP model integration for semantic embeddings
- `get_exif.py`: EXIF data extraction utilities
- `proc_imgs.py`: Face detection and processing pipeline

## Requirements

- Python 3.8+
- PyTorch
- transformers
- face_recognition
- LanceDB
- PIL/Pillow
- pyheif
- scikit-learn
- numpy
  (for a full list, install the requirements.txt)

## Installation

If you're starting from scratch, take a look at `setup.sh` which provides all of the commands needed to get it up and
running, starting from an Ubuntu 24.04 environment. Please note versions prior to 23.04 will NOT WORK! It also does not
work on Windows, sadly :/ (but you CAN use WSL).

```bash
pip install -r requirements.txt
```

## Usage

1. Place your images in a directory
2. Run the processing pipeline:

```bash
python main_load.py
```

## API Endpoints

The system provides REST API endpoints for:

- Semantic image search using natural language queries
- Face-based photo search
- Temporal search and filtering
- Individual photo retrieval
- Person management (naming, merging identities)

See the OpenAPI specification for detailed endpoint documentation.

## Database Schema

### People Table

- `people_id`: Unique identifier for each person
- `name`: Person's name (can be updated via API)

### Images Table

- `image_id`: Unique identifier for each image
- `vector`: CLIP embedding vector (512 dimensions)
- `image_path`: Path to the original image
- `people_ids`: List of people present in the image
- `date`: Timestamp from EXIF data
- `location`: Geographic location (if available in EXIF)
