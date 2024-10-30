from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import date, datetime
from pydantic import BaseModel, Field
import lancedb
import pandas as pd
from pathlib import Path
import logging
from get_emb import get_text_embedding
import os
import tempfile
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Photo Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configuration
DB_URI = "data/photos-256"
FACE_IMAGES_DIR = "cropped_faces_256"
IMAGES_PER_PAGE = 20
NUM_PROBES = 20  # For vector search
REFINE_FACTOR = 10  # For vector search refinement

THUMBNAIL_SIZES = {
    "small": (150, 150),
    "medium": (300, 300),
    "large": (500, 500)
}


class SearchRequest(BaseModel):
    query: Optional[str] = None
    start_date: Optional[str] = Field(None, description="ISO format date string")
    end_date: Optional[str] = Field(None, description="ISO format date string")
    people_ids: Optional[List[int]] = None
    page: int = 1
    per_page: int = IMAGES_PER_PAGE


class Person(BaseModel):
    people_id: int
    name: str
    photo_count: int
    face_image_url: str


class SearchResults(BaseModel):
    total: int
    page: int
    per_page: int
    results: List[dict]


def get_db():
    """Database connection factory"""
    try:
        db = lancedb.connect(DB_URI)
        return db
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")


def apply_filters(query: pd.DataFrame, start_date: Optional[date],
                  end_date: Optional[date], people_ids: Optional[List[int]]) -> pd.DataFrame:
    """Apply date and people filters to search results"""
    if start_date:
        query = query[query['date'] >= pd.Timestamp(start_date)]
    if end_date:
        query = query[query['date'] <= pd.Timestamp(end_date)]
    if people_ids:
        # Filter for images containing any of the specified people
        query = query[query['people_ids'].apply(
            lambda x: any(pid in x for pid in people_ids)
        )]
    return query


@app.post("/api/v1/search", response_model=SearchResults)
async def search_photos(search_request: SearchRequest):
    """
    Combined semantic, temporal, and people-based photo search endpoint using proper LanceDB where clause.
    """
    try:
        db = get_db()
        images_table = db["images"]

        # Build the where clause conditions
        where_conditions = []

        start_date = datetime.fromisoformat(search_request.start_date.replace('Z', '+00:00')) if search_request.start_date else None
        end_date = datetime.fromisoformat(search_request.end_date.replace('Z', '+00:00')) if search_request.end_date else None

        # Add date filters if provided, but handle null dates
        if start_date:
            where_conditions.append(
                f"(date >= TIMESTAMP '{start_date.date()}' OR date IS NULL)"
            )
        if end_date:
            where_conditions.append(
                f"(date <= TIMESTAMP '{end_date.date()}' OR date IS NULL)"
            )

        # Add people filter if provided
        if search_request.people_ids:
            people_conditions = [
                f"array_contains(people_ids, {pid})"
                for pid in search_request.people_ids
            ]
            where_conditions.append(f"({' OR '.join(people_conditions)})")

        # Combine all conditions with AND
        where_clause = " AND ".join(where_conditions) if where_conditions else None

        # Calculate pagination
        offset = (search_request.page - 1) * search_request.per_page

        # Execute search query
        if search_request.query:
            # Get embedding for semantic search
            query_emb = get_text_embedding(search_request.query)

            # Build and execute search with prefiltering
            if where_clause:
                results = (
                    images_table.search(query_emb)
                    .where(where_clause, prefilter=True)
                )
            else:
                results = images_table.search(query_emb)

            # Get total count first
            total_count = len(results.to_arrow())

            # Then get paginated results
            results_df = (
                results
                .limit(search_request.per_page)
                .offset(offset)
                .to_pandas()
            )
        else:
            # No semantic search, just filtering and pagination
            query = images_table
            if where_clause:
                query = query.search().where(where_clause)

            # Get total count
            total_count = len(query.to_arrow())

            # Get paginated results
            results_df = (
                query
                .limit(search_request.per_page)
                .offset(offset)
                .to_pandas()
            )

        # Format results according to API schema
        results = []
        for _, row in results_df.iterrows():
            date_value = row["date"]
            formatted_date = None

            # Only format valid dates after 1970
            if pd.notnull(date_value) and isinstance(date_value, pd.Timestamp):
                if date_value.year > 1970:
                    formatted_date = date_value.isoformat()
                else:
                    formatted_date = None

            results.append({
                "image_id": int(row["image_id"]),
                "date": formatted_date,  # Will be None for invalid/null dates
                "location": row["location"] if pd.notnull(row["location"]) else "",
                "people_ids": row["people_ids"].tolist() if isinstance(row["people_ids"], (list, pd.Series)) else [],
                "thumbnail_url": f"/api/v1/images/{row['image_id']}/thumbnail"
            })

        return SearchResults(
            total=total_count,
            page=search_request.page,
            per_page=search_request.per_page,
            results=results
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/people/{people_id}")
async def get_person(people_id: int):
    """
    Get details about a specific person, including their face image.
    """
    try:
        db = get_db()
        people_table = db["people"]
        images_table = db["images"]

        # Get person details
        person_df = people_table.to_pandas()
        person = person_df[person_df["people_id"] == people_id]

        if person.empty:
            raise HTTPException(status_code=404, detail="Person not found")

        # Count photos with this person
        images_df = images_table.to_pandas()
        photo_count = len(images_df[
                              images_df["people_ids"].apply(lambda x: people_id in x)
                          ])

        # Check if face image exists
        face_path = Path(FACE_IMAGES_DIR) / f"person_{people_id}.jpg"
        if not face_path.exists():
            raise HTTPException(status_code=404, detail="Face image not found")

        return Person(
            people_id=people_id,
            name=person.iloc[0]["name"],
            photo_count=photo_count,
            face_image_url=f"/api/v1/people/{people_id}/face"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get person details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/people/{people_id}/face")
async def get_person_face(people_id: int):
    """
    Get the face image for a specific person.
    """
    face_path = Path(FACE_IMAGES_DIR) / f"person_{people_id}.jpg"
    if not face_path.exists():
        raise HTTPException(status_code=404, detail="Face image not found")

    return FileResponse(face_path, media_type="image/jpeg")

class UpdatePersonRequest(BaseModel):
    name: str

@app.patch("/api/v1/people/{people_id}")
async def update_person(people_id: int, request: UpdatePersonRequest = Body(...)):
    """
    Update a person's name.
    """
    try:
        db = get_db()
        images_table = db.open_table("images")

        images_df = images_table.to_pandas()

        people_table = db.open_table("people")


        people_table.update(where=f"people_id = {people_id}", values={"name": request.name})
        return Person(
            people_id=people_id,
            name=request.name,
            photo_count=-1,
            face_image_url=f"/api/v1/people/{people_id}/face"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/people", response_model=List[Person])
async def list_people():
    """
    List all people detected in photos, including their photo counts and face image URLs.
    """
    try:
        db = get_db()
        people_table = db["people"]
        images_table = db["images"]

        # Get all people
        people_df = people_table.to_pandas()

        # Get image counts for each person using a single pass through the images table
        images_df = images_table.to_pandas()

        # Calculate photo counts for all people at once
        photo_counts = {}
        for _, row in images_df.iterrows():
            for person_id in row["people_ids"]:
                photo_counts[person_id] = photo_counts.get(person_id, 0) + 1

        # Build the response
        people_list = []
        for _, person in people_df.iterrows():
            person_id = person["people_id"]

            # Check if face image exists
            face_path = Path(FACE_IMAGES_DIR) / f"person_{person_id}.jpg"

            # Only include people who have a face image
            if face_path.exists():
                people_list.append(
                    Person(
                        people_id=person_id,
                        name=person["name"],
                        photo_count=photo_counts.get(person_id, 0),
                        face_image_url=f"/api/v1/people/{person_id}/face"
                    )
                )

        # Sort by photo count descending, then by name
        people_list.sort(key=lambda x: (-x.photo_count, x.name))
        print("People list", people_list)
        return people_list

    except Exception as e:
        logger.error(f"Failed to list people: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_image_path(image_id: int, db) -> str:
    """Get the image path from the database for a given image ID."""
    try:
        table = db["images"]
        result = (
            table.search()
            .where(f"image_id = {image_id}")
            .limit(1)
            .to_pandas()
        )

        if result.empty:
            raise HTTPException(status_code=404, detail="Image not found")

        return result.iloc[0]["image_path"]
    except Exception as e:
        logger.error(f"Failed to get image path: {e}")
        raise HTTPException(status_code=500, detail="Database error")


def create_thumbnail(image_path: str, size: tuple) -> str:
    """
    Create a thumbnail of the specified size while maintaining aspect ratio.
    Supports both regular image formats and HEIC files.
    Returns path to temporary thumbnail file.
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as img:
            # Convert to RGB if necessary (e.g., for PNGs with transparency or HEIC)
            if img.mode in ('RGBA', 'P', 'CMYK'):
                img = img.convert('RGB')

            # Calculate new dimensions maintaining aspect ratio
            orig_width, orig_height = img.size
            target_width, target_height = size

            # Calculate aspect ratios
            aspect = orig_width / orig_height
            target_aspect = target_width / target_height

            if aspect > target_aspect:
                # Image is wider than target
                new_width = target_width
                new_height = int(target_width / aspect)
            else:
                # Image is taller than target
                new_height = target_height
                new_width = int(target_height * aspect)

            # Resize with high-quality antialiasing
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Create temporary file for thumbnail
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            img.save(temp_file.name, "JPEG", quality=85, optimize=True)

            return temp_file.name

    except Exception as e:
        logger.error(f"Failed to create thumbnail for {image_path}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Thumbnail creation failed: {str(e)}"
        )

@app.get("/api/v1/images/{image_id}")
async def get_original_image(image_id: int):
    """Get the original image file."""
    try:
        db = get_db()
        image_path = get_image_path(image_id, db)

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")

        return FileResponse(
            image_path,
            media_type="image/jpeg",  # Adjust if you need to handle other formats
            filename=f"image_{image_id}{Path(image_path).suffix}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image")


@app.get("/api/v1/images/{image_id}/thumbnail")
async def get_image_thumbnail(
        image_id: int,
        size: str = Query("medium", enum=["small", "medium", "large"])
):
    """Get a thumbnail of the image at the specified size."""
    try:
        db = get_db()
        image_path = get_image_path(image_id, db)

        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")

        # Create thumbnail
        thumbnail_path = create_thumbnail(image_path, THUMBNAIL_SIZES[size])

        # Use FileResponse with cleanup callback
        def cleanup_thumbnail():
            try:
                os.unlink(thumbnail_path)
            except:
                pass

        return FileResponse(
            thumbnail_path,
            media_type="image/jpeg",
            filename=f"thumbnail_{image_id}.jpg",
            background=cleanup_thumbnail
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get thumbnail: {e}")
        raise HTTPException(status_code=500, detail="Failed to create thumbnail")


# Optional: Add a cleanup route for temporary files
@app.on_event("shutdown")
async def cleanup_temp_files():
    """Clean up any remaining temporary thumbnail files on shutdown."""
    temp_dir = tempfile.gettempdir()
    try:
        for file in Path(temp_dir).glob("*thumbnail_*.jpg"):
            try:
                os.unlink(file)
            except:
                pass
    except Exception as e:
        logger.error(f"Failed to clean up temporary files: {e}")
