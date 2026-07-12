from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import lancedb
import pandas as pd
import pyarrow.compute as pc
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

# Configuration (overridable via environment variables)
DB_URI = os.environ.get("PHOTO_DB_URI", "data/photos-256")
FACE_IMAGES_DIR = os.environ.get("PHOTO_FACES_DIR", "cropped_faces_256")
THUMBNAIL_CACHE_DIR = os.environ.get("PHOTO_THUMBNAIL_CACHE_DIR", "thumbnail_cache")
CORS_ORIGINS = os.environ.get("PHOTO_CORS_ORIGINS", "http://localhost:3000").split(",")

IMAGES_PER_PAGE = 20
NUM_PROBES = 20  # For vector search
REFINE_FACTOR = 10  # For vector search refinement

# Columns returned to clients; excludes the 512-dim embedding vectors so we
# never pull them out of the database for plain result listing.
IMAGE_RESULT_COLUMNS = ["image_id", "image_path", "people_ids", "date", "location"]

THUMBNAIL_SIZES = {
    "small": (150, 150),
    "medium": (300, 300),
    "large": (500, 500)
}

# Initialize FastAPI app
app = FastAPI(title="Photo Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class SearchRequest(BaseModel):
    query: Optional[str] = None
    start_date: Optional[str] = Field(None, description="ISO format date string")
    end_date: Optional[str] = Field(None, description="ISO format date string")
    people_ids: Optional[List[int]] = None
    page: int = Field(1, ge=1)
    per_page: int = Field(IMAGES_PER_PAGE, ge=1, le=100)


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


class UpdatePersonRequest(BaseModel):
    name: str


class MergePeopleRequest(BaseModel):
    source_id: int
    target_id: int


def get_db():
    """Database connection factory"""
    try:
        db = lancedb.connect(DB_URI)
        return db
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")


def format_image_date(value):
    """ISO-format a stored timestamp; None for missing/epoch-garbage dates"""
    if pd.notnull(value) and isinstance(value, pd.Timestamp) and value.year > 1970:
        return value.isoformat()
    return None


def row_to_image(row):
    """Convert an images-table row to the API's Image representation"""
    people_ids = row["people_ids"]
    if not isinstance(people_ids, list):
        people_ids = list(people_ids) if people_ids is not None else []

    return {
        "image_id": int(row["image_id"]),
        "date": format_image_date(row["date"]),
        "location": row["location"] if pd.notnull(row["location"]) else "",
        "people_ids": [int(pid) for pid in people_ids],
        "thumbnail_url": f"/api/v1/images/{int(row['image_id'])}/thumbnail"
    }


def person_photo_count(images_table, people_id: int) -> int:
    return images_table.count_rows(f"array_contains(people_ids, {people_id})")


def person_exists(people_table, people_id: int) -> bool:
    return people_table.count_rows(f"people_id = {people_id}") > 0


def get_person_name(people_table, people_id: int) -> Optional[str]:
    df = (
        people_table.search()
        .where(f"people_id = {people_id}")
        .limit(1)
        .to_pandas()
    )
    if df.empty:
        return None
    return df.iloc[0]["name"]


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

        # Total matches: a vector search ranks every row that passes the
        # filter, so the filtered row count is the total for both paths.
        total_count = images_table.count_rows(where_clause)

        # Calculate pagination
        offset = (search_request.page - 1) * search_request.per_page

        # Execute search query
        if search_request.query:
            # Get embedding for semantic search
            query_emb = get_text_embedding(search_request.query)

            search = (
                images_table.search(query_emb)
                .nprobes(NUM_PROBES)
                .refine_factor(REFINE_FACTOR)
            )
            if where_clause:
                search = search.where(where_clause, prefilter=True)
        else:
            # No semantic search, just filtering and pagination
            search = images_table.search()
            if where_clause:
                search = search.where(where_clause)

        # LanceDB 0.14 silently ignores offset() on non-vector queries, so
        # fetch the first offset+per_page rows (cheap: small columns only,
        # no vectors) and slice off the earlier pages.
        results_df = (
            search
            .select(IMAGE_RESULT_COLUMNS)
            .limit(offset + search_request.per_page)
            .to_pandas()
            .iloc[offset:]
        )

        return SearchResults(
            total=total_count,
            page=search_request.page,
            per_page=search_request.per_page,
            results=[row_to_image(row) for _, row in results_df.iterrows()]
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_stats():
    """
    Library-wide statistics: image/people counts and the covered date range.
    """
    try:
        db = get_db()
        images_table = db["images"]
        people_table = db["people"]

        total_images = images_table.count_rows(None)
        total_people = people_table.count_rows(None)
        images_with_location = images_table.count_rows("location != ''")

        earliest = latest = None
        if total_images:
            # Read only the date column, never the embedding vectors
            dates = images_table.to_lance().to_table(columns=["date"])["date"]
            min_date, max_date = pc.min(dates).as_py(), pc.max(dates).as_py()
            if min_date is not None:
                earliest = min_date.isoformat()
            if max_date is not None:
                latest = max_date.isoformat()

        return {
            "total_images": total_images,
            "total_people": total_people,
            "images_with_location": images_with_location,
            "earliest_date": earliest,
            "latest_date": latest,
        }

    except Exception as e:
        logger.error(f"Failed to compute stats: {e}")
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

        name = get_person_name(people_table, people_id)
        if name is None:
            raise HTTPException(status_code=404, detail="Person not found")

        # Check if face image exists
        face_path = Path(FACE_IMAGES_DIR) / f"person_{people_id}.jpg"
        if not face_path.exists():
            raise HTTPException(status_code=404, detail="Face image not found")

        return Person(
            people_id=people_id,
            name=name,
            photo_count=person_photo_count(images_table, people_id),
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


@app.patch("/api/v1/people/{people_id}")
async def update_person(people_id: int, request: UpdatePersonRequest = Body(...)):
    """
    Update a person's name.
    """
    try:
        db = get_db()
        people_table = db.open_table("people")

        if not person_exists(people_table, people_id):
            raise HTTPException(status_code=404, detail="Person not found")

        people_table.update(where=f"people_id = {people_id}", values={"name": request.name})

        return Person(
            people_id=people_id,
            name=request.name,
            photo_count=person_photo_count(db.open_table("images"), people_id),
            face_image_url=f"/api/v1/people/{people_id}/face"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def remove_person_from_images(images_table, people_id: int, replacement_id: Optional[int] = None) -> int:
    """Remove (or replace) a person id in every image's people_ids list.

    Returns the number of affected images.
    """
    affected_count = person_photo_count(images_table, people_id)
    if affected_count == 0:
        return 0

    affected = (
        images_table.search()
        .where(f"array_contains(people_ids, {people_id})")
        .select(["image_id", "people_ids"])
        .limit(affected_count)
        .to_pandas()
    )

    for _, row in affected.iterrows():
        ids = {int(pid) for pid in row["people_ids"]}
        ids.discard(people_id)
        if replacement_id is not None:
            ids.add(replacement_id)
        where = f"image_id = {int(row['image_id'])}"
        if ids:
            images_table.update(where=where, values={"people_ids": sorted(ids)})
        else:
            # An empty Python list can't be type-inferred by update();
            # make_array() writes a typed empty list instead.
            images_table.update(where=where, values_sql={"people_ids": "make_array()"})

    return affected_count


@app.post("/api/v1/people/merge")
async def merge_people(request: MergePeopleRequest):
    """
    Merge two person entries (for when the same person was detected as two
    different people). All of source's photos are re-attributed to target,
    then the source person is removed.
    """
    try:
        if request.source_id == request.target_id:
            raise HTTPException(status_code=400, detail="Cannot merge a person into themselves")

        db = get_db()
        people_table = db["people"]
        images_table = db["images"]

        target_name = get_person_name(people_table, request.target_id)
        if target_name is None:
            raise HTTPException(status_code=404, detail="Target person not found")
        if not person_exists(people_table, request.source_id):
            raise HTTPException(status_code=404, detail="Source person not found")

        remove_person_from_images(images_table, request.source_id, replacement_id=request.target_id)
        people_table.delete(f"people_id = {request.source_id}")

        # Remove the now-orphaned face crop of the source person
        source_face = Path(FACE_IMAGES_DIR) / f"person_{request.source_id}.jpg"
        if source_face.exists():
            source_face.unlink()

        return Person(
            people_id=request.target_id,
            name=target_name,
            photo_count=person_photo_count(images_table, request.target_id),
            face_image_url=f"/api/v1/people/{request.target_id}/face"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to merge people: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/people/{people_id}")
async def delete_person(people_id: int, permanent: bool = Query(False)):
    """
    Delete a person: removes them from all images and from the people table.
    With permanent=true, their cropped face image is also deleted from disk.
    """
    try:
        db = get_db()
        people_table = db["people"]
        images_table = db["images"]

        if not person_exists(people_table, people_id):
            raise HTTPException(status_code=404, detail="Person not found")

        affected = remove_person_from_images(images_table, people_id)
        people_table.delete(f"people_id = {people_id}")

        if permanent:
            face_path = Path(FACE_IMAGES_DIR) / f"person_{people_id}.jpg"
            if face_path.exists():
                face_path.unlink()

        return {
            "success": True,
            "message": f"Person {people_id} {'permanently deleted' if permanent else 'deleted'}",
            "affected_images": affected,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete person: {e}")
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

        # Get all people (small table)
        people_df = people_table.to_pandas()

        # Count photos per person in one pass over just the people_ids
        # column — the embedding vectors never leave the database.
        photo_counts = {}
        for ids in images_table.to_lance().to_table(columns=["people_ids"])["people_ids"].to_pylist():
            for person_id in ids or []:
                photo_counts[person_id] = photo_counts.get(person_id, 0) + 1

        # Build the response
        people_list = []
        for _, person in people_df.iterrows():
            person_id = int(person["people_id"])

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
        return people_list

    except Exception as e:
        logger.error(f"Failed to list people: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_image_row(image_id: int, db, columns: List[str]):
    """Fetch selected columns of one images-table row, or 404."""
    result = (
        db["images"].search()
        .where(f"image_id = {image_id}")
        .select(columns)
        .limit(1)
        .to_pandas()
    )

    if result.empty:
        raise HTTPException(status_code=404, detail="Image not found")

    return result.iloc[0]


def get_image_path(image_id: int, db) -> str:
    """Get the image path from the database for a given image ID."""
    try:
        return get_image_row(image_id, db, ["image_path"])["image_path"]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image path: {e}")
        raise HTTPException(status_code=500, detail="Database error")


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
            filename=f"image_{image_id}{Path(image_path).suffix}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image")


@app.get("/api/v1/images/{image_id}/details")
async def get_image_details(image_id: int):
    """Full metadata for one image, with resolved people names."""
    try:
        db = get_db()
        row = get_image_row(image_id, db, IMAGE_RESULT_COLUMNS)

        image = row_to_image(row)
        image["image_url"] = f"/api/v1/images/{image_id}"
        image["similar_url"] = f"/api/v1/images/{image_id}/similar"
        image["filename"] = Path(row["image_path"]).name

        # Resolve people names (people table is small)
        people_df = db["people"].to_pandas()
        names = dict(zip(people_df["people_id"].astype(int), people_df["name"]))
        image["people"] = [
            {
                "people_id": pid,
                "name": names.get(pid, ""),
                "face_image_url": f"/api/v1/people/{pid}/face",
            }
            for pid in image["people_ids"]
        ]

        return image

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get image details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/images/{image_id}/similar")
async def get_similar_images(image_id: int, limit: int = Query(12, ge=1, le=100)):
    """Find visually similar images using the stored CLIP embedding."""
    try:
        db = get_db()
        row = get_image_row(image_id, db, ["vector"])

        results_df = (
            db["images"].search(list(row["vector"]))
            .nprobes(NUM_PROBES)
            .refine_factor(REFINE_FACTOR)
            .where(f"image_id != {image_id}", prefilter=True)
            .select(IMAGE_RESULT_COLUMNS)
            .limit(limit)
            .to_pandas()
        )

        return {"results": [row_to_image(r) for _, r in results_df.iterrows()]}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find similar images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_or_create_thumbnail(image_id: int, image_path: str, size_name: str) -> str:
    """
    Return a cached thumbnail path, generating it on first request.

    Thumbnails are cached on disk keyed by image id and size, and
    regenerated if the source image is newer than the cached file.
    JPEG sources use draft-mode decoding, which decodes directly to a
    reduced resolution instead of loading the full image into memory.
    """
    cache_dir = Path(THUMBNAIL_CACHE_DIR) / size_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{image_id}.jpg"

    if cache_path.exists() and cache_path.stat().st_mtime >= os.path.getmtime(image_path):
        return str(cache_path)

    size = THUMBNAIL_SIZES[size_name]
    try:
        with Image.open(image_path) as img:
            img.draft("RGB", size)  # Fast reduced-resolution decode (JPEG only, no-op otherwise)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.thumbnail(size, Image.Resampling.LANCZOS)

            # Write atomically so a concurrent request never sees a partial file
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=cache_dir)
            os.close(fd)
            try:
                img.save(tmp_path, "JPEG", quality=85, optimize=True)
                os.replace(tmp_path, cache_path)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

        return str(cache_path)

    except Exception as e:
        logger.error(f"Failed to create thumbnail for {image_path}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Thumbnail creation failed: {str(e)}"
        )


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

        thumbnail_path = get_or_create_thumbnail(image_id, image_path, size)

        return FileResponse(
            thumbnail_path,
            media_type="image/jpeg",
            filename=f"thumbnail_{image_id}.jpg",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get thumbnail: {e}")
        raise HTTPException(status_code=500, detail="Failed to create thumbnail")
