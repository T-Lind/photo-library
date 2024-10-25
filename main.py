from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List, Optional
from datetime import date
from pydantic import BaseModel
import lancedb
import pandas as pd
from pathlib import Path
import logging
from get_emb import get_text_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Photo Search API")

# Configuration
DB_URI = "data/photos-256"
FACE_IMAGES_DIR = "cropped_faces_256"
IMAGES_PER_PAGE = 20
NUM_PROBES = 20  # For vector search
REFINE_FACTOR = 10  # For vector search refinement


class SearchRequest(BaseModel):
    query: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
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
    Combined semantic, temporal, and people-based photo search endpoint with optimized pre-filtering.
    """
    try:
        db = get_db()
        table = db["images"]

        # Build the where clause for pre-filtering
        where_conditions = []

        # Date filters
        if search_request.start_date:
            where_conditions.append(
                f"date >= TIMESTAMP '{search_request.start_date}'"
            )
        if search_request.end_date:
            where_conditions.append(
                f"date <= TIMESTAMP '{search_request.end_date}'"
            )

        # People filter using array_contains
        if search_request.people_ids:
            people_conditions = [
                f"array_contains(people_ids, {pid})"
                for pid in search_request.people_ids
            ]
            where_conditions.append(f"({' OR '.join(people_conditions)})")

        # Combine all conditions
        where_clause = " AND ".join(where_conditions) if where_conditions else None

        # Calculate pagination
        offset = (search_request.page - 1) * search_request.per_page

        if search_request.query:
            # Semantic search with pre-filtering
            query_emb = get_text_embedding(search_request.query)

            search_query = table.search(
                query_emb,
                vector_column_name="vector",  # Match schema field name
                where=where_clause,
                prefilter=True  # Enable pre-filtering
            )

            # Apply pagination after search
            results_df = search_query.limit(search_request.per_page) \
                .offset(offset) \
                .to_pandas()

            # Get total count with same filters
            total_count = len(table.search(
                query_emb,
                vector_column_name="vector",
                where=where_clause,
                prefilter=True
            ).to_pandas())

        else:
            # No semantic search, just filtered pagination
            base_query = table
            if where_clause:
                base_query = base_query.filter(where_clause)

            results_df = base_query.limit(search_request.per_page) \
                .offset(offset) \
                .to_pandas()

            total_count = len(base_query.to_pandas())

        # Format results according to API schema
        results = []
        for _, row in results_df.iterrows():
            results.append({
                "image_id": int(row["image_id"]),
                "date": row["date"].isoformat() if pd.notnull(row["date"]) else None,
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


@app.patch("/api/v1/people/{people_id}")
async def update_person(people_id: int, name: str):
    """
    Update a person's name.
    """
    try:
        db = get_db()
        people_table = db["people"]

        # Update the name
        people_df = people_table.to_pandas()
        if people_id not in people_df["people_id"].values:
            raise HTTPException(status_code=404, detail="Person not found")

        people_df.loc[people_df["people_id"] == people_id, "name"] = name

        # Write back to database
        people_table.delete()
        people_table = db.create_table("people", data=people_df)

        # Return updated person details
        return await get_person(people_id)

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

        return people_list

    except Exception as e:
        logger.error(f"Failed to list people: {e}")
        raise HTTPException(status_code=500, detail=str(e))