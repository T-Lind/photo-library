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
DB_URI = "data/photos"
FACE_IMAGES_DIR = "cropped_faces"
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
    Combined semantic, temporal, and people-based photo search endpoint.
    """
    try:
        db = get_db()
        table = db["images"]

        # Start with all images if no semantic query
        if search_request.query:
            query_emb = get_text_embedding(search_request.query)
            results_df = table.search(query_emb) \
                .limit(1000) \
                .nprobes(NUM_PROBES) \
                .refine_factor(REFINE_FACTOR) \
                .to_pandas()
        else:
            results_df = table.to_pandas()

        # Apply filters
        filtered_df = apply_filters(
            results_df,
            search_request.start_date,
            search_request.end_date,
            search_request.people_ids
        )

        # Pagination
        start_idx = (search_request.page - 1) * search_request.per_page
        end_idx = start_idx + search_request.per_page
        paginated_df = filtered_df.iloc[start_idx:end_idx]

        # Format results
        results = []
        for _, row in paginated_df.iterrows():
            results.append({
                "image_id": int(row["image_id"]),
                "date": row["date"].isoformat() if pd.notnull(row["date"]) else None,
                "location": row["location"] if pd.notnull(row["location"]) else "",
                "people_ids": row["people_ids"],
                "thumbnail_url": f"/api/v1/images/{row['image_id']}/thumbnail"
            })

        return SearchResults(
            total=len(filtered_df),
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