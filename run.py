import uvicorn
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from main import app

# Configuration
HOST = "localhost"  # This allows external access. Use "127.0.0.1" for local only
PORT = 5000
DB_URI = "data/photos-256"  # LanceDB database location
FACE_IMAGES_DIR = "cropped_faces_256"  # Directory for face images


def init_app():
    """Initialize the application, create directories and database if needed"""
    # Create necessary directories
    Path(DB_URI).mkdir(parents=True, exist_ok=True)
    Path(FACE_IMAGES_DIR).mkdir(parents=True, exist_ok=True)

    # Mount static files directory for serving images
    app.mount("/faces", StaticFiles(directory=FACE_IMAGES_DIR), name="faces")


def main():
    print("Initializing application...")
    init_app()

    print(f"Starting server on http://{HOST}:{PORT}")
    print("Documentation available at:")
    print(f"  - http://{HOST}:{PORT}/docs (Swagger UI)")
    print(f"  - http://{HOST}:{PORT}/redoc (ReDoc)")

    # Start the server
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,  # Enable auto-reload during development
        workers=1  # Number of worker processes
    )


if __name__ == "__main__":
    main()