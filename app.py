from fastapi import FastAPI, File, UploadFile, HTTPException, Depends,Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv
from pipeline import image_retrieval_pipeline, image_retrieval_pipeline_from_url
import logging
import shutil
from db import connect_to_mongodb, get_db, close_mongodb_connection
from models import ItemModel, SimilarItemsResponse
from bson import ObjectId
from typing import List, Optional

# Load environment variables
load_dotenv()

app = FastAPI(title="Clothing Image Retrieval API", 
              description="API for retrieving similar clothing items from MongoDB Atlas",
              version="1.0.0")

@app.get("/favicon.ico", include_in_schema=False)
async def ignore_favicon():
    return Response(status_code=204)  # No Content

# Setup static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================= Security Configurations =================
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# ===========================================================

# Startup and shutdown events
@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongodb()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongodb_connection()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse('static/favicon.ico')

@app.get("/")
async def root():
    return {
        "message": "Image Retrieval API is running on MongoDB Atlas!",
        "endpoints": {
            "retrieve_images_upload": "POST /retrieve",
            "retrieve_images_url": "POST /retrieve/url",
            "get_item": "GET /items/{item_id}",
            "docs": "GET /docs",
            "redoc": "GET /redoc"
        }
    }

@app.get("/items/{item_id}", response_model=ItemModel)
async def get_item(item_id: str, db = Depends(get_db)):
    """Get a single item by ID"""
    try:
        item = await db.items.find_one({"_id": ObjectId(item_id)})
        if not item:
            raise HTTPException(status_code=404, detail=f"Item with ID {item_id} not found")
        return item
    except Exception as e:
        logging.error(f"Error retrieving item: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve", response_model=List[ItemModel])
async def retrieve_images(file: UploadFile = File(...), db = Depends(get_db)):
    """Upload an image to find similar clothing items"""
    if not file.filename or not allowed_file(file.filename):
        logging.error("Invalid file type or empty filename")
        raise HTTPException(status_code=400, detail="Invalid file type or empty filename")

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logging.error(f"Failed to save the file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save the file")

    try:
        similar_item_ids = image_retrieval_pipeline(filepath)

        if not similar_item_ids:
            logging.error("No similar items found")
            raise HTTPException(status_code=404, detail="No similar items found")

        # Retrieve the actual items from MongoDB
        similar_items = []
        for item_id in similar_item_ids:
            try:
                # Try to convert to ObjectId if it's a valid ObjectId
                if ObjectId.is_valid(item_id):
                    item = await db.items.find_one({"_id": ObjectId(item_id)})
                else:
                    # If not a valid ObjectId, try as a string ID
                    item = await db.items.find_one({"_id": item_id})
                
                if item:
                    similar_items.append(item)
            except Exception as e:
                logging.warning(f"Error retrieving item {item_id}: {e}")

        if not similar_items:
            logging.error("Could not find any matching items in the database")
            raise HTTPException(status_code=404, detail="No matching items found in the database")

        return similar_items

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always try to remove the uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            logging.error(f"Failed to remove the file: {e}")

@app.post("/retrieve/url", response_model=List[ItemModel])
async def retrieve_images_by_url(image_url: str, db = Depends(get_db)):
    """Find similar clothing items from an image URL"""
    try:
        similar_item_ids = image_retrieval_pipeline_from_url(image_url)

        if not similar_item_ids:
            logging.error("No similar items found")
            raise HTTPException(status_code=404, detail="No similar items found")

        # Retrieve the actual items from MongoDB
        similar_items = []
        for item_id in similar_item_ids:
            try:
                # Try to convert to ObjectId if it's a valid ObjectId
                if ObjectId.is_valid(item_id):
                    item = await db.items.find_one({"_id": ObjectId(item_id)})
                else:
                    # If not a valid ObjectId, try as a string ID
                    item = await db.items.find_one({"_id": item_id})
                
                if item:
                    similar_items.append(item)
            except Exception as e:
                logging.warning(f"Error retrieving item {item_id}: {e}")

        if not similar_items:
            logging.error("Could not find any matching items in the database")
            raise HTTPException(status_code=404, detail="No matching items found in the database")

        return similar_items

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))