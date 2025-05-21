from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB connection string from environment variables
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "ecommerce")

client = None
db = None

async def connect_to_mongodb():
    """Connect to MongoDB Atlas"""
    global client, db
    try:
        client = AsyncIOMotorClient(MONGODB_URI)
        db = client[DB_NAME]
        # Verify connection
        await client.admin.command('ping')
        logging.info("Connected to MongoDB Atlas successfully")
        return db
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        raise

async def get_db():
    """Return database instance"""
    if db is None:
        await connect_to_mongodb()
    return db

async def close_mongodb_connection():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()
        logging.info("MongoDB connection closed")