import tensorflow as tf
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from transformers import SiglipVisionModel, AutoProcessor
import os
import json
import logging
import faiss
from dotenv import load_dotenv
import sys
import io

# Load environment variables
load_dotenv()

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
tf.get_logger().setLevel('ERROR')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Access environment variables
RESNET_MODEL_PATH = os.getenv('RESNET_MODEL_PATH', 'resnet_model.keras')
SIGLIP_MODEL_FOLDER = os.getenv('SIGLIP_MODEL_FOLDER', 'siglip_model')
CLASS_LABELS_PATH = os.getenv('CLASS_LABELS_PATH', 'class_labels.json')
EMBEDDINGS_FOLDER = os.getenv('EMBEDDINGS_FOLDER', 'embeddings')

# ========= Load models locally =========

# Custom function to load TensorFlow model with error handling
def load_tf_model_safely(model_path):
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Try different loading approaches based on the TensorFlow version
    try:
        # First approach: standard loading
        return tf.keras.models.load_model(model_path, compile=False)
    except (ImportError, TypeError) as e:
        logging.warning(f"Standard model loading failed: {e}. Trying alternative approach.")
        try:
            # Alternative for TensorFlow compatibility issues
            # Try loading with custom_objects for compatibility
            return tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={}
            )
        except Exception as e2:
            logging.error(f"All model loading attempts failed: {e2}")
            raise RuntimeError(f"Could not load model due to compatibility issues: {e2}")

# Load TensorFlow ResNet classifier
original_stdout = sys.stdout
sys.stdout = io.StringIO()  # Redirect stdout
try:
    classifier = load_tf_model_safely(RESNET_MODEL_PATH)
    logging.info("Loaded TensorFlow ResNet model successfully.")
finally:
    sys.stdout = original_stdout  # Restore stdout

# Load SiglipVisionModel and processor locally
try:
    embedding_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_FOLDER)
    processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_FOLDER)
    logging.info("Loaded Siglip model and processor successfully.")
except Exception as e:
    logging.error(f"Failed to load Siglip model: {e}")
    raise

# Load class labels locally
if not os.path.exists(CLASS_LABELS_PATH):
    logging.error(f"Class labels file not found: {CLASS_LABELS_PATH}")
    raise FileNotFoundError(f"Class labels file not found at {CLASS_LABELS_PATH}")

with open(CLASS_LABELS_PATH, "r") as f:
    lis_type = json.load(f)
logging.info("Loaded class labels successfully.")

# ========= Helper Functions =========

def preprocess_image_from_url(image_url, image_size=224):
    """Process image from URL instead of local file system"""
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        logging.error(f"Failed to preprocess image from URL: {e}")
        raise

def preprocess_image(image_path, image_size=224):
    """Process image from local file system or uploaded file"""
    try:
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        logging.error(f"Failed to preprocess image: {e}")
        raise

def classify_image_from_url(image_url):
    """Classify image from URL"""
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize((224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0) / 255.0  # Normalize
        
        # Suppress TensorFlow prediction output
        with tf.device('/CPU:0'):  # Force CPU to avoid more verbose GPU logs
            prediction = classifier.predict(image_array, verbose=0)
        
        category_idx = np.argmax(prediction)
        category_name = lis_type[category_idx]
        return category_name
    except Exception as e:
        logging.error(f"Failed to classify image from URL: {e}")
        raise

def classify_image(image_path):
    """Classify image from local file system or uploaded file"""
    try:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0) / 255.0  # Normalize
        
        # Suppress TensorFlow prediction output
        with tf.device('/CPU:0'):  # Force CPU to avoid more verbose GPU logs
            prediction = classifier.predict(image_array, verbose=0)
        
        category_idx = np.argmax(prediction)
        category_name = lis_type[category_idx]
        return category_name
    except Exception as e:
        logging.error(f"Failed to classify image: {e}")
        raise

def extract_embedding_from_url(image_url):
    """Extract embedding from URL"""
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = embedding_model(**inputs)

        return outputs.pooler_output.squeeze(0).cpu().numpy()
    except Exception as e:
        logging.error(f"Failed to extract embedding from URL: {e}")
        raise

def extract_embedding(image_path):
    """Extract embedding from local file system or uploaded file"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = embedding_model(**inputs)

        return outputs.pooler_output.squeeze(0).cpu().numpy()
    except Exception as e:
        logging.error(f"Failed to extract embedding: {e}")
        raise

def build_faiss_index(embeddings):
    try:
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index
    except Exception as e:
        logging.error(f"Failed to build FAISS index: {e}")
        raise

def find_similar_images_faiss(query_embedding, index, item_ids, top_n=5):
    """Find similar items using FAISS and return their IDs"""
    try:
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = index.search(query_embedding, top_n)
        
        similar_item_ids = [item_ids[i] for i in indices[0]]
        return similar_item_ids
    except Exception as e:
        logging.error(f"Failed to find similar images: {e}")
        raise

def load_category_embeddings(category_folder):
    """Load embeddings and item IDs for a category"""
    try:
        embeddings = []
        item_ids = []

        category_path = os.path.join(EMBEDDINGS_FOLDER, category_folder)
        if not os.path.exists(category_path):
            logging.warning(f"Category embeddings folder not found: {category_path}")
            return np.array([]), []

        for file_name in os.listdir(category_path):
            if file_name.endswith(".npy"):
                file_path = os.path.join(category_path, file_name)
                emb = np.load(file_path)
                embeddings.append(emb)

                # Use the filename (without extension) as the item ID
                item_id = os.path.splitext(file_name)[0]
                item_ids.append(item_id)

        logging.info(f"Loaded {len(embeddings)} embeddings for category: {category_folder}")
        return np.array(embeddings), item_ids
    except Exception as e:
        logging.error(f"Failed to load category embeddings: {e}")
        raise

# ========= Main Pipeline =========

def image_retrieval_pipeline_from_url(image_url, top_n=10):
    """Image retrieval pipeline that processes images from URLs"""
    try:
        category_name = classify_image_from_url(image_url)
        logging.info(f"Predicted Category: {category_name}")
        
        category_embeddings, item_ids = load_category_embeddings(category_name)
        
        if len(category_embeddings) == 0:
            logging.warning(f"No embeddings found for category: {category_name}")
            return []
        
        index = build_faiss_index(category_embeddings)
        
        query_embedding = extract_embedding_from_url(image_url)
        similar_item_ids = find_similar_images_faiss(query_embedding, index, item_ids, top_n)
        
        return similar_item_ids
    except Exception as e:
        logging.error(f"An error occurred in the retrieval pipeline: {e}")
        raise

def image_retrieval_pipeline(image_path, top_n=10):
    """Image retrieval pipeline that processes uploaded images"""
    try:
        category_name = classify_image(image_path)
        logging.info(f"Predicted Category: {category_name}")
        
        category_embeddings, item_ids = load_category_embeddings(category_name)
        
        if len(category_embeddings) == 0:
            logging.warning(f"No embeddings found for category: {category_name}")
            return []
        
        index = build_faiss_index(category_embeddings)
        
        query_embedding = extract_embedding(image_path)
        similar_item_ids = find_similar_images_faiss(query_embedding, index, item_ids, top_n)
        
        return similar_item_ids
    except Exception as e:
        logging.error(f"An error occurred in the retrieval pipeline: {e}")
        raise