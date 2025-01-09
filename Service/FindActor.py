import logging
from supabase import create_client, ClientOptions
from sqlalchemy import create_engine, text
from typing import List
from Domain.domain import TemplateMemeResponse
import face_recognition
from io import BytesIO
from PIL import Image

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Supabase setup
SUPABASE_URL = "https://ixnbfvyeniksbqcfdmdo.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml4bmJmdnllbmlrc2JxY2ZkbWRvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE0MDE3NjgsImV4cCI6MjA0Njk3Nzc2OH0.h4JtVbwtKAe38yvtOLYvZIbhmMy6v2QCVg51Q11ubYg"
options = ClientOptions(schema="dc")  
supabase = create_client(SUPABASE_URL, SUPABASE_KEY, options=options)

# Helper Functions
def extract_features(image_bytes: bytes):
    """Extract face features from image bytes."""
    #logger.info(image_bytes)
    image = face_recognition.load_image_file(BytesIO(image_bytes))
    #logger.info(f"loaded image {image}")
    features = face_recognition.face_encodings(image)
    #logger.info(f"encoding {features}")
    if not features:
        raise ValueError("No face detected in the uploaded image.")
    return features[0]

def calculate_similarity(features1, features2):
    """Calculate similarity between two feature vectors."""
    return 1 - face_recognition.face_distance([features2], features1)[0]

async def get_similar_images_from_storage(image_bytes: bytes, limit:int=10) -> List[TemplateMemeResponse]:
    """Find similar images in the database and return details."""   

    input_features = extract_features(image_bytes)
    similar_images = []
    response = supabase.storage.from_("meta-data").list(path="memes")  
    #logger.info(f"supa list {response}")
    if not response:
        logger.error("Error: No images found in Supabase storage.")
        raise ValueError("Error fetching images from Supabase storage.")    
    limited_response = response[:limit]
    #logger.info(f"Fetched {len(response)} images from Supabase storage.")
    
    # Loop through all images and compare features
    for file_info in limited_response:
        image_path = file_info['name']
        #logger.debug(f"Checking image: {image_path}")  # Debugging: Image being checked          
     
        #logger.info(f"Found meme image: {image_path}")
        img_response = supabase.storage.from_("meta-data/memes").download(image_path)   
        logger.info(f"Successfully downloaded: {image_path}")     
       
        image = Image.open(BytesIO(img_response))
        logger.info(f"Loaded image {image_path}: size={image.size}, format={image.format}")
        try:               
            db_features = extract_features(img_response)
            similarity = calculate_similarity(input_features, db_features)
            logger.debug(f"Calculated similarity for {image_path}: {similarity}")  # Debugging: similarity value

            if similarity >= 0.7:
                logger.info(f"Image {image_path} is similar with similarity {similarity}")
                similar_images.append({
                    "image_name": image_path,
                    "similarity": similarity,
                })
        except Exception as e:
                logger.error(f"Skipping image {image_path} due to error: {e}")
    
    similar_images = sorted(similar_images, key=lambda x: x["similarity"], reverse=True)[:5]
    logger.info(f"Top 5 similar images: {similar_images}")  # Debugging: Top similar images after sorting

    # Fetch URLs and additional details using Supabase query
    responses = []
    for img in similar_images:
        image_url = supabase.storage.from_("meta-data/memes").get_public_url(img["image_name"])
        file_extension = img["image_name"].split('.')[-1].lower()
        if file_extension in ['jpg', 'png']:
            # Construct the image URL
            image_url = supabase.storage.from_("meta-data/memes").get_public_url(img["image_name"])        
            # Set the mime_type based on the file extension
            mime_type = "image/" + file_extension
            responses.append(TemplateMemeResponse(             
                image_url=image_url,  
                mime_type=mime_type,  
                similarity=img["similarity"]                
            ))
    if not responses:
        logger.warning("No similar images found.")

    return responses