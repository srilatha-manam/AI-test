import os
import psycopg2
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from io import BytesIO
import pickle
import logging
import numpy as np
import tensorflow_hub as hub

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load pickle model
MODEL_PATH = "Artifacts/model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
        logger.info("Model loaded successfully from pickle.")
else:
    logger.error(f"Model file not found at {MODEL_PATH}.")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

# PostgreSQL setup
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'root',
    'host': 'localhost',
    'port': 5432
}

# Connect to local PostgreSQL
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    logger.info("Successfully connected to PostgreSQL database.")
except Exception as e:
    logger.error(f"Error connecting to PostgreSQL database: {e}")
    raise

def preprocess_text(text: str) -> str:
    """Preprocess input text by removing punctuation and converting to lowercase."""
    logger.debug(f"Preprocessing text: {text}")
    text = text.lower()
    return re.sub(r'[^\w\s]', '', text)

def local_overlay_dialog_on_image(prompt: str, fixed_size=(200, 200)) -> BytesIO:
    """Generate a meme by overlaying text on an image and return it as a byte stream."""
    start_time = time.time()
    logger.info(f"Processing prompt: {prompt}")

    try:
        query_embedding = model.encode(preprocess_text(prompt), convert_to_numpy=True)
        logger.debug(f"Query embedding: {query_embedding[:5]}...")  # Log part of the embedding for debugging
    except Exception as e:
        logger.error(f"Error encoding prompt text: {e}")
        raise

    try:
        # Fetch dialogs from local PostgreSQL
        cursor.execute("""SELECT dialog_id, text, meme_id FROM dialogs;""")
        dialogs = cursor.fetchall()
        logger.debug(f"Fetched dialogs: {dialogs}")
    except Exception as e:
        logger.error(f"Error fetching dialogs from PostgreSQL: {e}")
        raise

    if not dialogs:
        logger.error("No dialogs found in local PostgreSQL.")
        raise ValueError("No dialogs found in local PostgreSQL.")

    # Calculate cosine similarity
    results = []
    for dialog in dialogs:
        try:
            dialog_embedding = model.encode(preprocess_text(dialog[1]), convert_to_numpy=True)
            similarity = cosine_similarity([query_embedding], [dialog_embedding])[0][0]
            results.append((dialog[0], dialog[1], dialog[2], similarity))
        except Exception as e:
            logger.error(f"Error calculating similarity for dialog {dialog[0]}: {e}")
            continue

    logger.debug(f"Cosine similarities: {[(r[0], r[1], r[3]) for r in results]}")

    results.sort(key=lambda x: x[3], reverse=True)

    top_result = results[0]
    highest_similarity_meme_id = top_result[2]
    caption = top_result[1]

    logger.info(f"Best match: Meme ID = {highest_similarity_meme_id}, Caption = {caption}")

    try:
        # Fetch the image path pattern from memes_dc table
        cursor.execute("""SELECT image_path FROM public.movie_dc WHERE meme_id::text = %s;""", (highest_similarity_meme_id,))
        image_path_row = cursor.fetchone()

        if not image_path_row:
            logger.error(f"Image path for Meme ID {highest_similarity_meme_id} not found in memes_dc table.")
            raise ValueError(f"Image path for Meme ID {highest_similarity_meme_id} not found in memes_dc table.")

        image_path_pattern = image_path_row[0]
        logger.debug(f"Image path pattern from memes_dc table: {image_path_pattern}")

        # Extract the file name (pattern) from the image_path
        pattern = os.path.basename(image_path_pattern)  # Extract file name from path
        logger.debug(f"Extracted pattern: {pattern}")

        meme_image_path = None
        for filename in os.listdir('memes'):
            # Match based on the pattern extracted
            if filename == pattern:
                meme_image_path = os.path.join('memes', filename)
                logger.info(f"Found matching meme image: {meme_image_path}")
                break

        if not meme_image_path:
            logger.error(f"No matching meme image found in local 'memes' folder for meme ID {highest_similarity_meme_id}.")
            raise ValueError(f"No matching meme image found in local 'memes' folder for meme ID {highest_similarity_meme_id}.")

    except Exception as e:
        logger.error(f"Error processing meme image path: {e}")
        raise

    try:
        # Open and process the image
        img = Image.open(meme_image_path).resize(fixed_size)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        position = (50, 50)
        draw.text(position, caption, font=font, fill=(255, 255, 255))

        # Save image to a byte stream
        byte_stream = BytesIO()
        img.save(byte_stream, format="PNG")
        byte_stream.seek(0)
        logger.info("Generated meme successfully.")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

    end_time = time.time()
    latency = (end_time - start_time)
    logger.info(f"Total processing time: {latency:.2f} s")

    return {
        "image_stream": byte_stream,
        "latency": f"{latency:.2f} s"
    }

def generate_memes(image_path: str, text: str, fixed_size=(200, 200)) -> BytesIO:
    """
    Generate an image with overlaid text and return as a byte stream.
    """
    try:
        # Open the image, resize it, and overlay the text
        img = Image.open(image_path).resize(fixed_size)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        position = (10, 10)  # Position of the text on the image
        draw.text(position, text, font=font, fill=(255, 255, 255))  # Text overlay

        byte_stream = BytesIO()
        img.save(byte_stream, format="PNG")
        byte_stream.seek(0)
        return byte_stream
    except Exception as e:
        logger.error(f"Error generating meme for {image_path}: {e}")
        raise

def search_by_text(prompt: str, fixed_size=(200, 200)) -> bytes:
    """
    Search for top 5 matches by comparing prompt text with the text column in the database and return images with overlaid text.
    """
    start_time = time.time()

    # Encode the prompt to get its embedding using Universal Sentence Encoder
    try:
        query_embedding = model_embedding([prompt]).numpy().squeeze().tolist()
        logger.info(f"Encoded query embedding for prompt: {prompt}")
    except Exception as e:
        logger.error(f"Error encoding prompt text: {e}")
        raise

    try:
        # Fetch records from the database (text and image path)
        cursor.execute("SELECT text, image_path FROM Supabase_text_embeddings;")
        records = cursor.fetchall()
        logger.debug(f"Fetched {len(records)} records for embedding search.")
    except Exception as e:
        logger.error(f"Error fetching records for embedding search: {e}")
        raise

    # List to store search results
    results = []
    for record in records:
        try:
            text, image_path = record
            # Encode the text from the database to get its embedding
            record_embedding = model_embedding([text]).numpy().squeeze().tolist()

            # Calculate the cosine similarity between the prompt and the record's text
            similarity = cosine_similarity([query_embedding], [record_embedding])[0][0]

            # Extract the image name from image_path and search for it in the memes folder
            image_name = os.path.basename(image_path)
            meme_image_path = None
            for filename in os.listdir('memes'):
                # Match based on the pattern extracted from the image_path
                if filename == image_name:
                    meme_image_path = os.path.join('memes', filename)
                    logger.info(f"Found matching meme image: {meme_image_path}")
                    break

            if not meme_image_path:
                logger.error(f"No matching meme image found in local 'memes' folder for image path {image_path}.")
                raise ValueError(f"No matching meme image found in local 'memes' folder for image path {image_path}.")

            # Add result with internal similarity calculation
            results.append((text, meme_image_path, similarity))  # Only store text, meme_image_path, and similarity
        except Exception as e:
            logger.error(f"Error calculating similarity for record: {e}")
            continue

    # Sort results by similarity in descending order (to display the most relevant memes first)
    results.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity score
    top_results = results[:5]  # Get the top 5 results
    logger.info(f"Top 5 results for embedding search: {[(r[0], r[2]) for r in top_results]}")

    # Generate images for the top results with overlaid text
    images = _generate_images_text(top_results, fixed_size)
    latency = time.time() - start_time  # Measure the time it took
    logger.info(f"search_by_embedding took {latency:.2f} seconds.")
    
    return images
def _generate_images_text(top_results: list, fixed_size=(200, 200)) -> list:
    """
    Generate images with overlaid text for the top results.
    """
    images = []
    for result in top_results:
        text, meme_image_path, similarity = result
        try:
            # Open the image, resize it, and overlay the text
            img = Image.open(meme_image_path).resize(fixed_size)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            position = (10, 10)  # Position of the text on the image
            draw.text(position, text, font=font, fill=(255, 255, 255))  # Text overlay

            byte_stream = BytesIO()
            img.save(byte_stream, format="PNG")
            byte_stream.seek(0)

            images.append({
                "image_stream": byte_stream
            })
            logger.info(f"Generated image for text '{text}'")
        except Exception as e:
            logger.error(f"Error generating image for text '{text}': {e}")

    return images

model_embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def search_by_embedding(prompt: str, fixed_size=(200, 200)) -> list:
    """
    Search for top 5 matches by comparing prompt embedding with the embedding column and return images with overlaid text.
    """
    start_time = time.time()

    # Encode the prompt to get its embedding using Universal Sentence Encoder
    try:
        query_embedding = model_embedding([prompt]).numpy().squeeze().tolist()
        logger.info(f"Encoded query embedding for prompt: {prompt}")
    except Exception as e:
        logger.error(f"Error encoding prompt text: {e}")
        raise

    try:
        # Fetch records from the database
        cursor.execute("SELECT text, image_path, embedding FROM Supabase_text_embeddings;")
        records = cursor.fetchall()
        logger.debug(f"Fetched {len(records)} records for embedding search.")
    except Exception as e:
        logger.error(f"Error fetching records for embedding search: {e}")
        raise

    # List to store search results
    results = []
    for record in records:
        try:
            text, image_path, embedding = record
            record_embedding = np.array(embedding)  # Convert to numpy array for cosine similarity
            similarity = cosine_similarity([query_embedding], [record_embedding])[0][0]

            # Extract the image name from image_path and search for it in the memes folder
            image_name = os.path.basename(image_path)
            meme_image_path = None
            for filename in os.listdir('memes'):
                # Match based on the pattern extracted from the image_path
                if filename == image_name:
                    meme_image_path = os.path.join('memes', filename)
                    logger.info(f"Found matching meme image: {meme_image_path}")
                    break

            if not meme_image_path:
                logger.error(f"No matching meme image found in local 'memes' folder for image path {image_path}.")
                raise ValueError(f"No matching meme image found in local 'memes' folder for image path {image_path}.")

            # Add result with internal similarity calculation
            results.append((text, meme_image_path, similarity))  # Only store text, meme_image_path, and similarity
        except Exception as e:
            logger.error(f"Error calculating similarity for record: {e}")
            continue

    # Sort results by similarity in descending order (to display the most relevant memes first)
    results.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity score
    top_results = results[:5]  # Get the top 5 results
    logger.info(f"Top 5 results for embedding search: {[(r[0], r[2]) for r in top_results]}")

    # Generate images for the top results with overlaid text
    images = _generate_images(top_results, fixed_size)
    latency = time.time() - start_time  # Measure the time it took
    logger.info(f"search_by_embedding took {latency:.2f} seconds.")
    
    return images

def _generate_images(top_results: list, fixed_size=(200, 200)) -> list:
    """
    Generate images with overlaid text for the top results.
    """
    images = []
    for result in top_results:
        text, meme_image_path, similarity = result  # Unpack 3 values: text, meme_image_path, similarity
        try:
            # Open the image, resize it, and overlay the text
            img = Image.open(meme_image_path).resize(fixed_size)
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()
            position = (10, 10)  # Position of the text on the image
            draw.text(position, text, font=font, fill=(255, 255, 255))  # Text overlay

            byte_stream = BytesIO()
            img.save(byte_stream, format="PNG")
            byte_stream.seek(0)

            images.append({
                "image_stream": byte_stream
            })
            logger.info(f"Generated image with overlayed text: {text}")
        except Exception as e:
            logger.error(f"Error generating image with overlay for text: {text}, {e}")

    return images