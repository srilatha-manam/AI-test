import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from supabase import create_client, ClientOptions
import requests
from io import BytesIO
import pickle
import re
import numpy as np
from io import BytesIO
import time 

# Load pickle model
MODEL_PATH = "Artifacts/model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

# Supabase setup
SUPABASE_URL = "https://ixnbfvyeniksbqcfdmdo.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml4bmJmdnllbmlrc2JxY2ZkbWRvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzE0MDE3NjgsImV4cCI6MjA0Njk3Nzc2OH0.h4JtVbwtKAe38yvtOLYvZIbhmMy6v2QCVg51Q11ubYg"
options = ClientOptions(schema="dc")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY,options=options)

def preprocess_text(text: str) -> str:
    """Preprocess input text by removing punctuation and converting to lowercase."""
    text = text.lower()
    return re.sub(r'[^\w\s]', '', text)


def overlay_dialog_on_image(prompt: str, fixed_size=(200, 200)) -> BytesIO:
    """Generate a meme by overlaying text on an image and return it as a byte stream."""
    start_time = time.time()
    query_embedding = model.encode(preprocess_text(prompt), convert_to_numpy=True)

    # Fetch dialogs from Supabase
    dialogs = supabase.table('dialogs').select('dialog_id', 'text', 'meme_id').execute().data
    if not dialogs:
        raise ValueError("No dialogs found in Supabase.")

    # Calculate cosine similarity
    results = []
    for dialog in dialogs:
        dialog_embedding = model.encode(preprocess_text(dialog['text']), convert_to_numpy=True)
        similarity = cosine_similarity([query_embedding], [dialog_embedding])[0][0]
        results.append((dialog['dialog_id'], dialog['text'], dialog.get('meme_id'), similarity))
    results.sort(key=lambda x: x[3], reverse=True)

    top_result = results[0]
    highest_similarity_meme_id = top_result[2]
    caption = top_result[1]

    # Fetch image for the meme
    meme_query = supabase.table('memes_dc').select('image_path').eq('meme_id', highest_similarity_meme_id).execute().data
    if not meme_query:
        raise ValueError("No matching meme found.")
    image_path = meme_query[0]['image_path']

    # Download and process image
    image_url = f"{SUPABASE_URL}/{image_path}"
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch image. HTTP status code: {response.status_code}")
    img = Image.open(BytesIO(response.content)).resize(fixed_size)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    position = (50, 50)
    draw.text(position, caption, font=font, fill=(255, 255, 255))

    # Save image to a byte stream
    byte_stream = BytesIO()
    img.save(byte_stream, format="PNG")
    byte_stream.seek(0)
    end_time = time.time()  
    latency = (end_time - start_time) 
    return {
        "image_stream": byte_stream,
        "latency": f"{latency:.2f} s"  
    }