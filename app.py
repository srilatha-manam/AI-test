from fastapi import FastAPI, Response,File, UploadFile, HTTPException
from Domain.domain import MemeRequest, TemplateMemeResponse
from fastapi.responses import JSONResponse
from Service.service import overlay_dialog_on_image
from Service.servicecodewithlocal import local_overlay_dialog_on_image
from Service.FindActor import get_similar_images_from_storage
from typing import List
from fastapi.logger import logger
import base64
from Service.servicecodewithlocal import search_by_text
from Service.servicecodewithlocal import search_by_embedding

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Meme Generator API"}

#find actor with image similarity

@app.post("/find-actor", response_model=List[TemplateMemeResponse])
async def find_actor(image: UploadFile = File(...)):
    try:
        logger.info("Start processing image...")
        image_bytes = await image.read()    
        logger.info(f"Image size: {len(image_bytes)} bytes")
        # Get similar images from storage
        actor_info = await get_similar_images_from_storage(image_bytes)
        logger.info("End processing image.")

        if not actor_info:
            raise HTTPException(status_code=404, detail="No matching actor or images found.")

        return actor_info  # Return the list of similar images

    except Exception as e:
        logger.error(f"Error in find_actor: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate-meme")
def generate_meme(request: MemeRequest):
        try:
            result = overlay_dialog_on_image(request.text)
            image_stream = result["image_stream"]
            latency = result["latency"]
            headers = {"X-Latency": latency}
            return Response(content=image_stream.getvalue(), media_type="image/png",headers=headers)
        except Exception as e:
            return {"error": str(e)}
        
@app.post("/local-test-generate-meme")
def generate_meme(request: MemeRequest):
    try:
        # Call the overlay_dialog_on_image function and get the result
        result = local_overlay_dialog_on_image(request.text)
        
        # Extract image stream and latency from the result
        image_stream = result["image_stream"]
        latency = result["latency"]
        
        # Add the latency header to the response
        headers = {"X-Latency": latency}
        
        # Return the generated meme image with the latency header
        return Response(content=image_stream.getvalue(), media_type="image/png", headers=headers)
    
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/local-search-by-text")
def search_text(request: MemeRequest):
    try:
        # Extract the 'text' from the request
        prompt = request.text       

        # Call the search_by_text function and get the result
        images = search_by_text(prompt)

        # Assuming each image in `images` is a dictionary containing "image_stream"
        # and the image stream is in bytes
        if images:
            # Pick the first image to return (or you can iterate over images to return multiple)
            first_image_stream = images[0]["image_stream"]

            # Add the latency header to the response
            latency = images[0].get("latency", 0)  # Assuming latency is part of the response
            headers = {"X-Latency": str(latency)}

            # Return the generated meme image with the media type "image/png"
            return Response(content=first_image_stream.getvalue(), media_type="image/png", headers=headers)
        else:
            raise HTTPException(status_code=404, detail="No matching images found.")

    except Exception as e:
        logger.error(f"Error in search_by_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/local-search-by-embedding")
def search_embedding(request: MemeRequest):
    try:
        # Extract the 'text' from the request
        prompt = request.text      

        # Call the search_by_embedding function and get the result
        images = search_by_embedding(prompt)
        
        # Assuming each image in `images` is a dictionary containing "image_stream"
        # and the image stream is in bytes
        if images:
            # Pick the first image to return (or you can iterate over images to return multiple)
            first_image_stream = images[0]["image_stream"]
            
            # Return the generated meme image with the media type "image/png"
            return Response(content=first_image_stream.getvalue(), media_type="image/png")
        else:
            raise HTTPException(status_code=404, detail="No matching images found.")
    
    except Exception as e:
        logger.error(f"Error in search_by_embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))