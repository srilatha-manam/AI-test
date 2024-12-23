from fastapi import FastAPI, Response
from Domain.domain import MemeRequest
from Service.service import overlay_dialog_on_image

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Meme Generator API"}

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
