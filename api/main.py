from fastapi import FastAPI, File, UploadFile
from grounded_sam_wrapper import Grounded_Sam_wrapper
import os
from fastapi.responses import JSONResponse
import numpy as np 
import time


app = FastAPI()

# Load the ML model when the server starts

cwd = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(cwd)

grounded_sam_wrapper = Grounded_Sam_wrapper(
    config = os.path.join(parent_directory,"GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    grounded_checkpoint = os.path.join(parent_directory,"groundingdino_swint_ogc.pth"), 
    sam_checkpoint = os.path.join(parent_directory,"sam_vit_h_4b8939.pth"), 
    sam_hq_checkpoint = None, 
    use_sam_hq = None, 
    output_dir = "../outputs", 
    box_threshold = 0.3, 
    text_threshold = 0.3, 
    device = "cuda" 
    )

grounded_sam_wrapper.load()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/uploadImage/")
async def create_upload_file(file: UploadFile = File(...)):
    start = time.time()
    
    try:
        file_path = os.path.join(parent_directory, "outputs", file.filename)
        with open(file_path, "wb") as image_file:
            image_file.write(file.file.read())
        masks, pred = grounded_sam_wrapper.run(file_path, "bed")
        end = time.time()
        print(end - start)
        masks = masks.cpu().numpy()
        masks = np.squeeze(masks, (0,1))
        masks = np.argwhere(masks & ~np.roll(masks, 1, axis=(0, 1)))
        random_samples = np.random.choice(masks.shape[0], size=100, replace=False)
        masks = masks[random_samples]
        masks = masks.tolist()
        end2 = time.time()
        print(end2 - end) 
        
        return JSONResponse(content = {"message": {"masks": masks}}, status_code = 200)
            
    except Exception as e:
        return JSONResponse(content={"message": "Error uploading image", "error": str(e)}, status_code=500)
    
    