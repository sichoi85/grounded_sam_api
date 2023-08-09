from fastapi import FastAPI, File, UploadFile
from grounded_sam_wrapper import Grounded_Sam_wrapper
import os
from fastapi.responses import JSONResponse

app = FastAPI()

# Load the ML model when the server starts

cwd = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(cwd)

grounded_sam_wrapper = Grounded_Sam_wrapper(
    config = os.path.join(parent_directory,"GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    grounded_checkpoint = "groundingdino_swint_ogc.pth", 
    sam_checkpoint = "sam_vit_h_4b8939.pth", 
    sam_hq_checkpoint = "sam_vit_h_4b8939.pth", 
    use_sam_hq = "sam_vit_h_4b8939.pth", 
    output_dir = "../outputs", 
    box_threshold = 0.3, 
    text_threshold = 0.3, 
    device = "cuda" 
    )

grounded_sam_wrapper.load()

x,y = grounded_sam_wrapper.run("../outputs/room_furnished.png", "bed.")
print(x.cpu().numpy().tolist())
print(y)