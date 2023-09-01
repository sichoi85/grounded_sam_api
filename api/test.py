from fastapi import FastAPI, File, UploadFile
from grounded_sam_wrapper import Grounded_Sam_wrapper
import os
from fastapi.responses import JSONResponse
import numpy as np

app = FastAPI()

# Load the ML model when the server starts

cwd = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(cwd)

grounded_sam_wrapper = Grounded_Sam_wrapper(
    config = os.path.join(parent_directory,"GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    grounded_checkpoint = os.path.join(parent_directory,"groundingdino_swint_ogc.pth"), 
    sam_checkpoint = os.path.join(parent_directory,"sam_vit_h_4b8939.pth"), 
    sam_hq_checkpoint = os.path.join(parent_directory,"sam_vit_h_4b8939.pth"), 
    use_sam_hq = os.path.join(parent_directory,"sam_vit_h_4b8939.pth"), 
    output_dir = "../outputs", 
    box_threshold = 0.3, 
    text_threshold = 0.3, 
    device = "cuda" 
    )

grounded_sam_wrapper.load()

x,y = grounded_sam_wrapper.run(os.path.join(cwd,"inputs/emptyroomtouse.jpg"), "bed.")
x = x.cpu().numpy()
x = np.squeeze(x, (0,1))
x = np.argwhere(x & ~np.roll(x, 1, axis=(0, 1)))
print(x.shape)

print(y)