# the base image
FROM friday0013/grounded-sam

# Set environment variables (optional)
ENV APP_PORT=8080

# Create a directory to work in
WORKDIR /app

# Clone the grounded_sam_api repository
RUN git clone https://github.com/sichoi85/grounded_sam_api .

# Download the required model files
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Install dependencies (you may need to adjust this depending on your app)
RUN pip install -r requirements.txt

# Expose the port your FastAPI app will run on
EXPOSE $APP_PORT

# Start the FastAPI app with UVicorn
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]