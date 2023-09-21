# the base image
FROM friday0013/grounded-sam:v0

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

# Set environment variables (optional)
ENV APP_PORT=8080

# # Create a non-root user
RUN useradd -ms /bin/bash myuser

# Set environment variables for AWS CLI
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION


# # Switch to the non-root user
USER myuser

# # Set the working directory
WORKDIR /home/myuser
# Clone the grounded_sam_api repository
RUN git clone https://github.com/sichoi85/grounded_sam_api 

# # Set the working directory
WORKDIR /home/myuser/grounded_sam_api

ENV PYTHONPATH=$PYTHONPATH:/home/myuser/grounded_sam_api

# Install AWS CLI (if not already installed)
# Install AWS CLI (if not already installed) with elevated privileges
USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y awscli

# # Switch to the non-root user
USER myuser
# Create the destination directory if it doesn't exist
RUN mkdir -p /home/myuser/.cache/huggingface/hub/models--bert-base-uncased/

# Download everything from S3
RUN aws s3 cp s3://grounded-sam /home/myuser/.cache/huggingface/hub/models--bert-base-uncased/ --recursive

# Download the required model files
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Install dependencies (you may need to adjust this depending on your app)
RUN pip install -r requirements.txt

# Expose the port your FastAPI app will run on
EXPOSE $APP_PORT

WORKDIR /home/myuser/grounded_sam_api/api


# Start the FastAPI app with UVicorn
CMD ["uvicorn", "main:app", "--reload", "--port", "8080"]