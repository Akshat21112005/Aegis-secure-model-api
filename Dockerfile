# 1. Base Image: Use a modern, slim Python version
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements file first to use Docker's build cache
COPY requirements.txt requirements.txt

# 4. Install all your Python libraries
# This step might take a while because of PyTorch and other ML libs
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy ALL other files from your repo into the container
# (This includes app.py, config.py, models.py, finetuned_bert/, Message_model/, etc.)
COPY . .

# 6. This is the production command to run your FastAPI app
# It uses Gunicorn as the server, with a Uvicorn worker for FastAPI.
# It binds to the $PORT (8080) provided by Cloud Run.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 -k uvicorn.workers.UvicornWorker app:app
