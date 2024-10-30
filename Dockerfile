# Use a base image with Python
FROM python:3.10-slim

# Install git and other system dependencies
RUN apt-get update && apt-get install -y \
    git \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/lauracabayol/TEMPS.git
# Set the working directory
WORKDIR /app

# Copy the application file(s)
COPY app.py .

# Create models directory and copy model files
RUN mkdir -p data/models

# Copy model files - make sure these files exist in your repository
COPY data/models/modelF_DA.pt data/models/
COPY data/models/modelZ_DA.pt data/models/
# Expose the port the app runs on (if needed)
EXPOSE 7860

# Updated command with correct argument names
CMD ["python", "app.py", "--port", "7860", "--server-address", "0.0.0.0"]