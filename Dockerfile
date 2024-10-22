# Use a base image with Python
FROM python:3.10

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install necessary packages, including git
RUN apt-get update && apt-get upgrade -y && apt-get install -y git

# Switch to the "user" user
USER user

# Set the working directory to the user's app directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app, setting the owner to the user
COPY --chown=user . $HOME/app

# Install the necessary GitHub repositories
RUN pip install --user git+https://github.com/lauracabayol/TEMPS.git

# Create a directory for the models
RUN mkdir -p $HOME/app/models

# Copy model files into the models directory
COPY --chown=user data/models/modelZ_DA.pt $HOME/app/models/
COPY --chown=user data/models/modelF_DA.pt $HOME/app/models/

# Set the command to run your app, using the Hugging Face port
CMD ["python", "app.py", "--port", "$PORT", "--server-name", "0.0.0.0"]

