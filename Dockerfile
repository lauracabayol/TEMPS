ROM python:3.9-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Switch to the "user" user
USER user

# Set the working directory to the user's app directory
WORKDIR $HOME/app

# Copy pyproject.toml first, setting the owner to the user
COPY --chown=user pyproject.toml .

# Install the project and its dependencies
RUN pip install --no-cache-dir .

# Copy the rest of the application
COPY --chown=user . .

# Expose the port
EXPOSE 7860

# Set the command to run your app
CMD ["python", "app.py", "--server-port", "7860", "--server-address", "0.0.0.0"]