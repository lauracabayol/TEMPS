FROM pytorch/pytorch:latest

WORKDIR /code

# Install the repository directly from GitHub
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

    # Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

RUN pip install git+https://github.com/lauracabayol/TEMPS.git

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app
# setting the owner to the user
COPY --chown=user . $HOME/app

# Expose the port the app runs on (if needed)
EXPOSE 7860

# Set the command to run your app
CMD ["python", "app.py", "--server-port", "7860", "--server-address","127.0.0.1"]