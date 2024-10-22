FROM pytorch/pytorch:latest

# Install necessary packages, including git
RUN apt-get update && apt-get upgrade -y && apt-get install -y git

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Install the necessary GitHub repositories
RUN pip install git+https://github.com/lauracabayol/TEMPS.git
RUN pip install -e /home/user/TEMPS

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's app directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app, setting the owner to the user
COPY --chown=user . $HOME/app

# Expose the port (not mandatory as Hugging Face manages this, but can remain for clarity)
EXPOSE 7860

# Set the command to run your app, using the Hugging Face port
CMD ["python", "temps/app.py", "--port", "$PORT", "--server-name", "0.0.0.0"]