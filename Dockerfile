# Use a base image with Python
FROM python:3.10

RUN pip install git+https://github.com/lauracabayol/TEMPS.git
# Set the working directory
WORKDIR /app

# Copy the application file(s)
COPY app.py .

# Expose the port the app runs on (if needed)
EXPOSE 7860

# Set the command to run your app
CMD ["python", "app.py", "--server-port", "7860", "--server-address","127.0.0.1"]