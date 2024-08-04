# Use an official TensorFlow runtime as a parent image
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-6

# Set the working directory
WORKDIR /app

# Copies the requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . /app

# Set the entrypoint
ENTRYPOINT ["python", "Training_Job_Emotion_Detection_AI.py"]

# Build your Docker image
docker build -t gcr.io/528224663118/emotion-recognition-trainer:v1 .

# Push it to Google Container Registry
docker push gcr.io/528224663118/emotion-recognition-trainer:v1
