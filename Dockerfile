# Use an official Python runtime with GPU support as a parent image
FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install any needed system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg

# Install Python 3.10
RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev

# Install pip for Python 3.10
RUN apt-get -y install python3-pip

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Install any needed Python packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED 1

# Run app.py when the container launches
# CMD python3 -u app.py

#COPY Deforum_Stable_Diffusion.py .

ENTRYPOINT ["python3", "Deforum_Stable_Diffusion.py" ]