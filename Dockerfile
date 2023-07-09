# Use an official Python runtime as a parent image
FROM nvidia/cuda:11.6.0-base-ubuntu20.04

# Set the working directory to /app
WORKDIR /app

# Install Python 3.9
RUN apt-get update && apt-get install -y python3.9

# Install Pip
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip

# OS install cuda toolkit
RUN apt-get install -y --fix-missing cuda-toolkit-11.6

# Fix CUDA Version
ENV cuda_version=cuda11.6

# Copy the Pipfile and Pipfile.lock to the container
COPY Pipfile Pipfile.lock ./

# Install pipenv and use it to install dependencies
RUN pip install pipenv && pipenv install --system --deploy

# Dependencies for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the contents of the current directory into the container at /app
COPY . /app

# Install Jupyter Notebook
RUN pip install jupyter

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Start Jupyter Notebook when the container launches
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
