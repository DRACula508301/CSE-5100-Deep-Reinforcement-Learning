# Start from an NVIDIA image that has Python 3.12 and CUDA 12.1+
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set up the working directory
WORKDIR /app

# The base image uses Conda, so we'll install uv into that env
RUN pip install uv

# Copy your (now updated) project file
COPY pyproject.toml .

# Install all your dependencies using uv
# This will use the "pytorch-cu121" index you defined
RUN uv pip install --system .

# (Optional) Copy the rest of your project code
# COPY . .

# Set a default command to keep the container running
CMD ["sleep", "infinity"]