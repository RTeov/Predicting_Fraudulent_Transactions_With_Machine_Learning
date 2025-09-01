# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose a port (optional, for web apps or APIs)
# EXPOSE 8080

# Default command (change as needed for your entry point)
CMD ["bash"]
