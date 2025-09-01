
# Use the official Python image (bookworm variant) as a base
FROM python:3.10-bookworm


# Set the working directory
WORKDIR /app


# Install system dependencies (for scientific libraries, if needed)
RUN apt-get update && apt-get install -y \
	build-essential \
	git \
	&& rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code
COPY . .


# Environment variables for AWS (can be overridden at runtime)
ENV AWS_DEFAULT_REGION=us-east-1
ENV AWS_REGION=us-east-1

# Expose a port (for web apps or APIs, e.g., FastAPI/Flask)
# EXPOSE 8080

# Healthcheck (optional, for ECS)
# HEALTHCHECK CMD curl --fail http://localhost:8080/health || exit 1


# Default command (override in ECS task definition or Lambda handler)
CMD ["bash"]

# For AWS Lambda (if using AWS Lambda with container images)
# Uncomment the following line and set your handler if using AWS Lambda
# CMD ["python", "your_lambda_handler.py"]

# For SageMaker (if using as a custom container)
# Add SageMaker-specific entrypoint if needed
# ENTRYPOINT ["python", "serve.py"]
