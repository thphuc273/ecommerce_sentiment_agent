#!/bin/bash

# Create directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/checkpoints

echo "Setting up the e-commerce sentiment analysis system..."
echo "------------------------------------------------"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Create the network if it doesn't exist
docker network create ecommerce_net 2>/dev/null || true

# Check if NVIDIA Container Toolkit is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected! For GPU acceleration, edit docker-compose.yml to uncomment the GPU section."
else
    echo "No NVIDIA GPU detected. System will run in CPU-only mode."
fi

echo "Building and starting services..."
docker-compose build
docker-compose up -d

echo ""
echo "Service endpoints:"
echo "- PostgreSQL: localhost:5433"
echo "- Data Collection: http://localhost:9001"
echo "- Data Processing: http://localhost:8002"
echo "- Model Training: http://localhost:8003"
echo "- Retrieval: http://localhost:8004"
echo "- Inference: http://localhost:8005"
echo "- Frontend: http://localhost:8006"
echo ""
echo "To view logs: docker-compose logs -f [service_name]"
echo "To stop all services: docker-compose down"