#!/bin/bash

# Validate Environment Configuration Script

echo "🔍 Validating environment configuration..."
echo "============================================"

# Load .env file
if [ ! -f ".env" ]; then
    echo ".env file not found. Run ./scripts/setup_env.sh first"
    exit 1
fi

# Source the .env file
source .env

# Validation checks
errors=0

# Check Hugging Face configuration
echo "🤗 Checking Hugging Face configuration..."
if [[ -z "$HF_TOKEN" || "$HF_TOKEN" == "your_huggingface_token_here" || "$HF_TOKEN" == "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" ]]; then
    echo "HF_TOKEN is not set or still has placeholder value"
    errors=$((errors + 1))
else
    echo "HF_TOKEN is configured"
fi

if [[ -z "$HF_USERNAME" || "$HF_USERNAME" == "your_username" ]]; then
    echo "HF_USERNAME is not set or still has placeholder value"
    errors=$((errors + 1))
else
    echo "HF_USERNAME is configured"
fi

# Check database configuration
echo "Checking database configuration..."
if [[ -z "$DATABASE_URL" ]]; then
    echo "DATABASE_URL is not set"
    errors=$((errors + 1))
else
    echo "DATABASE_URL is configured"
fi

# Check service URLs
echo "Checking service URLs..."
required_urls=("RETRIEVAL_SERVICE_URL" "INFERENCE_SERVICE_URL")
for url_var in "${required_urls[@]}"; do
    if [[ -z "${!url_var}" ]]; then
        echo "$url_var is not set"
        errors=$((errors + 1))
    else
        echo "$url_var is configured"
    fi
done

# Check directories exist
echo "Checking required directories..."
required_dirs=("./data" "./models" "./logs")
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Creating missing directory: $dir"
        mkdir -p "$dir"
    else
        echo "Directory exists: $dir"
    fi
done

echo ""
if [ $errors -eq 0 ]; then
    echo "All environment variables are properly configured!"
    echo ""
    echo "You can now run:"
    echo "   docker-compose up -d"
    echo ""
else
    echo "Found $errors configuration errors. Please fix them in .env file"
    echo ""
    echo "Need help? Check the documentation or run:"
    echo "   ./scripts/setup_env.sh"
    echo ""
    exit 1
fi