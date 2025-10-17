#!/bin/bash

# Deploy to Hugging Face Spaces Script
# This script prepares and deploys the application to Hugging Face Spaces

set -e

echo "🚀 Starting Hugging Face Spaces Deployment..."

# Configuration
HF_SPACE_NAME="ecommerce-sentiment-analysis"
HF_USERNAME=${HF_USERNAME:-"your-username"}  # Set this environment variable
SPACE_DIR="hf_space_deployment"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable is not set."
    echo "Please set your Hugging Face token:"
    echo "export HF_TOKEN=your_hugging_face_token"
    exit 1
fi

# Check if HF_USERNAME is set
if [ "$HF_USERNAME" = "your-username" ]; then
    echo "❌ Error: Please set the HF_USERNAME environment variable to your Hugging Face username."
    echo "export HF_USERNAME=your_actual_username"
    exit 1
fi

echo "📦 Preparing deployment files..."

# Clean up any existing deployment directory
rm -rf "$SPACE_DIR"
mkdir -p "$SPACE_DIR"

# Copy application files
echo "📋 Copying application files..."
cp hf_space_app.py "$SPACE_DIR/app.py"
cp hf_space_requirements.txt "$SPACE_DIR/requirements.txt"
cp hf_space_README.md "$SPACE_DIR/README.md"

# Copy model files if they exist
if [ -d "models/checkpoints/final_model" ]; then
    echo "🤖 Copying trained model files..."
    cp -r models/checkpoints/final_model "$SPACE_DIR/"
else
    echo "⚠️  No trained model found, will use pre-trained model"
fi

# Copy data files if they exist
if [ -d "data/processed" ]; then
    echo "📊 Copying processed data files..."
    mkdir -p "$SPACE_DIR/data/processed"
    
    if [ -f "data/processed/embeddings.npy" ]; then
        cp data/processed/embeddings.npy "$SPACE_DIR/data/processed/"
        echo "✅ Copied embeddings.npy"
    fi
    
    if [ -f "data/processed/reviews.csv" ]; then
        cp data/processed/reviews.csv "$SPACE_DIR/data/processed/"
        echo "✅ Copied reviews.csv"
    fi
else
    echo "⚠️  No processed data found, similarity search will be limited"
fi

# Create .gitattributes for large files
cat > "$SPACE_DIR/.gitattributes" << EOF
*.npy filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
EOF

echo "🔑 Authenticating with Hugging Face..."

# Install huggingface_hub if not already installed
pip install huggingface_hub

# Create deployment script
cat > "$SPACE_DIR/deploy.py" << EOF
#!/usr/bin/env python3

import os
import sys
from huggingface_hub import HfApi, login

def main():
    # Login to Hugging Face
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("❌ Error: HF_TOKEN environment variable not found")
        sys.exit(1)
    
    login(token=token)
    print("✅ Successfully authenticated with Hugging Face")
    
    # Initialize API
    api = HfApi()
    
    username = os.environ.get('HF_USERNAME', 'your-username')
    space_name = '$HF_SPACE_NAME'
    repo_id = f"{username}/{space_name}"
    
    print(f"🚀 Deploying to space: {repo_id}")
    
    try:
        # Try to create the space
        api.create_repo(
            repo_id=repo_id,
            repo_type='space',
            space_sdk='gradio',
            private=False
        )
        print(f"✅ Created new Hugging Face Space: {repo_id}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"ℹ️  Space already exists: {repo_id}")
        else:
            print(f"⚠️  Could not create space: {e}")
    
    # Upload all files
    try:
        api.upload_folder(
            folder_path='.',
            repo_id=repo_id,
            repo_type='space',
            commit_message='Deploy E-commerce Sentiment Analysis App'
        )
        print(f"🎉 Successfully deployed to: https://huggingface.co/spaces/{repo_id}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Change to deployment directory and run deployment
cd "$SPACE_DIR"

echo "🌐 Deploying to Hugging Face Spaces..."
python deploy.py

echo "🎉 Deployment completed successfully!"
echo "🔗 Your space should be available at: https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME"

# Clean up
cd ..
echo "🧹 Cleaning up deployment files..."
# Uncomment the next line if you want to remove the deployment directory after deployment
# rm -rf "$SPACE_DIR"

echo "✅ Deployment script completed!"