#!/usr/bin/env python3
"""
Deploy to Hugging Face Spaces script
"""
import os
import subprocess
import sys
from huggingface_hub import HfApi, create_repo

def deploy_to_hf():
    """Deploy the application to Hugging Face Spaces"""
    print("Deploying to Hugging Face Spaces...")
    
    # Get credentials from environment
    hf_token = os.environ.get('HF_TOKEN')
    hf_username = os.environ.get('HF_USERNAME')
    
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set")
        sys.exit(1)
        
    if not hf_username:
        print("Error: HF_USERNAME environment variable not set")
        sys.exit(1)
        
    repo_id = f"{hf_username}/ecommerce-sentiment-analysis"
    space_files = {
        "hf_space_app.py": "app.py",
        "hf_space_requirements.txt": "requirements.txt",
        "hf_space_README.md": "README.md"
    }
    
    try:
        # Initialize API
        api = HfApi(token=hf_token)
        
        # Create or get repository
        try:
            create_repo(
                repo_id=repo_id,
                token=hf_token,
                repo_type="space",
                space_sdk="gradio",
                exist_ok=True
            )
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Repository creation warning (may already exist): {str(e)}")
        
        # Upload files
        for source, destination in space_files.items():
            if os.path.exists(source):
                api.upload_file(
                    path_or_fileobj=source,
                    path_in_repo=destination,
                    repo_id=repo_id,
                    repo_type="space",
                    token=hf_token
                )
                print(f"Uploaded {source} as {destination}")
            else:
                print(f"Warning: {source} not found, skipping")
        
        # Upload data directory if it exists
        data_dir = "data/processed"
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if os.path.isfile(file_path):
                    hf_path = os.path.join("data/processed", file)
                    # Create directory first
                    try:
                        api.upload_file(
                            path_or_fileobj=file_path,
                            path_in_repo=hf_path,
                            repo_id=repo_id,
                            repo_type="space",
                            token=hf_token
                        )
                        print(f"Uploaded {file_path}")
                    except Exception as e:
                        print(f"Error uploading {file_path}: {str(e)}")
        
        # Upload model files if they exist
        model_dir = "models/checkpoints/final_model"
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                file_path = os.path.join(model_dir, file)
                if os.path.isfile(file_path):
                    try:
                        api.upload_file(
                            path_or_fileobj=file_path,
                            path_in_repo=file,
                            repo_id=repo_id,
                            repo_type="space",
                            token=hf_token
                        )
                        print(f"Uploaded model file {file}")
                    except Exception as e:
                        print(f"Error uploading model file {file}: {str(e)}")
        
        print(f"🎉 Successfully deployed to https://huggingface.co/spaces/{repo_id}")
        return True
        
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Load environment variables
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
    
    # Deploy to HF Spaces
    deploy_to_hf()