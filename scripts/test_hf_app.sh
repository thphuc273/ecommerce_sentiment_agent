#!/bin/bash

# Local Testing Script for HF Spaces App
# This script tests the Hugging Face Spaces application locally

set -e

echo "Starting local testing for HF Spaces app..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_test" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_test
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_test/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r hf_space_requirements.txt

# Install additional testing dependencies
pip install pytest pytest-cov

# Run basic import test
echo "🔍 Testing imports..."
python3 -c "
try:
    import gradio as gr
    import torch
    import transformers
    import numpy as np
    import pandas as pd
    print('All required packages imported successfully')
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
"

# Test model loading
echo "Testing model initialization..."
python3 -c "
import sys
sys.path.append('.')

# Test basic model loading functionality
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    
    # Test sentiment model
    model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Test pipeline
    sentiment_pipeline = pipeline(
        'sentiment-analysis',
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True
    )
    
    # Test with sample text
    test_text = 'This is a great product!'
    result = sentiment_pipeline(test_text)
    
    print('Sentiment analysis model loaded and tested successfully')
    print(f'Sample result: {result}')
    
except Exception as e:
    print(f'Model loading error: {e}')
    exit(1)
"

# Test Gradio app syntax
echo "🎨 Testing Gradio app syntax..."
python3 -c "
import ast
import sys

try:
    with open('hf_space_app.py', 'r') as f:
        source = f.read()
    
    # Parse the file to check for syntax errors
    ast.parse(source)
    print('Gradio app syntax is valid')
    
except SyntaxError as e:
    print(f'Syntax error in hf_space_app.py: {e}')
    exit(1)
except FileNotFoundError:
    print('hf_space_app.py not found')
    exit(1)
"

# Test data file handling
echo "📊 Testing data file handling..."
python3 -c "
import os
import numpy as np
import pandas as pd

# Test with mock data if real data doesn't exist
embeddings_path = 'data/processed/embeddings.npy'
reviews_path = 'data/processed/reviews.csv'

if os.path.exists(embeddings_path) and os.path.exists(reviews_path):
    try:
        embeddings = np.load(embeddings_path)
        reviews_df = pd.read_csv(reviews_path)
        print(f'Data files loaded: {embeddings.shape} embeddings, {len(reviews_df)} reviews')
    except Exception as e:
        print(f'Data loading error (will use fallback): {e}')
else:
    print('Data files not found, app will run without similarity search')

print('Data handling test completed')
"

# Run the app in test mode (non-blocking)
echo "🚀 Testing Gradio app launch..."

# Start the app in background with a timeout mechanism
python3 hf_space_app.py &
APP_PID=$!

# Wait for app to start
sleep 10

# Check if the app is running
if ps -p $APP_PID > /dev/null 2>&1; then
    echo "Gradio app launched successfully"
    # Kill the app
    kill $APP_PID 2>/dev/null || true
    wait $APP_PID 2>/dev/null || true
else
    echo "Gradio app failed to launch"
    exit 1
fi

# Alternative timeout function for systems without timeout command
kill_after_timeout() {
    local pid=$1
    local timeout_duration=$2
    sleep $timeout_duration
    if ps -p $pid > /dev/null 2>&1; then
        kill $pid 2>/dev/null || true
    fi
}

# Run unit tests if they exist
if [ -d "tests" ]; then
    echo "Running unit tests..."
    python -m pytest tests/ -v
else
    echo "No unit tests found, skipping..."
fi

# Test deployment files
echo "Checking deployment files..."

files=("hf_space_app.py" "hf_space_requirements.txt" "hf_space_README.md")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "$file exists"
    else
        echo "$file missing"
        exit 1
    fi
done

# Check README format
echo "Validating README format..."
python3 -c "
import re

with open('hf_space_README.md', 'r') as f:
    content = f.read()

# Check for required YAML frontmatter
if not content.startswith('---'):
    print('README missing YAML frontmatter')
    exit(1)

# Check for required fields in frontmatter
required_fields = ['title:', 'emoji:', 'sdk:', 'app_file:']
for field in required_fields:
    if field not in content:
        print(f'README missing required field: {field}')
        exit(1)

print('README format is valid')
"

# Deactivate virtual environment
deactivate

echo ""
echo "All tests passed! Your HF Spaces app is ready for deployment."
echo ""
echo "Summary:"
echo "Dependencies installed successfully"
echo "Model loading works"
echo "Gradio app syntax is valid"
echo "Data handling implemented"
echo "App launches successfully"
echo "Deployment files are ready"
echo ""
echo "To deploy to HuggingFace Spaces, run:"
echo "   export HF_TOKEN=your_token_here"
echo "   export HF_USERNAME=your_username_here"
echo "   ./scripts/deploy_to_hf.sh"