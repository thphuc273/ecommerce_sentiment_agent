#!/bin/bash

# Setup Environment Variables Script for E-commerce Sentiment Analysis

echo "🔧 Setting up environment variables for E-commerce Sentiment Analysis"
echo "============================================================================="

# Check if .env already exists
if [ -f ".env" ]; then
    echo ".env file already exists. Creating backup..."
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
    echo "Backup created: .env.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Copy template to .env
if [ -f ".env.example" ]; then
    cp .env.example .env
    echo "Created .env from template"
else
    echo ".env.example not found. Creating basic .env..."
    cat > .env << 'EOF'
# Basic .env configuration
HF_TOKEN=your_huggingface_token_here
HF_USERNAME=your_username
DATABASE_URL=postgres://postgres:postgres@localhost:5433/ecommerce_sentiment
LOG_LEVEL=INFO
EOF
fi

echo ""
echo "Please update the following values in your .env file:"
echo ""
echo "1. HF_TOKEN - Get from https://huggingface.co/settings/tokens"
echo "2. HF_USERNAME - Your Hugging Face username"
echo "3. Other configuration values as needed"
echo ""
echo "To get your Hugging Face token:"
echo "   1. Go to https://huggingface.co/settings/tokens"
echo "   2. Click 'New token'"
echo "   3. Select 'Write' permissions"
echo "   4. Copy the token to your .env file"
echo ""

# Check if user wants to open the file for editing
read -p "Do you want to open .env for editing now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v code &> /dev/null; then
        code .env
    elif command -v nano &> /dev/null; then
        nano .env
    elif command -v vim &> /dev/null; then
        vim .env
    else
        echo "Please edit .env manually with your preferred editor"
    fi
fi

echo ""
echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "   1. Edit .env with your actual values"
echo "   2. Run: docker-compose up -d"
echo "   3. Test your deployment with: ./scripts/test_hf_app.sh"