# 📊 E-commerce Sentiment Analysis System

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10-green)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

## 🧠 Overview

The E-commerce Sentiment Analysis System is an AI-powered platform that provides deep insights into customer sentiment from product reviews across multiple e-commerce platforms. By leveraging advanced natural language processing and computer vision techniques, the system analyzes both text and image data to extract sentiment, identify trends, and surface similar customer experiences.

Key features include:
- Automated collection and processing of product reviews
- Sentiment analysis using state-of-the-art transformer models
- Vector-based similarity search for finding related reviews
- Multi-modal analysis supporting both text and image inputs
- Interactive web interface for exploring sentiment patterns
- Scalable microservices architecture for production deployment

## 🧩 Architecture

The system follows a microservices architecture pattern, with each component responsible for a specific function:

### Microservices

**1. Data Collection Service**
- Collects product reviews from various sources (Amazon datasets)
- Performs initial cleaning and normalization
- Stores raw data for further processing
- Endpoints: `/collect`, `/health`

**2. Data Processing Service**
- Generates embeddings for text and image content
- Handles preprocessing, tokenization, and feature extraction
- Creates datasets suitable for model training
- Endpoints: `/process`, `/embeddings`, `/health`

**3. Model Training Service**
- Fine-tunes transformer models for sentiment classification
- Supports distributed training across multiple GPUs
- Logs training metrics and saves model checkpoints
- Implements early stopping and model evaluation
- Endpoints: `/train`, `/status`, `/health`

**4. Retrieval Service**
- Manages vector indices for efficient similarity search
- Provides fast nearest-neighbor lookup using FAISS
- Supports both text and multi-modal queries
- Returns the most relevant reviews for a given query
- Endpoints: `/search_similar`, `/build_index`, `/health`

**5. Inference Service**
- Performs real-time sentiment analysis on new reviews
- Integrates with the retrieval service to provide context
- Handles both text and image inputs
- Generates sentiment summaries and confidence scores
- Endpoints: `/analyze`, `/health`

**6. Frontend Service**
- Provides an interactive user interface using Gradio
- Visualizes sentiment analysis results
- Allows users to submit reviews and view similar content
- Presents sentiment trends and insights
- Accessible via web browser

**7. Database (PostgreSQL)**
- Stores review data, embeddings, and analysis results
- Ensures data persistence and reliability
- Supports complex queries for analytics

### System Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Data     │     │    Data     │     │   Model     │
│ Collection  ├────►│ Processing  ├────►│  Training   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Frontend   │◄────│  Inference  │◄────│  Retrieval  │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                   ▲                   ▲
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                     ┌─────────────┐
                     │  Database   │
                     └─────────────┘
```

## ⚙️ Tech Stack

### Backend
- **Python 3.10**: Core programming language
- **FastAPI**: High-performance API framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformer models for NLP
- **FAISS**: Efficient similarity search and clustering
- **NumPy/Pandas**: Data processing and manipulation
- **Uvicorn**: ASGI server for FastAPI

### Frontend
- **Gradio**: Interactive UI components for ML applications
- **Matplotlib/Seaborn**: Data visualization

### Data Storage
- **PostgreSQL**: Primary relational database
- **NumPy arrays**: Efficient storage for embeddings

### MLOps & Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **AWS Elastic Beanstalk**: PaaS for deployment
- **AWS CodePipeline**: CI/CD pipeline
- **AWS S3**: Artifact and model storage
- **AWS CloudWatch**: Monitoring and logging

### Development Tools
- **Pytest**: Testing framework
- **Jupyter Notebooks**: Research and evaluation

## 🗂️ Project Structure

```
ecommerce_sentiment_agent/
├── data/
│   ├── raw/                 # Original unprocessed data
│   └── processed/           # Cleaned and processed data
│       ├── embeddings.npy   # Vector embeddings
│       └── reviews.csv      # Processed reviews
├── docker-compose.yml       # Multi-service configuration
├── eval/
│   └── robustness_tests.ipynb  # Test notebooks
├── logs/
│   └── train_metrics.json   # Training logs
├── models/
│   └── checkpoints/         # Saved model weights
├── README.md                # Project documentation
├── requirements.txt         # Core dependencies
├── services/
│   ├── data_collection/     # Review collection service
│   ├── data_processing/     # Embedding generation service
│   ├── frontend/            # Gradio web interface
│   ├── inference/           # Sentiment analysis service
│   ├── model_training/      # Model training service
│   └── retrieval/           # Vector search service
├── setup.sh                 # Setup script
└── tests/                   # Test suite
```

## 🚀 Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Git
- 8GB+ RAM recommended
- NVIDIA GPU (optional, for faster training)
  - If you have an NVIDIA GPU and want to use it, you'll need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and uncomment the GPU section in docker-compose.yml

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/ecommerce_sentiment_agent.git
cd ecommerce_sentiment_agent
```

2. **Run the setup script:**
```bash
chmod +x setup.sh
./setup.sh
```

3. **Access the services:**
- Frontend UI: http://localhost:8006
- Inference API: http://localhost:8005
- Retrieval API: http://localhost:8004
- Model Training API: http://localhost:8003
- Data Processing API: http://localhost:8002
- Data Collection API: http://localhost:9001

### Manual Setup

1. **Create necessary directories:**
```bash
mkdir -p data/raw data/processed models/checkpoints
```

2. **Build and start all services:**
```bash
docker-compose build
docker-compose up -d
```

3. **Check service status:**
```bash
docker-compose ps
```

4. **View logs:**
```bash
docker-compose logs -f [service_name]
```

5. **Stop all services:**
```bash
docker-compose down
```

## ☁️ Deployment Guide


## 🧪 Model Training and Evaluation

### Training Process

The sentiment analysis model is fine-tuned using a multi-stage process:

1. **Data Preparation:**
   - Tokenization and preprocessing of review text
   - Stratified splitting into train/validation/test sets
   - Generation of image embeddings (for multi-modal models)

2. **Model Training:**
   - Base model: DistilBERT (for text-only) and CLIP (for multi-modal)
   - Fine-tuning with sentiment labels (negative, neutral, positive)
   - Hyperparameters:
     - Batch size: 32
     - Learning rate: 2e-5
     - Epochs: 3-5
     - Optimizer: AdamW with weight decay

3. **Evaluation Metrics:**
   - **Accuracy**: Overall classification rate
   - **F1-Score**: Harmonic mean of precision and recall
   - **ROC-AUC**: Area under the receiver operating characteristic curve
   - **Latency**: Prediction time per sample

### Model Performance

| Model            | Accuracy | F1-Score | ROC-AUC | Latency (ms) |
|------------------|----------|----------|---------|--------------|
| DistilBERT       | 0.92     | 0.91     | 0.97    | 15           |
| BERT-base        | 0.94     | 0.93     | 0.98    | 25           |
| CLIP (text+img)  | 0.89     | 0.88     | 0.95    | 40           |

### Robustness Testing

The models are evaluated against various types of noise:
- Typos and spelling errors
- Truncated reviews
- Case variations
- Punctuation changes
- Emoji and special characters

## 🧭 API Endpoints

### Data Collection Service

**Trigger data collection:**
```
POST /collect
```

Response:
```json
{
  "message": "Data collection process started in the background"
}
```

### Inference Service

**Analyze sentiment:**
```
POST /analyze
Content-Type: application/json

{
  "query_text": "This product exceeded my expectations! The quality is outstanding.",
  "image_url": "https://example.com/product_image.jpg" // Optional
}
```

Response:
```json
{
  "sentiment": "positive",
  "confidence": 0.95,
  "summary": "Strongly positive, with overwhelmingly similar opinions from related reviews.",
  "similar_reviews": [
    {
      "review_text": "Great product, really happy with my purchase!",
      "sentiment": "positive",
      "similarity_score": 0.87
    },
    {
      "review_text": "Excellent quality and fast shipping. Will buy again.",
      "sentiment": "positive",
      "similarity_score": 0.78
    }
  ]
}
```

### Retrieval Service

**Search similar reviews:**
```
POST /search_similar
Content-Type: application/json

{
  "query_text": "Battery life is disappointing",
  "top_k": 5
}
```

Response:
```json
{
  "results": [
    {
      "review_text": "The battery drains very quickly, quite disappointed.",
      "sentiment": "negative",
      "similarity_score": 0.92
    },
    {
      "review_text": "Battery performance is terrible, doesn't last a day.",
      "sentiment": "negative",
      "similarity_score": 0.88
    }
  ]
}
```

### Model Training Service

**Start training:**
```
POST /train
Content-Type: application/json

{
  "model_name": "distilbert-base-uncased",
  "epochs": 3,
  "batch_size": 32,
  "learning_rate": 2e-5
}
```

Response:
```json
{
  "training_id": "train_20251016_001",
  "status": "started",
  "estimated_completion_time": "2025-10-16T15:30:00Z"
}
```

## � CI/CD and Deployment

### GitHub Actions Pipeline

The project includes a comprehensive CI/CD pipeline that automatically:

1. **Testing**: Runs unit tests, integration tests, and linting
2. **Security**: Performs vulnerability scanning with Trivy
3. **Building**: Creates and pushes Docker images to Docker Hub
4. **Deployment**: Automatically deploys to Hugging Face Spaces on main branch

#### Required GitHub Secrets

Set these secrets in your GitHub repository settings:

- `HF_TOKEN`: Your Hugging Face API token
- `HF_USERNAME`: Your Hugging Face username
- `DOCKER_USERNAME`: Docker Hub username (optional)
- `DOCKER_PASSWORD`: Docker Hub password (optional)

### Hugging Face Spaces Deployment

Deploy the sentiment analysis interface to Hugging Face Spaces for public access:

#### Quick Deploy

```bash
# Set environment variables
export HF_TOKEN=your_hugging_face_token
export HF_USERNAME=your_hf_username

# Test locally first
./scripts/test_hf_app.sh

# Deploy to HF Spaces
./scripts/deploy_to_hf.sh
```

#### Manual Setup

1. **Create a new Space** on Hugging Face
2. **Set SDK to Gradio** in Space settings
3. **Upload files**:
   - `hf_space_app.py` → `app.py`
   - `hf_space_requirements.txt` → `requirements.txt`
   - `hf_space_README.md` → `README.md`
4. **Add data files** (optional):
   - `data/processed/embeddings.npy`
   - `data/processed/reviews.csv`

#### Features of HF Spaces App

- **Interactive UI**: Beautiful Gradio interface
- **Real-time Analysis**: Instant sentiment analysis
- **Sample Reviews**: Pre-loaded examples
- **Similar Reviews**: Shows related reviews from database
- **Mobile Friendly**: Responsive design for all devices

### Docker Deployment

Deploy the full microservices stack using Docker:

```bash
# Build and start all services
docker-compose up --build

# Scale specific services
docker-compose up --scale inference=3 --scale retrieval=2

# Production deployment with external database
docker-compose -f docker-compose.prod.yml up
```

### Cloud Deployment Options

#### AWS ECS/EKS
- Use provided Docker images
- Configure load balancers for each service
- Set up RDS for PostgreSQL database

#### Google Cloud Run
- Deploy each service as a Cloud Run service
- Use Cloud SQL for database
- Configure service-to-service authentication

#### Azure Container Instances
- Deploy using Azure Container Groups
- Use Azure Database for PostgreSQL
- Set up Application Gateway for load balancing

## �📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Phuc Nguyen
## 🙏 Acknowledgements

- HuggingFace for the transformer models
- Facebook Research for FAISS
- FastAPI for the web framework
