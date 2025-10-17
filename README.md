# 📊 E-commerce Sentiment Analysis System

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11-green)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

## Overview

The E-commerce Sentiment Analysis System is an AI-powered platform that provides deep insights into customer sentiment from product reviews across multiple e-commerce platforms. By leveraging advanced natural language processing and computer vision techniques, the system analyzes both text and image data to extract sentiment, identify trends, and surface similar customer experiences.

Key features include:
- Automated collection and processing of product reviews
- Sentiment analysis using state-of-the-art transformer models
- Vector-based similarity search for finding related reviews
- Multi-modal analysis supporting both text and image inputs
- Interactive web interface for exploring sentiment patterns
- Scalable microservices architecture for production deployment

## System Architecture

The system implements a modern, scalable microservices architecture with continuous integration and deployment capabilities. Each component is designed to be independently deployable, maintainable, and scalable.

### Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                        CI/CD Pipeline (GitHub Actions)                 │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        Containerized Services (Docker)                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│  │    Data     │     │    Data     │     │   Model     │              │
│  │ Collection  ├────►│ Processing  ├────►│  Training   │              │
│  └─────────────┘     └─────────────┘     └─────────────┘              │
│                                                 │                      │
│                                                 │                      │
│                                                 ▼                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│  │  Frontend   │◄────│  Inference  │◄────│  Retrieval  │              │
│  └─────────────┘     └─────────────┘     └─────────────┘              │
│         │                   │                   │                      │
└─────────┼───────────────────┼───────────────────┼──────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│                       Persistence Layer                                │
│  ┌─────────────────────────────┐   ┌─────────────────────────────┐    │
│  │     PostgreSQL Database     │   │     Vector Store (FAISS)     │    │
│  └─────────────────────────────┘   └─────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘
          │                                       │
          ▼                                       ▼
┌───────────────────────────────┐   ┌───────────────────────────────────┐
│      Hugging Face Spaces      │   │         Docker Registry           │
│    (Public Demo Deployment)   │   │    (Container Image Storage)      │
└───────────────────────────────┘   └───────────────────────────────────┘
```

### Microservices Components

#### 1. Data Collection Service
**Purpose**: Acquires and normalizes review data from various e-commerce platforms
- **Technology**: Python, FastAPI, BeautifulSoup, Requests
- **Key Features**:
  - Dataset ingestion from Amazon review datasets
  - Web scraping capabilities for fresh data
  - Initial data validation and cleaning
  - Streaming data pipeline integration
- **Endpoints**: 
  - `GET /health`: Service health check
  - `POST /collect`: Trigger data collection job

#### 2. Data Processing Service
**Purpose**: Transforms raw data into model-ready features and embeddings
- **Technology**: Python, FastAPI, PyTorch, Transformers
- **Key Features**:
  - Text preprocessing (tokenization, normalization)
  - Image feature extraction using CLIP
  - Embedding generation for vectorization
  - Dataset splitting and preparation
- **Endpoints**:
  - `GET /health`: Service health check
  - `POST /process`: Process raw data into features

#### 3. Model Training Service
**Purpose**: Trains and fine-tunes sentiment analysis models
- **Technology**: Python, FastAPI, PyTorch, Transformers, MLflow
- **Key Features**:
  - Fine-tuning transformer models (DistilBERT, BERT)
  - Hyperparameter optimization
  - Multi-GPU distributed training support
  - Model evaluation and validation
  - Training metrics logging and visualization
- **Endpoints**:
  - `GET /health`: Service health check
  - `POST /train`: Start model training job


#### 4. Retrieval Service
**Purpose**: Enables efficient semantic search across review embeddings
- **Technology**: Python, FastAPI, FAISS, NumPy
- **Key Features**:
  - Fast approximate nearest neighbor search
  - Vector index management with FAISS
  - Multi-modal embedding support
  - Configurable similarity metrics
- **Endpoints**:
  - `GET /health`: Service health check
  - `POST /search_similar`: Find similar reviews


#### 5. Inference Service
**Purpose**: Performs real-time sentiment analysis and review similarity
- **Technology**: Python, FastAPI, PyTorch, Transformers
- **Key Features**:
  - Real-time sentiment classification
  - Confidence scoring
  - Integration with retrieval service
  - Multi-modal input support (text and images)
  - Batch prediction capabilities
- **Endpoints**:
  - `GET /health`: Service health check
  - `POST /analyze`: Analyze sentiment of review

#### 6. Frontend Service
**Purpose**: Provides user interface for interacting with the system
- **Technology**: Python, Gradio
- **Key Features**:
  - Interactive web interface with Gradio
  - Real-time sentiment analysis demo
- **Access**: Web browser via http://localhost:8006 or https://huggingface.co/spaces/Felix273/ecommerce-sentiment-analysis

### Persistence Layer

#### PostgreSQL Database
- Stores structured review data and metadata
- Manages user accounts and preferences
- Tracks analysis history and results
- Ensures ACID compliance for critical operations

#### Vector Store (FAISS)
- Memory-mapped vector indices for fast similarity search
- Specialized data structure for embedding storage
- Optimized for high-dimensional nearest neighbor search
- Supports both CPU and GPU acceleration

### DevOps Infrastructure

#### CI/CD Pipeline (GitHub Actions)
- **Testing Phase**:
  - Automated unit and integration testing
  - Code quality checks with flake8
  - Security scanning with Trivy and CodeQL
- **Building Phase**:
  - Docker image building for all microservices
  - Multi-platform support (linux/amd64, linux/arm64)
  - Image tagging and versioning
- **Deployment Phase**:
  - Automatic deployment to Hugging Face Spaces
  - Docker Hub image publishing
  - Integration testing in production environment
  - Deployment notifications

#### Containerization (Docker)
- Individual containers for each microservice
- Docker Compose for local orchestration
- Volume mounting for persistent data
- Network isolation and service discovery

### Deployment Targets

#### Development Environment
- Local Docker Compose setup
- Hot-reloading for rapid development
- Local PostgreSQL database
- Debugging and profiling tools

#### Production Environment
- Hugging Face Spaces for public demo
- Docker Hub for container registry

## Dataset Description

The system uses a carefully curated dataset of e-commerce product reviews to train and evaluate sentiment analysis models. This dataset serves as the foundation for the entire sentiment analysis pipeline.

### Dataset Overview

- **Source**: Amazon Product Reviews dataset, a widely-used benchmark for sentiment analysis
- **Size**: Over 1,000 product reviews spanning multiple product categories
- **Structure**: CSV format with review text, star ratings, and sentiment labels
- **Labels**: Three-class sentiment categorization (positive, neutral, negative)
- **Features**: 
  - `review_body`: Full text content of customer reviews
  - `stars`: Numeric rating (1-5 stars) given by customers
  - `sentiment`: Derived sentiment label (positive, neutral, negative)

### Data Processing Pipeline

1. **Collection Phase**:
   - Raw review data is collected from Amazon dataset
   - Initial filtering removes uninformative or extremely short reviews
   - Basic data cleaning to handle special characters and formatting issues

2. **Preprocessing Phase**:
   - Text normalization (lowercase, punctuation handling, etc.)
   - Tokenization using transformer model tokenizers
   - Star ratings are converted to sentiment categories:
     - 1-2 stars: Negative
     - 3 stars: Neutral
     - 4-5 stars: Positive

3. **Feature Extraction**:
   - Text embeddings are generated using transformer models
   - Embeddings are stored in NumPy arrays for efficient retrieval
   - Vector dimensionality: 512 (using CLIP model encodings)

4. **Storage**:
   - Processed reviews stored in `data/processed/reviews.csv`
   - Embeddings stored in `data/processed/embeddings.npy`
   - Vector indices built using FAISS for similarity search

### Dataset Statistics

- **Distribution**: Approximately 60% positive, 15% neutral, 25% negative reviews
- **Average review length**: ~120 words per review
- **Vocabulary size**: ~25,000 unique tokens
- **Languages**: Primarily English language reviews
- **Time period**: Reviews spanning multiple years for diverse product categories

### Data Quality and Ethics

- **Privacy**: All personally identifiable information (PII) removed from reviews
- **Bias mitigation**: Dataset balanced across product categories to reduce bias
- **Quality control**: Manual verification of a subset of sentiment labels
- **Reproducibility**: Fixed random seeds used in train/validation/test splits

## Tech Stack

### Backend Technologies
- **Python 3.11**: Core programming language
- **FastAPI**: High-performance API framework
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformer models for NLP
- **FAISS**: Efficient similarity search and clustering
- **NumPy/Pandas**: Data processing and manipulation
- **Uvicorn**: ASGI server for FastAPI

### Frontend Technologies
- **Gradio**: Interactive UI components for ML applications
- **Matplotlib/Seaborn**: Data visualization

### Data Storage
- **PostgreSQL**: Primary relational database
- **FAISS Indices**: Vector similarity search
- **NumPy arrays**: Efficient storage for embeddings

### MLOps & Deployment
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **GitHub Actions**: CI/CD pipeline
- **Hugging Face Spaces**: Public ML demo deployment
- **Docker Hub**: Container registry

### Development Tools
- **Pytest**: Testing framework
- **Jupyter Notebooks**: Research and evaluation
- **Flake8**: Code quality checking
- **Trivy & CodeQL**: Security scanning

## 🗂️ Project Structure

```
ecommerce_sentiment_agent/
├── .github/
│   └── workflows/            # GitHub Actions workflow definitions
│       └── ci-cd.yml         # CI/CD pipeline configuration
├── data/
│   ├── raw/                  # Original unprocessed data
│   └── processed/            # Cleaned and processed data
│       ├── embeddings.npy    # Vector embeddings
│       └── reviews.csv       # Processed reviews
├── docker-compose.yml        # Multi-service configuration
├── .env.example              # Environment variables template
├── eval/
│   └── robustness_tests.ipynb # Test notebooks
├── hf_space_app.py           # Hugging Face Spaces application
├── hf_space_README.md        # README for Hugging Face Spaces
├── hf_space_requirements.txt # Dependencies for Hugging Face deployment
├── logs/
│   └── train_metrics.json    # Training logs
├── models/
│   └── checkpoints/          # Saved model weights
│       └── final_model/      # Production model files
├── README.md                 # Project documentation
├── requirements.txt          # Core dependencies
├── scripts/
│   ├── deploy_to_hf.sh       # Hugging Face deployment script
│   ├── setup_env.sh          # Environment setup script
│   ├── setup_repo.sh         # Repository initialization script
│   ├── test_hf_app.sh        # Test Hugging Face app locally
│   └── validate_env.sh       # Environment validation script
├── services/
│   ├── data_collection/      # Review collection service
│   │   ├── app.py            # FastAPI application
│   │   ├── Dockerfile        # Container definition
│   │   └── requirements.txt  # Service dependencies
│   ├── data_processing/      # Embedding generation service
│   ├── frontend/             # Gradio web interface
│   ├── inference/            # Sentiment analysis service
│   ├── model_training/       # Model training service
│   └── retrieval/            # Vector search service
├── setup.sh                  # Main setup script
└── tests/                    # Test suite
    ├── requirements.txt      # Test dependencies
    └── test_inference.py     # Inference service tests
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

### Data Collection Service (Port 8001)

- `GET /health`: Health check
- `POST /collect`: Trigger data collection
- `GET /status/{job_id}`: Check collection status

**Example - Trigger data collection:**
```
POST /collect
```

Response:
```json
{
  "message": "Data collection process started in the background"
}
```

### Data Processing Service (Port 8002)

- `GET /health`: Health check
- `POST /process`: Process raw data
- `POST /embeddings`: Generate embeddings
- `GET /status/{job_id}`: Check processing status

### Model Training Service (Port 8003)

- `GET /health`: Health check
- `POST /train`: Start model training
- `GET /status/{job_id}`: Check training status
- `GET /metrics/{job_id}`: Get training metrics
- `GET /models`: List available models

### Retrieval Service (Port 8004)

- `GET /health`: Health check
- `POST /search_similar`: Find similar reviews
- `POST /build_index`: Build vector index
- `GET /status`: Get index statistics

### Inference Service (Port 8005)

- `GET /health`: Health check
- `POST /analyze`: Analyze sentiment
- `POST /batch_analyze`: Process multiple reviews
- `GET /models`: List available models

**Example - Analyze sentiment:**
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

## 🔐 Environment Configuration

### Setting Up .env File

This project uses environment variables for configuration. Follow these steps to set up your environment:

1. **Create a .env file** at the project root:
```bash
# Create a .env file from template
cp .env.example .env
# OR run the setup script
./scripts/setup_env.sh
```

2. **Configure essential variables** in your .env file:
```bash
# Hugging Face configuration
HF_TOKEN=your_huggingface_token_here  # Get from https://huggingface.co/settings/tokens
HF_USERNAME=your_username

# Database configuration
DATABASE_URL=postgres://postgres:postgres@localhost:5433/ecommerce_sentiment
POSTGRES_PASSWORD=postgres

# Service URLs for local development
RETRIEVAL_SERVICE_URL=http://localhost:8004
INFERENCE_SERVICE_URL=http://localhost:8005
```

3. **Validate your configuration**:
```bash
./scripts/validate_env.sh
```

### Getting Your Hugging Face Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name: `ecommerce-sentiment-analysis`
4. Role: `Write` (required for deployment)
5. Copy the token to your .env file

### Environment Variables Reference

| Variable | Purpose | Default |
|----------|---------|---------|
| `HF_TOKEN` | Authentication for HF deployment | (required for deployment) |
| `HF_USERNAME` | Your HF username | (required for deployment) |
| `DATABASE_URL` | PostgreSQL connection string | postgres://postgres:postgres@localhost:5433/ecommerce_sentiment |
| `RETRIEVAL_SERVICE_URL` | URL for retrieval service | http://localhost:8004 |
| `INFERENCE_SERVICE_URL` | URL for inference service | http://localhost:8005 |
| `MODEL_NAME` | Model for sentiment analysis | distilbert-base-uncased |
| `LOG_LEVEL` | Logging verbosity | INFO |

## 🚀 CI/CD and Deployment

### GitHub Actions Pipeline

The project includes a comprehensive CI/CD pipeline that automatically:

1. **Testing Phase**:
   - Runs unit and integration tests on Python 3.11
   - Performs code quality checks with flake8
   - Ensures critical functionality works before deployment

2. **Security Phase**:
   - Scans for vulnerabilities with Trivy
   - Performs code security analysis with CodeQL
   - Identifies potential security issues early

3. **Build Phase**:
   - Creates Docker images for all microservices
   - Builds for multiple platforms (amd64, arm64)
   - Tags images based on branch and commit

4. **Deployment Phase**:
   - Pushes images to Docker Hub (if configured)
   - Deploys to Hugging Face Spaces automatically
   - Runs integration tests on the deployed application

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


##�📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributors

- Phuc Nguyen
