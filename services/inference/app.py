from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import os
import json
import torch
import numpy as np
import requests
from typing import List, Dict, Optional, Any
import logging
import uvicorn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("inference")

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis Inference Service",
    description="Microservice for analyzing sentiment of text with context from similar reviews",
    version="1.0.0",
)

# Set up paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pth")
FINAL_MODEL_DIR = os.path.join(MODELS_DIR, "final_model")

# Service URLs
RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://retrieval:8000")

# Model settings
MODEL_NAME = "distilbert-base-uncased"
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Better pre-trained sentiment model
NUM_LABELS = 3
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Global variables
tokenizer = None
model = None
sentiment_pipeline = None

class AnalysisRequest(BaseModel):
    query_text: str
    image_url: Optional[HttpUrl] = None

class SimilarReview(BaseModel):
    review_text: str
    sentiment: str
    similarity_score: float

class AnalysisResponse(BaseModel):
    sentiment: str
    confidence: float
    summary: str
    similar_reviews: List[SimilarReview]

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global tokenizer, model, sentiment_pipeline
    
    try:
        # Try to load the fine-tuned model
        if os.path.exists(FINAL_MODEL_DIR):
            logger.info(f"Loading fine-tuned model from {FINAL_MODEL_DIR}")
            model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_DIR)
            tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)
        elif os.path.exists(MODEL_PATH):
            logger.info(f"Loading model weights from {MODEL_PATH}")
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        else:
            # Fall back to a pre-trained sentiment analysis model
            logger.warning(f"Fine-tuned model not found, using pre-trained sentiment model: {SENTIMENT_MODEL_NAME}")
            try:
                model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
                tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
            except Exception as e:
                logger.error(f"Error loading pre-trained sentiment model: {str(e)}")
                logger.warning(f"Falling back to basic model: {MODEL_NAME}")
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Set up sentiment pipeline
        sentiment_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None  # Use top_k=None instead of return_all_scores=True
        )
        logger.info("Sentiment analysis model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        # Continue without model - we'll handle this in the endpoint

def fetch_similar_reviews(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch similar reviews from the retrieval service.
    
    Args:
        query_text: The query text to find similar reviews for
        top_k: Number of similar reviews to fetch
        
    Returns:
        List of similar review objects
    """
    try:
        url = f"{RETRIEVAL_SERVICE_URL}/search_similar"
        payload = {
            "query_text": query_text,
            "top_k": top_k
        }
        
        logger.info(f"Fetching similar reviews from {url}")
        
        # Reduce timeout to fail faster if the service is not responding
        response = requests.post(url, json=payload, timeout=5)
        
        if response.status_code == 200:
            results = response.json().get("results", [])
            logger.info(f"Retrieved {len(results)} similar reviews")
            return results
        else:
            logger.error(f"Error fetching similar reviews: {response.status_code}, {response.text}")
            return []
            
    except requests.exceptions.Timeout:
        logger.error("Timeout connecting to retrieval service. Check if the service is running.")
        return []
    except requests.exceptions.ConnectionError:
        logger.error("Connection error to retrieval service. Check if the service is running on the correct port.")
        return []
    except Exception as e:
        logger.error(f"Error connecting to retrieval service: {str(e)}")
        return []

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text using the loaded model.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dict with sentiment analysis results
    """
    if sentiment_pipeline is None:
        logger.error("Sentiment model not initialized")
        return {"label": "neutral", "score": 0.5}
    
    try:
        # Get sentiment predictions
        results = sentiment_pipeline(text)
        
        # Debug logging to understand the structure
        logger.info(f"Pipeline results type: {type(results)}")
        logger.info(f"Pipeline results: {results}")
        
        # Handle different return formats from the pipeline
        if isinstance(results, list) and len(results) > 0:
            logger.info(f"First result type: {type(results[0])}")
            if isinstance(results[0], list):
                # If results is a list of lists (newer transformers versions)
                scores = results[0]  # First (and only) input's results
                logger.info(f"Using nested list format, scores: {scores}")
            elif isinstance(results[0], dict):
                # If results is a list of dictionaries
                scores = results
                logger.info(f"Using list of dicts format, scores: {scores}")
            else:
                # Fallback for unexpected format
                logger.error(f"Unexpected results[0] format: {type(results[0])}, value: {results[0]}")
                return {"label": "neutral", "score": 0.5}
        elif isinstance(results, list) and len(results) == 0:
            logger.error("Pipeline returned empty list")
            return {"label": "neutral", "score": 0.5}
        else:
            # Fallback for completely unexpected format
            logger.error(f"Unexpected results format: {type(results)}, value: {results}")
            return {"label": "neutral", "score": 0.5}
            
        # Validate scores format before processing
        if not scores or not isinstance(scores, list):
            logger.error(f"Invalid scores format: {scores}")
            return {"label": "neutral", "score": 0.5}
            
        # Validate each score item
        for i, score in enumerate(scores):
            if not isinstance(score, dict):
                logger.error(f"Score item {i} is not a dict: {type(score)}, value: {score}")
                return {"label": "neutral", "score": 0.5}
            if "score" not in score:
                logger.error(f"Score item {i} missing 'score' key: {score}")
                return {"label": "neutral", "score": 0.5}
                
        highest_score = max(scores, key=lambda x: x.get("score", 0) if isinstance(x, dict) else 0)
        
        # Ensure highest_score is a dictionary
        if not isinstance(highest_score, dict):
            logger.error(f"Expected dict for highest_score, got {type(highest_score)}")
            return {"label": "neutral", "score": 0.5}
        label_map = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral", 
            "LABEL_2": "positive"
        }
        
        label = highest_score.get("label", "")
        
        # Handle different label formats
        if isinstance(label, int) and label in ID2LABEL:
            # Direct integer label
            label = ID2LABEL[label]
        elif isinstance(label, str):
            # String label handling
            if label.isdigit() and int(label) in ID2LABEL:
                # Numeric string like "0", "1", "2"
                label = ID2LABEL[int(label)]
            elif label in label_map:
                # Format like "LABEL_0", "LABEL_1"
                label = label_map[label]
            elif label.lower() in ["positive", "neutral", "negative"]:
                # Already in the desired format
                label = label.lower()
            else:
                # Unknown format
                logger.warning(f"Unknown label format: {label}, defaulting to 'neutral'")
                label = "neutral"
        else:
            # Completely unexpected type
            logger.error(f"Unexpected label type: {type(label)}")
            label = "neutral"
        
        return {
            "label": label,
            "score": highest_score.get("score", 0.5)
        }
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return {"label": "neutral", "score": 0.5}

def generate_summary(sentiment_result: Dict[str, Any], similar_reviews: List[Dict[str, Any]]) -> str:
    """
    Generate a summary of the sentiment analysis results.
    
    Args:
        sentiment_result: Result from sentiment analysis
        similar_reviews: List of similar reviews
        
    Returns:
        String summary
    """
    # Map labels to human-readable sentiment
    label_map = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }
    
    # Count sentiments of similar reviews
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for review in similar_reviews:
        sentiment = review.get("sentiment", "neutral")
        
        # Map the sentiment if it's a label
        if isinstance(sentiment, str):
            if sentiment in label_map:
                sentiment = label_map[sentiment]
            elif sentiment.isdigit() and int(sentiment) in [0, 1, 2]:
                # Map numeric strings: 0 -> negative, 1 -> neutral, 2 -> positive
                sentiment = ["negative", "neutral", "positive"][int(sentiment)]
            elif sentiment.lower() not in sentiment_counts:
                # Default for unknown sentiments
                logger.warning(f"Unknown sentiment value: {sentiment}, defaulting to 'neutral'")
                sentiment = "neutral"
            else:
                # Normalize to lowercase
                sentiment = sentiment.lower()
        elif isinstance(sentiment, int) and 0 <= sentiment <= 2:
            # Direct integer sentiment
            sentiment = ["negative", "neutral", "positive"][sentiment]
        else:
            # Default for unexpected types
            logger.warning(f"Unexpected sentiment type: {type(sentiment)}, defaulting to 'neutral'")
            sentiment = "neutral"
            
        sentiment_counts[sentiment] += 1
    
    # Determine most common sentiment
    # Handle the case where there are no similar reviews
    if not similar_reviews:
        # If no similar reviews, just return the query sentiment
        query_sentiment = sentiment_result["label"]
        # Map the sentiment if it's a label
        if query_sentiment in label_map:
            query_sentiment = label_map[query_sentiment]
        return f"This review is {query_sentiment}, but no similar reviews were found for comparison."
    
    most_common = max(sentiment_counts.items(), key=lambda x: x[1])
    most_common_sentiment = most_common[0]
    most_common_count = most_common[1]
    
    # Generate summary
    query_sentiment = sentiment_result["label"]
    query_confidence = sentiment_result["score"]
    
    if query_sentiment == most_common_sentiment and query_confidence > 0.7:
        prefix = "Strongly"
    else:
        prefix = "Mostly"
    
    # Avoid division by zero
    review_count = len(similar_reviews)
    if review_count > 0:
        ratio = most_common_count / review_count
    else:
        ratio = 0
    
    if ratio > 0.7:
        strength = "overwhelmingly"
    elif ratio > 0.5:
        strength = "generally"
    else:
        strength = "somewhat"
    
    return f"{prefix} {query_sentiment}, with {strength} similar opinions from related reviews."

@app.post("/analyze")
async def analyze(request: AnalysisRequest) -> Dict:
    """
    Analyze sentiment of query text with context from similar reviews.
    
    Args:
        request: AnalysisRequest with query_text and optional image_url
        
    Returns:
        Dict with sentiment analysis results and similar reviews
    """
    try:
        # Validate input
        if not request.query_text or len(request.query_text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Query text cannot be empty"
            )
        
        # Analyze sentiment of query text first - this is essential
        logger.info(f"Analyzing sentiment for query: {request.query_text}")
        sentiment_result = analyze_sentiment(request.query_text)
        
        # Try to fetch similar reviews - this is optional
        try:
            similar_reviews = fetch_similar_reviews(request.query_text)
        except Exception as e:
            logger.error(f"Error fetching similar reviews: {str(e)}")
            similar_reviews = []  # Continue with empty reviews
            
        # Format similar reviews - safely handle missing fields
        formatted_reviews = []
        for review in similar_reviews:
            try:
                formatted_reviews.append(
                    SimilarReview(
                        review_text=review.get("review_text", "No text available"),
                        sentiment=review.get("sentiment", "neutral"),
                        similarity_score=float(review.get("similarity_score", 0.0))
                    )
                )
            except Exception as e:
                logger.warning(f"Error formatting review: {str(e)}")
                # Skip this review and continue
        
        # Generate summary - safely handling potential division by zero
        try:
            summary = generate_summary(sentiment_result, similar_reviews)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            summary = f"Sentiment is {sentiment_result['label']} with confidence {sentiment_result['score']:.2f}"
        
        return {
            "sentiment": sentiment_result["label"],
            "confidence": sentiment_result["score"],
            "summary": summary,
            "similar_reviews": formatted_reviews
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing analysis request: {str(e)}"
        )

@app.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint.
    
    Returns:
        Dict: Health status
    """
    model_status = "loaded" if model is not None else "not_loaded"
    tokenizer_status = "loaded" if tokenizer is not None else "not_loaded"
    
    # Check retrieval service connectivity
    retrieval_status = "unknown"
    try:
        response = requests.get(f"{RETRIEVAL_SERVICE_URL}/health", timeout=2)
        if response.status_code == 200:
            retrieval_status = "connected"
        else:
            retrieval_status = f"error_code_{response.status_code}"
    except Exception:
        retrieval_status = "not_connected"
    
    return {
        "status": "healthy",
        "components": {
            "model": model_status,
            "tokenizer": tokenizer_status,
            "retrieval_service": retrieval_status
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
