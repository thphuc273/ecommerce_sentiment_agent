from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel
import torch
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("data_processing")

# Create FastAPI app
app = FastAPI(
    title="Data Processing Service",
    description="Microservice for processing text and image data into embeddings",
    version="1.0.0",
)

# Ensure output directory exists (relative to project)
# When running in Docker, the file is at /app/data/processed/reviews.csv
if os.path.exists("/app/data/processed"):
    PROCESSED_DIR = "/app/data/processed"
else:
    PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Constants
BERT_MODEL_NAME = "bert-base-uncased"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_LENGTH = 128

# Initialize models and tokenizers
@app.on_event("startup")
async def load_models():
    global bert_tokenizer, clip_processor, clip_model
    
    logger.info(f"Loading BERT tokenizer: {BERT_MODEL_NAME}")
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    logger.info(f"Loading CLIP model and processor: {CLIP_MODEL_NAME}")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

async def process_data():
    """
    Process reviews data:
    1. Tokenize text using BERT tokenizer
    2. Extract image features using CLIP
    3. Combine embeddings
    4. Save as numpy array
    """
    try:
        # Check if reviews data exists
        input_path = os.path.join(PROCESSED_DIR, "reviews.csv")
        if not os.path.exists(input_path):
            logger.error(f"Reviews data not found at {input_path}")
            return False
        
        logger.info(f"Loading reviews from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} reviews")
        
        # Process text data with BERT tokenizer
        logger.info("Tokenizing text data with BERT")
        encoded_texts = bert_tokenizer(
            df["review_body"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # For demonstration purposes, we'll generate random image embeddings
        # In a real scenario, you would process actual product images with CLIP
        logger.info("Generating placeholder image embeddings with CLIP")
        image_embedding_size = 512  # CLIP's embedding size
        num_samples = len(df)
        
        # Create dummy image embeddings (in a real scenario, use actual images)
        dummy_image_embeddings = torch.randn(num_samples, image_embedding_size)
        
        # Combine embeddings (in practice, you would use actual CLIP embeddings)
        logger.info("Combining text and image embeddings")
        
        # Use input_ids from tokenizer output as our text representation
        # Shape should be [num_samples, max_length]
        input_ids = encoded_texts["input_ids"]
        logger.info(f"Input IDs shape: {input_ids.shape}")
        
        # Convert to float for numerical operations
        text_embeddings = input_ids.float()
        
        # If text embeddings is 2D (expected), we proceed normally
        if len(text_embeddings.shape) == 2:
            # It's already [num_samples, max_length]
            text_embeddings_reshaped = text_embeddings
        else:
            # If it's 3D, flatten the last two dimensions
            logger.info(f"Reshaping 3D tensor to 2D")
            text_embeddings_reshaped = text_embeddings.reshape(num_samples, -1)
        
        logger.info(f"Reshaped text embeddings: {text_embeddings_reshaped.shape}")
        
        # Normalize to match CLIP's embedding size using interpolation
        text_embeddings_normalized = torch.nn.functional.interpolate(
            text_embeddings_reshaped.unsqueeze(1),  # Add channel dimension [batch, 1, features]
            size=image_embedding_size,  # Target size
            mode='linear'
        ).squeeze(1)  # Remove channel dimension [batch, features]
        
        logger.info(f"Normalized text embeddings: {text_embeddings_normalized.shape}")
        
        # Concatenate text and image embeddings
        combined_embeddings = torch.cat(
            [text_embeddings_normalized, dummy_image_embeddings], 
            dim=1
        )
        
        # Convert to numpy and save
        output_path = os.path.join(PROCESSED_DIR, "embeddings.npy")
        logger.info(f"Saving embeddings to {output_path}")
        np.save(output_path, combined_embeddings.numpy())
        logger.info(f"Successfully saved embeddings with shape {combined_embeddings.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return False

@app.post("/process")
@app.get("/process")
async def process_data_endpoint(background_tasks: BackgroundTasks) -> Dict:
    """
    API endpoint to trigger data processing.
    Supports both GET and POST methods.
    
    Returns:
        Dict: Status message
    """
    background_tasks.add_task(process_data)
    return JSONResponse(
        content={"message": "Data processing started in background"},
        status_code=202
    )

@app.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint.
    
    Returns:
        Dict: Health status
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
