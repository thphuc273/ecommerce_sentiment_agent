from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import datasets
import pandas as pd
import os
import logging
from typing import Dict
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("data_collection")

# Create FastAPI app
app = FastAPI(
    title="Data Collection Service",
    description="Microservice for collecting and preprocessing product reviews",
    version="1.0.0",
)

# Ensure output directory exists (relative to project)
PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
os.makedirs(PROCESSED_DIR, exist_ok=True)

def map_sentiment(stars: int) -> str:
    """Map star ratings to sentiment labels.
    
    Args:
        stars (int): Star rating (1-5)
        
    Returns:
        str: Sentiment label (positive/neutral/negative)
    """
    if stars is None:
        return "neutral"
    try:
        stars = int(stars)
    except Exception:
        return "neutral"
    if stars >= 4:
        return "positive"
    elif stars <= 2:
        return "negative"
    else:
        return "neutral"

async def collect_and_process_data():
    """
    Load, clean, and save product reviews.
    Supports 'amazon_polarity' and falls back to 'amazon_reviews_multi'.
    Keeps only review text and a sentiment label.
    """
    try:
        logger.info("Loading dataset 'amazon_polarity'...")
        # Try loading amazon_polarity; fall back to amazon_reviews_multi if needed
        try:
            dataset = datasets.load_dataset("amazon_polarity")
            # amazon_polarity has splits 'train' and 'test'; use train
            reviews = dataset["train"].select(range(1000))
            dataset_type = "amazon_polarity"
        except Exception:
            logger.info("Failed to load 'amazon_polarity', falling back to 'amazon_reviews_multi'")
            dataset = datasets.load_dataset("amazon_reviews_multi", "en")
            reviews = dataset["train"].select(range(1000))
            dataset_type = "amazon_reviews_multi"

        logger.info("Processing reviews...")
        # Extract relevant fields and convert to DataFrame
        data = []
        # Field mappings depending on dataset
        for review in reviews:
            if dataset_type == "amazon_polarity":
                # amazon_polarity has fields: 'title', 'content', 'label'
                content = review.get("content") or review.get("review_body")
                label = review.get("label")  # 0 or 1 (negative/positive)
                # Map numeric label to sentiment to keep interface similar
                if label is None:
                    sentiment = "neutral"
                else:
                    sentiment = "positive" if int(label) == 1 else "negative"
                stars = None
                data.append({
                    "review_body": content,
                    "stars": stars,
                    "sentiment": sentiment
                })
            else:
                # amazon_reviews_multi fields: 'review_body', 'stars'
                data.append({
                    "review_body": review.get("review_body"),
                    "stars": review.get("stars"),
                    "sentiment": map_sentiment(review.get("stars"))
                })

        df = pd.DataFrame(data)

        # Save to CSV
        output_path = os.path.join(PROCESSED_DIR, "reviews.csv")
        logger.info(f"Saving processed data to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} reviews")

        return True
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        return False

@app.post("/collect")
@app.get("/collect")
async def collect_data(background_tasks: BackgroundTasks) -> Dict:
    """
    API endpoint to trigger data collection process.
    Supports both GET and POST methods.
    
    Returns:
        Dict: Status message
    """
    background_tasks.add_task(collect_and_process_data)
    return JSONResponse(
        content={"message": "Data collection process started in the background"},
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
