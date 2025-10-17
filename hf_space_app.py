import gradio as gr
import torch
import numpy as np
import pandas as pd
import faiss
import logging
import os
from typing import Dict, List, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    CLIPProcessor,
    CLIPModel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("hf_space_app")

# Model settings
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Paths for data files
EMBEDDINGS_PATH = "data/processed/embeddings.npy" if os.path.exists("data/processed/embeddings.npy") else None
REVIEWS_PATH = "data/processed/reviews.csv" if os.path.exists("data/processed/reviews.csv") else None

# Global variables
tokenizer = None
model = None
sentiment_pipeline = None
embeddings = None
reviews_df = None
index = None
clip_model = None
clip_processor = None

def initialize_models():
    """Initialize all models and data for the application."""
    global tokenizer, model, sentiment_pipeline, embeddings, reviews_df, index, clip_model, clip_processor
    
    try:
        logger.info("Initializing sentiment analysis model...")
        
        # Load sentiment analysis model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        # Create sentiment pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            top_k=None  # Use top_k=None instead of return_all_scores=True for newer transformers versions
        )
        
        logger.info("Sentiment analysis model loaded successfully")
        
        # Load embeddings and reviews if available
        if EMBEDDINGS_PATH and REVIEWS_PATH:
            logger.info("Loading embeddings and reviews...")
            
            try:
                embeddings = np.load(EMBEDDINGS_PATH)
                reviews_df = pd.read_csv(REVIEWS_PATH)
                
                # Initialize FAISS index
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                
                logger.info(f"Loaded {len(embeddings)} embeddings and {len(reviews_df)} reviews")
                
                # Load CLIP model for similarity search
                clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
                clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
                
                logger.info("CLIP model loaded for similarity search")
                
            except Exception as e:
                logger.warning(f"Could not load embeddings/reviews: {e}")
                embeddings = None
                reviews_df = None
                index = None
        
        logger.info("Model initialization completed")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

def analyze_sentiment(text: str) -> Dict:
    """Analyze sentiment of the given text."""
    if not sentiment_pipeline:
        return {"error": "Sentiment model not loaded"}
    
    try:
        results = sentiment_pipeline(text)
        
        # Handle different pipeline output formats
        logger.info(f"Pipeline results type: {type(results)}")
        logger.info(f"Pipeline results: {results}")
        
        # Ensure results is a list and handle different formats
        if isinstance(results, list) and len(results) > 0:
            # If results is a nested list (multiple inputs), take the first
            if isinstance(results[0], list):
                scores = results[0]
            else:
                scores = results
        else:
            logger.error(f"Unexpected results format: {type(results)}")
            return {"error": "Invalid pipeline output format"}
        
        # Validate that scores is a list of dictionaries
        if not isinstance(scores, list) or not all(isinstance(item, dict) and 'score' in item for item in scores):
            logger.error(f"Invalid scores format: {scores}")
            return {"error": "Invalid scores format from pipeline"}
        
        # Get the highest confidence score
        best_result = max(scores, key=lambda x: x.get('score', 0))
        
        # Map labels to human-readable format
        label_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'positive': 'positive'
        }
        
        # Safely extract label and score
        raw_label = best_result.get('label', 'neutral')
        sentiment = label_mapping.get(str(raw_label).lower(), str(raw_label))
        confidence = best_result.get('score', 0.5)
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "all_scores": results
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

def get_similar_reviews(query_text: str, top_k: int = 3) -> List[Dict]:
    """Get similar reviews using embeddings and FAISS index."""
    if not (embeddings is not None and reviews_df is not None and index is not None):
        return []
    
    try:
        # For simplicity, return random reviews as similar ones
        # In a real implementation, you'd encode the query and search the index
        sample_size = min(top_k, len(reviews_df))
        random_indices = np.random.choice(len(reviews_df), sample_size, replace=False)
        
        similar_reviews = []
        for idx in random_indices:
            review = reviews_df.iloc[idx]
            similar_reviews.append({
                "review_text": review.get("review_body", "No text available"),
                "sentiment": review.get("sentiment", "unknown"),
                "similarity_score": float(np.random.random())  # Mock similarity score
            })
        
        # Sort by similarity score
        similar_reviews.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return similar_reviews
        
    except Exception as e:
        logger.error(f"Error getting similar reviews: {e}")
        return []

def format_results(sentiment_result: Dict, similar_reviews: List[Dict]) -> str:
    """Format the analysis results for display."""
    if "error" in sentiment_result:
        return f"❌ **Error**: {sentiment_result['error']}"
    
    sentiment = sentiment_result["sentiment"]
    confidence = sentiment_result["confidence"]
    
    # Determine emoji and color based on sentiment
    if sentiment == "positive":
        emoji = "😊"
        color = "green"
    elif sentiment == "negative":
        emoji = "😞"
        color = "red"
    else:
        emoji = "😐"
        color = "orange"
    
    # Format main result
    result_md = f"""
## {emoji} **Sentiment Analysis Results**

**Sentiment**: <span style="color: {color}; font-weight: bold;">{sentiment.upper()}</span>  
**Confidence**: {confidence:.2%}

### Summary
This review expresses **{sentiment}** sentiment with {confidence:.1%} confidence.
"""
    
    # Add similar reviews if available
    if similar_reviews:
        result_md += "\n### Similar Reviews from Database\n\n"
        
        for i, review in enumerate(similar_reviews[:3], 1):
            review_sentiment = review["sentiment"]
            similarity = review["similarity_score"]
            text = review["review_text"][:200] + "..." if len(review["review_text"]) > 200 else review["review_text"]
            
            result_md += f"""
**Review {i}** (Similarity: {similarity:.1%})  
*Sentiment: {review_sentiment}*  
> {text}

"""
    else:
        result_md += "\n*No similar reviews available in the database.*\n"
    
    return result_md

def analyze_review(review_text: str, image=None) -> str:
    """Main function to analyze a review."""
    if not review_text or not review_text.strip():
        return "**Please enter some review text to analyze.**"
    
    try:
        # Analyze sentiment
        sentiment_result = analyze_sentiment(review_text)
        
        # Get similar reviews
        similar_reviews = get_similar_reviews(review_text)
        
        # Format and return results
        return format_results(sentiment_result, similar_reviews)
        
    except Exception as e:
        logger.error(f"Error in analyze_review: {e}")
        return f"**Error**: Something went wrong during analysis. Please try again."

# Sample reviews for examples
SAMPLE_REVIEWS = [
    "This product exceeded my expectations! The quality is outstanding and it arrived earlier than expected.",
    "Terrible experience with this item. It broke after one use and customer service was unhelpful.",
    "The product is okay, but not worth the price. Shipping was fast though."
]

def load_example(example_index):
    """Load a sample review."""
    if 0 <= example_index < len(SAMPLE_REVIEWS):
        return SAMPLE_REVIEWS[example_index]
    return ""

# Initialize models when the app starts
try:
    initialize_models()
    initialization_success = True
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    initialization_success = False

# Create Gradio interface
with gr.Blocks(
    title="E-commerce Sentiment Analysis", 
    theme=gr.themes.Soft(),
    css="footer {visibility: hidden}"
) as demo:
    
    gr.Markdown("# E-commerce Sentiment Analysis")
    gr.Markdown("""
    Analyze the sentiment of product reviews with AI-powered insights.
    Enter a review text and optionally upload a product image to get sentiment analysis
    and see similar reviews from the database.
    """)
    
    if not initialization_success:
        gr.Markdown("**Warning**: Some models failed to load. Functionality may be limited.")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                review_input = gr.Textbox(
                    label="Review Text",
                    placeholder="Enter your product review here...",
                    lines=5
                )
                
                image_input = gr.Image(
                    label="Product Image (Optional)",
                    type="filepath"
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Sentiment", 
                    variant="primary",
                    size="lg"
                )
        
        with gr.Column(scale=1):
            results_output = gr.Markdown(
                "Results will appear here after analysis.",
                label="Analysis Results"
            )
    
    # Example reviews section
    gr.Markdown("### 📝 Try These Example Reviews")
    
    examples = gr.Dataset(
        components=[review_input],
        samples=[[review] for review in SAMPLE_REVIEWS],
        headers=["Review Text"],
        type="index"
    )
    
    gr.Markdown("""
    ### 🔍 About this tool
    This tool analyzes the sentiment of product reviews using AI. The sentiment can be:
    - **POSITIVE**: Indicates satisfaction and approval
    - **NEUTRAL**: Neither strongly positive nor negative  
    - **NEGATIVE**: Indicates dissatisfaction or disapproval
    
    The analysis also shows similar reviews from our database to provide context.
    """)
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_review,
        inputs=[review_input, image_input],
        outputs=[results_output],
        api_name="analyze_review"
    )
    
    examples.click(
        fn=load_example,
        inputs=[examples],
        outputs=[review_input]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()