import gradio as gr
import requests
import json
from typing import Optional, Dict, Any, List, Tuple
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("frontend")

# Map model labels to human-readable sentiments
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

# Define the inference service URL
INFERENCE_SERVICE_URL = os.environ.get("INFERENCE_SERVICE_URL", "http://inference:8000")
logger.info(f"Using inference service at: {INFERENCE_SERVICE_URL}")

def analyze_review(review_text: str, image: Optional[Any] = None) -> str:
    """
    Send the review text and optional image to the inference service
    and return the sentiment analysis results.
    
    Args:
        review_text: The text of the review to analyze
        image: Optional image file uploaded by the user
    
    Returns:
        Formatted text with sentiment analysis results and similar reviews
    """
    if not review_text or len(review_text.strip()) == 0:
        return "Please enter some review text to analyze."
    
    # Prepare the request payload
    payload = {"query_text": review_text}
    
    # If an image was uploaded, we'd need to handle it
    # For now, we'll just acknowledge it but not send it
    # as the current inference service doesn't handle images directly
    image_note = ""
    if image is not None:
        image_note = "\n\nNote: Image was uploaded but not processed in this version."
    
    try:
        # Call the inference service
        logger.info(f"Sending request to inference service: {review_text[:50]}...")
        response = requests.post(
            f"{INFERENCE_SERVICE_URL}/analyze", 
            json=payload,
            timeout=10
        )
        
        logger.info(f"Response status: {response.status_code}")
        
        # Check if request was successful
        if response.status_code == 200:
            try:
                result = response.json()
                logger.info(f"Received response: {result}")
                
                # Safely extract sentiment with fallback
                sentiment_label = result.get('sentiment', 'neutral')
                if sentiment_label in LABEL_MAP:
                    human_sentiment = LABEL_MAP[sentiment_label].upper()
                else:
                    # If label is not in our map, use as is
                    human_sentiment = str(sentiment_label).upper()
            except Exception as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                return f"❌ Error: Failed to parse response from inference service"
            
            # Format the output with safe access
            output = f"### Sentiment Analysis Results\n\n"
            output += f"**Sentiment:** {human_sentiment}\n"
            output += f"**Confidence:** {result.get('confidence', 0.0):.2f}\n\n"
            output += f"**Summary:** {result.get('summary', 'No summary available')}\n\n"
            output += f"**Review Text:** {review_text}\n\n"
            
            # Add similar reviews with safe access
            output += "### Top Similar Reviews\n\n"
            
            similar_reviews = result.get('similar_reviews', [])
            if not similar_reviews or len(similar_reviews) == 0:
                output += "No similar reviews found.\n"
            else:
                try:
                    # Take only top 3 reviews
                    top_reviews = similar_reviews[:3] if isinstance(similar_reviews, list) else []
                    for i, review in enumerate(top_reviews, 1):
                        if not isinstance(review, dict):
                            logger.warning(f"Review {i} is not a dict: {type(review)}")
                            continue
                            
                        # Convert the sentiment label if needed with safe access
                        sentiment = review.get('sentiment', 'neutral')
                        if sentiment in LABEL_MAP:
                            sentiment = LABEL_MAP[sentiment]
                        
                        similarity_score = review.get('similarity_score', 0.0)
                        review_text_snippet = review.get('review_text', 'No text available')
                        
                        output += f"**Review {i}** ({str(sentiment).capitalize()})\n"
                        output += f"Similarity Score: {similarity_score:.2f}\n"
                        output += f"Text: {str(review_text_snippet)[:200]}...\n\n"
                except Exception as e:
                    logger.error(f"Error processing similar reviews: {str(e)}")
                    output += f"Error displaying similar reviews: {str(e)}\n"
            
            return output + image_note
        else:
            error_msg = f"Error from inference service: {response.status_code}"
            try:
                error_detail = response.json().get("detail", "No details provided")
                error_msg += f" - {error_detail}"
            except:
                pass
            logger.error(error_msg)
            return f"Error: {error_msg}"
            
    except requests.exceptions.Timeout:
        error_msg = "Timeout connecting to inference service. Is the service running?"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except requests.exceptions.ConnectionError:
        error_msg = "Connection error. Is the inference service running on port 8004?"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="E-commerce Sentiment Analysis") as demo:
        gr.Markdown("# E-commerce Sentiment Analysis")
        gr.Markdown("""
        Analyze the sentiment of product reviews with AI-powered insights.
        Enter a review text and optionally upload a product image to get sentiment analysis
        and see similar reviews from the database.
        """)
        
        with gr.Row():
            with gr.Column():
                review_input = gr.Textbox(
                    label="Review Text", 
                    placeholder="Enter your product review here...",
                    lines=5
                )
                image_input = gr.Image(
                    label="Product Image (Optional)", 
                    type="filepath",
                    visible=True
                )
                analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
            
            with gr.Column():
                result_output = gr.Markdown(
                    label="Analysis Results",
                    value="Results will appear here after analysis."
                )
                
        gr.Markdown("""
        ### About this tool
        This tool analyzes the sentiment of product reviews using AI. The sentiment can be:
        - **POSITIVE**: Indicates satisfaction and approval
        - **NEUTRAL**: Neither strongly positive nor negative
        - **NEGATIVE**: Indicates dissatisfaction or disapproval
        
        The analysis also shows similar reviews from our database to provide context.
        """)
        
        analyze_btn.click(
            fn=analyze_review, 
            inputs=[review_input, image_input], 
            outputs=result_output
        )
        
        gr.Examples(
            [
                ["This product exceeded my expectations! The quality is outstanding and it arrived earlier than expected.", None],
                ["Terrible experience with this item. It broke after one use and customer service was unhelpful.", None],
                ["The product is okay, but not worth the price. Shipping was fast though.", None]
            ],
            inputs=[review_input, image_input],
        )
        
    return demo

if __name__ == "__main__":
    # Update the requirements file
    try:
        with open("requirements.txt", "w") as f:
            f.write("fastapi>=0.95.0\n")
            f.write("uvicorn>=0.21.1\n")
            f.write("gradio>=3.32.0\n")
            f.write("requests>=2.28.2\n")
    except Exception as e:
        logger.error(f"Error updating requirements.txt: {e}")
    
    # Launch the Gradio app
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Make the server publicly available
        server_port=8000,       # Use port 8000 for the frontend (mapped to 8006 on host)
        share=False             # Don't create a public link
    )
