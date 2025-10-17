from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import pandas as pd
import numpy as np
import torch
import logging
import json
import time
from typing import Dict, List, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EvalPrediction,
    TrainerCallback  # Base class for callbacks
)
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_training")

# Create FastAPI app
app = FastAPI(
    title="Model Training Service",
    description="Microservice for training sentiment classification models",
    version="1.0.0",
)

# Create necessary directories
# When running in Docker, use absolute paths
if os.path.exists("/app/data/processed"):
    PROCESSED_DIR = "/app/data/processed"
    MODELS_DIR = "/app/models/checkpoints"
    LOGS_DIR = "/app/logs"
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Constants
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
BATCH_SIZE = 8
NUM_EPOCHS = 3
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "sentiment_model.pth")
METRICS_PATH = os.path.join(LOGS_DIR, "train_metrics.json")

# Training state
training_logs = []
training_metrics = {
    "accuracy": 0.0,
    "f1": 0.0,
    "loss": 0.0,
    "epoch": 0,
    "status": "Not started"
}

def compute_metrics(eval_pred) -> Dict:
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: Evaluation predictions from Trainer,
                 can be tuple (predictions, labels) or EvalPrediction object
        
    Returns:
        Dict of metrics (accuracy, f1)
    """
    # Handle different formats of eval_pred based on transformers version
    if isinstance(eval_pred, tuple):
        # Older transformers versions may pass tuple (predictions, labels)
        predictions, labels = eval_pred
    else:
        # Newer versions use EvalPrediction object
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Ensure predictions are the right shape (get class predictions)
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        "accuracy": float(acc),
        "f1": float(f1)
    }

class SimpleCallback(TrainerCallback):
    """
    Simple callback that logs training progress and metrics.
    Inherits from TrainerCallback for compatibility with any transformers version.
    """
    def __init__(self):
        self.global_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of a training step.
        """
        global training_logs, training_metrics
        
        self.global_step += 1
        if self.global_step % 10 == 0:  # Log every 10 steps
            # Safely check for log_history attribute
            log_history = getattr(state, "log_history", [])
            current_loss = None
            
            # Safely extract loss value if available
            if log_history and isinstance(log_history, list) and log_history:
                if isinstance(log_history[-1], dict) and "loss" in log_history[-1]:
                    current_loss = log_history[-1]["loss"]
            
            log_entry = {
                "step": self.global_step,
                "timestamp": time.time(),
                "loss": current_loss
            }
            
            training_logs.append(log_entry)
            if current_loss is not None:
                training_metrics["loss"] = current_loss
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after evaluation.
        """
        global training_metrics
        
        if metrics:
            training_logs.append({
                "step": self.global_step,
                "timestamp": time.time(),
                **metrics
            })
            
            # Update metrics
            if "eval_accuracy" in metrics:
                training_metrics["accuracy"] = metrics["eval_accuracy"]
            if "eval_f1" in metrics:
                training_metrics["f1"] = metrics["eval_f1"]
            if "eval_loss" in metrics:
                training_metrics["loss"] = metrics["eval_loss"]

def convert_labels_to_ids(sentiment_labels: List[str]) -> List[int]:
    """
    Convert sentiment labels to numeric IDs.
    
    Args:
        sentiment_labels: List of sentiment labels (positive/neutral/negative)
        
    Returns:
        List of label IDs (0: negative, 1: neutral, 2: positive)
    """
    label_map = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }
    
    return [label_map.get(label, 1) for label in sentiment_labels]  # Default to neutral if unknown

async def train_model():
    """
    Train a sentiment classification model.
    """
    global training_logs, training_metrics
    
    try:
        # Update training status
        training_metrics["status"] = "Loading data"
        
        # Load dataset
        logger.info("Loading dataset from CSV")
        reviews_path = os.path.join(PROCESSED_DIR, "reviews.csv")
        
        if not os.path.exists(reviews_path):
            logger.error(f"Reviews data not found at {reviews_path}")
            training_metrics["status"] = "Failed: Data not found"
            return False
        
        df = pd.read_csv(reviews_path)
        logger.info(f"Loaded {len(df)} reviews")
        
        # Convert sentiment labels to IDs
        logger.info("Converting labels to IDs")
        df["label"] = convert_labels_to_ids(df["sentiment"].tolist())
        
        # Create Hugging Face dataset
        dataset = Dataset.from_pandas(df)
        
        # Split dataset into train and validation (80/20)
        dataset = dataset.train_test_split(test_size=0.2)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        
        # Load tokenizer and model
        training_metrics["status"] = "Loading model and tokenizer"
        logger.info(f"Loading tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        logger.info(f"Loading model: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=NUM_LABELS
        )
        
        # Tokenize function
        def tokenize_function(examples):
            return tokenizer(
                examples["review_body"],
                padding="max_length",
                truncation=True,
                max_length=128
            )
        
        # Tokenize datasets
        logger.info("Tokenizing datasets")
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        # Training arguments - compatible with transformers library requirements
        training_args = TrainingArguments(
            output_dir=os.path.join(MODELS_DIR, "results"),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            weight_decay=0.01,
            logging_dir=os.path.join(LOGS_DIR, "tensorboard"),
            logging_steps=10,
            # Set evaluation and save strategies to be compatible
            eval_steps=100,     # Evaluate every 100 steps
            save_steps=100,     # Save every 100 steps (same as eval_steps)
            do_eval=True,       # Enable evaluation during training
            # Remove load_best_model_at_end to avoid strategy mismatch issues
            # We'll manually select the best model after training
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[SimpleCallback()]
        )
        
        # Start training
        training_metrics["status"] = "Training in progress"
        logger.info("Starting training")
        trainer.train()
        
        # Final evaluation to get the best metrics
        logger.info("Running final evaluation")
        training_metrics["status"] = "Final evaluation"
        eval_results = trainer.evaluate()
        
        # Save model - using the final trained model
        logger.info(f"Saving model checkpoint to {CHECKPOINT_PATH}")
        training_metrics["status"] = "Saving model"
        
        # Save the entire model (not just state_dict for better compatibility)
        model.save_pretrained(os.path.join(MODELS_DIR, "final_model"))
        tokenizer.save_pretrained(os.path.join(MODELS_DIR, "final_model"))
        
        # Also save state_dict as requested
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        
        # Update and save metrics
        training_metrics.update({
            "accuracy": float(eval_results.get("eval_accuracy", 0.0)),
            "f1": float(eval_results.get("eval_f1", 0.0)),
            "loss": float(eval_results.get("eval_loss", 0.0)),
            "status": "Completed"
        })
        
        # Save metrics to file
        with open(METRICS_PATH, "w") as f:
            json.dump(training_metrics, f, indent=4)
            
        logger.info(f"Training completed. Metrics: {training_metrics}")
        return True
    
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        training_metrics["status"] = f"Failed: {str(e)}"
        return False

@app.post("/train")
@app.get("/train")
async def train_endpoint(background_tasks: BackgroundTasks) -> Dict:
    """
    API endpoint to start model training.
    Supports both GET and POST methods.
    
    Returns:
        Dict: Status message
    """
    # Reset training logs and metrics
    global training_logs, training_metrics
    training_logs = []
    training_metrics = {
        "accuracy": 0.0,
        "f1": 0.0,
        "loss": 0.0,
        "epoch": 0,
        "status": "Starting"
    }
    
    background_tasks.add_task(train_model)
    return JSONResponse(
        content={"message": "Model training started in background"},
        status_code=202
    )

@app.get("/status")
async def status_endpoint() -> Dict:
    """
    API endpoint to get training status and logs.
    
    Returns:
        Dict: Training status and logs
    """
    return {
        "metrics": training_metrics,
        "logs": training_logs[-20:] if training_logs else []  # Return last 20 logs
    }

@app.get("/health")
async def health_check() -> Dict:
    """
    Health check endpoint.
    
    Returns:
        Dict: Health status
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
