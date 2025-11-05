import os
import re
import json
import joblib
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import google.generativeai as genai
from urllib.parse import urlparse
from dotenv import load_dotenv
import config
from models import get_ml_models, get_dl_models, FinetunedBERT
from feature_extraction import process_row
import sys

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.join(config.BASE_DIR, 'Message_model'))
from predict import PhishingPredictor

app = FastAPI(
    title="Phishing Detection API",
    description="Advanced phishing detection system using multiple ML/DL models and Gemini AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageInput(BaseModel):
    text: str
    metadata: Optional[Dict] = {}

class PredictionResponse(BaseModel):
    confidence: float
    reasoning: str
    highlighted_text: str
    final_decision: str

ml_models = {}
dl_models = {}
bert_model = None
semantic_model = None
gemini_model = None

MODEL_BOUNDARIES = {
    'logistic': 0.5,
    'svm': 0.5,
    'xgboost': 0.5,
    'attention_blstm': 0.5,
    'rcnn': 0.5,
    'bert': 0.5,
    'semantic': 0.5
}

def load_models():
    """Load all models at startup"""
    global ml_models, dl_models, bert_model, semantic_model, gemini_model
    
    print("Loading models...")
    
    models_dir = config.MODELS_DIR
    for model_name in ['logistic', 'svm', 'xgboost']:
        model_path = os.path.join(models_dir, f'{model_name}.joblib')
        if os.path.exists(model_path):
            ml_models[model_name] = joblib.load(model_path)
            print(f"✓ Loaded {model_name} model")
        else:
            print(f"⚠ Warning: {model_name} model not found at {model_path}")
    
    for model_name in ['attention_blstm', 'rcnn']:
        model_path = os.path.join(models_dir, f'{model_name}.pt')
        if os.path.exists(model_path):
            model_template = get_dl_models(input_dim=len(config.NUMERICAL_FEATURES))
            dl_models[model_name] = model_template[model_name]
            dl_models[model_name].load_state_dict(torch.load(model_path, map_location='cpu'))
            dl_models[model_name].eval()
            print(f"✓ Loaded {model_name} model")
        else:
            print(f"⚠ Warning: {model_name} model not found at {model_path}")
    
    bert_path = os.path.join(config.BASE_DIR, 'finetuned_bert')
    if os.path.exists(bert_path):
        try:
            bert_model = FinetunedBERT(bert_path)
            print("✓ Loaded BERT model")
        except Exception as e:
            print(f"⚠ Warning: Could not load BERT model: {e}")
    
    semantic_model_path = os.path.join(config.BASE_DIR, 'Message_model', 'final_semantic_model')
    if os.path.exists(semantic_model_path) and os.listdir(semantic_model_path):
        try:
            semantic_model = PhishingPredictor(model_path=semantic_model_path)
            print("✓ Loaded semantic model")
        except Exception as e:
            print(f"⚠ Warning: Could not load semantic model: {e}")
    else:
        checkpoint_path = os.path.join(config.BASE_DIR, 'Message_model', 'training_checkpoints', 'checkpoint-30')
        if os.path.exists(checkpoint_path):
            try:
                semantic_model = PhishingPredictor(model_path=checkpoint_path)
                print("✓ Loaded semantic model from checkpoint")
            except Exception as e:
                print(f"⚠ Warning: Could not load semantic model from checkpoint: {e}")
    
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✓ Initialized Gemini API")
    else:
        print("⚠ Warning: GEMINI_API_KEY not set. Set it as environment variable.")
        print("   Example: export GEMINI_API_KEY='your-api-key-here'")

def parse_message(text: str) -> tuple:
    """Parse message to extract URLs and clean text"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s])?'
    urls = re.findall(url_pattern, text)
    
    cleaned_text = re.sub(url_pattern, '', text)
    
    cleaned_text = ' '.join(cleaned_text.lower().split())
    
    cleaned_text = re.sub(r'[^a-z0-9\s.,!?-]', '', cleaned_text)
    
    cleaned_text = re.sub(r'([.,!?])+', r'\1', cleaned_text)
    
    cleaned_text = ' '.join(cleaned_text.split())
    
    return urls, cleaned_text

def extract_url_features(urls: List[str]) -> pd.DataFrame:
    """Extract 28 features from URLs"""
    if not urls:
        return pd.DataFrame()
    
    df = pd.DataFrame({'url': urls})
    
    whois_cache = {}
    ssl_cache = {}
    
    feature_list = []
    for _, row in df.iterrows():
        features = process_row(row, whois_cache, ssl_cache)
        feature_list.append(features)
    
    features_df = pd.DataFrame(feature_list)
    
    result_df = pd.concat([df, features_df], axis=1)
    
    return result_df

def custom_boundary(raw_score: float, boundary: float) -> float:
    """Apply custom boundary scaling: (raw_score - boundary) * 100"""
    return (raw_score - boundary) * 100

def get_model_predictions(features_df: pd.DataFrame, message_text: str) -> Dict:
    """Get predictions from all models"""
    predictions = {}
    
    numerical_features = config.NUMERICAL_FEATURES
    X = features_df[numerical_features].fillna(-1).values
    
    for model_name, model in ml_models.items():
        try:
            raw_score = model.predict_proba(X)[0][1]
            scaled_score = custom_boundary(raw_score, MODEL_BOUNDARIES[model_name])
            predictions[model_name] = {
                'raw_score': float(raw_score),
                'scaled_score': float(scaled_score)
            }
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    for model_name, model in dl_models.items():
        try:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                raw_score = model(X_tensor).item()
            scaled_score = custom_boundary(raw_score, MODEL_BOUNDARIES[model_name])
            predictions[model_name] = {
                'raw_score': float(raw_score),
                'scaled_score': float(scaled_score)
            }
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    if bert_model and len(features_df) > 0:
        try:
            urls = features_df['url'].tolist()
            raw_scores = bert_model.predict_proba(urls)
            avg_raw_score = np.mean([score[1] for score in raw_scores])
            scaled_score = custom_boundary(avg_raw_score, MODEL_BOUNDARIES['bert'])
            predictions['bert'] = {
                'raw_score': float(avg_raw_score),
                'scaled_score': float(scaled_score)
            }
        except Exception as e:
            print(f"Error with BERT: {e}")
    
    if semantic_model and message_text:
        try:
            result = semantic_model.predict(message_text)
            raw_score = result['phishing_probability']
            scaled_score = custom_boundary(raw_score, MODEL_BOUNDARIES['semantic'])
            predictions['semantic'] = {
                'raw_score': float(raw_score),
                'scaled_score': float(scaled_score),
                'confidence': result['confidence']
            }
        except Exception as e:
            print(f"Error with semantic model: {e}")
    
    return predictions

def get_gemini_final_decision(urls: List[str], features_df: pd.DataFrame, 
                              message_text: str, predictions: Dict, 
                              original_text: str) -> Dict:
    """Use Gemini as final judge to make decision"""
    
    if not gemini_model:
        avg_scaled_score = np.mean([p['scaled_score'] for p in predictions.values()])
        confidence = min(100, max(0, 50 + avg_scaled_score))
        
        return {
            "confidence": round(confidence, 2),
            "reasoning": "Gemini API not available. Using average model scores.",
            "highlighted_text": original_text,
            "final_decision": "phishing" if avg_scaled_score > 0 else "legitimate"
        }
    
    url_features_summary = "No URL features available"
    if len(features_df) > 0:
        feature_summary_parts = []
        for idx, row in features_df.iterrows():
            url = row.get('url', 'Unknown')
            feature_summary_parts.append(f"URL: {url}")
            feature_summary_parts.append(f"  - Length: {row.get('url_length', 'N/A')}, Dots: {row.get('count_dot', 'N/A')}, Special chars: {row.get('count_special_chars', 'N/A')}")
            feature_summary_parts.append(f"  - Domain age: {row.get('domain_age_days', 'N/A')} days, SSL valid: {row.get('cert_has_valid_hostname', 'N/A')}")
        url_features_summary = "\n".join(feature_summary_parts)
    
    context = f"""You are a phishing detection system. Analyze this message and respond with ONLY a JSON object.

MESSAGE DATA:
Original: {original_text}
Cleaned: {message_text}
URLs Found: {', '.join(urls) if urls else 'None'}

URL FEATURES:
{url_features_summary}

MODEL SCORES (positive=phishing, negative=legitimate):
{json.dumps(predictions, indent=2)}

ANALYSIS REQUIRED:
Examine the message for phishing indicators: urgency, suspicious URLs, credential requests, typos, threats, promises.
Consider all model predictions and URL features.

OUTPUT FORMAT (respond with ONLY this JSON, nothing else):
{{
  "confidence": 85.5,
  "reasoning": "Brief explanation of why this is/isn't phishing based on the data above",
  "highlighted_text": "Complete original message text with suspicious parts marked like $$this$$",
  "final_decision": "phishing" or "legitimate"
}}"""

    try:
        generation_config = {
            'temperature': 0.3,
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 1024,
        }
        response = gemini_model.generate_content(
            context,
            generation_config=generation_config
        )
        response_text = response.text.strip()
        
        if 'json' in response_text:
            response_text = response_text.split('json')[1].split('')[0].strip()
        elif '' in response_text:
            response_text = response_text.split('')[1].split('')[0].strip()
        
        if not response_text.startswith('{'):
            import re
            json_match = re.search(r'\{[^{}](?:\{[^{}]\}[^{}])\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            else:
                raise ValueError(f"Could not find JSON in Gemini response: {response_text[:200]}")
        
        result = json.loads(response_text)
        
        required_fields = ['confidence', 'reasoning', 'highlighted_text', 'final_decision']
        if not all(field in result for field in required_fields):
            raise ValueError("Missing required fields in Gemini response")
        
        result['confidence'] = float(result['confidence'])
        
        return result
    
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        avg_scaled_score = np.mean([p['scaled_score'] for p in predictions.values()])
        confidence = min(100, max(0, 50 + avg_scaled_score))
        
        return {
            "confidence": round(confidence, 2),
            "reasoning": f"Gemini API error: {str(e)}. Using average model scores for decision.",
            "highlighted_text": original_text,
            "final_decision": "phishing" if avg_scaled_score > 0 else "legitimate"
        }

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()
    print("\n" + "="*60)
    print("Phishing Detection API is ready!")
    print("="*60)
    print("API Documentation: http://localhost:8000/docs")
    print("="*60 + "\n")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Phishing Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = {
        "ml_models": list(ml_models.keys()),
        "dl_models": list(dl_models.keys()),
        "bert_model": bert_model is not None,
        "semantic_model": semantic_model is not None,
        "gemini_model": gemini_model is not None
    }
    
    return {
        "status": "healthy",
        "models_loaded": models_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(message_input: MessageInput):
    """
    Predict if a message is phishing or legitimate.
    
    This endpoint:
    1. Parses the message to extract URLs and clean text
    2. Extracts 28 features from URLs
    3. Gets predictions from all ML/DL models
    4. Applies custom boundary scaling
    5. Uses Gemini AI as final judge
    6. Returns comprehensive analysis
    """
    try:
        original_text = message_input.text
        
        if not original_text or not original_text.strip():
            raise HTTPException(status_code=400, detail="Message text cannot be empty")
        
        urls, cleaned_text = parse_message(original_text)
        
        features_df = pd.DataFrame()
        if urls:
            features_df = extract_url_features(urls)
        
        predictions = {}
        if len(features_df) > 0:
            predictions = get_model_predictions(features_df, cleaned_text)
        elif cleaned_text:
            if semantic_model:
                result = semantic_model.predict(cleaned_text)
                raw_score = result['phishing_probability']
                scaled_score = custom_boundary(raw_score, MODEL_BOUNDARIES['semantic'])
                predictions['semantic'] = {
                    'raw_score': float(raw_score),
                    'scaled_score': float(scaled_score),
                    'confidence': result['confidence']
                }
        
        if not predictions:
            raise HTTPException(
                status_code=500, 
                detail="No models available for prediction. Please ensure models are trained and loaded."
            )
        
        final_result = get_gemini_final_decision(
            urls, features_df, cleaned_text, predictions, original_text
        )
        
        return PredictionResponse(**final_result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
