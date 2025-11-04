import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import os

class PhishingPredictor:
    def __init__(self, model_path="final_semantic_model"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def predict(self, text):
        if not text or not text.strip():
            return {
                "text": text,
                "phishing_probability": 0.0,
                "prediction": "ham",
                "confidence": "low"
            }
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()
        
        probabilities = softmax(logits, axis=1)[0]
        
        phishing_prob = float(probabilities[1])
        
        prediction = "phishing" if phishing_prob > 0.5 else "ham"
        
        confidence_score = max(phishing_prob, 1 - phishing_prob)
        if confidence_score > 0.8:
            confidence = "high"
        elif confidence_score > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "text": text,
            "phishing_probability": round(phishing_prob, 4),
            "ham_probability": round(float(probabilities[0]), 4),
            "prediction": prediction,
            "confidence": confidence,
            "confidence_score": round(confidence_score, 4)
        }
    
    def predict_batch(self, texts):
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def main():
    try:
        predictor = PhishingPredictor()
        
        print("\n" + "="*60)
        print("SMS PHISHING DETECTION SYSTEM")
        print("="*60)
        print("Enter SMS messages to analyze (type 'quit' to exit)")
        print("Type 'batch' to analyze multiple messages at once")
        print("-"*60)
        
        while True:
            user_input = input("\nEnter SMS message: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'batch':
                print("\nBatch mode - Enter messages (empty line to finish):")
                messages = []
                while True:
                    msg = input(f"Message {len(messages) + 1}: ").strip()
                    if not msg:
                        break
                    messages.append(msg)
                
                if messages:
                    results = predictor.predict_batch(messages)
                    print(f"\n{'='*60}")
                    print("BATCH RESULTS")
                    print(f"{'='*60}")
                    for i, result in enumerate(results, 1):
                        print(f"\nMessage {i}: {result['text'][:50]}...")
                        print(f"Prediction: {result['prediction'].upper()}")
                        print(f"Phishing Probability: {result['phishing_probability']:.1%}")
                        print(f"Confidence: {result['confidence'].upper()}")
                        print("-" * 40)
                else:
                    print("No messages entered.")
            
            elif user_input:
                result = predictor.predict(user_input)
                
                print(f"\n{'='*60}")
                print("PREDICTION RESULT")
                print(f"{'='*60}")
                print(f"Message: {result['text']}")
                print(f"Prediction: {result['prediction'].upper()}")
                print(f"Phishing Probability: {result['phishing_probability']:.1%}")
                print(f"Ham Probability: {result['ham_probability']:.1%}")
                print(f"Confidence: {result['confidence'].upper()} ({result['confidence_score']:.1%})")
                
                prob = result['phishing_probability']
                if prob > 0.7:
                    print("🚨 HIGH RISK - Likely phishing!")
                elif prob > 0.3:
                    print("⚠️  MEDIUM RISK - Be cautious")
                else:
                    print("✅ LOW RISK - Appears legitimate")
            
            else:
                print("Please enter a message or 'quit' to exit.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train.py first to create the model.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()