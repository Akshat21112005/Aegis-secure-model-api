#!/usr/bin/env python3
"""
Test script for the Phishing Detection API
Run this after starting the API server with: python app.py
"""

import requests
import json
import time
from typing import Dict, Any

API_URL = "http://localhost:8000"

def print_separator():
    print("\n" + "="*80 + "\n")

def test_health():
    """Test the health endpoint"""
    print("🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"Status: {data.get('status')}")
            print(f"Models loaded: {json.dumps(data.get('models_loaded'), indent=2)}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        print("Make sure the API is running: python app.py")
        return False

def test_predict(text: str, description: str) -> Dict[Any, Any]:
    """Test the prediction endpoint"""
    print(f"🔍 Testing: {description}")
    print(f"Message: {text[:100]}...")
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text, "metadata": {}},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Decision: {result.get('final_decision').upper()}")
            print(f"   Confidence: {result.get('confidence')}%")
            print(f"   Reasoning: {result.get('reasoning')[:100]}...")
            if '$$' in result.get('highlighted_text', ''):
                print(f"   ⚠️  Highlighted suspicious parts detected")
            return result
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return {}
    except requests.Timeout:
        print("⏱️  Request timed out (this can happen with URL feature extraction)")
        return {}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

def main():
    print_separator()
    print("🚀 PHISHING DETECTION API TEST SUITE")
    print_separator()
    
    # Test 1: Health Check
    if not test_health():
        print("\n⚠️  API is not responding. Please start it first:")
        print("   python app.py")
        return
    
    print_separator()
    time.sleep(1)
    
    # Test 2: Obvious Phishing with URL
    test_predict(
        "URGENT! Your account has been suspended. Click here to verify immediately: http://suspicious-bank-login.com/verify",
        "Obvious phishing with suspicious URL"
    )
    
    print_separator()
    time.sleep(1)
    
    # Test 3: Prize/Lottery Scam
    test_predict(
        "Congratulations! You've won $5,000 in our lottery. Claim your prize now by clicking: http://bit.ly/claim-prize-2024",
        "Prize/lottery scam"
    )
    
    print_separator()
    time.sleep(1)
    
    # Test 4: Package Delivery Scam
    test_predict(
        "Your package delivery failed. Please reschedule at: http://fedex-rescheduling.com/track",
        "Package delivery scam"
    )
    
    print_separator()
    time.sleep(1)
    
    # Test 5: Legitimate Message (No URL)
    test_predict(
        "Hi, can we meet tomorrow at 3pm for coffee? Let me know if that works for you.",
        "Legitimate message without URL"
    )
    
    print_separator()
    time.sleep(1)
    
    # Test 6: Legitimate Message (With Real URL)
    test_predict(
        "Here's the article I mentioned about machine learning: https://www.tensorflow.org/tutorials/quickstart/beginner",
        "Legitimate message with real URL"
    )
    
    print_separator()
    time.sleep(1)
    
    # Test 7: Banking Phishing
    test_predict(
        "Chase Bank Alert: Unusual activity detected. Verify your identity: http://chase-secure-verify.net/login",
        "Banking phishing attempt"
    )
    
    print_separator()
    time.sleep(1)
    
    # Test 8: Message with No URLs or Obvious Indicators
    test_predict(
        "Your order has been confirmed. Thank you for shopping with us!",
        "Ambiguous legitimate message"
    )
    
    print_separator()
    print("✨ Test suite completed!")
    print_separator()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite error: {e}")
