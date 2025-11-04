import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from models import get_dl_models, PhishingDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def prepare_data(df, numerical_cols):
    print("Preparing data for DL training...")
    X = df[numerical_cols].fillna(-1).values
    y = df['label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_dl_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_accuracy = correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return model

def main():
    print("--- Starting DL Model Training ---")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    try:
        df = pd.read_csv(config.ENGINEERED_TRAIN_FILE)
    except FileNotFoundError:
        print(f"Error: '{config.ENGINEERED_TRAIN_FILE}' not found.")
        print("Please run `python data_pipeline.py` first.")
        return
    
    X_scaled, y, scaler = prepare_data(df, config.NUMERICAL_FEATURES)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y,
        test_size=config.ML_TEST_SIZE,
        random_state=config.ML_MODEL_RANDOM_STATE,
        stratify=y
    )
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    
    train_dataset = PhishingDataset(X_train, y_train)
    val_dataset = PhishingDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config.DL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.DL_BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_dim = X_train.shape[1]
    dl_models = get_dl_models(input_dim)
    
    for name, model in dl_models.items():
        print(f"\n--- Training {name} ---")
        
        trained_model = train_dl_model(
            model, train_loader, val_loader, device,
            epochs=config.DL_EPOCHS,
            lr=config.DL_LEARNING_RATE
        )
        
        save_path = os.path.join(config.MODELS_DIR, f"{name}.pt")
        torch.save(trained_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    scaler_path = os.path.join(config.MODELS_DIR, "dl_scaler.pkl")
    import joblib
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    print("\n--- DL Model Training Complete ---")

if __name__ == "__main__":
    main()
