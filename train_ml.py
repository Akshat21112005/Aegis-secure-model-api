import os
import pandas as pd
import joblib
import config
from models import get_ml_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def prepare_data(df, numerical_cols, categorical_cols):
    print("Preparing data for ML training...")
    X = df[numerical_cols + categorical_cols]
    y = df['label']
    numerical_transformer = Pipeline(steps=[
        ('imputer', 'passthrough'),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', 'passthrough'),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    X.loc[:, numerical_cols] = X.loc[:, numerical_cols].fillna(-1)
    X.loc[:, categorical_cols] = X.loc[:, categorical_cols].fillna('N/A')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor, X, y

def main():
    print("--- Starting ML Model Training ---")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    try:
        df = pd.read_csv(config.ENGINEERED_TRAIN_FILE)
    except FileNotFoundError:
        print(f"Error: '{config.ENGINEERED_TRAIN_FILE}' not found.")
        print("Please run `python data_pipeline.py` first.")
        return

    preprocessor, X, y = prepare_data(
        df, 
        config.NUMERICAL_FEATURES, 
        config.CATEGORICAL_FEATURES
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config.ML_TEST_SIZE, 
        random_state=config.ML_MODEL_RANDOM_STATE,
        stratify=y
    )
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    ml_models = get_ml_models()
    for name, model in ml_models.items():
        print(f"\n--- Training {name} ---")
        
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy for {name}: {val_accuracy:.4f}")
        
        save_path = os.path.join(config.MODELS_DIR, f"{name}.joblib")
        joblib.dump(model_pipeline, save_path)
        print(f"Model saved to {save_path}")

    print("\n--- ML Model Training Complete ---")

if __name__ == "__main__":
    main()
