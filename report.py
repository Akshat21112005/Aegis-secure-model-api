import os
import pandas as pd
import numpy as np
import joblib
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import config
from models import get_dl_models, PhishingDataset, FinetunedBERT

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4',
    'tertiary': '#45B7D1',
    'quaternary': '#FFA07A',
    'quinary': '#98D8C8',
    'bg': '#F7F7F7',
    'text': '#2C3E50'
}

MODEL_THRESHOLDS = {
    'attention_blstm': 0.8,
    'rcnn': 0.8,
    'logistic': 0.5,
    'svm': 0.5,
    'xgboost': 0.5,
    'bert': 0.5
}

def load_sample_data(sample_fraction=0.05):
    print(f"Loading {sample_fraction*100}% sample from data...")
    
    if os.path.exists(config.ENGINEERED_TEST_FILE):
        df = pd.read_csv(config.ENGINEERED_TEST_FILE)
        print(f"Loaded test data: {len(df)} samples")
    elif os.path.exists(config.ENGINEERED_TRAIN_FILE):
        df = pd.read_csv(config.ENGINEERED_TRAIN_FILE)
        print(f"Loaded train data: {len(df)} samples")
    else:
        data_files = [
            os.path.join(config.DATA_DIR, 'url_data_labeled.csv'),
            os.path.join(config.DATA_DIR, 'data_bal - 20000.csv')
        ]
        df = None
        for file in data_files:
            if os.path.exists(file):
                df = pd.read_csv(file)
                print(f"Loaded raw data: {len(df)} samples")
                break
        
        if df is None:
            raise FileNotFoundError("No data file found!")
    
    sample_size = max(int(len(df) * sample_fraction), config.REPORT_SAMPLE_SIZE)
    sample_size = min(sample_size, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    print(f"Sampled {len(df_sample)} URLs for report generation")
    return df_sample

def prepare_ml_data(df):
    X = df[config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES]
    y = df['label'].values
    
    X.loc[:, config.NUMERICAL_FEATURES] = X.loc[:, config.NUMERICAL_FEATURES].fillna(-1)
    X.loc[:, config.CATEGORICAL_FEATURES] = X.loc[:, config.CATEGORICAL_FEATURES].fillna('N/A')
    
    return X, y

def prepare_dl_data(df):
    X = df[config.NUMERICAL_FEATURES].fillna(-1).values
    y = df['label'].values
    
    scaler_path = os.path.join(config.MODELS_DIR, "dl_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def predict_ml_models(X, y):
    predictions = {}
    scores = {}
    
    ml_models = ['logistic', 'svm', 'xgboost']
    
    for model_name in ml_models:
        model_path = os.path.join(config.MODELS_DIR, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            print(f"WARNING: Model {model_name} not found, skipping...")
            continue
        
        print(f"Loading {model_name} model...")
        model = joblib.load(model_path)
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        predictions[model_name] = y_pred
        scores[model_name] = y_proba
        
        acc = accuracy_score(y, y_pred)
        print(f"  {model_name} accuracy: {acc:.4f}")
    
    return predictions, scores

def predict_dl_models(X, y):
    predictions = {}
    scores = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X.shape[1]
    
    dl_models_dict = get_dl_models(input_dim)
    
    for model_name, model in dl_models_dict.items():
        model_path = os.path.join(config.MODELS_DIR, f"{model_name}.pt")
        if not os.path.exists(model_path):
            print(f"WARNING: Model {model_name} not found, skipping...")
            continue
        
        print(f"Loading {model_name} model...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(X_tensor).cpu().numpy().flatten()
        
        threshold = MODEL_THRESHOLDS.get(model_name, 0.5)
        y_pred = (outputs > threshold).astype(int)
        
        predictions[model_name] = y_pred
        scores[model_name] = outputs
        
        acc = accuracy_score(y, y_pred)
        print(f"  {model_name} accuracy: {acc:.4f} (threshold: {threshold})")
        
        del model, X_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return predictions, scores

def predict_bert_model(df, y):
    bert_path = os.path.join(config.BASE_DIR, 'finetuned_bert')
    if not os.path.exists(bert_path):
        print(f"WARNING: BERT model not found at {bert_path}, skipping...")
        return None, None
    
    if 'url' not in df.columns:
        print("WARNING: 'url' column not found in data, skipping BERT...")
        return None, None
    
    try:
        print("Loading BERT model...")
        bert_model = FinetunedBERT(bert_path)
        
        urls = df['url'].tolist()
        
        batch_size = 32
        all_preds = []
        all_probas = []
        
        print(f"Processing {len(urls)} URLs in batches of {batch_size}...")
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i+batch_size]
            batch_preds = bert_model.predict(batch_urls)
            batch_probas = bert_model.predict_proba(batch_urls)[:, 1]
            all_preds.extend(batch_preds)
            all_probas.extend(batch_probas)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        y_pred = 1-np.array(all_preds)
        y_proba = 1-np.array(all_probas)
        
        acc = accuracy_score(y, y_pred)
        print(f"  BERT accuracy: {acc:.4f}")
        
        return y_pred, y_proba
        
    except torch.cuda.OutOfMemoryError:
        print("WARNING: CUDA out of memory for BERT model, skipping...")
        print("  Try reducing batch size or use CPU by setting CUDA_VISIBLE_DEVICES=''")
        return None, None
    except Exception as e:
        print(f"WARNING: Error loading BERT model: {e}")
        return None, None

def plot_confusion_matrices(y_true, all_predictions, save_dir):
    print("\nGenerating confusion matrices...")
    
    n_models = len(all_predictions)
    if n_models == 0:
        print("No predictions to plot!")
        return
    
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    
    for idx, (model_name, y_pred) in enumerate(all_predictions.items()):
        ax = axes[idx]
        
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': 14, 'weight': 'bold'})
        
        ax.set_title(f'{model_name.upper()} Confusion Matrix', 
                     fontsize=14, fontweight='bold', color=COLORS['text'])
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xticklabels(['Legitimate (0)', 'Phishing (1)'])
        ax.set_yticklabels(['Legitimate (0)', 'Phishing (1)'])
    
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved confusion matrices to {save_path}")
    plt.close()

def plot_accuracy_comparison(y_true, all_predictions, save_dir):
    print("\nGenerating accuracy comparison plot...")
    
    if len(all_predictions) == 0:
        print("No predictions to plot!")
        return
    
    accuracies = {}
    for model_name, y_pred in all_predictions.items():
        acc = accuracy_score(y_true, y_pred)
        accuracies[model_name] = acc
    
    models = list(accuracies.keys())
    accs = list(accuracies.values())
    
    colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], 
                   COLORS['quaternary'], COLORS['quinary']]
    bar_colors = [colors_list[i % len(colors_list)] for i in range(len(models))]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(models, accs, color=bar_colors, edgecolor='black', linewidth=2, alpha=0.8)
    
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.set_xlabel('Models', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.set_title('Model Accuracy Comparison', fontsize=18, fontweight='bold', 
                 color=COLORS['text'], pad=20)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'accuracy_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved accuracy comparison to {save_path}")
    plt.close()

def plot_score_vs_label(y_true, all_scores, save_dir):
    print("\nGenerating score vs label scatter plots...")
    
    if len(all_scores) == 0:
        print("No scores to plot!")
        return
    
    n_models = len(all_scores)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    colors_map = {0: COLORS['secondary'], 1: COLORS['primary']}
    
    for idx, (model_name, scores) in enumerate(all_scores.items()):
        ax = axes[idx]
        
        for label in [0, 1]:
            mask = y_true == label
            label_name = 'Legitimate' if label == 0 else 'Phishing'
            ax.scatter(np.where(mask)[0], scores[mask], 
                      c=colors_map[label], label=label_name, 
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        threshold = MODEL_THRESHOLDS.get(model_name, 0.5)
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({threshold})', alpha=0.7)
        
        ax.set_title(f'{model_name.upper()} Prediction Scores', 
                     fontsize=14, fontweight='bold', color=COLORS['text'])
        ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Prediction Score', fontsize=11, fontweight='bold')
        ax.set_ylim([-0.1, 1.1])
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'score_vs_label.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved score vs label plots to {save_path}")
    plt.close()

def main():
    print("="*60)
    print("PHISHING DETECTION MODEL EVALUATION REPORT")
    print("="*60)
    print("\nCustom Thresholds Configuration:")
    for model, threshold in MODEL_THRESHOLDS.items():
        print(f"   • {model}: {threshold}")
    print()
    
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    df = load_sample_data(sample_fraction=0.05)
    
    all_predictions = {}
    all_scores = {}
    
    X_ml, y = prepare_ml_data(df)
    ml_preds, ml_scores = predict_ml_models(X_ml, y)
    all_predictions.update(ml_preds)
    all_scores.update(ml_scores)
    
    X_dl, y_dl = prepare_dl_data(df)
    dl_preds, dl_scores = predict_dl_models(X_dl, y_dl)
    all_predictions.update(dl_preds)
    all_scores.update(dl_scores)
    
    bert_pred, bert_score = predict_bert_model(df, y)
    if bert_pred is not None:
        all_predictions['bert'] = bert_pred
        all_scores['bert'] = bert_score
    
    if len(all_predictions) == 0:
        print("\nWARNING: No models found! Please train models first.")
        print("Run: python train_ml.py && python train_dl.py")
        return
    
    plot_confusion_matrices(y, all_predictions, config.REPORTS_DIR)
    plot_accuracy_comparison(y, all_predictions, config.REPORTS_DIR)
    plot_score_vs_label(y, all_scores, config.REPORTS_DIR)
    
    print("\n" + "="*60)
    print("REPORT GENERATION COMPLETE!")
    print(f"All visualizations saved to: {config.REPORTS_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()