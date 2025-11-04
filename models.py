from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import config
import os

def get_ml_models():
    models = {
        'logistic': LogisticRegression(
            max_iter=1000, 
            random_state=config.ML_MODEL_RANDOM_STATE
        ),
        'svm': SVC(
            probability=True,
            random_state=config.ML_MODEL_RANDOM_STATE
        ),
        'xgboost': XGBClassifier(
            n_estimators=100,
            random_state=config.ML_MODEL_RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }
    return models

class Attention_BLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super(Attention_BLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        out = self.fc(context_vector)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.output(out)
        out = self.sigmoid(out)
        
        return out

class RCNN(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_filters=100, filter_sizes=[3, 4, 5], dropout=0.5):
        super(RCNN, self).__init__()
        
        self.lstm = nn.LSTM(1, embed_dim // 2, batch_first=True, bidirectional=True)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * num_filters, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        x = x.unsqueeze(2)
        
        lstm_out, _ = self.lstm(x)
        
        lstm_out = lstm_out.permute(0, 2, 1)
        
        conv_outs = [torch.relu(conv(lstm_out)) for conv in self.convs]
        
        pooled = [torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in conv_outs]
        
        cat = torch.cat(pooled, dim=1)
        
        out = self.fc(cat)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.output(out)
        out = self.sigmoid(out)
        
        return out

class FinetunedBERT:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(config.BASE_DIR, 'finetuned_bert')
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            local_files_only=True
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, urls):
        if isinstance(urls, str):
            urls = [urls]
        
        encodings = self.tokenizer(
            urls,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        encodings = {key: val.to(self.device) for key, val in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, urls):
        if isinstance(urls, str):
            urls = [urls]
        
        encodings = self.tokenizer(
            urls,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        encodings = {key: val.to(self.device) for key, val in encodings.items()}
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            probas = torch.softmax(outputs.logits, dim=1)
        
        return probas.cpu().numpy()

class PhishingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx].view(-1)

def get_dl_models(input_dim):
    models = {
        'attention_blstm': Attention_BLSTM(input_dim),
        'rcnn': RCNN(input_dim)
    }
    return models
