import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 1. Triplet Dataset (positive는 prediction이 같고, negative는 prediction이 다를 때만 사용)
class TripletDataset(Dataset):
    def __init__(self, triplet_path, tokenizer, max_length=128):
        self.samples = []
        with open(triplet_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # positive: original_prediction == positive_prediction
            # negative: original_prediction != negative_prediction
            self.samples = [item for item in data if item['original_prediction'] == item['positive_prediction'] and item['original_prediction'] != item['negative_prediction']]
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Filtered triplet dataset: {len(self.samples)} samples (pos pred==orig, neg pred!=orig)")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        anchor = sample['anchor']
        positive = sample['positive']
        negative = sample['negative']
        
        anchor_enc = self.tokenizer(anchor, truncation=True, padding='max_length', 
                                   max_length=self.max_length, return_tensors='pt')
        pos_enc = self.tokenizer(positive, truncation=True, padding='max_length', 
                                max_length=self.max_length, return_tensors='pt')
        neg_enc = self.tokenizer(negative, truncation=True, padding='max_length', 
                                max_length=self.max_length, return_tensors='pt')
        
        return {
            'anchor_input_ids': anchor_enc['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_enc['attention_mask'].squeeze(0),
            'pos_input_ids': pos_enc['input_ids'].squeeze(0),
            'pos_attention_mask': pos_enc['attention_mask'].squeeze(0),
            'neg_input_ids': neg_enc['input_ids'].squeeze(0),
            'neg_attention_mask': neg_enc['attention_mask'].squeeze(0)
        }

# 2. Classification Dataset (for downstream task)
class ClassificationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
        self.samples = []
        df = pd.read_csv(data_path)
        for _, row in df.iterrows():
            text = row['text']
            label = row['label']
            self.samples.append({'text': text, 'label': label})
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        text = self.samples[idx]['text']
        label = self.samples[idx]['label']
        enc = self.tokenizer(text, truncation=True, padding='max_length', 
                            max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 3. Contrastive Loss (margin-based)
def contrastive_loss(anchor, positive, negative, margin=1.0):
    """Cosine similarity 기반 margin loss"""
    pos_sim = F.cosine_similarity(anchor, positive, dim=1)
    neg_sim = F.cosine_similarity(anchor, negative, dim=1)
    # anchor-positive는 1에 가깝게, anchor-negative는 0 또는 -1에 가깝게
    loss = F.relu(margin - pos_sim + neg_sim).mean()
    return loss

# 4. Joint Model: BERT encoder + classification head
class JointBertModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(cls_emb)
        return cls_emb, logits

# 5. Joint Training loop
def train_joint(model, triplet_loader, cls_loader, device, epochs=5, alpha=0.5, lr=2e-5, margin=1.0):
    optimizer = AdamW(model.parameters(), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        triplet_iter = iter(triplet_loader)
        cls_iter = iter(cls_loader)
        n_batches = min(len(triplet_loader), len(cls_loader))
        
        for _ in tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{epochs}'):
            # Triplet batch (for contrastive loss)
            try:
                triplet_batch = next(triplet_iter)
            except StopIteration:
                triplet_iter = iter(triplet_loader)
                triplet_batch = next(triplet_iter)
                
            anchor_ids = triplet_batch['anchor_input_ids'].to(device)
            anchor_mask = triplet_batch['anchor_attention_mask'].to(device)
            pos_ids = triplet_batch['pos_input_ids'].to(device)
            pos_mask = triplet_batch['pos_attention_mask'].to(device)
            neg_ids = triplet_batch['neg_input_ids'].to(device)
            neg_mask = triplet_batch['neg_attention_mask'].to(device)
            
            anchor_emb, _ = model(anchor_ids, anchor_mask)
            pos_emb, _ = model(pos_ids, pos_mask)
            neg_emb, _ = model(neg_ids, neg_mask)
            
            lcl = contrastive_loss(anchor_emb, pos_emb, neg_emb, margin=margin)
            
            # Classification batch (for CE loss)
            try:
                cls_batch = next(cls_iter)
            except StopIteration:
                cls_iter = iter(cls_loader)
                cls_batch = next(cls_iter)
                
            input_ids = cls_batch['input_ids'].to(device)
            attention_mask = cls_batch['attention_mask'].to(device)
            labels = cls_batch['label'].to(device)
            
            _, logits = model(input_ids, attention_mask)
            lce = ce_loss_fn(logits, labels)
            
            # Joint loss
            loss = alpha * lcl + (1 - alpha) * lce
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / n_batches
        print(f'Epoch {epoch+1} avg loss: {avg_loss:.4f}')
    
    # Save final model after training
    torch.save(model.state_dict(), 'best_joint_bert.pt')
    print('Training completed! Model saved.')

def evaluate(model, loader, device):
    """Evaluate classification performance"""
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()
            
            _, logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            
            preds.extend(pred)
            trues.extend(labels)
    
    acc = accuracy_score(trues, preds)
    print(f'Test accuracy: {acc:.4f}')
    print(classification_report(trues, preds))
    return acc

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load triplet data for contrastive learning
    triplet_path = 'data/Loop/loop_1/triplets_expert_all.json'
    print(f'Loading triplet data from: {triplet_path}')
    
    # Create triplet dataset
    triplet_dataset = TripletDataset(triplet_path, tokenizer)
    
    # Load classification data for downstream task
    train_cls_dataset = ClassificationDataset('data/processed/train.csv', tokenizer)
    test_cls_dataset = ClassificationDataset('data/processed/test.csv', tokenizer)
    
    # Create data loaders
    triplet_loader = DataLoader(triplet_dataset, batch_size=16, shuffle=True)
    train_cls_loader = DataLoader(train_cls_dataset, batch_size=16, shuffle=True)
    test_cls_loader = DataLoader(test_cls_dataset, batch_size=32, shuffle=False)
    
    print(f'Triplet dataset size: {len(triplet_dataset)}')
    print(f'Classification dataset - Train: {len(train_cls_dataset)}, Test: {len(test_cls_dataset)}')
    
    # Create model
    model = JointBertModel()
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Train
    print('Starting joint training...')
    train_joint(model, triplet_loader, train_cls_loader, device, epochs=10, alpha=0.5, lr=2e-5, margin=1.0)
    
    # Test
    print('Loading best model and evaluating on test set...')
    model.load_state_dict(torch.load('best_joint_bert.pt'))
    evaluate(model, test_cls_loader, device)
    
    print('Joint training and evaluation completed!')

if __name__ == '__main__':
    main() 