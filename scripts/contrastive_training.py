import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# 1. Triplet Dataset
class TripletDataset(Dataset):
    def __init__(self, triplet_path, tokenizer, max_length=128):
        self.samples = []
        with open(triplet_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.samples = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
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

# 2. Contrastive Loss (margin-based)
def triplet_loss(anchor, positive, negative, margin=0.5):
    """Triplet loss with margin"""
    pos_sim = F.cosine_similarity(anchor, positive, dim=1)
    neg_sim = F.cosine_similarity(anchor, negative, dim=1)
    
    # anchor-positive는 높게, anchor-negative는 낮게
    loss = F.relu(margin - pos_sim + neg_sim).mean()
    return loss

# 3. Model: BERT encoder + classification head
class ContrastiveBertModel(nn.Module):
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

# 4. Training loop
def train_contrastive(model, train_loader, val_loader, device, epochs=5, lr=2e-5, margin=0.5):
    optimizer = AdamW(model.parameters(), lr=lr)
    model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            anchor_ids = batch['anchor_input_ids'].to(device)
            anchor_mask = batch['anchor_attention_mask'].to(device)
            pos_ids = batch['pos_input_ids'].to(device)
            pos_mask = batch['pos_attention_mask'].to(device)
            neg_ids = batch['neg_input_ids'].to(device)
            neg_mask = batch['neg_attention_mask'].to(device)
            
            # Get embeddings
            anchor_emb, _ = model(anchor_ids, anchor_mask)
            pos_emb, _ = model(pos_ids, pos_mask)
            neg_emb, _ = model(neg_ids, neg_mask)
            
            # Calculate triplet loss
            loss = triplet_loss(anchor_emb, pos_emb, neg_emb, margin=margin)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                anchor_ids = batch['anchor_input_ids'].to(device)
                anchor_mask = batch['anchor_attention_mask'].to(device)
                pos_ids = batch['pos_input_ids'].to(device)
                pos_mask = batch['pos_attention_mask'].to(device)
                neg_ids = batch['neg_input_ids'].to(device)
                neg_mask = batch['neg_attention_mask'].to(device)
                
                anchor_emb, _ = model(anchor_ids, anchor_mask)
                pos_emb, _ = model(pos_ids, pos_mask)
                neg_emb, _ = model(neg_ids, neg_mask)
                
                loss = triplet_loss(anchor_emb, pos_emb, neg_emb, margin=margin)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'contrastive_bert_model.pth')
            print(f'New best model saved with val loss: {best_val_loss:.4f}')
    
    print('Training completed!')

def evaluate_embeddings(model, data_loader, device):
    """Evaluate embedding quality"""
    model.eval()
    similarities = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating embeddings'):
            anchor_ids = batch['anchor_input_ids'].to(device)
            anchor_mask = batch['anchor_attention_mask'].to(device)
            pos_ids = batch['pos_input_ids'].to(device)
            pos_mask = batch['pos_attention_mask'].to(device)
            neg_ids = batch['neg_input_ids'].to(device)
            neg_mask = batch['neg_attention_mask'].to(device)
            
            anchor_emb, _ = model(anchor_ids, anchor_mask)
            pos_emb, _ = model(pos_ids, pos_mask)
            neg_emb, _ = model(neg_ids, neg_mask)
            
            pos_sim = F.cosine_similarity(anchor_emb, pos_emb, dim=1)
            neg_sim = F.cosine_similarity(anchor_emb, neg_emb, dim=1)
            
            similarities.extend([
                {'type': 'positive', 'similarity': sim.item()} 
                for sim in pos_sim
            ])
            similarities.extend([
                {'type': 'negative', 'similarity': sim.item()} 
                for sim in neg_sim
            ])
    
    # Calculate statistics
    pos_sims = [s['similarity'] for s in similarities if s['type'] == 'positive']
    neg_sims = [s['similarity'] for s in similarities if s['type'] == 'negative']
    
    print(f'Positive similarity - Mean: {np.mean(pos_sims):.4f}, Std: {np.std(pos_sims):.4f}')
    print(f'Negative similarity - Mean: {np.mean(neg_sims):.4f}, Std: {np.std(neg_sims):.4f}')
    print(f'Positive > Negative: {sum(1 for p, n in zip(pos_sims, neg_sims) if p > n)}/{len(pos_sims)}')

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load triplet data
    triplet_path = 'data/Loop/triplets_expert_all.json'
    print(f'Loading triplet data from: {triplet_path}')
    
    # Create datasets
    dataset = TripletDataset(triplet_path, tokenizer)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f'Dataset size: {len(dataset)}')
    print(f'Train size: {len(train_dataset)}')
    print(f'Val size: {len(val_dataset)}')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create model
    model = ContrastiveBertModel()
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Train
    print('Starting training...')
    train_contrastive(model, train_loader, val_loader, device, epochs=5, lr=2e-5, margin=0.5)
    
    # Evaluate
    print('Evaluating embeddings...')
    model.load_state_dict(torch.load('contrastive_bert_model.pth'))
    evaluate_embeddings(model, val_loader, device)
    
    print('Training and evaluation completed!')

if __name__ == '__main__':
    main() 