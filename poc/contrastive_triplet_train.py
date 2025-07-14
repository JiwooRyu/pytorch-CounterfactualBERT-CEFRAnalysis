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

# 1. Triplet Dataset
class TripletDataset(Dataset):
    def __init__(self, triplet_path, tokenizer, max_length=128):
        self.samples = []
        with open(triplet_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj)
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        anchor = self.samples[idx]['anchor']
        positive = self.samples[idx]['positive']
        hard_negative = self.samples[idx]['hard_negative']
        anchor_enc = self.tokenizer(anchor, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        pos_enc = self.tokenizer(positive, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        neg_enc = self.tokenizer(hard_negative, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'anchor_input_ids': anchor_enc['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_enc['attention_mask'].squeeze(0),
            'pos_input_ids': pos_enc['input_ids'].squeeze(0),
            'pos_attention_mask': pos_enc['attention_mask'].squeeze(0),
            'neg_input_ids': neg_enc['input_ids'].squeeze(0),
            'neg_attention_mask': neg_enc['attention_mask'].squeeze(0)
        }

# 2. Downstream Classification Dataset
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
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Contrastive Loss (margin-based)
def contrastive_loss(anchor, positive, negative, margin=1.0):
    # Cosine similarity 기반 margin loss
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    # anchor-positive는 1에 가깝게, anchor-negative는 0 또는 -1에 가깝게
    loss = F.relu(margin - pos_sim + neg_sim).mean()
    return loss

# 3. Model: BERT encoder + classification head
class JointBertModel(nn.Module):
    def __init__(self, model_dir, num_labels=2):
        super().__init__()
        # encoder만 불러오기
        self.bert = BertModel.from_pretrained(model_dir)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(cls_emb)
        return cls_emb, logits

# 4. Training loop

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
        for _ in tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{epochs}'):  # joint batch
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
    # Save final model after last epoch
    torch.save(model.state_dict(), 'best_joint_bert.pt')
    print('Final model saved.')

def evaluate(model, loader, device):
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
    print(f'Validation accuracy: {acc:.4f}')
    print(classification_report(trues, preds))
    return acc

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dir = 'models/baseline_bert'
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    # Triplet dataset
    triplet_path = 'poc/counterfactual_triplets.jsonl'
    triplet_dataset = TripletDataset(triplet_path, tokenizer)
    triplet_loader = DataLoader(triplet_dataset, batch_size=16, shuffle=True)
    # Downstream classification dataset (CSV)
    train_dataset = ClassificationDataset('data/processed/train.csv', tokenizer)
    test_dataset = ClassificationDataset('data/processed/test.csv', tokenizer)
    cls_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    # Model
    model = JointBertModel(model_dir, num_labels=2)
    # Train
    train_joint(model, triplet_loader, cls_loader, device, epochs=5, alpha=0.5, lr=2e-5, margin=1.0)
    # Test
    print('Loading best model and evaluating on test set...')
    model.load_state_dict(torch.load('best_joint_bert.pt'))
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main() 