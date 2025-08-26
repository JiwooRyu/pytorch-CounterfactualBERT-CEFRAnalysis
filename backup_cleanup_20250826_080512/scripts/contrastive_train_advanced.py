import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TripletAllDataset(Dataset):
    """모든 유효 트립렛을 사용. 필수 키(anchor, positive, negative)만 확인.
    original_prediction/positive_prediction/negative_prediction을 pseudo-label로 활용 가능.
    """

    def __init__(self, path: str, tokenizer: BertTokenizer, max_length: int = 256):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        required = {"anchor", "positive", "negative"}
        self.samples: List[Dict] = [d for d in data if required.issubset(d.keys())]
        self.tokenizer = tokenizer
        self.max_length = max_length
        # label 매핑 (있으면 사용)
        self.label_map = {"B1": 0, "B2": 1}
        print(f"Loaded triplets: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        anchor, positive, negative = s["anchor"], s["positive"], s["negative"]
        # pseudo labels (없으면 -1)
        o = s.get("original_prediction")
        p = s.get("positive_prediction")
        n = s.get("negative_prediction")
        o_id = self.label_map.get(o, -1)
        p_id = self.label_map.get(p, -1)
        n_id = self.label_map.get(n, -1)

        a_enc = self.tokenizer(anchor, truncation=True, padding='max_length',
                               max_length=self.max_length, return_tensors='pt')
        p_enc = self.tokenizer(positive, truncation=True, padding='max_length',
                               max_length=self.max_length, return_tensors='pt')
        n_enc = self.tokenizer(negative, truncation=True, padding='max_length',
                               max_length=self.max_length, return_tensors='pt')

        return {
            'anchor_input_ids': a_enc['input_ids'].squeeze(0),
            'anchor_attention_mask': a_enc['attention_mask'].squeeze(0),
            'pos_input_ids': p_enc['input_ids'].squeeze(0),
            'pos_attention_mask': p_enc['attention_mask'].squeeze(0),
            'neg_input_ids': n_enc['input_ids'].squeeze(0),
            'neg_attention_mask': n_enc['attention_mask'].squeeze(0),
            'o_label': torch.tensor(o_id, dtype=torch.long),
            'p_label': torch.tensor(p_id, dtype=torch.long),
            'n_label': torch.tensor(n_id, dtype=torch.long),
        }


class EncoderWithProjection(nn.Module):
    """BERT 인코더 + 투영(프로젝션) 헤드: contrastive representation 전용"""

    def __init__(self, model_name: str = 'bert-base-uncased', proj_dim: int = 256):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        z = self.proj(cls)
        return z


def cosine_triplet_loss(a, p, n, margin: float = 0.5):
    a = F.normalize(a, p=2, dim=1)
    p = F.normalize(p, p=2, dim=1)
    n = F.normalize(n, p=2, dim=1)
    pos_sim = F.cosine_similarity(a, p, dim=1)
    neg_sim = F.cosine_similarity(a, n, dim=1)
    return F.relu(margin - pos_sim + neg_sim).mean()


def info_nce_inbatch(a, p, temperature: float = 0.07):
    a = F.normalize(a, p=2, dim=1)
    p = F.normalize(p, p=2, dim=1)
    logits = torch.matmul(a, p.t()) / temperature  # [B, B]
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)


def hard_negative_from_batch(a, n_all):
    """배치 내에서 가장 유사한 네거티브를 선택 (하드 네거티브 마이닝)."""
    a_n = F.normalize(a, p=2, dim=1)
    n_n = F.normalize(n_all, p=2, dim=1)
    sim = torch.matmul(a_n, n_n.t())  # [B, B]
    diag = torch.eye(sim.size(0), device=sim.device).bool()
    sim = sim.masked_fill(diag, -1e9)
    idx = sim.argmax(dim=1)
    hard_negs = n_all[idx]
    return hard_negs


def train(
    triplet_path: str = 'data/Loop/loop_3/accumulated_loop2_3.json',
    model_name: str = 'bert-base-uncased',
    proj_dim: int = 256,
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 2e-5,
    margin: float = 0.5,
    alpha: float = 0.6,
    temperature: float = 0.07,
    max_length: int = 256,
    seed: int = 42,
    out_path: str = 'best_contrastive_encoder.pt',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    set_seed(seed)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = TripletAllDataset(triplet_path, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = EncoderWithProjection(model_name=model_name, proj_dim=proj_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=(device == 'cuda'))

    total_steps = len(loader) * epochs
    warmup = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        alpha_t = float(epoch + 1) / float(epochs) * alpha

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            a_ids = batch['anchor_input_ids'].to(device)
            a_mask = batch['anchor_attention_mask'].to(device)
            p_ids = batch['pos_input_ids'].to(device)
            p_mask = batch['pos_attention_mask'].to(device)
            n_ids = batch['neg_input_ids'].to(device)
            n_mask = batch['neg_attention_mask'].to(device)

            optimizer.zero_grad()
            # FP16/bf16는 임베딩 계산에만 적용하고, 손실은 FP32로 계산
            with autocast(enabled=(device == 'cuda')):
                a_z = model(a_ids, a_mask)
                p_z = model(p_ids, p_mask)
                n_z = model(n_ids, n_mask)

            a_z32 = a_z.float(); p_z32 = p_z.float(); n_z32 = n_z.float()
            # Triplet margin + In-batch InfoNCE + Hard negative mining
            loss_triplet = cosine_triplet_loss(a_z32, p_z32, n_z32, margin=margin)
            loss_infonce = info_nce_inbatch(a_z32, p_z32, temperature=temperature)

            hn = hard_negative_from_batch(a_z32.detach(), n_z32.detach())
            loss_triplet_hn = cosine_triplet_loss(a_z32, p_z32, hn, margin=margin)

            lcl = alpha_t * (0.5 * loss_triplet + 0.5 * loss_triplet_hn + 1.0 * loss_infonce)

            scaler.scale(lcl).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += lcl.item()

        print(f"Epoch {epoch+1} loss: {epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"Saved encoder to {out_path}")


class ClsDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: BertTokenizer, max_length: int = 256):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = int(self.df.iloc[idx]['label'])
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BertClassifier(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.classifier(cls)
        return logits


def load_bert_from_contrastive_encoder(encoder_ckpt: str, target_bert: BertModel):
    """EncoderWithProjection 체크포인트에서 BERT 가중치만 로드"""
    state = torch.load(encoder_ckpt, map_location='cpu')
    # EncoderWithProjection: keys start with 'bert.' and 'proj.'
    bert_state = {k.replace('bert.', ''): v for k, v in state.items() if k.startswith('bert.')}
    missing, unexpected = target_bert.load_state_dict(bert_state, strict=False)
    print(f"Loaded BERT from encoder. missing={missing}, unexpected={unexpected}")


def fine_tune_and_eval(
    encoder_ckpt: str,
    model_name: str,
    train_csv: str,
    test_csv: str,
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 2e-5,
    max_length: int = 256,
    seed: int = 42,
    out_path: str = 'best_joint_cls.pt',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(seed)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_ds = ClsDataset(train_csv, tokenizer, max_length=max_length)
    test_ds = ClsDataset(test_csv, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = BertClassifier(model_name=model_name).to(device)
    # contrastive encoder에서 BERT만 이니셜라이즈
    load_bert_from_contrastive_encoder(encoder_ckpt, model.bert)

    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=(device == 'cuda'))
    total_steps = len(train_loader) * epochs
    warmup = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"CLS Train {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast(enabled=(device == 'cuda')):
                logits = model(input_ids, attention_mask)
                loss = ce(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
        print(f"CLS Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")

        # Eval
        acc = evaluate_classifier(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path)
            print(f"Saved best classifier to {out_path} (acc={best_acc:.4f})")

    print(f"Final best acc: {best_acc:.4f}")


def evaluate_classifier(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()
            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels)
    acc = accuracy_score(trues, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(trues, preds))
    return acc


def main():
    # 1) Contrastive training
    train(
        triplet_path='data/Loop/loop_3/accumulated_loop2_3.json',
        model_name='bert-base-uncased',
        proj_dim=256,
        batch_size=16,
        epochs=10,
        lr=2e-5,
        margin=0.5,
        alpha=0.6,
        temperature=0.07,
        max_length=256,
        seed=42,
        out_path='best_contrastive_encoder.pt',
    )

    # 2) Classification fine-tuning + evaluation
    fine_tune_and_eval(
        encoder_ckpt='best_contrastive_encoder.pt',
        model_name='bert-base-uncased',
        train_csv='data/processed/train.csv',
        test_csv='data/processed/test.csv',
        batch_size=16,
        epochs=5,
        lr=2e-5,
        max_length=256,
        seed=42,
        out_path='best_joint_cls.pt',
    )


if __name__ == '__main__':
    main()


