import json
import random
from typing import Dict, List, Optional

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


class TripletDataset(Dataset):
    def __init__(self, path: str, tokenizer: BertTokenizer, max_length: int = 256):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        required = {"anchor", "positive", "negative"}
        self.samples: List[Dict] = [d for d in data if required.issubset(d.keys())]
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loaded triplets: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        a, p, n = s["anchor"], s["positive"], s["negative"]
        a_enc = self.tokenizer(a, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        p_enc = self.tokenizer(p, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        n_enc = self.tokenizer(n, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'anchor_input_ids': a_enc['input_ids'].squeeze(0),
            'anchor_attention_mask': a_enc['attention_mask'].squeeze(0),
            'pos_input_ids': p_enc['input_ids'].squeeze(0),
            'pos_attention_mask': p_enc['attention_mask'].squeeze(0),
            'neg_input_ids': n_enc['input_ids'].squeeze(0),
            'neg_attention_mask': n_enc['attention_mask'].squeeze(0),
        }


class EncoderWithProjection(nn.Module):
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


class MemoryBank:
    def __init__(self, dim: int, capacity_batches: int = 8, batch_size: int = 16, device: str = 'cuda'):
        self.capacity = capacity_batches * batch_size
        self.device = device
        self.buffer: Optional[torch.Tensor] = None  # [K, D]
        self.ptr = 0
        self.dim = dim

    def push(self, feats: torch.Tensor):
        # 메모리뱅크는 항상 FP32로 저장해 dtype 불일치 방지
        feats = feats.detach().to(self.device).float()
        if self.buffer is None:
            self.buffer = torch.zeros((self.capacity, self.dim), device=self.device, dtype=torch.float32)
        bs = feats.size(0)
        if bs >= self.capacity:
            self.buffer = feats[-self.capacity:]
            self.ptr = 0
            return
        end = self.ptr + bs
        if end <= self.capacity:
            self.buffer[self.ptr:end] = feats
        else:
            first = self.capacity - self.ptr
            self.buffer[self.ptr:] = feats[:first]
            self.buffer[:bs-first] = feats[first:]
        self.ptr = (self.ptr + bs) % self.capacity

    def get_hard_negs(self, anchors: torch.Tensor) -> Optional[torch.Tensor]:
        if self.buffer is None or (self.buffer == 0).all():
            return None
        # 연산은 FP32로 수행
        a_n = F.normalize(anchors.float(), p=2, dim=1)
        b_n = F.normalize(self.buffer.float(), p=2, dim=1)
        sim = a_n @ b_n.t()  # [B, K]
        idx = sim.argmax(dim=1)
        return self.buffer[idx]


def cosine_triplet_loss(a, p, n, margin: float = 0.5, weight: Optional[torch.Tensor] = None):
    a = F.normalize(a, p=2, dim=1)
    p = F.normalize(p, p=2, dim=1)
    n = F.normalize(n, p=2, dim=1)
    pos_sim = F.cosine_similarity(a, p, dim=1)
    neg_sim = F.cosine_similarity(a, n, dim=1)
    loss = F.relu(margin - pos_sim + neg_sim)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def info_nce(a, p, temperature: float = 0.07, weight: Optional[torch.Tensor] = None):
    a = F.normalize(a, p=2, dim=1)
    p = F.normalize(p, p=2, dim=1)
    logits = (a @ p.t()) / temperature  # [B, B]
    labels = torch.arange(a.size(0), device=a.device)
    loss = F.cross_entropy(logits, labels, reduction='none')
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def quality_weight(a, p, n, k: float = 10.0, tau_pos: float = 0.35, tau_neg: float = 0.65, tau_gap: float = 0.05):
    # FP32에서 계산
    a = F.normalize(a.float(), p=2, dim=1)
    p = F.normalize(p.float(), p=2, dim=1)
    n = F.normalize(n.float(), p=2, dim=1)
    s_pos = F.cosine_similarity(a, p, dim=1)
    s_neg = F.cosine_similarity(a, n, dim=1)
    gap = s_pos - s_neg
    # 필터링 마스크 (low quality 샘플 배제)
    mask = (s_pos >= tau_pos) & (s_neg <= tau_neg) & (gap >= tau_gap)
    # 가중치: margin 큰 샘플에 더 큰 가중치
    w = torch.sigmoid(k * gap)
    return w.detach(), mask.detach()


def train_efficiency(
    triplet_path: str = 'data/Loop/loop_3/accumulated_loop2_3.json',
    model_name: str = 'bert-base-uncased',
    proj_dim: int = 256,
    batch_size: int = 16,
    epochs: int = 8,
    lr: float = 2e-5,
    margin: float = 0.5,
    temperature: float = 0.07,
    alpha: float = 0.6,
    max_length: int = 256,
    seed: int = 42,
    out_path: str = 'best_contrastive_encoder_eff.pt',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(seed)
    print(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    dataset = TripletDataset(triplet_path, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = EncoderWithProjection(model_name=model_name, proj_dim=proj_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=(device == 'cuda'))
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    # 메모리 뱅크 (배치 8개 분량)
    mem_bank = MemoryBank(dim=proj_dim, capacity_batches=8, batch_size=batch_size, device=device)

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
            # 임베딩만 autocast, 손실은 FP32
            with autocast(enabled=(device == 'cuda')):
                a_z = model(a_ids, a_mask)
                p_z = model(p_ids, p_mask)
                n_z = model(n_ids, n_mask)

            # 품질 가중치/필터
            w, m = quality_weight(a_z, p_z, n_z)
            if m.sum().item() == 0:
                # 모두 저품질인 경우 스킵하되 메모리뱅크는 업데이트
                mem_bank.push(n_z.detach())
                scheduler.step()
                continue

            a_z = a_z[m]; p_z = p_z[m]; n_z = n_z[m]; w = w[m]

            # 하드 네거티브: 메모리 뱅크에서 선택하여 추가 트립렛 손실
            hn = mem_bank.get_hard_negs(a_z)
            if hn is not None:
                hn = hn.to(a_z.dtype)

            # 손실 (FP32)
            a32, p32, n32 = a_z.float(), p_z.float(), n_z.float()
            loss_trip = cosine_triplet_loss(a32, p32, n32, margin=margin, weight=w)
            loss_nce = info_nce(a32, p32, temperature=temperature, weight=w)
            if hn is not None:
                loss_trip_hn = cosine_triplet_loss(a32, p32, hn.float(), margin=margin, weight=w)
                loss = alpha_t * (0.5 * loss_trip + 0.5 * loss_trip_hn + loss_nce)
            else:
                loss = alpha_t * (loss_trip + loss_nce)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 메모리 뱅크 갱신(neg 임베딩 저장)
            mem_bank.push(n_z.detach())

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {epoch_loss/len(loader):.4f}")

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
    state = torch.load(encoder_ckpt, map_location='cpu')
    bert_state = {k.replace('bert.', ''): v for k, v in state.items() if k.startswith('bert.')}
    missing, unexpected = target_bert.load_state_dict(bert_state, strict=False)
    print(f"Loaded BERT from encoder. missing={missing}, unexpected={unexpected}")


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
    out_path: str = 'best_joint_cls_eff.pt',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(seed)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_ds = ClsDataset(train_csv, tokenizer, max_length=max_length)
    test_ds = ClsDataset(test_csv, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = BertClassifier(model_name=model_name).to(device)
    load_bert_from_contrastive_encoder(encoder_ckpt, model.bert)

    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=(device == 'cuda'))
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
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
            loss = ce(logits.float(), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
        print(f"CLS Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")

        acc = evaluate_classifier(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path)
            print(f"Saved best classifier to {out_path} (acc={best_acc:.4f})")

    print(f"Final best acc: {best_acc:.4f}")

def main():
    train_efficiency(
        triplet_path='data/Loop/loop_3/accumulated_loop2_3.json',
        model_name='bert-base-uncased',
        proj_dim=256,
        batch_size=16,
        epochs=8,
        lr=2e-5,
        margin=0.5,
        temperature=0.07,
        alpha=0.6,
        max_length=256,
        seed=42,
        out_path='best_contrastive_encoder_eff.pt',
    )

    fine_tune_and_eval(
        encoder_ckpt='best_contrastive_encoder_eff.pt',
        model_name='bert-base-uncased',
        train_csv='data/processed/train.csv',
        test_csv='data/processed/test.csv',
        batch_size=16,
        epochs=5,
        lr=2e-5,
        max_length=256,
        seed=42,
        out_path='best_joint_cls_eff.pt',
    )


if __name__ == '__main__':
    main()


