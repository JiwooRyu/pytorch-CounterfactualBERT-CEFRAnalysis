import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from typing import List, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class ContrastivePairDataset(Dataset):
    """Contrastive learning을 위한 문장 쌍 데이터셋"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # JSONL 파일에서 데이터 로드
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Anchor 문장 토크나이징
        anchor_encoding = self.tokenizer(
            item['anchor'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Negative 문장 토크나이징
        negative_encoding = self.tokenizer(
            item['counterfactual'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(0),
            'negative_input_ids': negative_encoding['input_ids'].squeeze(0),
            'negative_attention_mask': negative_encoding['attention_mask'].squeeze(0),
            'anchor_label': torch.tensor(item['anchor_label'], dtype=torch.long),
            'counterfactual_label': torch.tensor(item['counterfactual_label'], dtype=torch.long),
            'control_code': item['control_code'],
            'original_id': item['original_id']
        }

class ContrastiveBERT(nn.Module):
    """Contrastive learning을 위한 BERT 모델"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        super(ContrastiveBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooler = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Pooler output 사용 (BERT의 기본 풀링 방식)
        pooled_output = self.pooler(outputs.pooler_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output

def cosine_similarity_loss(anchor_embeddings, negative_embeddings):
    """Cosine similarity를 기반으로 한 contrastive loss"""
    # Cosine similarity 계산
    cos_sim = nn.functional.cosine_similarity(anchor_embeddings, negative_embeddings, dim=1)
    # Similarity를 최소화 (임베딩이 멀어지도록)
    loss = torch.mean(cos_sim)
    return loss

def train_contrastive_model(model, dataloader, optimizer, device, num_epochs=3):
    """Contrastive learning 훈련"""
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # 데이터를 디바이스로 이동
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)
            
            # Forward pass
            anchor_embeddings = model(anchor_input_ids, anchor_attention_mask)
            negative_embeddings = model(negative_input_ids, negative_attention_mask)
            
            # Loss 계산
            loss = cosine_similarity_loss(anchor_embeddings, negative_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    return model

def get_embeddings(model, dataset, device):
    """모델을 사용하여 임베딩 추출"""
    model.eval()
    anchor_embeddings = []
    negative_embeddings = []
    labels = []
    control_codes = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            
            anchor_input_ids = item['anchor_input_ids'].unsqueeze(0).to(device)
            anchor_attention_mask = item['anchor_attention_mask'].unsqueeze(0).to(device)
            negative_input_ids = item['negative_input_ids'].unsqueeze(0).to(device)
            negative_attention_mask = item['negative_attention_mask'].unsqueeze(0).to(device)
            
            # 임베딩 추출
            anchor_emb = model(anchor_input_ids, anchor_attention_mask)
            negative_emb = model(negative_input_ids, negative_attention_mask)
            
            anchor_embeddings.append(anchor_emb.cpu().numpy())
            negative_embeddings.append(negative_emb.cpu().numpy())
            labels.append(item['anchor_label'].item())
            control_codes.append(item['control_code'])
    
    return (np.vstack(anchor_embeddings), 
            np.vstack(negative_embeddings), 
            np.array(labels), 
            control_codes)

def plot_tsne(embeddings_before, embeddings_after, labels, control_codes, save_path='contrastive_tsne.png'):
    """t-SNE를 사용한 임베딩 시각화"""
    
    # t-SNE로 2차원 축소
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    # 학습 전 임베딩
    embeddings_2d_before = tsne.fit_transform(embeddings_before)
    
    # 학습 후 임베딩
    embeddings_2d_after = tsne.fit_transform(embeddings_after)
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 학습 전
    scatter1 = ax1.scatter(embeddings_2d_before[:, 0], embeddings_2d_before[:, 1], 
                           c=labels, cmap='viridis', alpha=0.7)
    ax1.set_title('Before Contrastive Learning', fontsize=14)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.grid(True, alpha=0.3)
    
    # 학습 후
    scatter2 = ax2.scatter(embeddings_2d_after[:, 0], embeddings_2d_after[:, 1], 
                           c=labels, cmap='viridis', alpha=0.7)
    ax2.set_title('After Contrastive Learning', fontsize=14)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.grid(True, alpha=0.3)
    
    # 범례 추가
    legend1 = ax1.legend(*scatter1.legend_elements(), title="Labels")
    legend2 = ax2.legend(*scatter2.legend_elements(), title="Labels")
    ax1.add_artist(legend1)
    ax2.add_artist(legend2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Control code별 분포도 시각화
    unique_codes = list(set(control_codes))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_codes)))
    
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 학습 전 control code별 분포
    for i, code in enumerate(unique_codes):
        mask = [c == code for c in control_codes]
        if sum(mask) > 0:
            ax3.scatter(embeddings_2d_before[mask, 0], embeddings_2d_before[mask, 1], 
                       c=[colors[i]], label=code, alpha=0.7)
    
    ax3.set_title('Before Learning - Control Codes', fontsize=14)
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 학습 후 control code별 분포
    for i, code in enumerate(unique_codes):
        mask = [c == code for c in control_codes]
        if sum(mask) > 0:
            ax4.scatter(embeddings_2d_after[mask, 0], embeddings_2d_after[mask, 1], 
                       c=[colors[i]], label=code, alpha=0.7)
    
    ax4.set_title('After Learning - Control Codes', fontsize=14)
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('contrastive_tsne_control_codes.png', dpi=300, bbox_inches='tight')
    plt.show()

def contrastive_collate_fn(batch):
    # 텐서 필드는 torch.stack, 문자열/리스트 필드는 리스트로 묶음
    elem = batch[0]
    collated = {}
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch])
        else:
            collated[key] = [d[key] for d in batch]
    return collated

class ClassificationDataset(Dataset):
    """CEFR 분류용 데이터셋 (data/processed/*.jsonl)"""
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # 'text', 'label' 필드 필요
                if 'text' in item and 'label' in item:
                    self.data.append(item)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

def classification_collate_fn(batch):
    elem = batch[0]
    collated = {}
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch])
        else:
            collated[key] = [d[key] for d in batch]
    return collated

class BertForClassification(nn.Module):
    def __init__(self, bert_encoder, num_labels=3):
        super().__init__()
        self.bert = bert_encoder
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits

def train_classifier(model, dataloader, optimizer, device, num_epochs=3):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'[Classifier] Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}')

def evaluate_classifier(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    from sklearn.metrics import classification_report
    print(f'[Classifier] Accuracy: {acc:.4f}')
    print(classification_report(all_labels, all_preds, digits=4))
    return acc

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='contrastive', choices=['contrastive', 'classification', 'both'])
    parser.add_argument('--processed_dir', type=str, default='data/processed')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 경로
    data_path = 'poc/contrastive_data.jsonl'
    model_save_path = 'models/contrastive_bert'
    
    # 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # contrastive 실험
    if args.mode in ['contrastive', 'both']:
        dataset = ContrastivePairDataset(data_path, tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=contrastive_collate_fn)
        print(f"Dataset size: {len(dataset)}")
        model = ContrastiveBERT('bert-base-uncased')
        model.to(device)
        print("Extracting embeddings before training...")
        embeddings_before, _, labels, control_codes = get_embeddings(model, dataset, device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        print("Starting contrastive learning...")
        model = train_contrastive_model(model, dataloader, optimizer, device, num_epochs=3)
        print("Extracting embeddings after training...")
        embeddings_after, _, _, _ = get_embeddings(model, dataset, device)
        os.makedirs(model_save_path, exist_ok=True)
        model.bert.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to {model_save_path}")
        print("Creating t-SNE visualizations...")
        plot_tsne(embeddings_before, embeddings_after, labels, control_codes)
        from sklearn.metrics.pairwise import cosine_similarity
        cos_sim_before = np.mean([cosine_similarity([embeddings_before[i]], [embeddings_before[i+len(embeddings_before)//2]])[0,0] 
                                 for i in range(len(embeddings_before)//2)])
        cos_sim_after = np.mean([cosine_similarity([embeddings_after[i]], [embeddings_after[i+len(embeddings_after)//2]])[0,0] 
                                for i in range(len(embeddings_after)//2)])
        print(f"Average cosine similarity before training: {cos_sim_before:.4f}")
        print(f"Average cosine similarity after training: {cos_sim_after:.4f}")
        print(f"Similarity reduction: {cos_sim_before - cos_sim_after:.4f}")

    # 분류기 실험
    if args.mode in ['classification', 'both']:
        print("\n[Classification] data/processed 데이터셋으로 분류기 학습/평가...")
        train_path = os.path.join(args.processed_dir, 'train.jsonl')
        valid_path = os.path.join(args.processed_dir, 'valid.jsonl')
        test_path = os.path.join(args.processed_dir, 'test.jsonl')
        train_set = ClassificationDataset(train_path, tokenizer)
        valid_set = ClassificationDataset(valid_path, tokenizer)
        test_set = ClassificationDataset(test_path, tokenizer)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=classification_collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, collate_fn=classification_collate_fn)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=classification_collate_fn)
        # (1) 사전학습 BERT로 분류기 학습/평가
        print("\n[Classifier] Pretrained BERT encoder로 분류기 학습/평가")
        bert_encoder = BertModel.from_pretrained('bert-base-uncased')
        classifier = BertForClassification(bert_encoder, num_labels=3).to(device)
        optimizer = optim.AdamW(classifier.parameters(), lr=2e-5)
        train_classifier(classifier, train_loader, optimizer, device, num_epochs=3)
        print("[Validation]")
        evaluate_classifier(classifier, valid_loader, device)
        print("[Test]")
        evaluate_classifier(classifier, test_loader, device)
        # (2) contrastive로 파인튜닝된 BERT encoder로 분류기 학습/평가
        if os.path.exists(model_save_path):
            print("\n[Classifier] Contrastive로 파인튜닝된 BERT encoder로 분류기 학습/평가")
            bert_encoder = BertModel.from_pretrained(model_save_path)
            classifier = BertForClassification(bert_encoder, num_labels=3).to(device)
            optimizer = optim.AdamW(classifier.parameters(), lr=2e-5)
            train_classifier(classifier, train_loader, optimizer, device, num_epochs=3)
            print("[Validation]")
            evaluate_classifier(classifier, valid_loader, device)
            print("[Test]")
            evaluate_classifier(classifier, test_loader, device)
        else:
            print("[경고] contrastive로 파인튜닝된 BERT encoder가 없습니다. 먼저 --mode contrastive로 학습하세요.")

if __name__ == "__main__":
    main()
