import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

class CEFRDataset(Dataset):
    """CEFR 데이터셋 클래스"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data():
    """데이터 로드"""
    print("데이터 로드 중...")
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    print(f"Train: {train_df.shape[0]}개")
    print(f"Test: {test_df.shape[0]}개")
    return train_df, test_df

def create_data_loaders(train_df, test_df, tokenizer, batch_size=16):
    """데이터 로더 생성"""
    print("데이터 로더 생성 중...")
    # 데이터셋 생성
    train_dataset = CEFRDataset(
        texts=train_df['text'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer
    )
    test_dataset = CEFRDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer
    )
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, device, tokenizer, epochs=3, learning_rate=2e-5):
    """모델 훈련"""
    print("모델 훈련 시작...")
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    train_losses = []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        # 훈련
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc="Training")
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_progress.set_postfix({'loss': loss.item()})
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Train Loss: {avg_train_loss:.4f}")
        # 최고 성능 모델 저장 (마지막 epoch 기준)
        if epoch == epochs - 1:
            os.makedirs('models/baseline_bert', exist_ok=True)
            model.save_pretrained('models/baseline_bert')
            tokenizer.save_pretrained('models/baseline_bert')
            print("모델 및 토크나이저 저장됨!")
    return train_losses

def evaluate_model(model, test_loader, device):
    """모델 평가"""
    print("모델 평가 중...")
    
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # 성능 지표 계산
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['Level 3', 'Level 4'])
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    print(f"\n=== 모델 성능 ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(report)
    
    # Confusion Matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Level 3', 'Level 4'], 
                yticklabels=['Level 3', 'Level 4'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/baseline_bert/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, report, conf_matrix

def main():
    """메인 함수"""
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    # 데이터 로드
    train_df, test_df = load_data()
    # 토크나이저 및 모델 로드
    print("BERT 모델 로드 중...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    model.to(device)
    # 데이터 로더 생성
    train_loader, test_loader = create_data_loaders(
        train_df, test_df, tokenizer, batch_size=16
    )
    # 모델 훈련
    train_losses = train_model(
        model, train_loader, device, tokenizer, epochs=10
    )
    # 모델 평가
    accuracy, report, conf_matrix = evaluate_model(model, test_loader, device)
    print("\n=== 훈련 완료 ===")
    print("모델이 models/baseline_bert/에 저장되었습니다.")
    print("Confusion Matrix가 models/baseline_bert/confusion_matrix.png에 저장되었습니다.")

if __name__ == "__main__":
    main()
