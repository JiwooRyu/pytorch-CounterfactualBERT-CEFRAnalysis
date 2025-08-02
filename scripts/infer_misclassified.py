import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import os
from tqdm import tqdm

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

class CEFRDataset(Dataset):
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 데이터 로드
    base_df = pd.read_csv('data/processed/base.csv')
    print(f"총 {len(base_df)}개의 문장을 처리합니다.")
    
    # 모델 및 토크나이저 로드
    model_dir = 'models/baseline_bert'
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # 데이터셋/로더
    test_dataset = CEFRDataset(
        texts=base_df['text'].values,
        labels=base_df['label'].values,
        tokenizer=tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 예측 및 confidence score 계산
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Infer'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            probs = softmax(logits)
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            
            all_probs.extend(confs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # 결과 데이터프레임 생성
    df_result = base_df.copy()
    df_result['pred'] = all_preds
    df_result['confidence'] = all_probs
    df_result['correct'] = (df_result['label'] == df_result['pred'])
    
    # 저장
    os.makedirs('outputs/errors', exist_ok=True)
    df_result.to_csv('outputs/errors/base_sentence_full.csv', index=False)
    print('저장 완료: outputs/errors/base_sentence_full.csv')
    print(f"총 {len(df_result)}개의 문장이 처리되었습니다.")
    print(f"정확도: {df_result['correct'].mean():.4f}")
    print(f"평균 confidence: {df_result['confidence'].mean():.4f}")

if __name__ == "__main__":
    main()
