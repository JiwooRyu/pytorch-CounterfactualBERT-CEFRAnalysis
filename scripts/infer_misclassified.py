import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
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
    test_df = pd.read_csv('data/processed/test.csv')
    # 모델 및 토크나이저 로드
    model_dir = 'models/baseline_distilbert'
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    # 데이터셋/로더
    test_dataset = CEFRDataset(
        texts=test_df['text'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    # 예측 및 confidence score 계산
    all_probs = []
    all_preds = []
    all_labels = []
    all_texts = []
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
            all_texts.extend([t for t in batch['input_ids']])
    # 낮은 confidence 순으로 정렬
    df_result = test_df.copy()
    df_result['pred'] = all_preds
    df_result['confidence'] = all_probs
    # 정답/오답 여부
    df_result['correct'] = (df_result['label'] == df_result['pred'])
    # 낮은 confidence 순으로 300개
    df_lowconf = df_result.sort_values('confidence', ascending=True).head(300)
    # 저장
    os.makedirs('outputs/errors', exist_ok=True)
    df_lowconf.to_csv('outputs/errors/low_confidence_300.csv', index=False)
    print('저장 완료: outputs/errors/low_confidence_300.csv')

if __name__ == "__main__":
    main()
