import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
from tqdm import tqdm

# 경로 설정
MODEL_DIR = 'models/baseline_distilbert'
DATA_PATH = 'data/counterfactuals/counterfactuals_example_10.json'
OUTPUT_PATH = 'outputs/analysis/counterfactual_predictions.json'

# 예측 함수
def predict_label(text, tokenizer, model, device):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    return pred

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 모델 및 토크나이저 로드
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    # 데이터 로드
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    for item in tqdm(data, desc='Predicting'):
        original_sentence = item['original_sentence']
        original_id = item['original_id']
        # original 예측
        orig_pred = predict_label(original_sentence, tokenizer, model, device)
        # counterfactual 예측
        cf_results = []
        for cf_type in ['lexical','shuffle','restructure','negation','quantifier','resemantic','insert','delete']:
            cf_sentence = item[cf_type]
            cf_pred = predict_label(cf_sentence, tokenizer, model, device)
            is_label_flipped = (cf_pred != orig_pred)
            cf_results.append({
                'type': cf_type,
                'sentence': cf_sentence,
                'prediction': cf_pred,
                'is_label_flipped': is_label_flipped
            })
        results.append({
            'original_id': original_id,
            'original_sentence': original_sentence,
            'original_prediction': orig_pred,
            'counterfactuals': cf_results
        })
    # 저장
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
    print(f'결과가 {OUTPUT_PATH}에 저장되었습니다.')

if __name__ == '__main__':
    main()
