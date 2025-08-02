import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
from tqdm import tqdm

# 경로 설정
MODEL_DIR = 'models/baseline_bert'
DATA_PATHS = {
    '10': 'data/counterfactuals/counterfactuals_example_10.json',
    '300': 'data/counterfactuals/counterfactual_example_300.json',
    '528': 'data/counterfactuals/original_528_base.json'
}
OUTPUT_DIR = 'outputs/analysis'

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

def process_dataset(data_path, output_path, tokenizer, model, device):
    """특정 데이터셋에 대해 예측을 수행합니다"""
    # 데이터 로드
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    for item in tqdm(data, desc=f'Predicting {os.path.basename(data_path)}'):
        # 데이터 형식에 따라 다른 처리
        if 'original_sentence' in item:
            # 기존 형식 (10, 300)
            original_sentence = item['original_sentence']
            original_id = item['original_id']
        else:
            # 새로운 형식 (528)
            original_sentence = item['original']
            original_id = item['sentence_id']
        
        # original 예측
        orig_pred = predict_label(original_sentence, tokenizer, model, device)
        
        # counterfactual 예측
        cf_results = []
        cf_types = ['lexical','shuffle','restructure','negation','quantifier','resemantic','insert','delete']
        
        for cf_type in cf_types:
            cf_sentence = item[cf_type]
            cf_pred = predict_label(cf_sentence, tokenizer, model, device)
            is_label_flipped = (cf_pred != orig_pred)
            cf_results.append({
                'type': cf_type,
                'sentence': cf_sentence,
                'prediction': cf_pred,
                'is_label_flipped': is_label_flipped
            })
        
        result_item = {
            'original_id': original_id,
            'original_sentence': original_sentence,
            'original_prediction': orig_pred,
            'counterfactuals': cf_results
        }
        
        # 528 데이터의 경우 추가 정보 포함
        if 'original_prediction' in item:
            result_item['original_prediction_ground_truth'] = int(item['original_prediction'])
        
        results.append(result_item)
    
    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
    print(f'결과가 {output_path}에 저장되었습니다.')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 디바이스: {device}')
    
    # 모델 및 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    
    # 각 데이터셋에 대해 예측 수행
    for dataset_size, data_path in DATA_PATHS.items():
        print(f'\n=== {dataset_size}개 데이터셋 처리 중 ===')
        output_path = os.path.join(OUTPUT_DIR, f'counterfactual_predictions_{dataset_size}.json')
        
        # 파일 존재 여부 확인
        if not os.path.exists(data_path):
            print(f'경고: {data_path} 파일이 존재하지 않습니다. 건너뜁니다.')
            continue
            
        process_dataset(data_path, output_path, tokenizer, model, device)
    
    print('\n=== 모든 예측 완료 ===')

if __name__ == '__main__':
    main()
