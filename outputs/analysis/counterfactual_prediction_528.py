import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
from tqdm import tqdm

def main():
    print("스크립트 시작...")
    
    # 경로 설정
    MODEL_DIR = 'models/baseline_bert'
    DATA_PATH = 'data/counterfactuals/original_528_base.json'
    OUTPUT_PATH = 'outputs/analysis/counterfactual_predictions_528.json'
    
    print(f"모델 디렉토리: {MODEL_DIR}")
    print(f"데이터 경로: {DATA_PATH}")
    print(f"출력 경로: {OUTPUT_PATH}")
    
    # 데이터 파일 존재 확인
    if not os.path.exists(DATA_PATH):
        print(f"오류: {DATA_PATH} 파일이 존재하지 않습니다.")
        return
    
    # 모델 디렉토리 존재 확인
    if not os.path.exists(MODEL_DIR):
        print(f"오류: {MODEL_DIR} 디렉토리가 존재하지 않습니다.")
        return
    
    print("파일 확인 완료")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 디바이스: {device}')
    
    # 모델 및 토크나이저 로드
    try:
        print("모델 로딩 중...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()
        print("모델 로딩 완료")
    except Exception as e:
        print(f"모델 로딩 오류: {e}")
        return
    
    # 데이터 로드
    try:
        print("데이터 로딩 중...")
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"데이터 로딩 완료: {len(data)}개 항목")
    except Exception as e:
        print(f"데이터 로딩 오류: {e}")
        return
    
    # 예측 함수
    def predict_label(text, tokenizer, model, device):
        try:
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
        except Exception as e:
            print(f"예측 오류: {e}")
            return 0
    
    results = []
    cf_types = ['lexical', 'shuffle', 'restructure', 'negation', 'quantifier', 'resemantic', 'insert', 'delete']
    
    print("예측 시작...")
    for i, item in enumerate(tqdm(data, desc='Predicting original_528_base')):
        if i % 50 == 0:
            print(f"처리 중: {i}/{len(data)}")
            
        sentence_id = item['sentence_id']
        original_sentence = item['original']
        original_prediction = int(item['original_prediction'])
        
        # original 예측 (검증용)
        orig_pred = predict_label(original_sentence, tokenizer, model, device)
        
        # counterfactual 예측
        cf_results = []
        
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
        
        results.append({
            'original_id': sentence_id,
            'original_sentence': original_sentence,
            'original_prediction': orig_pred,
            'original_prediction_ground_truth': original_prediction,
            'counterfactuals': cf_results
        })
    
    print("예측 완료")
    
    # 저장
    try:
        print("파일 저장 중...")
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as fout:
            json.dump(results, fout, ensure_ascii=False, indent=2)
        print(f'결과가 {OUTPUT_PATH}에 저장되었습니다.')
    except Exception as e:
        print(f"파일 저장 오류: {e}")
        return
    
    # 통계 출력
    total_items = len(results)
    total_cf = total_items * len(cf_types)
    flipped_count = sum(1 for item in results for cf in item['counterfactuals'] if cf['is_label_flipped'])
    flip_rate = (flipped_count / total_cf) * 100
    
    print(f'\n=== 예측 통계 ===')
    print(f'총 문장 수: {total_items}')
    print(f'총 counterfactual 수: {total_cf}')
    print(f'Label이 뒤바뀐 counterfactual 수: {flipped_count}')
    print(f'Flip rate: {flip_rate:.2f}%')

if __name__ == '__main__':
    main() 