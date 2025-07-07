## prompt_template.txt를 바탕으로
## prompt_with_original.jsonl 파일을 생성하는 스크립트

import json
import os
import pandas as pd

# 입력 파일 경로
TEMPLATE_PATH = 'outputs/prompt/prompt_template.txt'
ORIGINAL_DATA_PATH = 'outputs/errors/low_confidence_300.csv'
OUTPUT_PATH = 'outputs/prompt/prompts_with_original.jsonl'

def main():
    # 프롬프트 템플릿 읽기
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # original_data.json 읽기
    df = pd.read_csv(ORIGINAL_DATA_PATH)
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 각 샘플에 대해 프롬프트 생성 및 저장
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fout:
        for _, row in df.iterrows():
            original_id = row['Unnamed: 0']
            original_text = row['text']
            prompt = template.replace('{original_text}', original_text)
            out = {
                'original_id': original_id,
                'original_text': original_text,
                'prompt': prompt
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
    print(f'프롬프트 300개가 {OUTPUT_PATH}에 저장되었습니다.')

    print(df.columns)

if __name__ == '__main__':
    main()

