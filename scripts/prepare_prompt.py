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
            original_text = str(row['text'])
            prompt = template.replace('{original_text}', original_text)
            out = {
                'original_id': original_id,
                'original_text': original_text,
                'prompt': prompt
            }
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
    print(f'프롬프트 300개가 {OUTPUT_PATH}에 저장되었습니다.')

    # original_id, original_text만 추출해서 json 저장
    minimal_list = []
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as fin:
        for line in fin:
            item = json.loads(line)
            minimal_list.append({
                'original_id': item['original_id'],
                'original_text': item['original_text']
            })
    minimal_output_path = 'outputs/errors/low_confidence_300.json'
    with open(minimal_output_path, 'w', encoding='utf-8') as fout:
        json.dump(minimal_list, fout, ensure_ascii=False, indent=2)
    print(f'original_id, original_text만 저장된 파일이 {minimal_output_path}에 저장되었습니다.')

    print(df.columns)

if __name__ == '__main__':
    main()

