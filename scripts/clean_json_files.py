import json

def clean_json_file(input_path, output_path):
    """JSON 파일에서 part1_selected_sentences만 추출하여 새로운 파일 생성"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # part1_selected_sentences만 추출
        if 'part1_selected_sentences' in data:
            cleaned_data = {
                "part1_selected_sentences": data['part1_selected_sentences']
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            
            print(f"{input_path} -> {output_path}: {len(data['part1_selected_sentences'])}개 항목")
            return True
        else:
            print(f"{input_path}: part1_selected_sentences를 찾을 수 없음")
            return False
            
    except Exception as e:
        print(f"{input_path}: 오류 - {e}")
        return False

def main():
    files_to_clean = [
        ('data/Loop/loop_1/loop1_id1.json', 'data/Loop/loop_1/loop1_id1_clean.json'),
        ('data/Loop/loop_1/loop1_id3.json', 'data/Loop/loop_1/loop1_id3_clean.json'),
        ('data/Loop/loop_1/loop1_id4.json', 'data/Loop/loop_1/loop1_id4_clean.json'),
        ('data/Loop/loop_1/loop1_id5.json', 'data/Loop/loop_1/loop1_id5_clean.json')
    ]
    
    print("JSON 파일 정리 중...")
    for input_path, output_path in files_to_clean:
        clean_json_file(input_path, output_path)
    
    print("정리 완료!")

if __name__ == "__main__":
    main() 