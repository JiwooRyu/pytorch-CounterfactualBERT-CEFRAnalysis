import json
import re

def parse_txt_to_json(input_file, output_file):
    """
    original_528_base.txt 파일을 읽어서 JSON 형태로 변환합니다.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # JSON 객체들을 분리 (각 객체는 {로 시작하고 }로 끝남)
    # 정규식을 사용하여 JSON 객체들을 추출
    json_objects = []
    
    # 중괄호로 둘러싸인 JSON 객체들을 찾기
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(pattern, content)
    
    for match in matches:
        try:
            # JSON 파싱 시도
            obj = json.loads(match)
            json_objects.append(obj)
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            print(f"문제가 있는 텍스트: {match[:100]}...")
            continue
    
    # 전체 JSON 배열로 만들기
    result = json_objects
    
    # JSON 파일로 저장 (들여쓰기와 정렬 포함)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"총 {len(result)}개의 JSON 객체가 {output_file}에 저장되었습니다.")
    return result

if __name__ == "__main__":
    input_file = "data/counterfactuals/original_528_base.txt"
    output_file = "data/counterfactuals/original_528_base.json"
    
    result = parse_txt_to_json(input_file, output_file)
    
    # 첫 번째 객체의 구조 확인
    if result:
        print("\n첫 번째 객체 구조:")
        print(json.dumps(result[0], indent=2, ensure_ascii=False)) 