import json
import random
from collections import defaultdict

def load_data(file_path):
    """데이터 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_data():
    """데이터 분할"""
    # 데이터 로드
    data = load_data('outputs/analysis/counterfactual_predictions_528.json')
    print(f"총 데이터 개수: {len(data)}")
    
    # 랜덤 seed 설정
    random.seed(42)
    
    # 40개 anchor를 랜덤으로 선택
    anchors = random.sample(data, 40)
    print(f"선택된 anchor 개수: {len(anchors)}")
    
    # 나머지 데이터에서 480개 선택
    remaining_data = [item for item in data if item not in anchors]
    selected_normal = random.sample(remaining_data, 480)
    print(f"선택된 일반 문장 개수: {len(selected_normal)}")
    
    # 전체 선택된 데이터
    all_selected = anchors + selected_normal
    print(f"총 선택된 데이터: {len(all_selected)}")
    
    # 3개 루프로 분할 (각 루프: 40개 anchor + 160개 일반 = 200개)
    loops = []
    for i in range(3):
        start_idx = i * 160
        end_idx = (i + 1) * 160
        loop_data = anchors + selected_normal[start_idx:end_idx]
        loops.append(loop_data)
        print(f"Loop {i+1}: {len(loop_data)}개 (anchor: 40개, 일반: {len(selected_normal[start_idx:end_idx])}개)")
    
    return loops, anchors, selected_normal

def save_data(loops, anchors, selected_normal):
    """데이터 저장"""
    # 각 루프별 데이터 저장
    for i, loop_data in enumerate(loops):
        filename = f'outputs/analysis/loop_{i+1}_data.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(loop_data, f, ensure_ascii=False, indent=2)
        print(f"저장됨: {filename}")
    
    # anchor 데이터 저장
    with open('outputs/analysis/anchor_data.json', 'w', encoding='utf-8') as f:
        json.dump(anchors, f, ensure_ascii=False, indent=2)
    print("저장됨: outputs/analysis/anchor_data.json")
    
    # 선택된 일반 데이터 저장
    with open('outputs/analysis/selected_normal_data.json', 'w', encoding='utf-8') as f:
        json.dump(selected_normal, f, ensure_ascii=False, indent=2)
    print("저장됨: outputs/analysis/selected_normal_data.json")
    
    # 전체 선택된 데이터 저장
    all_data = anchors + selected_normal
    with open('outputs/analysis/all_selected_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print("저장됨: outputs/analysis/all_selected_data.json")

def main():
    """메인 함수"""
    print("=== 데이터 분할 시작 ===")
    
    # 데이터 분할
    loops, anchors, selected_normal = split_data()
    
    # 데이터 저장
    save_data(loops, anchors, selected_normal)
    
    print("\n=== 분할 완료 ===")
    print("생성된 파일:")
    print("- loop_1_data.json (200개)")
    print("- loop_2_data.json (200개)")
    print("- loop_3_data.json (200개)")
    print("- anchor_data.json (40개)")
    print("- selected_normal_data.json (480개)")
    print("- all_selected_data.json (520개)")

if __name__ == "__main__":
    main() 