import json
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Tuple

def parse_counterfactual_data(json_path: str):
    """
    counterfactual_predictions_300.json 데이터를 파싱하여 negative pairs를 생성합니다.
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        negative_pairs: (anchor, counterfactual) 쌍의 리스트
    """
    # JSON 데이터 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    negative_pairs = []
    control_code_counts = Counter()
    
    print(f"총 {len(data)}개의 원본 문장을 처리합니다...")
    
    for item in data:
        original_id = item['original_id']
        original_sentence = item['original_sentence']
        original_prediction = item['original_prediction']
        
        # counterfactuals 리스트에서 label이 flipped된 것만 선택
        for cf in item['counterfactuals']:
            if cf['is_label_flipped']:
                # Negative pair 생성
                negative_pair = {
                    'anchor': original_sentence,
                    'counterfactual': cf['sentence'],
                    'anchor_label': original_prediction,
                    'counterfactual_label': cf['prediction'],
                    'control_code': cf['type'],
                    'original_id': original_id
                }
                
                negative_pairs.append(negative_pair)
                control_code_counts[cf['type']] += 1
    
    print(f"총 {len(negative_pairs)}개의 negative pairs가 생성되었습니다.")
    
    return negative_pairs, control_code_counts

def save_negative_pairs(negative_pairs: List[Dict[str, Any]], output_path: str):
    """
    Negative pairs를 JSONL 형식으로 저장합니다.
    
    Args:
        negative_pairs: 저장할 negative pairs 리스트
        output_path: 출력 파일 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in negative_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"Negative pairs가 {output_path}에 저장되었습니다.")

def print_control_code_statistics(control_code_counts: Counter):
    """
    Control code별 통계를 출력합니다.
    
    Args:
        control_code_counts: control code별 개수를 담은 Counter
    """
    print("\n=== Control Code별 통계 ===")
    print(f"{'Control Code':<15} {'Count':<10} {'Percentage':<12}")
    print("-" * 40)
    
    total = sum(control_code_counts.values())
    
    for control_code, count in control_code_counts.most_common():
        percentage = (count / total) * 100
        print(f"{control_code:<15} {count:<10} {percentage:.1f}%")
    
    print(f"\n총 negative pairs: {total}개")

def analyze_label_distribution(negative_pairs: List[Dict[str, Any]]):
    """
    Label 분포를 분석합니다.
    
    Args:
        negative_pairs: negative pairs 리스트
    """
    anchor_labels = [pair['anchor_label'] for pair in negative_pairs]
    counterfactual_labels = [pair['counterfactual_label'] for pair in negative_pairs]
    
    anchor_label_counts = Counter(anchor_labels)
    counterfactual_label_counts = Counter(counterfactual_labels)
    
    print("\n=== Label 분포 분석 ===")
    print(f"Anchor 문장 레이블 분포:")
    for label, count in anchor_label_counts.items():
        print(f"  Label {label}: {count}개")
    
    print(f"\nCounterfactual 문장 레이블 분포:")
    for label, count in counterfactual_label_counts.items():
        print(f"  Label {label}: {count}개")
    
    # Label flip 패턴 분석
    flip_patterns = []
    for pair in negative_pairs:
        pattern = f"{pair['anchor_label']} -> {pair['counterfactual_label']}"
        flip_patterns.append(pattern)
    
    flip_pattern_counts = Counter(flip_patterns)
    print(f"\nLabel flip 패턴:")
    for pattern, count in flip_pattern_counts.items():
        print(f"  {pattern}: {count}개")

def main():
    """메인 실행 함수"""
    # 파일 경로 설정
    input_path = "outputs/analysis/counterfactual_predictions_300.json"
    output_path = "poc/contrastive_data.jsonl"
    
    print("=== Negative Pairs 생성 시작 ===")
    print(f"입력 파일: {input_path}")
    print(f"출력 파일: {output_path}")
    
    # Negative pairs 생성
    negative_pairs, control_code_counts = parse_counterfactual_data(input_path)
    
    # Control code별 통계 출력
    print_control_code_statistics(control_code_counts)
    
    # Label 분포 분석
    analyze_label_distribution(negative_pairs)
    
    # Negative pairs 저장
    save_negative_pairs(negative_pairs, output_path)
    
    print("\n=== 분석 완료 ===")
    print(f"생성된 파일: {output_path}")
    print(f"총 negative pairs: {len(negative_pairs)}개")

if __name__ == "__main__":
    main()
