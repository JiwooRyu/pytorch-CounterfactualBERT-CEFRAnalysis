import json
import os

def load_data(file_path):
    """JSON 파일을 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✅ {file_path} 로드 완료: {len(data)}개 항목")
            return data
    except Exception as e:
        print(f"❌ {file_path} 로드 실패: {e}")
        return []

def merge_triplets():
    """triplets_expert_all.json과 triplets_loop2_all.json을 합칩니다."""
    
    print("🔄 Triplet 데이터 병합 중...\n")
    
    # 1. 두 파일 로드
    triplets_expert = load_data('data/Loop/loop_1/triplets_expert_all.json')
    triplets_loop2 = load_data('data/Loop/loop_2/triplets_loop2_all.json')
    
    if not triplets_expert or not triplets_loop2:
        print("❌ 데이터 로드 실패로 병합을 중단합니다.")
        return
    
    # 2. 데이터 구조 확인
    print(f"\n📊 데이터 구조 분석:")
    if triplets_expert:
        print(f"  triplets_expert_all.json 첫 번째 항목 키: {list(triplets_expert[0].keys())}")
    if triplets_loop2:
        print(f"  triplets_loop2_all.json 첫 번째 항목 키: {list(triplets_loop2[0].keys())}")
    
    # 3. 데이터 병합
    print(f"\n🔗 데이터 병합 중...")
    merged_data = []
    
    # triplets_expert_all.json 데이터 추가
    for item in triplets_expert:
        # source 정보 추가
        item_copy = item.copy()
        item_copy['source'] = 'expert_all'
        merged_data.append(item_copy)
    
    # triplets_loop2_all.json 데이터 추가
    for item in triplets_loop2:
        # source 정보 추가
        item_copy = item.copy()
        item_copy['source'] = 'loop2_all'
        merged_data.append(item_copy)
    
    print(f"  triplets_expert_all.json: {len(triplets_expert)}개")
    print(f"  triplets_loop2_all.json: {len(triplets_loop2)}개")
    print(f"  병합된 총 데이터: {len(merged_data)}개")
    
    # 4. 중복 제거 (original_id 기준)
    print(f"\n🧹 중복 제거 중...")
    unique_data = {}
    duplicates = 0
    
    for item in merged_data:
        original_id = item['original_id']
        if original_id not in unique_data:
            unique_data[original_id] = item
        else:
            duplicates += 1
            # source가 다르면 두 source 모두 표시
            if 'source' in unique_data[original_id]:
                if isinstance(unique_data[original_id]['source'], list):
                    unique_data[original_id]['source'].append(item['source'])
                else:
                    unique_data[original_id]['source'] = [unique_data[original_id]['source'], item['source']]
    
    final_data = list(unique_data.values())
    print(f"  중복 제거 전: {len(merged_data)}개")
    print(f"  중복 제거 후: {len(final_data)}개")
    print(f"  제거된 중복: {duplicates}개")
    
    # 5. 통계 분석
    print(f"\n📈 병합된 데이터 통계:")
    
    # source 분포
    source_counts = {}
    for item in final_data:
        source = item.get('source', 'unknown')
        if isinstance(source, list):
            for s in source:
                source_counts[s] = source_counts.get(s, 0) + 1
        else:
            source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"  Source 분포:")
    for source, count in source_counts.items():
        print(f"    {source}: {count}개")
    
    # original_id 분포
    expert_ids = set(item['original_id'] for item in triplets_expert)
    loop2_ids = set(item['original_id'] for item in triplets_loop2)
    common_ids = expert_ids.intersection(loop2_ids)
    
    print(f"  ID 분포:")
    print(f"    expert_all에만 있는 ID: {len(expert_ids - loop2_ids)}개")
    print(f"    loop2_all에만 있는 ID: {len(loop2_ids - expert_ids)}개")
    print(f"    공통 ID: {len(common_ids)}개")
    
    # 6. 결과 저장
    output_path = 'data/Loop/loop_2/accumulated_loop2.json'
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 병합된 데이터가 {output_path}에 저장되었습니다.")
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        return
    
    # 7. 샘플 출력
    print(f"\n🔍 샘플 데이터:")
    if final_data:
        sample = final_data[0]
        print(f"  Original ID: {sample['original_id']}")
        print(f"  Source: {sample['source']}")
        print(f"  Anchor: {sample['anchor'][:100]}...")
        print(f"  Positive: {sample['positive'][:100]}...")
        print(f"  Negative: {sample['negative'][:100]}...")
    
    print(f"\n🎉 Triplet 데이터 병합 완료!")

if __name__ == "__main__":
    merge_triplets()
