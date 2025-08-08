import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_similarity(model, sentence1, sentence2):
    embeddings = model.encode([sentence1, sentence2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

def main():
    print("모델 로딩 중...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("전문가 데이터 로딩 중...")
    expert_files = [
        'data/Loop/loop_1/loop1_id1_clean.json',
        'data/Loop/loop_1/loop1_id3_clean.json', 
        'data/Loop/loop_1/loop1_id4_clean.json',
        'data/Loop/loop_1/loop1_id5_clean.json'
    ]
    
    expert_data = []
    for file_path in expert_files:
        data = load_data(file_path)
        expert_data.extend(data['part1_selected_sentences'])
        print(f"  {file_path}: {len(data['part1_selected_sentences'])}개")
    
    print(f"총 전문가 데이터: {len(expert_data)}개")
    
    print("Loop 데이터 로딩 중...")
    loop_data = load_data('data/Loop/loop_1_data.json')
    print(f"Loop 데이터: {len(loop_data)}개")
    
    # 공통 ID 찾기
    expert_ids = set(item['original_id'] for item in expert_data)
    loop_ids = set(item['original_id'] for item in loop_data)
    common_ids = expert_ids.intersection(loop_ids)
    print(f"공통 ID 개수: {len(common_ids)}개")
    
    print("Triplet 생성 중...")
    triplets = []
    
    for i, item in enumerate(loop_data):
        if item['original_id'] not in common_ids:
            continue
        if i % 20 == 0:
            print(f"  진행률: {i}/{len(loop_data)}")
        original_sentence = item['original_sentence']
        counterfactuals = item['counterfactuals']
        # Positive 찾기 (label_flipped=False인 것 중 가장 유사한 것)
        positive_candidates = [cf for cf in counterfactuals if not cf['is_label_flipped']]
        if not positive_candidates:
            continue
        best_positive = None
        best_positive_sim = -1
        for cf in positive_candidates:
            sim = calculate_similarity(model, original_sentence, cf['sentence'])
            if sim > best_positive_sim:
                best_positive_sim = sim
                best_positive = cf
        # Negative 후보 (같은 original_id의 전문가 문장 모두)
        same_id_experts = [expert_item for expert_item in expert_data if expert_item['original_id'] == item['original_id']]
        if not same_id_experts:
            continue
        for expert_item in same_id_experts:
            negative_sim = calculate_similarity(model, original_sentence, expert_item['cf_sentence'])
            triplet = {
                'original_id': item['original_id'],
                'anchor': original_sentence,
                'positive': best_positive['sentence'],
                'negative': expert_item['cf_sentence'],
                'positive_similarity': float(best_positive_sim),
                'negative_similarity': float(negative_sim),
                'original_prediction': item['original_prediction'],
                'positive_prediction': best_positive['prediction'],
                'negative_prediction': expert_item['cf_prediction'],
                'negative_source': expert_item['original_id'],
                'negative_cf_type': expert_item['cf_type'],
                'negative_is_label_flipped': expert_item['is_label_flipped']
            }
            triplets.append(triplet)
    print(f"\n=== Triplet 생성 결과 ===")
    print(f"총 triplet 개수: {len(triplets)}")
    if triplets:
        positive_similarities = [t['positive_similarity'] for t in triplets]
        negative_similarities = [t['negative_similarity'] for t in triplets]
        print(f"\nPositive similarity:")
        print(f"  평균: {np.mean(positive_similarities):.4f}")
        print(f"  표준편차: {np.std(positive_similarities):.4f}")
        print(f"  최소값: {np.min(positive_similarities):.4f}")
        print(f"  최대값: {np.max(positive_similarities):.4f}")
        print(f"\nNegative similarity:")
        print(f"  평균: {np.mean(negative_similarities):.4f}")
        print(f"  표준편차: {np.std(negative_similarities):.4f}")
        print(f"  최소값: {np.min(negative_similarities):.4f}")
        print(f"  최대값: {np.max(negative_similarities):.4f}")
        # 예측값 분포
        original_predictions = [t['original_prediction'] for t in triplets]
        positive_predictions = [t['positive_prediction'] for t in triplets]
        negative_predictions = [t['negative_prediction'] for t in triplets]
        print(f"\n예측값 분포:")
        print(f"Original - 0: {original_predictions.count(0)}, 1: {original_predictions.count(1)}")
        print(f"Positive - 0: {positive_predictions.count(0)}, 1: {positive_predictions.count(1)}")
        print(f"Negative - 0: {negative_predictions.count(0)}, 1: {negative_predictions.count(1)}")
        # Label flipped 분포
        label_flipped_count = sum(1 for t in triplets if t['negative_is_label_flipped'])
        not_flipped_count = len(triplets) - label_flipped_count
        print(f"\nNegative Label Flipped 분포:")
        print(f"  Label Flipped: {label_flipped_count}개 ({label_flipped_count/len(triplets)*100:.1f}%)")
        print(f"  Not Flipped: {not_flipped_count}개 ({not_flipped_count/len(triplets)*100:.1f}%)")
        # Counterfactual 타입 분포
        cf_types = [t['negative_cf_type'] for t in triplets]
        cf_type_counts = {}
        for cf_type in cf_types:
            cf_type_counts[cf_type] = cf_type_counts.get(cf_type, 0) + 1
        print(f"\nNegative Counterfactual 타입 분포:")
        for cf_type, count in cf_type_counts.items():
            print(f"  {cf_type}: {count}개")
        # 결과 저장
        output_path = 'data/Loop/triplets_expert_all.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(triplets, f, indent=2, ensure_ascii=False)
        print(f"\nTriplet 데이터가 {output_path}에 저장되었습니다.")
        # 샘플 출력
        print("\n=== 샘플 Triplet ===")
        sample = triplets[0]
        print(f"Original ID: {sample['original_id']}")
        print(f"Anchor: {sample['anchor']}")
        print(f"Positive: {sample['positive']}")
        print(f"Negative: {sample['negative']}")
        print(f"Positive Similarity: {sample['positive_similarity']:.4f}")
        print(f"Negative Similarity: {sample['negative_similarity']:.4f}")
        print(f"Negative Source: {sample['negative_source']}")
        print(f"Negative CF Type: {sample['negative_cf_type']}")
        print(f"Negative Label Flipped: {sample['negative_is_label_flipped']}")

if __name__ == "__main__":
    main() 