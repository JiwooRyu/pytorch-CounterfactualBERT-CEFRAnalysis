import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np


def safe_len(x: str) -> int:
    return len(x.split()) if isinstance(x, str) else 0


def analyze_triplets(path: str = 'data/Loop/loop_3/accumulated_loop2_3_clean.json'):
    with open(path, 'r', encoding='utf-8') as f:
        data: List[Dict] = json.load(f)

    print(f"File: {path}")
    print(f"Total items: {len(data)}")
    if not data:
        return

    # 필수 키 존재 여부
    required = {"anchor", "positive", "negative"}
    missing_req = sum(1 for d in data if not required.issubset(d.keys()))
    print(f"Missing required keys (anchor/positive/negative): {missing_req}")

    # 라벨 키 존재 여부 및 값 분포
    label_keys = ["original_prediction", "positive_prediction", "negative_prediction"]
    present_counts = {k: sum(1 for d in data if k in d) for k in label_keys}
    print(f"Label presence counts: {present_counts}")

    def get_label(d: Dict, key: str) -> str:
        v = d.get(key)
        return v if isinstance(v, str) else "<none>"

    label_set = Counter()
    pair_equal = 0
    pair_diff = 0
    tri_patterns = Counter()
    for d in data:
        o = get_label(d, 'original_prediction')
        p = get_label(d, 'positive_prediction')
        n = get_label(d, 'negative_prediction')
        label_set.update([o, p, n])
        if o == p and o != "<none>" and p != "<none>":
            pair_equal += 1
        if o != n and o != "<none>" and n != "<none>":
            pair_diff += 1
        tri_patterns.update([(o, p, n)])

    print(f"Unique labels (incl <none>): {dict(label_set)}")
    print(f"(o == p) count: {pair_equal}")
    print(f"(o != n) count: {pair_diff}")
    top_patterns = tri_patterns.most_common(10)
    print(f"Top label triplets (o,p,n): {top_patterns}")

    # 텍스트 동일/중복/길이 통계
    same_ap = sum(1 for d in data if d.get('anchor') == d.get('positive'))
    same_an = sum(1 for d in data if d.get('anchor') == d.get('negative'))
    same_pn = sum(1 for d in data if d.get('positive') == d.get('negative'))
    print(f"Identical pairs: anchor=positive {same_ap}, anchor=negative {same_an}, positive=negative {same_pn}")

    lens = {
        'anchor': [safe_len(d.get('anchor', '')) for d in data],
        'positive': [safe_len(d.get('positive', '')) for d in data],
        'negative': [safe_len(d.get('negative', '')) for d in data],
    }
    for k, arr in lens.items():
        arr_np = np.array(arr)
        print(f"Len({k}) -> mean {arr_np.mean():.1f}, p50 {np.percentile(arr_np,50):.1f}, p90 {np.percentile(arr_np,90):.1f}, max {arr_np.max()}")

    # 앵커별 분포, 중복도
    by_anchor = defaultdict(int)
    for d in data:
        a = d.get('anchor', '')
        by_anchor[a] += 1
    counts = np.array(list(by_anchor.values()))
    print(f"Unique anchors: {len(by_anchor)} (avg triplets per anchor {counts.mean():.2f}, p90 {np.percentile(counts,90):.1f}, max {counts.max()})")

    # s_pos/s_neg/gap 통계 (있을 경우)
    s_pos_vals = [d['s_pos'] for d in data if 's_pos' in d]
    s_neg_vals = [d['s_neg'] for d in data if 's_neg' in d]
    gap_vals = [d['gap'] for d in data if 'gap' in d]
    if gap_vals:
        print(f"s_pos mean {np.mean(s_pos_vals):.3f}, s_neg mean {np.mean(s_neg_vals):.3f}, gap mean {np.mean(gap_vals):.3f}")
        print(f"gap p10 {np.percentile(gap_vals,10):.3f}, p50 {np.percentile(gap_vals,50):.3f}, p90 {np.percentile(gap_vals,90):.3f}")
        bad_gap = sum(1 for g in gap_vals if g < 0)
        print(f"gap < 0 count: {bad_gap}")


if __name__ == '__main__':
    analyze_triplets()


