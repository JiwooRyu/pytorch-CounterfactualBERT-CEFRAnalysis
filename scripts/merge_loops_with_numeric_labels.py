import json
import os
from typing import Dict, List


LABEL_MAP = {"B1": 0, "B2": 1}


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_items(data) -> List[Dict]:
    # clean 파일은 dict 형태로 part1_selected_sentences를 포함
    if isinstance(data, dict) and 'part1_selected_sentences' in data:
        return data['part1_selected_sentences']
    # 리스트면 그대로
    if isinstance(data, list):
        return data
    return []


def to_numeric_label(lbl):
    if isinstance(lbl, str):
        return LABEL_MAP.get(lbl, None)
    if isinstance(lbl, (int, float)):
        return int(lbl)
    return None


def normalize_record(rec: Dict, loop_num: int) -> Dict:
    return {
        'loop': loop_num,
        'original_id': rec.get('original_id'),
        'original_sentence': rec.get('original_sentence'),
        'original_label': to_numeric_label(rec.get('original_prediction')),
        'cf_sentence': rec.get('cf_sentence'),
        'cf_label': to_numeric_label(rec.get('cf_prediction')),
        'cf_type': rec.get('cf_type'),
        'is_label_flipped': bool(rec.get('is_label_flipped')) if rec.get('is_label_flipped') is not None else None,
    }


def merge_all():
    files = []
    # loop1 id1~5
    files += [(f'data/Loop/loop_1/loop1_id{i}_clean.json', 1) for i in range(1, 6)]
    # loop2 id1~5
    files += [(f'data/Loop/loop_2/loop2_id{i}_clean.json', 2) for i in range(1, 6)]
    # loop3 id1~5
    files += [(f'data/Loop/loop_3/loop3_id{i}_clean.json', 3) for i in range(1, 6)]

    merged: List[Dict] = []
    missing = 0
    for path, loop_num in files:
        if not os.path.exists(path):
            print(f"⚠️ Missing: {path}")
            missing += 1
            continue
        data = load_json(path)
        items = extract_items(data)
        for rec in items:
            merged.append(normalize_record(rec, loop_num))

    out_path = 'data/Loop/loop123_merged_numeric.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path} (items={len(merged)}, missing_files={missing})")


if __name__ == '__main__':
    merge_all()


