import json
import os
from typing import Dict, List, Tuple

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().split())

def norm_label(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return 'B1' if int(x) == 0 else 'B2' if int(x) == 1 else None
    xs = str(x).strip().upper()
    if xs in ('B1', '0'):
        return 'B1'
    if xs in ('B2', '1'):
        return 'B2'
    return xs


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_negatives_from_clean(clean_files: List[str], loop_num: int) -> List[Dict]:
    negatives: List[Dict] = []
    for fp in clean_files:
        if not os.path.exists(fp):
            print(f"⚠️ Missing: {fp}")
            continue
        data = load_json(fp)
        items = data.get('part1_selected_sentences', []) if isinstance(data, dict) else []
        for r in items:
            oid = r.get('original_id')
            anchor = norm_text(r.get('original_sentence', ''))
            o_pred = r.get('original_prediction')
            neg = norm_text(r.get('cf_sentence', ''))
            n_pred = r.get('cf_prediction')
            n_type = r.get('cf_type')
            if not (oid and anchor and neg and o_pred and n_pred):
                continue
            negatives.append({
                'loop': loop_num,
                'original_id': str(oid),
                'anchor': anchor,
                'original_prediction': norm_label(o_pred),
                'negative': neg,
                'negative_prediction': norm_label(n_pred),
                'negative_cf_type': n_type,
            })
    return negatives


def build_positive_index(src_path: str) -> Dict[str, List[Dict]]:
    """Index by original_id: list of candidates {cf_sentence, cf_prediction, cf_type} from loop data file."""
    data = load_json(src_path)
    index: Dict[str, List[Dict]] = {}
    # 파일이 리스트라고 가정 (loop_1_data.json, loop_2_data.json, loop_3_converted.json 모두 리스트 형태)
    for r in data:
        oid = r.get('original_id')
        if oid is None:
            continue
        oid = str(oid)
        # 다양한 키 이름 대응
        cf_sent = norm_text(r.get('cf_sentence') or r.get('counterfactual_sentence') or r.get('cf') or '')
        # 레이블 키가 cf_prediction 또는 cf_label(숫자)일 수 있음
        raw_pred = r.get('cf_prediction')
        if raw_pred is None:
            raw_pred = r.get('cf_label')
        cf_pred = norm_label(raw_pred)
        cf_type = r.get('cf_type')
        if not (cf_sent and cf_pred):
            continue
        index.setdefault(oid, []).append({'cf_sentence': cf_sent, 'cf_prediction': cf_pred, 'cf_type': cf_type})
    return index


@torch.no_grad()
def pick_most_similar(anchor: str, candidates: List[str], model: SentenceTransformer) -> Tuple[str, int]:
    if not candidates:
        return "", -1
    if len(candidates) == 1:
        return candidates[0], 0
    vecs = model.encode([anchor] + candidates, convert_to_tensor=True, normalize_embeddings=True)
    a = vecs[0]
    C = vecs[1:]
    sims = (a @ C.T).cpu().tolist()
    best = int(max(range(len(candidates)), key=lambda i: sims[i]))
    return candidates[best], best


def build_triplets_loops123(
    out_path: str = 'data/Loop/triplets_loops123_from_rules.json',
):
    # 1) Negatives from clean files per loop
    neg_loop1 = load_negatives_from_clean([
        'data/Loop/loop_1/loop1_id1_clean.json',
        'data/Loop/loop_1/loop1_id2_clean.json',
        'data/Loop/loop_1/loop1_id3_clean.json',
        'data/Loop/loop_1/loop1_id4_clean.json',
        'data/Loop/loop_1/loop1_id5_clean.json',
    ], loop_num=1)
    neg_loop2 = load_negatives_from_clean([
        'data/Loop/loop_2/loop2_id1_clean.json',
        'data/Loop/loop_2/loop2_id2_clean.json',
        'data/Loop/loop_2/loop2_id3_clean.json',
        'data/Loop/loop_2/loop2_id4_clean.json',
        'data/Loop/loop_2/loop2_id5_clean.json',
    ], loop_num=2)
    neg_loop3 = load_negatives_from_clean([
        'data/Loop/loop_3/loop3_id1_clean.json',
        'data/Loop/loop_3/loop3_id2_clean.json',
        'data/Loop/loop_3/loop3_id3_clean.json',
        'data/Loop/loop_3/loop3_id4_clean.json',
        'data/Loop/loop_3/loop3_id5_clean.json',
    ], loop_num=3)

    negatives = neg_loop1 + neg_loop2 + neg_loop3
    print(f"Negatives loaded: loop1={len(neg_loop1)}, loop2={len(neg_loop2)}, loop3={len(neg_loop3)}, total={len(negatives)}")

    # 2) Positive candidates indices per loop source (use normalized datasets)
    pos_idx_loop1 = build_positive_index('data/Loop/loop_1/loop_1_data_normalized.json')
    pos_idx_loop2 = build_positive_index('data/Loop/loop_2/loop_2_data_normalized.json')
    pos_idx_loop3 = build_positive_index('data/Loop/loop_3/loop_3_data_normalized.json')

    # 간단한 로깅
    def _index_stats(idx, name):
        total = sum(len(v) for v in idx.values())
        print(f"[{name}] original_ids={len(idx)} candidates={total}")
    _index_stats(pos_idx_loop1, 'loop1_idx')
    _index_stats(pos_idx_loop2, 'loop2_idx')
    _index_stats(pos_idx_loop3, 'loop3_idx')

    model = SentenceTransformer('all-MiniLM-L6-v2')

    triplets: List[Dict] = []
    matched = 0
    for r in tqdm(negatives, desc='Building triplets(loop1/2/3)'):
        loop_num = r['loop']
        oid = r['original_id']
        anchor = r['anchor']
        o_pred = r['original_prediction']
        neg = r['negative']
        n_pred = r['negative_prediction']
        n_type = r['negative_cf_type']

        # select candidates with same label as anchor from the corresponding loop source
        if loop_num == 1:
            cands_src = pos_idx_loop1
        elif loop_num == 2:
            cands_src = pos_idx_loop2
        else:
            cands_src = pos_idx_loop3

        cands_all = cands_src.get(oid, [])
        cands = [c for c in cands_all if c.get('cf_prediction') == norm_label(o_pred)]
        pos_text = ""; pos_pred = None; pos_type = None
        if cands:
            pos_list = [c['cf_sentence'] for c in cands]
            pos_text, idx = pick_most_similar(anchor, pos_list, model)
            if idx >= 0:
                # 안전하게 텍스트 매칭으로도 보강
                selected = cands[idx]
                if pos_text and (selected.get('cf_sentence') != pos_text):
                    for cc in cands:
                        if cc.get('cf_sentence') == pos_text:
                            selected = cc
                            break
                pos_pred = norm_label(selected.get('cf_prediction'))
                pos_type = selected.get('cf_type')
                matched += 1
        # 디버그: 매칭 실패 시 간단 로그
        else:
            if cands_all:
                want = norm_label(o_pred)
                have_labels = {norm_label(x.get('cf_prediction')) for x in cands_all}
                print(f"[warn] no same-label positive for oid={oid} loop={loop_num} want={want} have={have_labels}")

        triplets.append({
            'loop': loop_num,
            'original_id': oid,
            'anchor': anchor,
            'original_prediction': o_pred,
            'positive': pos_text,
            'positive_prediction': pos_pred,
            'positive_cf_type': pos_type,
            'negative': neg,
            'negative_prediction': n_pred,
            'negative_cf_type': n_type,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path} (triplets={len(triplets)}, positives_matched={matched})")


if __name__ == '__main__':
    build_triplets_loops123()


