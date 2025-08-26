import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().split())


def load_merged(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    valid = []
    for d in data:
        if not isinstance(d, dict):
            continue
        o = d.get('original_sentence')
        c = d.get('cf_sentence')
        if not o or not c:
            continue
        nd = dict(d)
        nd['original_sentence'] = norm_text(o)
        nd['cf_sentence'] = norm_text(c)
        valid.append(nd)
    return valid


@torch.no_grad()
def embed(texts: List[str], model: SentenceTransformer, batch_size: int = 128) -> torch.Tensor:
    return model.encode(texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)


def select_negatives(
    in_path: str = 'data/Loop/loop123_merged_clean.json',
    out_path: str = 'data/Loop/negatives_selected.json',
    min_sim: float = 0.60,
    max_sim: float = 0.92,
    max_neg_per_anchor: int = 6,
    enforce_label_mismatch: bool = True,
    dedupe_threshold: float = 0.98,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    items = load_merged(in_path)
    from collections import defaultdict, Counter
    by_anchor: Dict[str, List[Dict]] = defaultdict(list)
    for d in items:
        by_anchor[d['original_sentence']].append(d)

    anchors = list(by_anchor.keys())
    print(f"Anchors: {len(anchors)}")

    uniq_texts: List[str] = []
    index: Dict[str, int] = {}
    for a, arr in by_anchor.items():
        if a not in index:
            index[a] = len(uniq_texts)
            uniq_texts.append(a)
        for d in arr:
            c = d['cf_sentence']
            if c not in index:
                index[c] = len(uniq_texts)
                uniq_texts.append(c)

    vecs = embed(uniq_texts, sbert)

    def vec(s: str) -> torch.Tensor:
        return vecs[index[s]]

    results: List[Dict] = []
    kept_total = 0

    for a in tqdm(anchors, desc='Selecting negatives'):
        a_vec = vec(a)
        cands = by_anchor[a]
        # 앵커 original_id 추출(가장 많이 등장하는 id 사용)
        id_counter = Counter([d.get('original_id') for d in cands if d.get('original_id') is not None])
        anchor_id = None
        if id_counter:
            anchor_id = id_counter.most_common(1)[0][0]

        filtered = []
        for d in cands:
            if enforce_label_mismatch:
                o_lab = d.get('original_prediction')
                c_lab = d.get('cf_prediction')
                if isinstance(o_lab, str) and isinstance(c_lab, str) and o_lab == c_lab:
                    continue
            sim = F.cosine_similarity(a_vec, vec(d['cf_sentence']), dim=0).item()
            if sim < min_sim or sim > max_sim:
                continue
            new_d = dict(d)
            new_d['sim'] = sim
            filtered.append(new_d)

        filtered.sort(key=lambda x: x['sim'], reverse=True)

        selected: List[Dict] = []
        selected_vecs: List[torch.Tensor] = []
        for d in filtered:
            c = d['cf_sentence']
            c_vec = vec(c)
            is_dup = False
            for sv in selected_vecs:
                if F.cosine_similarity(c_vec, sv, dim=0).item() >= dedupe_threshold:
                    is_dup = True
                    break
            if is_dup:
                continue
            selected.append(d)
            selected_vecs.append(c_vec)
            if len(selected) >= max_neg_per_anchor:
                break

        if not selected:
            continue

        results.append({
            'anchor': a,
            'original_id': anchor_id,
            'negatives': [
                {
                    'text': d['cf_sentence'],
                    'cf_type': d.get('cf_type'),
                    'sim': d.get('sim'),
                    'original_prediction': d.get('original_prediction'),
                    'cf_prediction': d.get('cf_prediction'),
                }
                for d in selected
            ]
        })
        kept_total += len(selected)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved negatives: anchors {len(results)}, total negatives {kept_total} -> {out_path}")


if __name__ == '__main__':
    select_negatives()

import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().split())


def load_merged(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 예상 포맷: 각 항목에 original_sentence, cf_sentence, original_prediction, cf_prediction, cf_type 등
    valid = []
    for d in data:
        if not isinstance(d, dict):
            continue
        o = d.get('original_sentence')
        c = d.get('cf_sentence')
        if not o or not c:
            continue
        nd = dict(d)
        nd['original_sentence'] = norm_text(o)
        nd['cf_sentence'] = norm_text(c)
        valid.append(nd)
    return valid


@torch.no_grad()
def embed(texts: List[str], model: SentenceTransformer, batch_size: int = 128) -> torch.Tensor:
    return model.encode(texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)


def select_negatives(
    in_path: str = 'data/Loop/loop123_merged_clean.json',
    out_path: str = 'data/Loop/negatives_selected.json',
    min_sim: float = 0.60,
    max_sim: float = 0.92,
    max_neg_per_anchor: int = 6,
    enforce_label_mismatch: bool = True,
    dedupe_threshold: float = 0.98,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    items = load_merged(in_path)
    # 앵커별 그룹핑
    from collections import defaultdict
    by_anchor: Dict[str, List[Dict]] = defaultdict(list)
    for d in items:
        by_anchor[d['original_sentence']].append(d)

    anchors = list(by_anchor.keys())
    print(f"Anchors: {len(anchors)}")

    # 모든 문장 임베딩 캐시(앵커 + 후보 cf)
    uniq_texts: List[str] = []
    index: Dict[str, int] = {}
    for a, arr in by_anchor.items():
        if a not in index:
            index[a] = len(uniq_texts)
            uniq_texts.append(a)
        for d in arr:
            c = d['cf_sentence']
            if c not in index:
                index[c] = len(uniq_texts)
                uniq_texts.append(c)

    vecs = embed(uniq_texts, sbert)

    def vec(s: str) -> torch.Tensor:
        return vecs[index[s]]

    results: List[Dict] = []
    kept_total = 0

    for a in tqdm(anchors, desc='Selecting negatives'):
        a_vec = vec(a)
        cands = by_anchor[a]
        # 앵커 original_id 추출(가장 많이 등장하는 id 사용)
        from collections import Counter
        id_counter = Counter([d.get('original_id') for d in cands if d.get('original_id') is not None])
        anchor_id = None
        if id_counter:
            anchor_id = id_counter.most_common(1)[0][0]

        # 라벨 불일치 필터(있으면 사용)
        filtered = []
        for d in cands:
            if enforce_label_mismatch:
                o_lab = d.get('original_prediction')
                c_lab = d.get('cf_prediction')
                if isinstance(o_lab, str) and isinstance(c_lab, str) and o_lab == c_lab:
                    continue
            # 유사도 필터
            sim = F.cosine_similarity(a_vec, vec(d['cf_sentence']), dim=0).item()
            if sim < min_sim or sim > max_sim:
                continue
            new_d = dict(d)
            new_d['sim'] = sim
            filtered.append(new_d)

        # 유사도 내림차순(하드 네거티브 우선)
        filtered.sort(key=lambda x: x['sim'], reverse=True)

        # 중복 제거(서로 너무 유사한 cf_sentence 제거)
        selected: List[Dict] = []
        selected_vecs: List[torch.Tensor] = []
        for d in filtered:
            c = d['cf_sentence']
            c_vec = vec(c)
            is_dup = False
            for sv in selected_vecs:
                if F.cosine_similarity(c_vec, sv, dim=0).item() >= dedupe_threshold:
                    is_dup = True
                    break
            if is_dup:
                continue
            selected.append(d)
            selected_vecs.append(c_vec)
            if len(selected) >= max_neg_per_anchor:
                break

        if not selected:
            continue

        results.append({
            'anchor': a,
            'original_id': anchor_id,
            'negatives': [
                {
                    'text': d['cf_sentence'],
                    'cf_type': d.get('cf_type'),
                    'sim': d.get('sim'),
                    'original_prediction': d.get('original_prediction'),
                    'cf_prediction': d.get('cf_prediction'),
                }
                for d in selected
            ]
        })
        kept_total += len(selected)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved negatives: anchors {len(results)}, total negatives {kept_total} -> {out_path}")


if __name__ == '__main__':
    select_negatives()


