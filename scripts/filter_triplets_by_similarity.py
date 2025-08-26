import json
import os
from typing import Any, Dict, List, Tuple
import argparse

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def norm_label(x: Any) -> int:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = int(x)
        return 0 if v == 0 else 1 if v == 1 else None
    xs = str(x).strip().upper()
    if xs in ("0", "B1"): return 0
    if xs in ("1", "B2"): return 1
    return None


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@torch.no_grad()
def embed_texts(texts: List[str], model: SentenceTransformer) -> torch.Tensor:
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=256)


def filter_triplets(
    src_path: str = 'data/Loop/triplets_loops123_from_rules.json',
    out_path: str = 'data/Loop/triplets_loops123_filtered.json',
    # 느슨한 기본값: 양 유지 우선, 명백한 노이즈만 제거
    s_pos_min_strict: float = 0.70,   # 양이 너무 낮은 양성만 컷
    s_neg_max_strict: float = 0.97,   # 음성이 앵커와 거의 동일할 때 컷
    gap_min_strict: float = -0.02,    # 음성이 양성보다 약간 높아도 허용, 심각한 역전만 컷
    topk_neg_per_anchor: int = 8,     # 앵커당 더 많은 네거티브 유지
    dedupe_threshold: float = 0.995,  # 거의 동일한 문장만 중복 제거
    mode: str = 'light',              # 'ultra' | 'light' | 'strict'
):
    data: List[Dict] = load_json(src_path)
    print(f"Loaded triplets: {len(data)}")

    # 전처리: 모드별 완화 수준
    cleaned: List[Dict] = []
    for t in data:
        a = norm_text(t.get('anchor'))
        p = norm_text(t.get('positive'))
        n = norm_text(t.get('negative'))
        if not (a and p and n):
            continue
        if mode == 'ultra':
            # 텍스트만 유효하면 통과 (라벨/동일성 제약 없음)
            t2 = dict(t)
            t2['_a'] = a; t2['_p'] = p; t2['_n'] = n
            cleaned.append(t2)
        else:
            al = norm_label(t.get('anchor_label') or t.get('original_prediction'))
            pl = norm_label(t.get('positive_label') or t.get('positive_prediction'))
            nl = norm_label(t.get('negative_label') or t.get('negative_prediction'))
            # anchor 라벨만 필수, p/n 라벨은 있으면만 검증
            if al is None:
                continue
            if (pl is not None and al != pl):
                continue
            if (nl is not None and al == nl):
                continue
            if a == p or a == n or p == n:
                continue
            t2 = dict(t)
            t2['_a'] = a; t2['_p'] = p; t2['_n'] = n
            t2['_al'] = al; t2['_pl'] = pl; t2['_nl'] = nl
            cleaned.append(t2)
    print(f"After basic validation: {len(cleaned)}")

    if not cleaned:
        print("No valid triplets after basic checks. Abort.")
        return

    if mode in ('ultra','light'):
        # SBERT 계산 없이 수량 보존: 유사도 컷 생략, 이후 단계에서 중복만 억제
        qualified = cleaned
        print(f"Light mode: skipping similarity thresholds. Qualified = {len(qualified)}")
    else:
        # SBERT 임베딩
        model = SentenceTransformer('all-MiniLM-L6-v2')
        uniq_texts: List[str] = []
        idx: Dict[str, int] = {}
        for t in cleaned:
            for s in (t['_a'], t['_p'], t['_n']):
                if s not in idx:
                    idx[s] = len(uniq_texts)
                    uniq_texts.append(s)
        vecs = embed_texts(uniq_texts, model)

        def v(s: str) -> torch.Tensor:
            return vecs[idx[s]]

        # 유사도 계산 및 완화 필터(명백한 노이즈만 제거)
        qualified: List[Dict] = []
        bad_counts = {"s_pos":0, "s_neg":0, "gap":0}
        for t in tqdm(cleaned, desc='Scoring'):
            a_v, p_v, n_v = v(t['_a']), v(t['_p']), v(t['_n'])
            s_pos = float(F.cosine_similarity(a_v, p_v, dim=0).item())
            s_neg = float(F.cosine_similarity(a_v, n_v, dim=0).item())
            gap = s_pos - s_neg
            # 명백한 노이즈 컷만 적용
            if s_pos < s_pos_min_strict:
                bad_counts["s_pos"] += 1; continue
            if s_neg > s_neg_max_strict:
                bad_counts["s_neg"] += 1; continue
            if gap < gap_min_strict:
                bad_counts["gap"] += 1; continue
            # 점수는 내부에서만 사용하고, 출력 스키마에는 포함하지 않음
            t['__s_pos'] = s_pos; t['__s_neg'] = s_neg; t['__gap'] = gap
            qualified.append(t)
        print(f"Qualified: {len(qualified)} | drop s_pos:{bad_counts['s_pos']} s_neg:{bad_counts['s_neg']} gap:{bad_counts['gap']}")

    if not qualified:
        print("No triplets pass similarity thresholds. Abort.")
        return

    # 앵커별 네거티브 정리(상위 K 유지 + 강한 중복만 제거)
    from collections import defaultdict
    by_anchor: Dict[Tuple[str,int], List[Dict]] = defaultdict(list)
    for t in qualified:
        key = (t['_a'], t.get('_al'))
        by_anchor[key].append(t)

    filtered: List[Dict] = []
    for key, items in by_anchor.items():
        # s_neg 높은 순(하드 네거티브) 정렬: light 모드에서는 원순서 유지
        if mode in ('ultra','light'):
            items_sorted = items
        else:
            items_sorted = sorted(items, key=lambda x: x['__s_neg'], reverse=True)
        selected: List[Dict] = []
        sel_vecs: List[torch.Tensor] = []
        seen_negs: set = set()
        for it in items_sorted:
            # 중복 제거
            if mode in ('ultra','light'):
                # 문자열 동일성 기반 중복만 제거
                if it['_n'] in seen_negs:
                    continue
                seen_negs.add(it['_n'])
            else:
                n_vec = v(it['_n'])
                dup = False
                for sv in sel_vecs:
                    if float(F.cosine_similarity(n_vec, sv, dim=0).item()) >= dedupe_threshold:
                        dup = True; break
                if dup:
                    continue
                sel_vecs.append(n_vec)
            selected.append(it)
            # ultra 모드: 상한 없음, light/strict: 상한 적용
            if mode != 'ultra':
                if len(selected) >= topk_neg_per_anchor:
                    break
        # 원래 스키마로 투영(원본 키 그대로, 추가 키 제거)
        for it in selected:
            # 원래 값으로 덮어쓰기
            it['anchor'] = it['_a']
            it['positive'] = it['_p']
            it['negative'] = it['_n']
            if 'anchor_label' not in it and it.get('_al') is not None:
                it['anchor_label'] = it['_al']
            if 'positive_label' not in it and it.get('_pl') is not None:
                it['positive_label'] = it['_pl']
            if 'negative_label' not in it and it.get('_nl') is not None:
                it['negative_label'] = it['_nl']
            # 임시/스코어 키 제거하여 원본 스키마 유지
            for k in list(it.keys()):
                if k.startswith('_') or k.startswith('__'):
                    it.pop(k, None)
                if k in ('s_pos','s_neg','gap'):
                    it.pop(k, None)
            out = it
            filtered.append(out)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"Saved filtered triplets: {len(filtered)} -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='data/Loop/triplets_loops123_from_rules.json')
    parser.add_argument('--out_path', type=str, default='data/Loop/triplets_loops123_filtered.json')
    parser.add_argument('--mode', type=str, default='ultra', choices=['ultra','light','strict'])
    args = parser.parse_args()
    filter_triplets(src_path=args.src_path, out_path=args.out_path, mode=args.mode)


