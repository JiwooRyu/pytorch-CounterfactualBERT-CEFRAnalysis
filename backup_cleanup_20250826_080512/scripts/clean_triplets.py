import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def load_triplets(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 필수 키 보장
    required = {"anchor", "positive", "negative"}
    data = [d for d in data if isinstance(d, dict) and required.issubset(d.keys())]
    return data


@torch.no_grad()
def embed_sentences(sentences: List[str], tokenizer: BertTokenizer, model: BertModel, device: str, batch_size: int = 64) -> torch.Tensor:
    all_vecs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Embedding"):
        batch = sentences[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        cls = out.last_hidden_state[:, 0, :]
        all_vecs.append(cls.cpu())
    return torch.cat(all_vecs, dim=0)


def build_embedding_cache(triplets: List[Dict], tokenizer: BertTokenizer, model: BertModel, device: str) -> Dict[str, torch.Tensor]:
    uniq = []
    seen = set()
    for t in triplets:
        for key in ("anchor", "positive", "negative"):
            s = t[key]
            if s not in seen:
                seen.add(s)
                uniq.append(s)
    vecs = embed_sentences(uniq, tokenizer, model, device)
    cache: Dict[str, torch.Tensor] = {}
    for s, v in zip(uniq, vecs):
        cache[s] = v
    return cache


def compute_scores(a: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> Tuple[float, float, float]:
    a = F.normalize(a.float(), p=2, dim=0)
    p = F.normalize(p.float(), p=2, dim=0)
    n = F.normalize(n.float(), p=2, dim=0)
    s_pos = torch.dot(a, p).item()
    s_neg = torch.dot(a, n).item()
    gap = s_pos - s_neg
    return s_pos, s_neg, gap


def clean_triplets(
    in_path: str = 'data/Loop/loop_3/accumulated_loop2_3.json',
    out_path: str = 'data/Loop/loop_3/accumulated_loop2_3_clean.json',
    tau_pos: float = 0.35,
    tau_neg: float = 0.70,
    tau_gap: float = 0.03,
    topk_per_anchor: int = 3,
    min_backup_gap: float = -0.05,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()

    triplets = load_triplets(in_path)
    print(f"Loaded {len(triplets)} triplets")

    # 중복/동일문장 제거를 위한 키
    def norm_text(s: str) -> str:
        return ' '.join(s.strip().split())

    seen_keys = set()
    prelim = []
    for t in triplets:
        a = norm_text(t['anchor']); p = norm_text(t['positive']); n = norm_text(t['negative'])
        # 동일문장 제거
        if a == p or a == n or p == n:
            continue
        key = (a, p, n)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        nt = dict(t)
        nt['anchor'] = a; nt['positive'] = p; nt['negative'] = n
        prelim.append(nt)

    print(f"After dups/self-eq filter: {len(prelim)}")

    # 임베딩 캐시
    cache = build_embedding_cache(prelim, tokenizer, model, device)

    # 1) 점수 계산
    scored = []
    for t in tqdm(prelim, desc='Scoring/Filtering'):
        a_v = cache[t['anchor']]
        p_v = cache[t['positive']]
        n_v = cache[t['negative']]
        s_pos, s_neg, gap = compute_scores(a_v, p_v, n_v)
        nt = dict(t)
        nt['s_pos'] = s_pos
        nt['s_neg'] = s_neg
        nt['gap'] = gap
        scored.append(nt)

    # 2) 앵커별 상위 k개 선택(우선: 엄격 기준 충족, 부족하면 완화 기준으로 보충)
    from collections import defaultdict
    by_anchor: Dict[str, List[Dict]] = defaultdict(list)
    for t in scored:
        by_anchor[t['anchor']].append(t)

    cleaned: List[Dict] = []
    for anchor, items in by_anchor.items():
        items_sorted = sorted(items, key=lambda x: x['gap'], reverse=True)
        strict = [x for x in items_sorted if x['s_pos'] >= tau_pos and x['s_neg'] <= tau_neg and x['gap'] >= tau_gap]
        take = strict[:topk_per_anchor]
        if len(take) < topk_per_anchor:
            # 완화 기준으로 보충 (gap이 너무 음수인 것은 제외)
            backup = [x for x in items_sorted if x['gap'] >= min_backup_gap]
            # 이미 담긴 것 제외
            seen = set((x['anchor'], x['positive'], x['negative']) for x in take)
            for x in backup:
                key = (x['anchor'], x['positive'], x['negative'])
                if key in seen:
                    continue
                take.append(x)
                seen.add(key)
                if len(take) >= topk_per_anchor:
                    break
        cleaned.extend(take)
    kept = len(cleaned)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"Saved cleaned triplets: {kept} → {out_path}")


if __name__ == '__main__':
    clean_triplets()


