import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_triplets(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    req = {"anchor", "positive", "negative"}
    data = [d for d in data if isinstance(d, dict) and req.issubset(d.keys())]
    return data


@torch.no_grad()
def predict_labels(texts: List[str], tokenizer, model, device: str, batch_size: int = 64) -> Tuple[List[str], List[float]]:
    labels: List[str] = []
    confs: List[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Labeling"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        logits = out.logits
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        conf = probs.max(dim=-1).values
        # id->label: assume index 0->B1, 1->B2 if config id2label exists use it
        if hasattr(model.config, 'id2label') and model.config.id2label:
            id2label = model.config.id2label
            batch_labels = [id2label[int(p)] for p in pred]
        else:
            batch_labels = ["B1" if int(p)==0 else "B2" for p in pred]
        labels.extend(batch_labels)
        confs.extend(conf.detach().cpu().tolist())
    return labels, confs


@torch.no_grad()
def embed(texts: List[str], sbert, batch_size: int = 128) -> torch.Tensor:
    vecs = sbert.encode(texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
    return vecs


def rebuild(
    in_path: str = 'data/Loop/loop_3/accumulated_loop2_3.json',
    out_path: str = 'data/Loop/loop_3/accumulated_loop2_3_labeled.json',
    min_conf: float = 0.60,
    topk_per_anchor: int = 4,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    triplets = load_triplets(in_path)
    print(f"Loaded {len(triplets)} raw triplets")

    # Collect unique texts
    uniq_texts = []
    index: Dict[str, int] = {}
    for t in triplets:
        for k in ("anchor", "positive", "negative"):
            s = t[k]
            if s not in index:
                index[s] = len(uniq_texts)
                uniq_texts.append(s)

    # Load models
    clf_path = 'models/contrastive_bert'
    tokenizer = AutoTokenizer.from_pretrained(clf_path)
    clf_model = AutoModelForSequenceClassification.from_pretrained(clf_path).to(device)
    clf_model.eval()
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Predict labels/conf
    uniq_labels, uniq_confs = predict_labels(uniq_texts, tokenizer, clf_model, device)

    # Embeddings for similarity
    uniq_vecs = embed(uniq_texts, sbert)

    # Helper getters
    def lab(s: str) -> str:
        return uniq_labels[index[s]]
    def conf(s: str) -> float:
        return float(uniq_confs[index[s]])
    def vec(s: str) -> torch.Tensor:
        return uniq_vecs[index[s]]

    # Group by anchor and select top-k consistent triplets
    from collections import defaultdict
    by_anchor = defaultdict(list)
    for t in triplets:
        a, p, n = t['anchor'], t['positive'], t['negative']
        o_lab, p_lab, n_lab = lab(a), lab(p), lab(n)
        o_conf, p_conf, n_conf = conf(a), conf(p), conf(n)
        # consistency rule: o==p, o!=n, min confidence
        if o_lab != p_lab:
            continue
        if o_lab == n_lab:
            continue
        if min(o_conf, p_conf, n_conf) < min_conf:
            continue
        # quality score: similarity gap + confidence gap
        sim_pos = F.cosine_similarity(vec(a), vec(p), dim=0).item()
        sim_neg = F.cosine_similarity(vec(a), vec(n), dim=0).item()
        gap = sim_pos - sim_neg
        conf_gap = (p_conf - n_conf)
        score = gap + 0.2 * conf_gap
        by_anchor[a].append({
            'anchor': a,
            'positive': p,
            'negative': n,
            'original_prediction': o_lab,
            'positive_prediction': p_lab,
            'negative_prediction': n_lab,
            'o_conf': o_conf,
            'p_conf': p_conf,
            'n_conf': n_conf,
            'sim_pos': sim_pos,
            'sim_neg': sim_neg,
            'gap': gap,
            'score': score,
        })

    rebuilt: List[Dict] = []
    for a, items in by_anchor.items():
        items.sort(key=lambda x: x['score'], reverse=True)
        rebuilt.extend(items[:topk_per_anchor])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(rebuilt, f, ensure_ascii=False, indent=2)
    print(f"Saved rebuilt triplets: {len(rebuilt)} -> {out_path}")


if __name__ == '__main__':
    rebuild()


