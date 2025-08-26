import json
import os
from typing import Dict, List, Any


def norm_text(s: Any) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def norm_label(x: Any) -> str:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return 'B1' if int(x) == 0 else 'B2' if int(x) == 1 else None
    xs = str(x).strip().upper()
    if xs in ('0', 'B1'): return 'B1'
    if xs in ('1', 'B2'): return 'B2'
    return xs


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, data: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _coerce_to_list_of_dicts(raw: Any) -> List[Dict]:
    """Try to coerce various JSON shapes into a flat list[dict]."""
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        # common containers
        for key in ('part1_selected_sentences', 'items', 'data', 'examples', 'records'):
            if key in raw and isinstance(raw[key], list):
                return [r for r in raw[key] if isinstance(r, dict)]
        # fallback: gather all list values
        acc = []
        for v in raw.values():
            if isinstance(v, list):
                acc.extend([r for r in v if isinstance(r, dict)])
        if acc:
            return acc
    return []


def normalize_records(raw: Any) -> List[Dict]:
    out: List[Dict] = []
    rows = _coerce_to_list_of_dicts(raw)
    if not rows:
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        oid = r.get('original_id')
        # sentence keys
        o_sent = r.get('original_sentence') or r.get('anchor') or r.get('original')
        # Case A: flat record with cf_* fields
        cf_sent_flat = (
            r.get('cf_sentence') or r.get('counterfactual_sentence') or r.get('cf')
            or r.get('positive') or r.get('negative')
        )
        # label keys
        o_pred = r.get('original_prediction') or r.get('anchor_label') or r.get('original_label')

        # Case B: nested counterfactuals list (loop_1/2/3 data format)
        if isinstance(r.get('counterfactuals'), list) and r['counterfactuals']:
            for cf in r['counterfactuals']:
                cf_sent = cf.get('sentence')
                cf_pred = cf.get('prediction')
                cf_type = cf.get('type')

                oid_str = str(oid) if oid is not None else None
                o_sent_n = norm_text(o_sent)
                cf_sent_n = norm_text(cf_sent)
                o_pred_n = norm_label(o_pred)
                cf_pred_n = norm_label(cf_pred)
                if not oid_str or not o_sent_n or not cf_sent_n:
                    continue
                out.append({
                    'original_id': oid_str,
                    'original_sentence': o_sent_n,
                    'original_prediction': o_pred_n,
                    'cf_sentence': cf_sent_n,
                    'cf_prediction': cf_pred_n,
                    'cf_type': cf_type,
                })
            continue

        # Case A handling (flat)
        cf_pred_flat = (
            r.get('cf_prediction') or r.get('positive_prediction') or r.get('negative_prediction')
            or r.get('cf_label')
        )
        cf_type = r.get('cf_type') or r.get('type')

        oid_str = str(oid) if oid is not None else None
        o_sent_n = norm_text(o_sent)
        cf_sent_n = norm_text(cf_sent_flat)
        o_pred_n = norm_label(o_pred)
        cf_pred_n = norm_label(cf_pred_flat)
        if not oid_str or not o_sent_n or not cf_sent_n:
            continue
        out.append({
            'original_id': oid_str,
            'original_sentence': o_sent_n,
            'original_prediction': o_pred_n,
            'cf_sentence': cf_sent_n,
            'cf_prediction': cf_pred_n,
            'cf_type': cf_type,
        })
    return out


def normalize_all():
    srcs = [
        ('data/Loop/loop_1/loop_1_data.json', 'data/Loop/loop_1/loop_1_data_normalized.json'),
        ('data/Loop/loop_2/loop_2_data.json', 'data/Loop/loop_2/loop_2_data_normalized.json'),
        ('data/Loop/loop_3/loop_3_data.json', 'data/Loop/loop_3/loop_3_data_normalized.json'),
    ]

    merged: List[Dict] = []
    for src, dst in srcs:
        try:
            raw = load_json(src)
        except Exception as e:
            print(f"❌ Load failed: {src} -> {e}")
            continue
        normed = normalize_records(raw)
        save_json(dst, normed)
        print(f"✅ Normalized {src} -> {dst} ({len(normed)} records)")
        # add loop tag inferred from path
        loop_num = 1 if 'loop_1' in dst or 'loop_1_data' in src else 2 if 'loop_2' in dst else 3
        for rec in normed:
            rec2 = dict(rec)
            rec2['loop'] = loop_num
            merged.append(rec2)

    save_json('data/Loop/loop_123_data_normalized_merged.json', merged)
    print(f"💾 Merged normalized -> data/Loop/loop_123_data_normalized_merged.json ({len(merged)} records)")


if __name__ == '__main__':
    normalize_all()


