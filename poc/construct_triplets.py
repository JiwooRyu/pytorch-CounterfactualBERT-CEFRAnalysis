import json
import torch
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from tqdm import tqdm

# File paths
PRED_PATH = 'outputs/analysis/counterfactual_predictions_300.json'
OUT_PATH = 'poc/counterfactual_triplets.jsonl'

# Load data
with open(PRED_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Load BERT
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').to(device)
bert.eval()

# Helper: get [CLS] embedding
@torch.no_grad()
def get_cls_emb(text):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
    outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)  # [768]

triplets = []

for item in tqdm(data, desc='Constructing triplets'):
    anchor = item['original_sentence']
    anchor_emb = get_cls_emb(anchor)
    positives = []
    negatives = []
    for cf in item['counterfactuals']:
        cf_sent = cf['sentence']
        is_flipped = cf['is_label_flipped']
        cf_emb = get_cls_emb(cf_sent)
        sim = F.cosine_similarity(anchor_emb, cf_emb, dim=0).item()
        if is_flipped:
            negatives.append((sim, cf_sent))
        else:
            positives.append((sim, cf_sent))
    if positives and negatives:
        # 가장 유사한 positive, hard negative 선택
        pos_sent = max(positives, key=lambda x: x[0])[1]
        neg_sent = max(negatives, key=lambda x: x[0])[1]
        triplets.append({
            'anchor': anchor,
            'positive': pos_sent,
            'hard_negative': neg_sent
        })

# Save as JSONL
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    for triplet in triplets:
        f.write(json.dumps(triplet, ensure_ascii=False) + '\n')

print(f"Saved {len(triplets)} triplets to {OUT_PATH}") 