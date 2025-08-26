import os
import json
import sys
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import argparse
import contextlib

# ensure project root on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from joint_contrastive_training import (
    JointBertModel,
    TripletDataset,
    ClassificationDataset,
    train_joint,
    evaluate,
)


def save_subset(data: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def eval_metrics(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float, List[List[int]], List[float]]:
    # Local eval (accuracy, macro-f1, confusion matrix, per-class acc)
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    model.eval()
    preds: List[int] = []
    trues: List[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()
            _, logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels)
    acc = float(accuracy_score(trues, preds))
    macro_f1 = float(f1_score(trues, preds, average='macro'))
    cm = confusion_matrix(trues, preds, labels=[0,1]).tolist()
    per_class_acc = []
    for i in range(2):
        row_sum = sum(cm[i])
        per_class_acc.append(float(cm[i][i] / row_sum) if row_sum > 0 else 0.0)
    return acc, macro_f1, cm, per_class_acc


def run_once(results_path: str = 'runs/ablation_results.csv'):
    base_triplet_path = 'data/Loop/triplets_loops123_filtered.json'
    with open(base_triplet_path, 'r', encoding='utf-8') as f:
        all_data: List[Dict] = json.load(f)

    # 분할: loop 1, loop 1+2, all
    l1 = [t for t in all_data if int(t.get('loop', 0)) == 1]
    l12 = [t for t in all_data if int(t.get('loop', 0)) in (1, 2)]
    l123 = all_data

    out_dir = 'data/Loop/ablation'
    p1 = os.path.join(out_dir, 'triplets_loop1.json')
    p12 = os.path.join(out_dir, 'triplets_loop12.json')
    p123 = os.path.join(out_dir, 'triplets_loop123.json')
    save_subset(l1, p1)
    save_subset(l12, p12)
    save_subset(l123, p123)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 분류 데이터 로더 (공통)
    train_cls_dataset = ClassificationDataset('data/processed/train.csv', tokenizer)
    test_cls_dataset = ClassificationDataset('data/processed/test.csv', tokenizer)
    train_cls_loader = DataLoader(train_cls_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_cls_loader = DataLoader(test_cls_dataset, batch_size=32, shuffle=False)

    # 0) Baseline: Contrastive 없이 분류만 학습 (조건 통일)
    print('================= baseline_cls_only =================')
    baseline_model = JointBertModel()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = AdamW(baseline_model.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = GradScaler(enabled=(device == 'cuda'))
    baseline_model.to(device)
    total_steps = len(train_cls_loader) * 10  # epochs=10, accumulate_steps=1
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    accumulate_steps = 1  # joint와 동일
    for epoch in range(10):  # baseline epochs 동일
        # suppress verbose epoch logs
        with open(os.devnull, 'w') as _null, contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
            baseline_model.train()
            running = 0.0
            for step, batch in enumerate(train_cls_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                with autocast(enabled=(device == 'cuda')):
                    _, logits = baseline_model(input_ids, attention_mask)
                    loss = ce_loss(logits, labels)
                scaler.scale(loss / accumulate_steps).backward()
                if (step + 1) % accumulate_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                running += loss.item()
    torch.save(baseline_model.state_dict(), 'best_joint_bert_baseline.pt')
    print('Saved baseline model -> best_joint_bert_baseline.pt')
    # concise metrics
    b_acc, b_f1, b_cm, b_pacc = eval_metrics(baseline_model, test_cls_loader, device)

    configs = [
        ('loop1_only', p1, 'best_joint_bert_loop1.pt'),
        ('loop12', p12, 'best_joint_bert_loop12.pt'),
        ('loop123', p123, 'best_joint_bert_loop123.pt'),
    ]

    loop_results: List[Tuple[str, float, float, List[List[int]], List[float]]] = []

    for tag, trip_path, ckpt in configs:
        print(f'================= {tag} =================')
        # Triplet 로더 (strict=True로 라벨 일관성 필터 활성화)
        triplet_dataset = TripletDataset(trip_path, tokenizer)
        # Note: joint_contrastive_training.py의 TripletDataset은 이미 라벨 일관성 필터가 활성화됨
        triplet_loader = DataLoader(triplet_dataset, batch_size=16, shuffle=True, drop_last=True)

        # 모델 생성 및 학습
        model = JointBertModel()
        # suppress epoch/progress logs during joint training
        with open(os.devnull, 'w') as _null, contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
            train_joint(
                model, triplet_loader, train_cls_loader, device,
                epochs=10, alpha=0.6, lr=2e-5, margin=0.3,
            )

        # 저장 및 평가
        torch.save(model.state_dict(), ckpt)
        print(f'Saved model -> {ckpt}')
        acc, f1, cm, pacc = eval_metrics(model, test_cls_loader, device)
        # 간결 출력
        print(f'[{tag}] acc={acc:.4f} macro_f1={f1:.4f} cm={cm} per_class_acc={pacc}')
        loop_results.append((tag, acc, f1, cm, pacc))

    # 결과 기록 (간결 형식 + 혼동행렬/클래스별 acc)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    header_needed = not os.path.exists(results_path)
    with open(results_path, 'a', encoding='utf-8') as f:
        if header_needed:
            f.write('timestamp,split,acc,macro_f1,cm_00,cm_01,cm_10,cm_11,cls_acc_0,cls_acc_1,baseline_acc,baseline_macro_f1,base_cm_00,base_cm_01,base_cm_10,base_cm_11,base_cls_acc_0,base_cls_acc_1\n')
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for tag, acc, f1, cm, pacc in loop_results:
            f.write(
                f"{ts},{tag},{acc:.4f},{f1:.4f},{cm[0][0]},{cm[0][1]},{cm[1][0]},{cm[1][1]},{pacc[0]:.4f},{pacc[1]:.4f},"
                f"{b_acc:.4f},{b_f1:.4f},{b_cm[0][0]},{b_cm[0][1]},{b_cm[1][0]},{b_cm[1][1]},{b_pacc[0]:.4f},{b_pacc[1]:.4f}\n"
            )
    print(f'Appended concise results to {results_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', action='store_true', help='Repeat indefinitely until runs/STOP exists')
    parser.add_argument('--interval', type=int, default=0, help='Seconds to sleep between repeats')
    args = parser.parse_args()

    if not args.repeat:
        run_once()
        return

    print('Repeat mode: will run until runs/STOP exists')
    while True:
        if os.path.exists('runs/STOP'):
            print('STOP file detected, exiting repeat loop')
            break
        run_once()
        if args.interval > 0:
            import time
            time.sleep(args.interval)


if __name__ == '__main__':
    main()


