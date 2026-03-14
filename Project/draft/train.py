import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score # <--- MỚI
import logging
import os

from src.dataset import CMMDDataset
from src.models import MammoAnalyzeModel
from src.utils import set_seed, save_checkpoint, compute_metrics

def setup_logger(task_id, shots, run_id):
    suffix = f"_{shots}shot_{run_id}" if shots else f"_full_{run_id}"
    log_file = f"log_task_{task_id}{suffix}.txt"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger()

def train_one_epoch(model, loader, optimizer, criterion, device, task_id):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        input_ids = None
        attention_mask = None
        if task_id == 3:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, task_id):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            input_ids = None
            attention_mask = None
            if task_id == 3:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
            outputs = model(images, input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1) 
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Tính các metrics cơ bản (Acc, AUC) từ utils
    metrics = compute_metrics(all_labels, all_probs)
    
    # --- [MỚI] TÍNH THÊM BALANCED ACCURACY ---
    y_pred = np.argmax(all_probs, axis=1)
    metrics['Balanced_Accuracy'] = balanced_accuracy_score(all_labels, y_pred)
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mammoclip_path', type=str, default=None)
    parser.add_argument('--data_csv', type=str, default='./data/cmmd_metadata.csv')
    parser.add_argument('--data_root', type=str, default='./data/CMMD')
    parser.add_argument('--shots', type=int, default=None)
    parser.add_argument('--split_file', type=str, default='./monte_carlo_splits.csv')
    parser.add_argument('--run_id', type=str, default='run_0')
    
    args = parser.parse_args()

    set_seed(42)
    logger = setup_logger(args.task, args.shots, args.run_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mode_str = f"{args.shots}-SHOT ({args.run_id})" if args.shots else f"FULL DATA ({args.run_id})"
    logger.info(f"STARTING TASK {args.task} in [{mode_str}] MODE")

    tokenizer = None
    if args.task == 3:
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    train_dataset = CMMDDataset(
        args.data_csv, args.data_root, tokenizer, 
        task_id=args.task, mode='train', shots=args.shots,
        split_file=args.split_file, run_id=args.run_id
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_dataset = CMMDDataset(
        args.data_csv, args.data_root, tokenizer, 
        task_id=args.task, mode='test', shots=None,
        split_file=args.split_file, run_id=args.run_id
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MammoAnalyzeModel(task_id=args.task, num_classes=len(train_dataset.label_map), mammoclip_path=args.mammoclip_path).to(device)

    backbone_ids = list(map(id, model.vision_encoder.parameters()))
    if hasattr(model, 'text_encoder'): backbone_ids += list(map(id, model.text_encoder.parameters()))
    backbone_params = [p for n, p in model.named_parameters() if id(p) in backbone_ids]
    head_params = [p for n, p in model.named_parameters() if id(p) not in backbone_ids]
            
    optimizer = optim.AdamW([{'params': backbone_params, 'lr': 1e-6}, {'params': head_params, 'lr': args.lr}], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    y_all = [train_dataset.label_map[str(label).strip()] for label in train_dataset.data['subtype']]
    class_weights = compute_class_weight('balanced', classes=np.unique(y_all), y=y_all)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.task)
        metrics = evaluate(model, test_loader, device, args.task)
        scheduler.step(loss)
        
        current_lr = optimizer.param_groups[1]['lr']
        log_msg = f"Ep {epoch+1} | Loss: {loss:.4f} | Acc: {metrics['Accuracy']:.4f} | BalAcc: {metrics['Balanced_Accuracy']:.4f} | AUC: {metrics['AUC']:.4f} | LR: {current_lr:.1e}"
        print(log_msg)
        logger.info(log_msg)
        
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(model, optimizer, epoch, args.task)
            logger.info(">> Best model saved!")

if __name__ == "__main__":
    main()