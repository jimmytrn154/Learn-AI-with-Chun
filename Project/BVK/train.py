import os
import warnings
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter

from src.dataset import CMMDDataset
from src.models import MammoAnalyzeModel

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.alpha = alpha 

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_tensor = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)
            alpha_t = alpha_tensor[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

def freeze_backbones(model):
    print("\n" + "="*60)
    print(" ❄️ KÍCH HOẠT PEFT & KIỂM TRA TRỌNG SỐ MÔ HÌNH ")
    print("="*60)
    
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    
    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
        for param in model.text_encoder.parameters():
            param.requires_grad = False
            
    modules_to_train = [
        getattr(model, 'graph_learner', None), 
        getattr(model, 'adapter', None), 
        getattr(model, 'text_proj', None), 
        getattr(model, 'classifier', None),
        getattr(model, 'late_fusion', None)
    ]
    
    for module in modules_to_train:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True
                
    if hasattr(model, 'logit_scale'):
        model.logit_scale.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"   - Tổng số tham số (Total)    : {total_params:,}")
    print(f"   - Tham số BỊ KHÓA (Xương sống): {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print(f"   - Tham số ĐƯỢC HỌC (Adapter)  : {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print("="*60 + "\n")

def init_base_img_features(model, train_loader, device):
    print("🔄 Đang trích xuất base_img_features cho GraphLearner từ tập Train...")
    model.eval()
    features_dict = {0: [], 1: [], 2: [], 3: []}
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting features", leave=False):
            imgs = batch['image'].to(device)
            labels = batch['label'].numpy()
            
            # Đưa qua Vision Backbone
            v_feat = F.adaptive_avg_pool2d(model.vision_encoder.extract_features(imgs), (1, 1)).flatten(1)
            img_feat = F.normalize(model.visual_projection(v_feat), dim=-1) # [B, 512]
            
            for i, label in enumerate(labels):
                if label in features_dict:
                    features_dict[label].append(img_feat[i].cpu())

    base_img_features = []
    for c in range(4):
        if len(features_dict[c]) > 0:
            class_feat = torch.stack(features_dict[c]).mean(dim=0)
            class_feat = F.normalize(class_feat, dim=-1)
            base_img_features.append(class_feat)
        else:
            base_img_features.append(torch.randn(512)) # Fallback an toàn nếu thiếu class

    print("✅ Đã hoàn tất trích xuất đặc trưng Graph ảnh!")
    return torch.stack(base_img_features).to(device)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_dir = f"runs/Task_{args.task_id}_{args.exp_name}/{args.run_id}"
    ckpt_dir = f"checkpoints/Task_{args.task_id}_{args.exp_name}/{args.run_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    tokenizer = AutoTokenizer.from_pretrained("/home/minhntn24/nhatminh/M2G-BRCA/data/huggingface_models/Bio_ClinicalBERT")
    
    # Thiết lập DataLoader sử dụng Monte Carlo Split
    train_loader = DataLoader(
        CMMDDataset(csv_file=args.data_csv, img_dir=args.img_dir, split_file=args.split_file, 
                    run_id=args.run_id, tokenizer=tokenizer, mode='train'), 
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        CMMDDataset(csv_file=args.data_csv, img_dir=args.img_dir, split_file=args.split_file, 
                    run_id=args.run_id, tokenizer=tokenizer, mode='val'), 
        batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Khởi tạo mô hình
    model = MammoAnalyzeModel(task_id=args.task_id, mammoclip_path=args.mammoclip_weights).to(device)
    
    # Nạp base_img_features động vào GraphLearner cho Task 2, 3
    if args.task_id in [2, 3]:
        real_img_features = init_base_img_features(model, train_loader, device)
        model.graph_learner.base_img_features.data = real_img_features

    freeze_backbones(model)
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    class_weights = [1.2, 0.5, 1.2, 2.6]
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, smoothing=0.1)
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_auc = 0.0
    epochs_no_improve = 0
    early_stop_patience = 50

    print(f"🚀 Bắt đầu huấn luyện {args.run_id} (Task {args.task_id}) | LR: {args.lr}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{args.epochs} [Train]", leave=False)
        for batch in pbar:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            ids = batch.get('input_ids', torch.zeros(1)).to(device)
            mask = batch.get('attention_mask', torch.zeros(1)).to(device)

            optimizer.zero_grad()
            logits = model(imgs, ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_probs, all_preds, all_labels = [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                labels = batch['label'].to(device)
                ids = batch.get('input_ids', torch.zeros(1)).to(device)
                mask = batch.get('attention_mask', torch.zeros(1)).to(device)

                logits = model(imgs, ids, mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader)
        
        try:
            val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            val_auc = 0.0
            
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = accuracy_score(all_labels, all_preds)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Metrics/AUC', val_auc, epoch)
        writer.add_scalar('Metrics/F1', val_f1, epoch)
        writer.add_scalar('Metrics/Accuracy', val_acc, epoch)

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Ep {epoch:03d} | LR: {current_lr:.1e} | Tr.Loss: {train_loss:.4f} | Val.Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f} | Acc: {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            print(f" >>> 🏆 ĐỈNH CAO MỚI: {best_auc:.4f} (Đã lưu Model!)")
            save_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc
            }, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"\n 🛑 KÍCH HOẠT EARLY STOPPING: Đã {early_stop_patience} Epochs không có sự đột phá.")
                break 

    writer.close()
    print(f"✅ Hoàn tất {args.run_id}. Best AUC đạt được: {best_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Đường dẫn đã được chuẩn hóa theo hệ thống
    parser.add_argument('--data_csv', type=str, default="/home/minhntn24/nhatminh/M2G-BRCA/test_mammo_clip/processed_cmmd.csv")
    parser.add_argument('--img_dir', type=str, default="/home/minhntn24/nhatminh/M2G-BRCA/test_mammo_clip/images")
    parser.add_argument('--split_file', type=str, default="/home/minhntn24/nhatminh/M2G-BRCA/data/monte_carlo_split.csv")
    parser.add_argument('--mammoclip_weights', type=str, default="/mnt/disk1/backup_user/minh.ntn/Mammo-CLIP/Pre-trained-checkpoints/b5-model-best-epoch-7.tar")
    parser.add_argument('--task_id', type=int, default=3)
    parser.add_argument('--exp_name', type=str, default="CLASS_WEIGHT_FIX") 
    parser.add_argument('--run_id', type=str, default="run_0")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5) 
    args = parser.parse_args()
    train(args)