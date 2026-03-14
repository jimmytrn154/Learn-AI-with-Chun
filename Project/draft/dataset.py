import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
import pydicom

class CMMDDataset(Dataset):
    def __init__(self, csv_file, root_dir, tokenizer=None, max_len=128, mode='train', task_id=1, shots=None, seed=42, split_file=None, run_id='run_0'):
        self.data_raw = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.task_id = task_id
        
        # 1. Cleaning cơ bản
        self.data_raw = self.data_raw.dropna(subset=['subtype']).reset_index(drop=True)

        # 2. XỬ LÝ SPLIT TỪ FILE NGOÀI (Monte Carlo)
        if split_file is not None and os.path.exists(split_file):
            split_df = pd.read_csv(split_file)
            
            # Chuẩn hóa tên cột để Merge (Metadata thường dùng 'ID' hoặc 'Subject ID', Split dùng 'PatientID')
            # Mục tiêu: Tạo cột 'PatientID' trong data_raw để khớp với split_df
            if 'PatientID' not in self.data_raw.columns:
                if 'ID' in self.data_raw.columns:
                    self.data_raw['PatientID'] = self.data_raw['ID']
                elif 'Subject ID' in self.data_raw.columns:
                    self.data_raw['PatientID'] = self.data_raw['Subject ID']
            
            # Kiểm tra xem run_id có tồn tại không
            if run_id not in split_df.columns:
                raise ValueError(f"Run ID '{run_id}' không tồn tại trong file split!")

            # Merge: Chỉ giữ lại những bệnh nhân có trong file split
            # self.data_raw sẽ được thêm cột 'run_0' (hoặc run_x)
            self.data_raw = self.data_raw.merge(split_df[['PatientID', run_id]], on='PatientID', how='inner')
            
            # Lọc dữ liệu theo Mode dựa trên cột run_id
            if mode == 'train':
                # Lấy các mẫu được đánh dấu là 'train'
                self.data = self.data_raw[self.data_raw[run_id] == 'train'].reset_index(drop=True)
            elif mode == 'test':
                # Lấy các mẫu được đánh dấu là 'test'
                self.data = self.data_raw[self.data_raw[run_id] == 'test'].reset_index(drop=True)
            else:
                # Fallback cho val
                self.data = self.data_raw[self.data_raw[run_id] == 'val'].reset_index(drop=True)

        else:
            print("[WARNING] Không tìm thấy split file. Đang chia ngẫu nhiên 80/20 (Không khuyến khích).")
            self.data_raw = self.data_raw.sample(frac=1, random_state=seed).reset_index(drop=True)
            split_idx = int(0.8 * len(self.data_raw))
            if mode == 'train':
                self.data = self.data_raw.iloc[:split_idx].reset_index(drop=True)
            else:
                self.data = self.data_raw.iloc[split_idx:].reset_index(drop=True)

        # 3. LỌC FEW-SHOT (Chỉ áp dụng trên tập TRAIN đã lọc ở bước 2)
        if shots is not None and mode == 'train':
            print(f"--- [FEW-SHOT] Selecting {shots}-shot from Train Split ({run_id}) ---")
            self.data = self._create_few_shot_data(self.data, shots, seed)
            print(f"--- Final Train Size: {len(self.data)} samples ---")
        elif mode == 'test':
            print(f"--- [TEST] Evaluation on Test Split ({run_id}): {len(self.data)} samples ---")

        self.label_map = {
            'Luminal A': 0, 'Luminal B': 1, 'HER2-enriched': 2, 'triple negative': 3
        }

        # 4. Transforms
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])
            ])

    def _create_few_shot_data(self, df, shots, seed):
        groups = df.groupby('subtype')
        sampled_dfs = []
        for name, group in groups:
            n_samples = min(len(group), shots)
            sampled_group = group.sample(n=n_samples, random_state=seed)
            sampled_dfs.append(sampled_group)
        return pd.concat(sampled_dfs).reset_index(drop=True)

    def __len__(self):
        return len(self.data)
    
    def load_dicom_image(self, path):
        if not os.path.exists(path):
            return Image.new('RGB', (224, 224))
        try:
            ds = pydicom.dcmread(path)
            img_array = ds.pixel_array.astype(float)
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = (img_array - img_min) / (img_max - img_min) * 255.0
            else:
                img_array = np.zeros_like(img_array)
            if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == "MONOCHROME1":
                img_array = 255.0 - img_array
            return Image.fromarray(img_array.astype(np.uint8)).convert('RGB')
        except:
            return Image.new('RGB', (224, 224))

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Path cleaning logic
        raw_path = str(row['dicom_path']).strip()
        garbage_prefixes = ["/mnt/disk1/backup_user/minh.ntn/CMMD/", "/mnt/disk1/backup_user/minh.ntn/", "CMMD/"]
        rel_path = raw_path
        for prefix in garbage_prefixes:
            if rel_path.startswith(prefix):
                rel_path = rel_path.replace(prefix, "")
        if rel_path.startswith('/'): rel_path = rel_path[1:]
        img_path = os.path.join(self.root_dir, rel_path)
        
        image = self.transform(self.load_dicom_image(img_path))
        subtype_str = str(row['subtype']).strip()
        label = torch.tensor(self.label_map.get(subtype_str, 0), dtype=torch.long)

        if self.task_id == 3 and self.tokenizer:
            clinical_text = str(row['text_report']) if ('text_report' in row and not pd.isna(row['text_report'])) else f"Patient diagnosed with {subtype_str}."
            inputs = self.tokenizer.encode_plus(clinical_text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
            return {'image': image, 'input_ids': inputs['input_ids'].flatten(), 'attention_mask': inputs['attention_mask'].flatten(), 'label': label}
        return {'image': image, 'label': label}