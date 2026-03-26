import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2 

import albumentations as A
from albumentations.pytorch import ToTensorV2

class CMMDDataset(Dataset):
    def __init__(self, csv_file, img_dir, split_file=None, run_id='run_0', tokenizer=None, mode='train'):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.mode = mode
        
        df = pd.read_csv(csv_file)
        
        # Lọc dữ liệu dựa trên file Monte Carlo Split
        if split_file and os.path.exists(split_file):
            split_df = pd.read_csv(split_file)
            
            if run_id in split_df.columns:
                # Lấy cột PatientID và cột ứng với run_id hiện tại (VD: 'run_0')
                split_info = split_df[['PatientID', run_id]].rename(columns={run_id: 'split'})
                
                # Merge thông tin split vào dataframe gốc theo PatientID
                df = df.merge(split_info, on='PatientID', how='inner')
                
                # Giữ lại các sample tương ứng với mode hiện tại (train/val/test)
                df = df[df['split'] == mode]
            else:
                raise ValueError(f"Không tìm thấy cột {run_id} trong file split {split_file}!")
        else:
            print(f"Đang dùng toàn bộ dữ liệu")
            
        self.data = df.reset_index(drop=True)
        self.input_size = 456 # Kích thước ảnh đầu vào (456x456) để phù hợp với EfficientNet-B5 pre-trained or B4??
        
        # 3. Chuẩn bị Albumentations
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(height=self.input_size, width=self.input_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5), 

                # Bổ sung ColorJitter theo đúng thiết lập của RSNA/VinDr
                A.ColorJitter(brightness=0.1, contrast=0.2, 
                              saturation=0.2, hue=0.1, 
                              p=0.5
                              ),
                
                # Mở rộng biên độ Affine và bổ sung Shear
                A.Affine(
                    rotate=(-20, 20), 
                    translate_percent=(0.1, 0.1), 
                    scale=(0.8, 1.2),
                    shear=(-20, 20),
                    p=0.5
                ),
                
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2), 
                
                # Giữ nguyên Normalize ImageNet để tương thích với EfficientNet pre-trained
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.input_size, width=self.input_size), 
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_id']}.png")
        
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=image)
            img_tensor = augmented['image']
        except Exception as e: 
            print(f"Lỗi đọc ảnh: {img_path} - {e}")
            img_tensor = torch.zeros((3, self.input_size, self.input_size))
        
        sample = {
            'image': img_tensor,
            'label': torch.tensor(row['label'], dtype=torch.long)
        }
        
        if self.tokenizer:
            text = str(row['text_prompt'])
            inputs = self.tokenizer(
                text, 
                padding='max_length', 
                truncation=True, 
                max_length=77, 
                return_tensors="pt"
            )
            sample.update({
                'input_ids': inputs['input_ids'].squeeze(0), 
                'attention_mask': inputs['attention_mask'].squeeze(0)
            })
            
        return sample

    def __len__(self): 
        return len(self.data)