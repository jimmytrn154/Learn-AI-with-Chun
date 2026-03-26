import os
import pydicom
import cv2
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

DATA_ROOT_DIR = "/home/minhntn24/nhatminh/M2G-BRCA/data/CMMD" 
ORIGINAL_CSV_PATH = "/home/minhntn24/nhatminh/M2G-BRCA/data/cmmd_metadata.csv" 
TEST_OUTPUT_DIR = "./test_mammo_clip"
IMG_OUTPUT_DIR = os.path.join(TEST_OUTPUT_DIR, "images")
PROCESSED_CSV_PATH = os.path.join(TEST_OUTPUT_DIR, "processed_cmmd.csv")

os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

# TỪ ĐIỂN MAP NHÃN SANG INTEGER (Bao hàm cả viết thường/viết hoa)
SUBTYPE_MAPPING = {
    'Luminal A': 0,
    'luminal a': 0,
    'Luminal B': 1,
    'luminal b': 1,
    'HER2': 2,
    'her2': 2,
    'HER2-enriched': 2, 
    'Triple Negative': 3,
    'triple negative': 3  
}

def crop_breast_region(img_array):
    _, thresh = cv2.threshold(img_array, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_array 
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img_array[y:y+h, x:x+w]

def process_dicom_to_png(dicom_path, save_path):
    dcm = pydicom.dcmread(dicom_path)
    pixel = dcm.pixel_array.astype(float)
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixel = np.max(pixel) - pixel
    pixel = (pixel - np.min(pixel)) / (np.max(pixel) - np.min(pixel) + 1e-6) * 255.0
    pixel_uint8 = pixel.astype(np.uint8)
    cropped_pixel = crop_breast_region(pixel_uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_img = clahe.apply(cropped_pixel)
    cv2.imwrite(save_path, final_img)
    return True

def clean_text_report(text):
    text = str(text)
    if text == "nan" or text.strip() == "":
        return "Mammogram."
    patterns = [
        r"Subtype\s*:[^.]*(\.|$)", r"Luminal\s*[AB]", r"HER2", 
        r"Triple\s*Negative", r"ER\s*(\+|\-)", r"PR\s*(\+|\-)", r"Ki-?67"
    ]
    for p in patterns: 
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print(f"Bắt đầu đọc metadata từ: {ORIGINAL_CSV_PATH}")
    df = pd.read_csv(ORIGINAL_CSV_PATH)
    
    processed_records = []
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Đang chuyển đổi dữ liệu"):
        raw_subtype = str(row.get('subtype', 'Unknown')).strip()
        
        # Chỉ lấy các bản ghi có subtype hợp lệ
        if raw_subtype not in SUBTYPE_MAPPING:
            continue
            
        label_idx = SUBTYPE_MAPPING[raw_subtype]
        
        raw_path = str(row.get('dicom_path', ''))
        if 'CMMD' in raw_path:
            relative_path = raw_path.split('CMMD/')[-1].lstrip('/')
            full_dicom_path = os.path.join(DATA_ROOT_DIR, relative_path)
        else:
            full_dicom_path = os.path.join(DATA_ROOT_DIR, raw_path)
        
        if not os.path.exists(full_dicom_path):
            error_count += 1
            continue
            
        image_id = str(row['SOPInstanceUID'])
        png_save_path = os.path.join(IMG_OUTPUT_DIR, f"{image_id}.png")
        
        try:
            process_dicom_to_png(full_dicom_path, png_save_path)
            clean_text = clean_text_report(row.get('text_report', ''))
            
            processed_records.append({
                'PatientID': row['PatientID'],
                'image_id': image_id,
                'image_path': png_save_path,
                'text_prompt': clean_text,
                'subtype': raw_subtype,
                'label': label_idx
            })
        except Exception as e:
            error_count += 1
            pass 
            
    new_df = pd.DataFrame(processed_records)
    new_df.to_csv(PROCESSED_CSV_PATH, index=False)
    
    print("\n" + "="*50)
    print(f"HOÀN THÀNH TIỀN XỬ LÝ (Base Data)!")
    print(f"- Số lượng ảnh hợp lệ/thành công: {len(processed_records)}")
    print(f"- Số lượng ảnh lỗi/không tìm thấy: {error_count}")
    print(f"- File dữ liệu nền tảng lưu tại: {PROCESSED_CSV_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()