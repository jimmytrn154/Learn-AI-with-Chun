pip install numpy==1.24.3  scipy==1.10.1 ultralytics==8.3.221 opencv-python==4.10.0.84 albumentations matplotlib seaborn tqdm filterpy scikit-learn --use-deprecated=legacy-resolver
# Imports
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm.notebook import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import shutil
import yaml

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
# Configuration
class Config:
    # Paths
    TRAIN_PATH = '/kaggle/working/train'  # Change to your train path
    TEST_PATH = '/kaggle/working/public_test'  # Change to your test path
    OUTPUT_DIR = '/kaggle/working/output'
    YOLO_DATASET_DIR = '/kaggle/working/yolo_dataset'
    
    # YOLO Training
    MODEL_SIZE = 'n'  # 'n', 's', 'm', 'l', 'x'
    EPOCHS = 1
    BATCH_SIZE = 16
    IMAGE_SIZE = 960
    DEVICE = 0
    
    # Few-Shot Detection
    BACKBONE = 'resnet50'  # 'resnet50', 'efficientnet', 'vit'
    CONF_THRESHOLD = 0.2  # YOLO confidence (low for high recall)
    SIMILARITY_THRESHOLD = 0.5  # Reference matching threshold
    ALPHA = 0.55  # Weight: alpha*YOLO + (1-alpha)*similarity
    
    # Post-processing
    MIN_SEQ_LENGTH = 5
    MAX_GAP = 3
    FRAME_SKIP = 1

config = Config()
Path(config.OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

print("✅ Configuration loaded")
print(f"   Train: {config.TRAIN_PATH}")
print(f"   Test: {config.TEST_PATH}")
print(f"   Device: {config.DEVICE}")

# Debug: Check dataset structure
import os
from pathlib import Path

train_path = Path(config.TRAIN_PATH)
print(f"📁 Checking dataset structure...")
print(f"   Train path exists: {train_path.exists()}")

if train_path.exists():
    # Check annotations
    ann_file = train_path / 'annotations' / 'annotations.json'
    print(f"   Annotations file: {ann_file.exists()}")
    
    if ann_file.exists():
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        print(f"   Total videos in annotations: {len(annotations)}")
        
        # Check first video
        if annotations:
            first_video = annotations[0]
            video_id = first_video['video_id']
            video_path = train_path / video_id / 'drone_video.mp4'
            ref_dir = train_path / video_id / 'object_images'
            
            print(f"\n   Checking first video: {video_id}")
            print(f"   Video exists: {video_path.exists()}")
            print(f"   Reference images dir: {ref_dir.exists()}")
            
            if ref_dir.exists():
                ref_images = list(ref_dir.glob('*.jpg')) + list(ref_dir.glob('*.png'))
                print(f"   Reference images: {len(ref_images)}")
            
            # Check annotations structure
            print(f"   Annotations groups: {len(first_video['annotations'])}")
            if first_video['annotations']:
                first_ann = first_video['annotations'][0]
                print(f"   Bboxes in first group: {len(first_ann['bboxes'])}")
                if first_ann['bboxes']:
                    print(f"   First bbox: {first_ann['bboxes'][0]}")
    
    # List video directories
    video_dirs = [d for d in train_path.iterdir() if d.is_dir() and d.name != 'annotations']
    print(f"\n   Video directories found: {len(video_dirs)}")
    if video_dirs:
        print(f"   First few: {[d.name for d in video_dirs[:3]]}")
else:
    print(f"❌ Train path not found: {config.TRAIN_PATH}")
    print(f"   Please update the TRAIN_PATH in config to point to your dataset")
    
    
def convert_to_yolo_format(dataset_path: str, output_dir: str, sample_rate: int = 5):
    """
    Convert drone dataset to YOLO format with robust error handling
    
    Args:
        dataset_path: Path to dataset with annotations
        output_dir: Output directory for YOLO dataset
        sample_rate: Sample every Nth frame
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    
    # Validate input
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    annotations_file = dataset_path / 'annotations' / 'annotations.json'
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    
    # Create YOLO directory structure
    print(f"📁 Creating YOLO directory structure...")
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"📄 Loading annotations from {annotations_file}")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"   Found {len(annotations)} videos in annotations")
    
    # Process each video
    video_ids = [ann['video_id'] for ann in annotations]
    train_ids = video_ids[:int(len(video_ids) * 0.8)]
    
    total_frames = 0
    videos_processed = 0
    videos_skipped = 0
    
    for video_ann in tqdm(annotations, desc="Converting to YOLO"):
        video_id = video_ann['video_id']
        split = 'train' if video_id in train_ids else 'val'
        
        # Find video file - try multiple locations
        video_file = dataset_path / video_id / 'drone_video.mp4'
        if not video_file.exists():
            # Try samples subdirectory
            video_file = dataset_path / 'samples' / video_id / 'drone_video.mp4'
        
        if not video_file.exists():
            print(f"\n⚠️ Video not found for {video_id}, skipping...")
            videos_skipped += 1
            continue
        
        # Create frame-to-bbox mapping
        frame_bboxes = {}
        for ann_group in video_ann['annotations']:
            if 'bboxes' not in ann_group:
                continue
            for bbox in ann_group['bboxes']:
                frame_idx = bbox['frame']
                if frame_idx not in frame_bboxes:
                    frame_bboxes[frame_idx] = []
                frame_bboxes[frame_idx].append(bbox)
        
        if not frame_bboxes:
            print(f"\n⚠️ No bboxes found for {video_id}, skipping...")
            videos_skipped += 1
            continue
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"\n⚠️ Cannot open video {video_id}, skipping...")
            videos_skipped += 1
            continue
        
        frame_idx = 0
        frames_saved = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames with annotations
            if frame_idx % sample_rate == 0 and frame_idx in frame_bboxes:
                h, w = frame.shape[:2]
                
                # Validate frame dimensions
                if h == 0 or w == 0:
                    frame_idx += 1
                    continue
                
                # Save image
                img_name = f"{video_id}_frame_{frame_idx:06d}.jpg"
                img_path = output_dir / 'images' / split / img_name
                success = cv2.imwrite(str(img_path), frame)
                
                if not success:
                    print(f"\n⚠️ Failed to save image: {img_path}")
                    frame_idx += 1
                    continue
                
                # Save labels (YOLO format: class x_center y_center width height)
                label_path = output_dir / 'labels' / split / img_name.replace('.jpg', '.txt')
                with open(label_path, 'w') as f:
                    for bbox in frame_bboxes[frame_idx]:
                        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                        
                        # Validate bbox
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Convert to YOLO format (normalized)
                        x_center = (x1 + x2) / 2 / w
                        y_center = (y1 + y2) / 2 / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        # Clip to [0, 1]
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))
                        
                        # Class 0 (single class)
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                frames_saved += 1
                total_frames += 1
            
            frame_idx += 1
        
        cap.release()
        
        if frames_saved > 0:
            videos_processed += 1
    
    # Verify we have data
    train_images = list((output_dir / 'images' / 'train').glob('*.jpg'))
    val_images = list((output_dir / 'images' / 'val').glob('*.jpg'))
    
    if len(train_images) == 0 and len(val_images) == 0:
        raise RuntimeError(
            f"No images were saved! Check:\n"
            f"  1. Dataset path: {dataset_path}\n"
            f"  2. Video files exist in {dataset_path}/[video_id]/drone_video.mp4\n"
            f"  3. Annotations contain valid bboxes"
        )
    
    # Create data.yaml with absolute path
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['object']
    }
    
    with open(output_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    print(f"\n✅ Conversion complete!")
    print(f"   Videos processed: {videos_processed}")
    print(f"   Videos skipped: {videos_skipped}")
    print(f"   Total frames saved: {total_frames}")
    print(f"   Train images: {len(train_images)}")
    print(f"   Val images: {len(val_images)}")
    print(f"   Train/Val split: {len(train_ids)}/{len(video_ids) - len(train_ids)} videos")
    print(f"   YOLO dataset: {output_dir.absolute()}")
    
    return str(output_dir / 'data.yaml')

# Convert dataset
print("🚀 Starting dataset conversion...")
data_yaml_path = convert_to_yolo_format(
    dataset_path=config.TRAIN_PATH,
    output_dir=config.YOLO_DATASET_DIR,
    sample_rate=10
)

def train_yolo_model(data_yaml: str, 
                     model_size: str = 'l',
                     epochs: int = 100,
                     batch_size: int = 32,
                     img_size: int = 960,
                     patience: int = 20,
                     use_pretrained: bool = True,
                     resume: bool = False,
                     project_name: str = 'runs/few_shot_train',
                     experiment_name: str = None):
    """
    Train YOLO model on converted dataset with comprehensive monitoring
    
    Args:
        data_yaml: Path to dataset YAML file
        model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        patience: Early stopping patience
        use_pretrained: Use pretrained weights
        resume: Resume from last checkpoint
        project_name: Project directory name
        experiment_name: Experiment name (auto-generated if None)
    
    Returns:
        tuple: (best_weights_path, results)
    """
    print("\n" + "="*70)
    print(f"  TRAINING YOLOv11{model_size.upper()}")
    print("="*70)
    
    # Validate data file
    if not Path(data_yaml).exists():
        print(f"\n❌ Data YAML not found: {data_yaml}")
        return None, None
    
    # Auto-generate experiment name if not provided
    if experiment_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"yolo11{model_size}_{timestamp}"
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  Model: YOLOv11{model_size}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Patience: {patience}")
    print(f"  Pretrained: {use_pretrained}")
    print(f"  Resume: {resume}")
    print(f"  Experiment: {experiment_name}")
    
    try:
        # Load model
        if resume:
            # Resume from last checkpoint
            last_checkpoint = Path(project_name) / experiment_name / 'weights' / 'last.pt'
            if last_checkpoint.exists():
                print(f"\n🔄 Resuming from checkpoint: {last_checkpoint}")
                model = YOLO(str(last_checkpoint))
            else:
                print(f"\n⚠️  Checkpoint not found, starting fresh training")
                model = YOLO(f'yolo11{model_size}.pt' if use_pretrained else f'yolo11{model_size}.yaml')
        else:
            model_path = f'yolo11{model_size}.pt' if use_pretrained else f'yolo11{model_size}.yaml'
            print(f"\n📦 Loading model: {model_path}")
            model = YOLO(model_path)
        
        print(f"\n🏋️ Starting training...")
        print(f"   This may take a while depending on dataset size and hardware")
        
        # Train with comprehensive augmentation
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=patience,
            device=config.DEVICE,
            save=True,
            project=project_name,
            name=experiment_name,
            exist_ok=True,
            pretrained=use_pretrained,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # Data augmentation
            augment=True,
            copy_paste=0.1,
            mosaic=0.5,
            mixup=0.1,
            # HSV augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            # Geometric augmentation
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.5,
            fliplr=0.5,
            # Validation
            val=True,
            plots=True,
            save_period=5,
            # Performance
            cache=False,  # Set to True if you have enough RAM
            workers=8,
            close_mosaic=10,
            amp=True  # Automatic Mixed Precision
        )
        
        # Get paths
        weights_dir = Path(project_name) / experiment_name / 'weights'
        best_weights = weights_dir / 'best.pt'
        last_weights = weights_dir / 'last.pt'
        
        if not best_weights.exists():
            print(f"\n⚠️  Best weights not found, using last weights")
            best_weights = last_weights
        
        print(f"\n" + "="*70)
        print("  ✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\n📊 Training Results:")
        print(f"  Best weights: {best_weights}")
        print(f"  Last weights: {last_weights}")
        print(f"  Results directory: {Path(project_name) / experiment_name}")
        
        # Display metrics if available
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\n🎯 Final Metrics:")
            if 'metrics/mAP50(B)' in metrics:
                print(f"  mAP50: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"  mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
            if 'metrics/precision(B)' in metrics:
                print(f"  Precision: {metrics['metrics/precision(B)']:.4f}")
            if 'metrics/recall(B)' in metrics:
                print(f"  Recall: {metrics['metrics/recall(B)']:.4f}")
        
        print(f"\n💡 Next Steps:")
        print(f"  1. Validate: model.val()")
        print(f"  2. Run inference on test set")
        print(f"  3. Train ensemble models with different sizes")
        
        return str(best_weights), results
        
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# Train model
yolo_weights, train_results = train_yolo_model(
    data_yaml=data_yaml_path,
    model_size=config.MODEL_SIZE,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    img_size=config.IMAGE_SIZE
)

class DeepFeatureExtractor:
    """Extract deep features for few-shot matching"""
    
    def __init__(self, backbone='resnet50', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = 2048
        elif backbone == 'efficientnet':
            efficientnet = models.efficientnet_b4(pretrained=True)
            self.model = nn.Sequential(*list(efficientnet.children())[:-1])
            self.feature_dim = 1792
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract feature vector from image"""
        if image is None or image.size == 0:
            return np.zeros(self.feature_dim)
        
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract
        with torch.no_grad():
            features = self.model(img_tensor)
        
        features = features.squeeze().cpu().numpy()
        features = features / (np.linalg.norm(features) + 1e-6)
        
        return features

print("✅ DeepFeatureExtractor defined")

class PrototypicalMatcher:
    """Prototypical Network-inspired matcher"""
    
    def __init__(self, feature_extractor: DeepFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.prototypes = {}
        self.support_features = {}
    
    def build_prototype(self, video_id: str, reference_images: List[str]):
        """Build prototype from reference images"""
        features = []
        
        for img_path in reference_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            feat = self.feature_extractor.extract(img)
            features.append(feat)
        
        if not features:
            self.prototypes[video_id] = np.zeros(self.feature_extractor.feature_dim)
            self.support_features[video_id] = []
            return
        
        prototype = np.mean(features, axis=0)
        prototype = prototype / (np.linalg.norm(prototype) + 1e-6)
        
        self.prototypes[video_id] = prototype
        self.support_features[video_id] = features
    
    def compute_similarity(self, frame: np.ndarray, bbox: Dict, video_id: str) -> float:
        """Compute similarity between detection and reference"""
        if video_id not in self.prototypes:
            return 0.5
        
        # Extract ROI
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        
        # Extract features
        roi_features = self.feature_extractor.extract(roi)
        
        # Compare with all support images, take max
        similarities = []
        for support_feat in self.support_features[video_id]:
            sim = cosine_similarity(
                roi_features.reshape(1, -1),
                support_feat.reshape(1, -1)
            )[0, 0]
            similarities.append(sim)
        
        similarity = max(similarities) if similarities else 0.0
        return np.clip(similarity, 0, 1)

print("✅ PrototypicalMatche defined")

class FewShotDroneDetector:
    """Complete few-shot detector"""
    
    def __init__(self, yolo_weights: str, backbone: str = 'resnet50',
                 conf_threshold: float = 0.10, similarity_threshold: float = 0.5,
                 alpha: float = 0.5):
        self.yolo = YOLO(yolo_weights)
        self.feature_extractor = DeepFeatureExtractor(backbone=backbone)
        self.matcher = PrototypicalMatcher(self.feature_extractor)
        
        self.conf_threshold = conf_threshold
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
    
    def register_video(self, video_id: str, reference_images_dir: Path):
        """Register reference images for a video"""
        ref_images = list(reference_images_dir.glob('*.jpg'))
        ref_images += list(reference_images_dir.glob('*.png'))
        
        if not ref_images:
            print(f"⚠️ No reference images for {video_id}")
            return
        
        self.matcher.build_prototype(video_id, ref_images)
        print(f"✓ Registered {len(ref_images)} images for {video_id}")
    
    def detect_frame(self, frame: np.ndarray, video_id: str) -> List[Dict]:
        """Detect objects in frame"""
        results = self.yolo(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                continue
            
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                yolo_conf = float(boxes.conf[i])
                
                bbox = {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)}
                
                # Compute similarity
                similarity = self.matcher.compute_similarity(frame, bbox, video_id)
                
                # Combined score
                combined_score = self.alpha * yolo_conf + (1 - self.alpha) * similarity
                
                if similarity >= self.similarity_threshold:
                    detections.append({
                        **bbox,
                        'yolo_conf': yolo_conf,
                        'similarity': similarity,
                        'score': combined_score
                    })
        
        return sorted(detections, key=lambda x: x['score'], reverse=True)
    
    def process_video(self, video_path: str, video_id: str, frame_skip: int = 1) -> List[Dict]:
        """Process entire video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_detections = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames // frame_skip, desc=f"Processing {video_id}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                detections = self.detect_frame(frame, video_id)
                for det in detections:
                    det['frame'] = frame_idx
                    all_detections.append(det)
                pbar.update(1)
            
            frame_idx += 1
        
        cap.release()
        pbar.close()
        
        return all_detections
    
    def to_submission_format(self, detections: List[Dict], 
                            min_length: int = 5, max_gap: int = 3) -> List[Dict]:
        """Convert to submission format"""
        if not detections:
            return []
        
        detections = sorted(detections, key=lambda x: x['frame'])
        
        sequences = []
        current_seq = []
        
        for det in detections:
            if not current_seq or det['frame'] - current_seq[-1]['frame'] <= max_gap:
                current_seq.append(det)
            else:
                if len(current_seq) >= min_length:
                    sequences.append(current_seq)
                current_seq = [det]
        
        if len(current_seq) >= min_length:
            sequences.append(current_seq)
        
        formatted = []
        for seq in sequences:
            filled = self._fill_gaps(seq, max_gap)
            bboxes = [{
                'frame': det['frame'],
                'x1': det['x1'], 'y1': det['y1'],
                'x2': det['x2'], 'y2': det['y2']
            } for det in filled]
            formatted.append({'bboxes': bboxes})
        
        return formatted
    
    def _fill_gaps(self, sequence: List[Dict], max_gap: int) -> List[Dict]:
        """Interpolate missing frames"""
        if len(sequence) <= 1:
            return sequence
        
        filled = [sequence[0]]
        
        for i in range(1, len(sequence)):
            prev = sequence[i-1]
            curr = sequence[i]
            gap = curr['frame'] - prev['frame']
            
            if gap > 1 and gap <= max_gap:
                for t in range(1, gap):
                    alpha = t / gap
                    interp = {
                        'frame': prev['frame'] + t,
                        'x1': int(prev['x1'] + alpha * (curr['x1'] - prev['x1'])),
                        'y1': int(prev['y1'] + alpha * (curr['y1'] - prev['y1'])),
                        'x2': int(prev['x2'] + alpha * (curr['x2'] - prev['x2'])),
                        'y2': int(prev['y2'] + alpha * (curr['y2'] - prev['y2'])),
                        'score': (prev['score'] + curr['score']) / 2
                    }
                    filled.append(interp)
            
            filled.append(curr)
        
        return filled

print("✅ FewShotDroneDetector defined")

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def evaluate_st_iou(predictions, ground_truth, iou_threshold=0.5):
    """Evaluate ST-IoU metric"""
    # Simplified ST-IoU evaluation
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred, gt in zip(predictions, ground_truth):
        pred_frames = {b['frame']: b for seq in pred for b in seq['bboxes']}
        gt_frames = {b['frame']: b for ann in gt for b in ann['bboxes']}
        
        all_frames = set(pred_frames.keys()) | set(gt_frames.keys())
        
        for frame in all_frames:
            if frame in pred_frames and frame in gt_frames:
                iou = calculate_iou(pred_frames[frame], gt_frames[frame])
                if iou >= iou_threshold:
                    total_tp += 1
                else:
                    total_fp += 1
            elif frame in pred_frames:
                total_fp += 1
            else:
                total_fn += 1
    
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }

print("✅ Evaluation functions defined")

# Evaluate on training data (optional)
def evaluate_on_train_data(detector, train_path, annotation_file):
    """Evaluate detector on training data"""
    print("📊 Evaluating on training data...")
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    all_predictions = []
    all_ground_truth = []
    
    for video_ann in annotations[:2]:  # Test on 2 videos for speed
        video_id = video_ann['video_id']
        video_file = Path(train_path) / video_id / 'drone_video.mp4'
        ref_dir = Path(train_path) / video_id / 'object_images'
        
        if not video_file.exists():
            continue
        
        # Register
        detector.register_video(video_id, ref_dir)
        
        # Process
        detections = detector.process_video(str(video_file), video_id, frame_skip=5)
        formatted = detector.to_submission_format(detections)
        
        all_predictions.append(formatted)
        all_ground_truth.append(video_ann['annotations'])
    
    # Evaluate
    metrics = evaluate_st_iou(all_predictions, all_ground_truth)
    
    print(f"\n✅ Evaluation Results:")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1-Score: {metrics['f1']:.3f}")
    
    return metrics

# Run evaluation (uncomment to test)
# detector = FewShotDroneDetector(
#     yolo_weights=yolo_weights,
#     backbone=config.BACKBONE,
#     conf_threshold=config.CONF_THRESHOLD,
#     similarity_threshold=config.SIMILARITY_THRESHOLD,
#     alpha=config.ALPHA
# )
# metrics = evaluate_on_train_data(
#     detector,
#     config.TRAIN_PATH,
#     Path(config.TRAIN_PATH) / 'annotations' / 'annotations.json'
# )

def run_inference_on_test(yolo_weights: str, test_path: str = None, 
                         output_path: str = None):
    """
    Run inference on test dataset using few-shot detector
    
    Args:
        yolo_weights: Path to YOLO model weights
        test_path: Path to test dataset (default: config.TEST_PATH)
        output_path: Path to save predictions (default: config.OUTPUT_DIR/submission_few_shot.json)
    """
    if test_path is None:
        test_path = config.TEST_PATH
    
    if output_path is None:
        output_path = str(Path(config.OUTPUT_DIR) / 'submission_few_shot.json')
    
    print("\n" + "="*70)
    print("  RUNNING TEST INFERENCE (FEW-SHOT)")
    print("="*70)
    
    # Validate paths
    if not Path(yolo_weights).exists():
        print(f"\n❌ Weights not found at {yolo_weights}")
        print("   Please train model first or provide correct path")
        return None
    
    if not Path(test_path).exists():
        print(f"\n❌ Test data not found at {test_path}")
        print("   Please upload test dataset")
        return None
    
    print(f"\n⚙️ Configuration:")
    print(f"  Weights: {yolo_weights}")
    print(f"  Test Path: {test_path}")
    print(f"  Backbone: {config.BACKBONE}")
    print(f"  YOLO Confidence: {config.CONF_THRESHOLD}")
    print(f"  Similarity Threshold: {config.SIMILARITY_THRESHOLD}")
    print(f"  Frame Skip: {config.FRAME_SKIP}")
    print(f"  Alpha: {config.ALPHA}")
    
    # Initialize detector
    detector = FewShotDroneDetector(
        yolo_weights='/kaggle/working/best (4).pt',
        backbone=config.BACKBONE,
        conf_threshold=config.CONF_THRESHOLD,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
        alpha=config.ALPHA
    )
    
    # Find all test videos - handle both flat and samples/ subdirectory
    test_path = Path(test_path)
    
    # Check if there's a samples subdirectory
    if (test_path / 'samples').exists():
        test_path = test_path / 'samples'
        print(f"  📁 Found samples subdirectory, using: {test_path}")
    
    video_dirs = sorted([d for d in test_path.iterdir() 
                        if d.is_dir() and d.name != '.ipynb_checkpoints'])
    
    if not video_dirs:
        print(f"\n❌ No video directories found in {test_path}")
        return None
    
    all_predictions = []
    
    print(f"\n🎬 Processing {len(video_dirs)} videos...")
    
    for video_dir in video_dirs:
        video_id = video_dir.name
        video_file = video_dir / 'drone_video.mp4'
        ref_dir = video_dir / 'object_images'
        
        if not video_file.exists():
            print(f"\n⚠️  {video_id}: Video not found, skipping...")
            all_predictions.append({'video_id': video_id, 'detections': []})
            continue
        
        print(f"\n📹 Processing: {video_id}")
        
        # Register reference images
        detector.register_video(video_id, ref_dir)
        
        # Process video
        detections = detector.process_video(
            str(video_file), 
            video_id, 
            frame_skip=config.FRAME_SKIP
        )
        
        # Format predictions
        formatted = detector.to_submission_format(
            detections,
            min_length=config.MIN_SEQ_LENGTH,
            max_gap=config.MAX_GAP
        )
        
        all_predictions.append({
            'video_id': video_id,
            'detections': formatted
        })
        
        total_frames = sum(len(d['bboxes']) for d in formatted)
        print(f"   ✓ {len(detections)} raw detections → {len(formatted)} sequences ({total_frames} frames)")
    
    # Save predictions
    with open(output_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\n✓ Test predictions saved to {output_path}")
    
    # Show statistics
    print(f"\n📊 Test Results:")
    total_detections = sum(len(r['detections']) for r in all_predictions)
    videos_with_detections = sum(1 for r in all_predictions if len(r['detections']) > 0)
    
    avg_frames_per_detection = []
    for r in all_predictions:
        for det in r['detections']:
            avg_frames_per_detection.append(len(det['bboxes']))
    
    print(f"  - Total videos: {len(all_predictions)}")
    print(f"  - Videos with detections: {videos_with_detections}")
    print(f"  - Videos without detections: {len(all_predictions) - videos_with_detections}")
    print(f"  - Total detection intervals: {total_detections}")
    
    if len(avg_frames_per_detection) > 0:
        print(f"  - Avg frames per detection: {np.mean(avg_frames_per_detection):.1f}")
        print(f"  - Min/Max frames: {np.min(avg_frames_per_detection)}/{np.max(avg_frames_per_detection)}")
    
    # Per-video breakdown
    print(f"\n📹 Per-Video Breakdown:")
    print(f"  {'Video ID':<30} {'Detections':>12} {'Total Frames':>12}")
    print("  " + "-"*55)
    for r in all_predictions:
        total_frames = sum(len(d['bboxes']) for d in r['detections'])
        print(f"  {r['video_id']:<30} {len(r['detections']):>12} {total_frames:>12}")
    
    print("\n" + "="*70)
    print("  ✓ TEST INFERENCE COMPLETE!")
    print("="*70)
    print(f"\n📥 Download: {output_path}")
    
    return all_predictions


# Run inference
predictions = run_inference_on_test(
    yolo_weights='/kaggle/working/best (4).pt',
    test_path=config.TEST_PATH,
    output_path=str(Path(config.OUTPUT_DIR) / 'submission6.json')
)

def verify_submission(submission_file: str):
    """Verify submission format"""
    print(f"🔍 Verifying submission: {submission_file}")
    
    with open(submission_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n📊 Submission Statistics:")
    print(f"   Total videos: {len(data)}")
    
    total_sequences = 0
    total_frames = 0
    videos_with_detections = 0
    
    for video_pred in data:
        video_id = video_pred['video_id']
        detections = video_pred['detections']
        
        if detections:
            videos_with_detections += 1
            total_sequences += len(detections)
            for seq in detections:
                total_frames += len(seq['bboxes'])
    
    print(f"   Videos with detections: {videos_with_detections}")
    print(f"   Total sequences: {total_sequences}")
    print(f"   Total frames detected: {total_frames}")
    
    # Check format
    errors = []
    for video_pred in data:
        if 'video_id' not in video_pred:
            errors.append("Missing 'video_id' field")
        if 'detections' not in video_pred:
            errors.append("Missing 'detections' field")
        
        for seq in video_pred.get('detections', []):
            if 'bboxes' not in seq:
                errors.append(f"Missing 'bboxes' in {video_pred['video_id']}")
            for bbox in seq.get('bboxes', []):
                required = ['frame', 'x1', 'y1', 'x2', 'y2']
                for field in required:
                    if field not in bbox:
                        errors.append(f"Missing '{field}' in bbox")
    
    if errors:
        print(f"\n❌ Format errors found:")
        for err in errors[:10]:
            print(f"   - {err}")
    else:
        print(f"\n✅ Submission format is valid!")
    
    return len(errors) == 0

# Verify
is_valid = verify_submission(str(Path(config.OUTPUT_DIR) / 'submission6.json'))

import matplotlib.pyplot as plt
import random

def visualize_test_results(predictions: list, test_path: str, num_videos: int = 6, num_frames: int = 10):
    """Visualize detections on a few frames from the test set."""
    print(f"\n🖼️ Visualizing detections for {num_videos} test video(s)...")
    test_path = Path(test_path)
    
    # Handle samples subdirectory
    if (test_path / 'samples').exists():
        test_path = test_path / 'samples'
        print(f"  📁 Using samples subdirectory: {test_path}")
    
    videos_to_show = [p for p in predictions if p['detections']][:num_videos]
    
    if not videos_to_show:
        print("❌ No detections found in the test set to visualize.")
        return
    
    for video_pred in videos_to_show:
        video_id = video_pred['video_id']
        video_file = test_path / video_id / 'drone_video.mp4'
        
        if not video_file.exists():
            print(f"⚠️  Video file not found for {video_id}, skipping visualization.")
            continue
            
        print(f"\n📹 Visualizing video: {video_id}")
        
        cap = cv2.VideoCapture(str(video_file))
        
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_file}")
            continue
        
        # Group detections by frame
        frame_to_bboxes = {}
        for seq in video_pred['detections']:
            for bbox in seq['bboxes']:
                frame_idx = bbox['frame']
                if frame_idx not in frame_to_bboxes:
                    frame_to_bboxes[frame_idx] = []
                frame_to_bboxes[frame_idx].append(bbox)
        
        if not frame_to_bboxes:
            print(f"   ℹ️  No frames with detections for {video_id}.")
            cap.release()
            continue
            
        frames_to_show = sorted(frame_to_bboxes.keys())
        sample_frames = random.sample(frames_to_show, min(num_frames, len(frames_to_show)))
        sample_frames.sort()
        
        print(f"   📊 Total frames with detections: {len(frames_to_show)}")
        print(f"   🎯 Showing frames: {sample_frames}")
        
        fig, axes = plt.subplots(1, len(sample_frames), figsize=(5*len(sample_frames), 5))
        if len(sample_frames) == 1:
            axes = [axes]
        
        for i, frame_idx in enumerate(sample_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"   ⚠️  Could not read frame {frame_idx}")
                continue
            
            # Draw all bboxes for this frame
            num_boxes = len(frame_to_bboxes[frame_idx])
            for bbox in frame_to_bboxes[frame_idx]:
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add bbox label
                label = f"Det"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_idx} | Detections: {num_boxes}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"Video: {video_id}\nFrame: {frame_idx} ({num_boxes} detections)", 
                             fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        cap.release()
        
    print("\n✓ Visualization complete!")


# Visualize results from the test run
if 'predictions' in locals() and predictions:
    visualize_test_results(predictions, config.TEST_PATH, num_videos=6, num_frames=9)
else:
    print("⚠️  Could not visualize test results because 'predictions' is not defined or empty.")
