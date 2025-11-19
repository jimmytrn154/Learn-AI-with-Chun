import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(0)
out = Path('feature_vectors')
out.mkdir(exist_ok=True)

def make_cat(cat, n=200):
    # base ranges altered per category to create separability
    
    if cat == 'table':
        sa = np.random.normal(500, 80, n).clip(50, 2000)
        vol = np.random.normal(200, 40, n).clip(5, 1000)
    elif cat == 'mug':
        sa = np.random.normal(120, 30, n).clip(10, 800)
        vol = np.random.normal(45, 15, n).clip(1, 400)
    elif cat == 'lamp':
        sa = np.random.normal(180, 60, n).clip(10, 1200)
        vol = np.random.normal(60, 25, n).clip(1, 500)
    elif cat == 'club':
        sa = np.random.normal(90, 25, n).clip(10, 600)
        vol = np.random.normal(35, 12, n).clip(1, 300)
    else: # dining
        sa = np.random.normal(300, 70, n).clip(20, 1500)
        vol = np.random.normal(120, 30, n).clip(1, 800)
        
    mean_curv = np.random.normal(25, 7, n).clip(0, 100) * (1 + (np.log1p(sa)/10 - 1))
    median_curv = mean_curv * (0.9 + 0.2 * np.random.rand(n))
    silhouette_complexity = np.random.normal(10, 5, n).clip(0, 50)
    skeleton_complexity = np.random.normal(8, 4, n).clip(0, 50)
    aspect_ratio_y = np.random.normal(1.0 + (vol/500), 0.15, n).clip(0.2, 10)
    hollow_ratio = np.random.beta(2, 5, n) * (1 + sa/2000)
    surface_to_volume_ratio = sa / (vol + 1e-6)
    
    df = pd.DataFrame({
        'mean_curvature': mean_curv,
        'median_curvature': median_curv,
        'surface_area': sa,
        'volume': vol,
        'silhouette_complexity': silhouette_complexity,
        'skeleton_complexity': skeleton_complexity,
        'aspect_ratio_y': aspect_ratio_y,
        'hollow_ratio': hollow_ratio,
        'surface_to_volume_ratio': surface_to_volume_ratio,
    })
    
    df['category'] = cat
    df['source_file'] = [f"{cat}_{i:04d}.obj" for i in range(len(df))]
    return df

cats = ['club', 'dining', 'lamp', 'mug', 'table']
for c in cats:
    df = make_cat(c, n=200)
    df.to_csv(out / f"{c}_features.csv", index=False)
    
print('Synthetic data written to', out)