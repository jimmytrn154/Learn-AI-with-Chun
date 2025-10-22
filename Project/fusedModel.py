# # Fused Clinical/Text/Image Pipeline — Updated Notebook
# **Includes:** robust image discovery, label cleaning & dropping, preflight checks, fused vector saving (raw + processed), and full metrics.
# 
# > Generated: 2025-10-19 18:43:06Z (UTC)

# Feature toggles/ Control the flow
USE_TABULAR = False   # <- turn OFF radiomics/clinical
USE_TEXT    = True
USE_IMAGE   = True

# ===== CELL 1 — CONFIG =====
IMG_ROOT = "/home/24chuong.ta/anaconda3/chun/DICOM 1-100/Image_Encoder_Outputs_Swin"   # top folder that contains {patient}_{case} dirs (e.g., 1062443_0062)
DATA_CSV = "data.csv"

ID_COL     = "ID"
LABEL_COL  = "Subtype"
GROUP_COL = "Patient Number"    # optional

VALID_VIEWS = {"RCC", "LCC", "RMLO", "LMLO"}

TEXT_COLUMNS = [
    "Gross Feature", "Calcification morphology", "BI-RAD",
    "Mass Shape", "Mass Margin", "Diagnosis", "Histologic grade"
]
ENCODER_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # or "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
BERT_MAX_LENGTH = 64
BATCH_SIZE = 16

NUMERIC_COLUMNS = [
    "Gross size",
    "her2 expression cd",
    "original_shape2D_Elongation",
    "original_shape2D_Perimeter",
    "original_shape2D_MinorAxisLength",
    "original_shape2D_MajorAxisLength",
    "original_shape2D_Sphericity",
    "original_shape2D_MeshSurface",
    "original_glrlm_GrayLevelNonUniformity",
    "original_glrlm_RunLengthNonUniformity",
    "original_glszm_LargeAreaEmphasis",
    "original_glszm_LargeAreaLowGrayLevelEmphasis",
    "original_glszm_ZoneVariance",
    "original_glszm_ZonePercentage",
    "original_gldm_DependenceNonUniformity",
    "original_gldm_DependenceNonUniformityNormalized",
    "original_gldm_DependenceVariance",
]

TEXT_PCA_DIM   = 96
IMG_PCA_DIM    = 64
FUSED_PCA_DIM  = 64

PER_REC_ROOT = "./outputs_per_record_IE_Anno" #Adjust by cases
GLOBAL_SUMMARY_PATH = f"{PER_REC_ROOT}/summary.json"
OVERWRITE_WITH_SUFFIX = True

TEST_SIZE = 0.20
RANDOM_SEED = 42

IMG_POOL = "mean"

DROP_LABELS = {"Unidentifiable", "Unknown", "", "NA", "nan", "None", "N/A"}

# ===== CELL 2 — Imports & Utils =====
import os, re, glob, json, math, time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, normalize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

np.random.seed(RANDOM_SEED)

def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def stats_for_vec(vec: np.ndarray):
    v = vec.astype(np.float32)
    return dict(
        embedding_dim = int(v.shape[0]),
        embedding_dtype = "float32",
        embedding_min = float(v.min()) if v.size else 0.0,
        embedding_max = float(v.max()) if v.size else 0.0,
        embedding_mean = float(v.mean()) if v.size else 0.0,
        embedding_std = float(v.std()) if v.size else 0.0,
        embedding_l2_norm = float(np.linalg.norm(v))
    )

def write_vector_and_json(folder, base_name, vec: np.ndarray, id_str: str, has_image: bool):
    ensure_dir(folder)
    ts = utc_now_iso().replace(":","").replace("-","")
    suffix = f"_{ts}" if OVERWRITE_WITH_SUFFIX else ""
    csv_path  = os.path.join(folder, f"{base_name}{suffix}.csv")
    json_path = os.path.join(folder, f"{base_name}{suffix}.json")
    np.savetxt(csv_path, vec.astype(np.float32)[None, :], delimiter=",")
    meta = {
        "timestamp_utc": utc_now_iso(),
        "id": id_str,
        "vector_type": base_name,
        "has_image_embedding": bool(has_image),
        **stats_for_vec(vec)
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    return csv_path, json_path, meta



# ===== CELL 3 — Load CSV, clean labels, parse ID =====
df = pd.read_csv(DATA_CSV)

TEXT_COLUMNS    = [c for c in TEXT_COLUMNS if c in df.columns]
NUMERIC_COLUMNS = [c for c in NUMERIC_COLUMNS if c in df.columns]
print(f"Using {len(TEXT_COLUMNS)} text columns:", TEXT_COLUMNS)
print(f"Using {len(NUMERIC_COLUMNS)} numeric features.")

def clean_label(x):
    if pd.isna(x): return ""
    s = str(x).replace("\u00A0"," ").replace("\u200B","")
    return s.strip()

df[LABEL_COL] = df[LABEL_COL].apply(clean_label)
before = len(df)
df = df[~df[LABEL_COL].isin(DROP_LABELS)].reset_index(drop=True)
after = len(df)
print(f"Dropped {before - after} rows by DROP_LABELS. Remaining: {after}")

ID_RE = re.compile(r"^\s*0*(?P<case>\d+)[_\-](?P<view>[A-Za-z]+)\s*$", re.IGNORECASE)

def parse_id(id_str):
    m = ID_RE.match(str(id_str))
    if not m: return None, None
    return int(m.group("case")), m.group("view").upper()

parsed = df[ID_COL].apply(parse_id)
df[["case_num","view"]] = pd.DataFrame(parsed.tolist(), index=df.index)
df["view"] = df["view"].str.upper()

exp_map = (df.dropna(subset=["case_num","view"])
             .groupby("case_num")["view"].nunique()
             .to_dict())
print("Example expected views per case:", dict(list(exp_map.items())[:5]))



# ===== CELL 4 — Robust Image Discovery =====
def case_candidates(case_num: int):
    root = Path(IMG_ROOT)
    if not root.exists(): return []
    return [str(d) for d in root.iterdir() if d.is_dir() and d.name.endswith(f"_{case_num:04d}")]

def list_view_subfolders(case_dir: str):
    vmap = {}
    p = Path(case_dir)
    if not p.exists(): return vmap
    for sub in p.iterdir():
        if not sub.is_dir(): continue
        parts = sub.name.split("_")
        if len(parts) >= 3 and parts[-1].lower() == "cropped":
            view = parts[-2].upper()
            if view in VALID_VIEWS:
                vmap[view] = str(sub)
    return vmap

def choose_case_dir_for_view(case_num: int, required_view: str):
    cands = case_candidates(case_num)
    if not cands: return None, {}
    scored = []
    for c in cands:
        vmap = list_view_subfolders(c)
        has_req = required_view.upper() in vmap
        scored.append((c, vmap, has_req, len(vmap)))
    with_view = [t for t in scored if t[2]]
    if with_view:
        best = sorted(with_view, key=lambda x: x[3], reverse=True)[0]
        return best[0], best[1]
    best = sorted(scored, key=lambda x: x[3], reverse=True)[0]
    return best[0], best[1]

def find_emb_file_in_cropped(cropped_dir: str):
    if not cropped_dir or not os.path.isdir(cropped_dir):
        return None
    p = Path(cropped_dir)
    npys = sorted(glob.glob(str(p / "*crop_emb.npy")))
    if npys: return npys[0]
    csvs = sorted(glob.glob(str(p / "*crop_emb.csv")) + glob.glob(str(p / "*crop_emb.CSV")))
    if csvs: return csvs[0]
    subdir = None
    for sub in p.iterdir():
        if sub.is_dir() and re.search(r"crop_emb", sub.name, re.IGNORECASE):
            subdir = sub; break
    if subdir:
        npys = sorted(glob.glob(str(subdir / "*.npy")))
        if npys: return npys[0]
        csvs = sorted(glob.glob(str(subdir / "*.csv")) + glob.glob(str(subdir / "*.CSV")))
        if csvs: return csvs[0]
    return None

def load_embedding_file(fpath: str):
    if fpath is None: return None
    if fpath.lower().endswith(".npy"):
        return np.load(fpath).astype(float).reshape(-1)
    arr = pd.read_csv(fpath, header=None).to_numpy().astype(float).reshape(-1)
    return arr



# ===== CELL 5 — Preflight =====
MISSING_REPORT = "./outputs/metrics/missing_image_embeddings.csv"
ensure_dir(os.path.dirname(MISSING_REPORT))

rows = []
miss = []
for i, id_str in enumerate(df[ID_COL].astype(str)):
    case, view = parse_id(id_str)
    if case is None or not isinstance(view, str):
        rows.append({"ID": id_str, "ok": False, "reason": "bad_id_format",
                     "case_dir": None, "cropped_dir": None, "file": None})
        miss.append(id_str)
        continue
    case_dir, vmap = choose_case_dir_for_view(case, view)
    cropped = vmap.get(view.upper()) if vmap else None
    fpath = find_emb_file_in_cropped(cropped) if cropped else None
    ok = fpath is not None
    rows.append({
        "ID": id_str, "case_num": case, "view": view, "ok": ok,
        "reason": None if ok else ("no_cropped_dir" if not cropped else "no_emb_file"),
        "case_dir": case_dir, "cropped_dir": cropped, "file": fpath
    })
    if not ok: miss.append(id_str)

rep = pd.DataFrame(rows)
n_all = len(rep); n_ok = int(rep["ok"].sum())
print(f"[Preflight] Embedded files found for {n_ok}/{n_all} records ({n_ok/n_all:.1%}).")
if n_ok < n_all:
    print(rep.loc[~rep["ok"], ["ID","reason","case_dir","cropped_dir"]].head(10).to_string(index=False))
ensure_dir("./outputs/metrics")
rep.to_csv(MISSING_REPORT, index=False)
print(f"Full preflight saved → {MISSING_REPORT}")



# ===== CELL 6 — Build Image Matrix =====
img_vecs, has_img = [], []
for _, r in df.iterrows():
    case, view = r.get("case_num"), r.get("view")
    case_dir, vmap = choose_case_dir_for_view(int(case), str(view))
    cropped_dir = vmap.get(str(view).upper()) if vmap else None
    fpath = find_emb_file_in_cropped(cropped_dir) if cropped_dir else None
    v = load_embedding_file(fpath) if fpath else None
    if v is None:
        img_vecs.append(None); has_img.append(0.0)
    else:
        img_vecs.append(v);    has_img.append(1.0)

max_dim = max((len(v) for v in img_vecs if isinstance(v, np.ndarray)), default=0)
X_img_raw_all = np.zeros((len(df), max_dim), dtype=float)
for i, v in enumerate(img_vecs):
    if isinstance(v, np.ndarray):
        d = min(max_dim, len(v)); X_img_raw_all[i, :d] = v[:d]
has_img_flag_all = np.array(has_img, dtype=np.float32).reshape(-1,1)

print("Image raw matrix:", X_img_raw_all.shape, "| coverage:", has_img_flag_all.mean())



# ===== CELL 7 — Text Encoder =====
HAS_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

class TextEncoder:
    def __init__(self, model_name=ENCODER_NAME, max_length=BERT_MAX_LENGTH, batch_size=BATCH_SIZE,
                 use_transformers=HAS_TRANSFORMERS):
        self.model_name=model_name; self.max_length=max_length; self.batch_size=batch_size
        self.use_transformers=use_transformers
        self.mode=None; self.device="cuda" if (use_transformers and HAS_TRANSFORMERS and torch.cuda.is_available()) else "cpu"
        self.tok=None; self.model=None; self.vec=None; self.svd=None

    def fit(self, texts):
        texts=[t if isinstance(t,str) else "" for t in texts]
        if self.use_transformers:
            try:
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
                self.mode = "bert"; return self
            except Exception as e:
                print("[warn] transformers unavailable, fallback to TF-IDF:", e)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vec = TfidfVectorizer(min_df=3, max_df=0.95, ngram_range=(1,2))
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(256, X.shape[1]-1), random_state=RANDOM_SEED).fit(X)
        self.mode="tfidf"; return self

    @torch.no_grad()
    def _encode_bert(self, texts):
        outs=[]
        for i in range(0, len(texts), self.batch_size):
            batch=[t if isinstance(t,str) else "" for t in texts[i:i+self.batch_size]]
            enc=self.tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            enc={k:v.to(self.device) for k,v in enc.items()}
            h=self.model(**enc).last_hidden_state
            mask=enc["attention_mask"].unsqueeze(-1)
            emb=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
            outs.append(emb.cpu().numpy())
        return np.vstack(outs)

    def transform(self, texts):
        texts=[t if isinstance(t,str) else "" for t in texts]
        if self.mode=="bert": return self._encode_bert(texts)
        X=self.vec.transform(texts)
        return self.svd.transform(X)



# ===== CELL 8 — 80/20 Split =====
labels_all = df[LABEL_COL].astype(str).values if LABEL_COL in df.columns else np.zeros(len(df), dtype=int)
sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
tr_idx, te_idx = next(sss.split(df, labels_all))

df_tr = df.iloc[tr_idx].reset_index(drop=True)
df_te = df.iloc[te_idx].reset_index(drop=True)
print(f"Train n={len(df_tr)}, Test n={len(df_te)}")



# ===== CELL 9 — Build Features =====

# ----- TABULAR (clinical + radiomics) -----
if USE_TABULAR and NUMERIC_COLUMNS:
    num_cols = NUMERIC_COLUMNS[:]
    cat_cols = []  # add if you have
    tab_pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp",  SimpleImputer(strategy="median")),
            ("sc",   StandardScaler(with_mean=True, with_std=True)),
        ]), num_cols),
        ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], remainder="drop")
    X_tab_tr = tab_pre.fit_transform(df_tr[num_cols + cat_cols])
    X_tab_te = tab_pre.transform(df_te[num_cols + cat_cols])
else:
    # empty placeholders when tabular disabled
    X_tab_tr = np.zeros((len(df_tr), 0), dtype=float)
    X_tab_te = np.zeros((len(df_te), 0), dtype=float)


# ----- TEXT -----
if USE_TEXT and TEXT_COLUMNS:
    texts_tr = df_tr[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1).tolist()
    texts_te = df_te[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1).tolist()
    txt_enc = TextEncoder(model_name=ENCODER_NAME, max_length=BERT_MAX_LENGTH, batch_size=BATCH_SIZE).fit(texts_tr)
    X_text_raw_tr = txt_enc.transform(texts_tr)   # e.g., 768
    X_text_raw_te = txt_enc.transform(texts_te)
    text_scaler = StandardScaler()
    X_text_tr_s = text_scaler.fit_transform(X_text_raw_tr)
    X_text_te_s = text_scaler.transform(X_text_raw_te)
    text_pca = PCA(n_components=TEXT_PCA_DIM, random_state=RANDOM_SEED)
    X_text_proc_tr = text_pca.fit_transform(X_text_tr_s)
    X_text_proc_te = text_pca.transform(X_text_te_s)
    print(f"Text PCA: kept {TEXT_PCA_DIM}, explains {text_pca.explained_variance_ratio_.sum():.2%}")
else:
    X_text_raw_tr = np.zeros((len(df_tr), 0)); X_text_raw_te = np.zeros((len(df_te), 0))
    X_text_proc_tr = X_text_raw_tr;            X_text_proc_te = X_text_raw_te


# IMAGE
# ----- IMAGE -----
if USE_IMAGE and X_img_raw_all.shape[1] > 0:
    X_img_raw_tr = X_img_raw_all[tr_idx]
    X_img_raw_te = X_img_raw_all[te_idx]
    has_img_tr   = has_img_flag_all[tr_idx]
    has_img_te   = has_img_flag_all[te_idx]

    X_img_raw_tr = normalize(X_img_raw_tr, norm="l2", axis=1)
    X_img_raw_te = normalize(X_img_raw_te, norm="l2", axis=1)

    img_scaler = StandardScaler()
    X_img_tr_s = img_scaler.fit_transform(X_img_raw_tr)
    X_img_te_s = img_scaler.transform(X_img_raw_te)

    if IMG_PCA_DIM < X_img_tr_s.shape[1]:
        img_pca = PCA(n_components=IMG_PCA_DIM, random_state=RANDOM_SEED)
        X_img_proc_tr = img_pca.fit_transform(X_img_tr_s)
        X_img_proc_te = img_pca.transform(X_img_te_s)
        print(f"Image PCA: kept {IMG_PCA_DIM}, explains {img_pca.explained_variance_ratio_.sum():.2%}")
    else:
        img_pca = None
        X_img_proc_tr = X_img_tr_s
        X_img_proc_te = X_img_te_s
else:
    # no image features
    X_img_raw_tr = np.zeros((len(df_tr), 0)); X_img_raw_te = np.zeros((len(df_te), 0))
    X_img_proc_tr = X_img_raw_tr;             X_img_proc_te = X_img_raw_te
    has_img_tr = np.zeros((len(df_tr),1), dtype=np.float32)
    has_img_te = np.zeros((len(df_te),1), dtype=np.float32)


print("Shapes → tabular:", X_tab_tr.shape, " | text_raw:", X_text_raw_tr.shape, " | text_proc:", X_text_proc_tr.shape,
      " | img_raw:", X_img_raw_tr.shape, " | img_proc:", X_img_proc_tr.shape)



# ===== CELL 10 — Fuse & Fused PCA =====
def build_fused_raw(X_tab, X_text_raw, X_img_raw, has_img):
    parts = [X_tab]
    if X_text_raw is not None and X_text_raw.shape[1] > 0: parts.append(X_text_raw)
    if X_img_raw is not None and X_img_raw.shape[1] > 0:   parts.append(X_img_raw)
    if has_img is not None:                                 parts.append(has_img)
    return np.hstack(parts)

def build_fused_proc_base(X_tab, X_text_proc, X_img_proc, has_img):
    parts = [X_tab]
    if X_text_proc is not None and X_text_proc.shape[1] > 0: parts.append(X_text_proc)
    if X_img_proc is not None and X_img_proc.shape[1] > 0:   parts.append(X_img_proc)
    if has_img is not None:                                   parts.append(has_img)
    return np.hstack(parts)

X_fused_raw_tr = build_fused_raw(X_tab_tr, X_text_raw_tr, X_img_raw_tr, has_img_tr)
X_fused_raw_te = build_fused_raw(X_tab_te, X_text_raw_te, X_img_raw_te, has_img_te)
print("fused_raw dims: train", X_fused_raw_tr.shape, "| test", X_fused_raw_te.shape)

X_fused_proc_base_tr = build_fused_proc_base(X_tab_tr, X_text_proc_tr, X_img_proc_tr, has_img_tr)
X_fused_proc_base_te = build_fused_proc_base(X_tab_te, X_text_proc_te, X_img_proc_te, has_img_te)
print("fused_proc_base dims: train", X_fused_proc_base_tr.shape, "| test", X_fused_proc_base_te.shape)

if FUSED_PCA_DIM and FUSED_PCA_DIM < X_fused_proc_base_tr.shape[1]:
    fused_pca = PCA(n_components=FUSED_PCA_DIM, random_state=RANDOM_SEED)
    X_fused_proc_tr = fused_pca.fit_transform(X_fused_proc_base_tr)
    X_fused_proc_te = fused_pca.transform(X_fused_proc_base_te)
    print(f"Fused PCA: kept {FUSED_PCA_DIM}, explains {fused_pca.explained_variance_ratio_.sum():.2%}")
else:
    fused_pca = None
    X_fused_proc_tr = X_fused_proc_base_tr
    X_fused_proc_te = X_fused_proc_base_te

print("fused_proc dims: train", X_fused_proc_tr.shape, "| test", X_fused_proc_te.shape)



# ===== CELL 11 — Save per-record fused vectors + summary =====
ensure_dir(PER_REC_ROOT)
all_summary = []

def write_split(df_split, X_fused_raw, X_fused_proc, has_img_split, split_tag: str):
    for i in range(len(df_split)):
        id_str = str(df_split.loc[i, ID_COL])
        rec_dir = os.path.join(PER_REC_ROOT, id_str)
        ensure_dir(rec_dir)

        v_raw  = X_fused_raw[i].astype(np.float32)
        v_proc = X_fused_proc[i].astype(np.float32)
        has_img = bool(has_img_split[i, 0]) if (has_img_split is not None and has_img_split.shape[1]==1) else True

        _, _, meta_raw  = write_vector_and_json(rec_dir, "fused_raw",  v_raw,  id_str, has_img)
        _, _, meta_proc = write_vector_and_json(rec_dir, "fused_proc", v_proc, id_str, has_img)

        all_summary.append({
            "id": id_str, "split": split_tag, "path": rec_dir, **meta_proc
        })

write_split(df_tr, X_fused_raw_tr, X_fused_proc_tr, has_img_tr, split_tag="train")
write_split(df_te, X_fused_raw_te, X_fused_proc_te, has_img_te, split_tag="test")

with open(GLOBAL_SUMMARY_PATH, "w") as f:
    json.dump(all_summary, f, indent=2)

print(f"Done. Wrote per-record fused vectors under: {PER_REC_ROOT}")
print(f"Global summary: {GLOBAL_SUMMARY_PATH}   (records: {len(all_summary)})")



# ===== CELL 12 — Metrics (RF + PR/CM) =====
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit

class_names = np.unique(df[LABEL_COL].astype(str).values)
C = len(class_names)

y_tr = pd.Categorical(df_tr[LABEL_COL].astype(str).values, categories=class_names).codes
y_te = pd.Categorical(df_te[LABEL_COL].astype(str).values, categories=class_names).codes

def expand_proba(proba, model_classes, n_classes):
    out = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
    out[:, model_classes] = proba
    return out

def show_confusion_matrix(y_true, y_pred, class_names, title, save_prefix=None, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    print(f"{title} (rows=true, cols=pred):\n", cm)
    fig, ax = plt.subplots(figsize=(6,5), dpi=140)
    im = ax.imshow(cm_disp, interpolation="nearest")
    ax.set_xticks(np.arange(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title + (" (normalized)" if normalize else ""))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm_disp[i,j]:.2f}" if normalize else f"{cm[i,j]}"
            ax.text(j, i, val, ha="center", va="center", fontsize=8)
    plt.tight_layout(); plt.show()
    if save_prefix:
        os.makedirs("./outputs/metrics", exist_ok=True)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(f"./outputs/metrics/{save_prefix}.csv", index=True)
        fig.savefig(f"./outputs/metrics/{save_prefix}.png", dpi=200)
        plt.close(fig)
    return cm

USE_FUSED_RAW = False
X_tr_eval = X_fused_raw_tr if USE_FUSED_RAW else X_fused_proc_tr
X_te_eval = X_fused_raw_te if USE_FUSED_RAW else X_fused_proc_te
vec_tag   = "fused_raw" if USE_FUSED_RAW else "fused_proc"

weights = compute_class_weight("balanced", classes=np.arange(C), y=y_tr)
class_weight = {i: w for i, w in enumerate(weights)}

rf = RandomForestClassifier(
    n_estimators=800, max_depth=None, min_samples_leaf=1, min_samples_split=2,
    max_features="sqrt", class_weight=class_weight, n_jobs=-1, random_state=RANDOM_SEED
)

# Argmax
rf.fit(X_tr_eval, y_tr)
proba_te_raw = rf.predict_proba(X_te_eval)
proba_te = expand_proba(proba_te_raw, rf.classes_, C)
y_arg    = proba_te.argmax(1)

acc_arg  = accuracy_score(y_te, y_arg)
f1m_arg  = f1_score(y_te, y_arg, average="macro")
f1u_arg  = f1_score(y_te, y_arg, average="micro")
f1w_arg  = f1_score(y_te, y_arg, average="weighted")
print(f"\n[{vec_tag}] Argmax → Acc={acc_arg:.3f} | F1(macro)={f1m_arg:.3f} | F1(micro)={f1u_arg:.3f} | F1(weighted)={f1w_arg:.3f}")

print("\nClassification report (argmax):")
print(classification_report(
    y_te, y_arg,
    labels=np.arange(C), target_names=class_names,
    digits=3, zero_division=0
))

_ = show_confusion_matrix(y_te, y_arg, class_names,
                          title=f"Confusion Matrix — {vec_tag} (argmax)",
                          save_prefix=f"{vec_tag}_cm_argmax", normalize=False)

# Threshold tuning
def tune_thresholds_holdout(X_tr, y_tr, model, n_classes, seed=RANDOM_SEED,
                            val_size=0.2, grid=np.linspace(0.4, 0.6, 9)):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_i, va_i = next(sss.split(X_tr, y_tr))
    X_in, y_in = X_tr[tr_i], y_tr[tr_i]
    X_va, y_va = X_tr[va_i], y_tr[va_i]
    model.fit(X_in, y_in)
    proba_va_raw = model.predict_proba(X_va)
    proba_va = expand_proba(proba_va_raw, model.classes_, n_classes)
    best = np.full(n_classes, 0.5, dtype=float)
    for c in range(n_classes):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            y_hat = proba_va.argmax(1).copy()
            y_hat = np.where(proba_va[:, c] >= t, c, y_hat)
            f1m = f1_score(y_va, y_hat, average="macro")
            if f1m > best_f1:
                best_f1, best_t = f1m, t
        best[c] = best_t
    return best

rf_tune = RandomForestClassifier(
    n_estimators=800, max_depth=None, min_samples_leaf=1, min_samples_split=2,
    max_features="sqrt", class_weight=class_weight, n_jobs=-1, random_state=RANDOM_SEED
)
thr = tune_thresholds_holdout(X_tr_eval, y_tr, rf_tune, n_classes=C)
print("Per-class thresholds (hold-out):", {class_names[i]: float(thr[i]) for i in range(C)})

def apply_thresholds(proba, thresholds):
    y = proba.argmax(1).copy()
    for c, t in enumerate(thresholds):
        y = np.where(proba[:, c] >= t, c, y)
    return y

y_thr = apply_thresholds(proba_te, thr)
acc_thr = accuracy_score(y_te, y_thr)
f1m_thr = f1_score(y_te, y_thr, average="macro")
f1u_thr = f1_score(y_te, y_thr, average="micro")
f1w_thr = f1_score(y_te, y_thr, average="weighted")
print(f"[{vec_tag}] Thresholded → Acc={acc_thr:.3f} | F1(macro)={f1m_thr:.3f} | F1(micro)={f1u_thr:.3f} | F1(weighted)={f1w_thr:.3f}")

print("\nClassification report (thresholded):")
print(classification_report(
    y_te, y_thr,
    labels=np.arange(C), target_names=class_names,
    digits=3, zero_division=0
))

_ = show_confusion_matrix(y_te, y_thr, class_names,
                          title=f"Confusion Matrix — {vec_tag} (thresholded)",
                          save_prefix=f"{vec_tag}_cm_thresholded", normalize=False)

# PR curves
Y_bin = pd.get_dummies(pd.Series(y_te), columns=range(C)).values
ap_per_class = {}
fig, ax = plt.subplots(figsize=(7,5), dpi=140)
for k, name in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(Y_bin[:, k], proba_te[:, k])
    ap = average_precision_score(Y_bin[:, k], proba_te[:, k])
    ap_per_class[name] = ap
    ax.plot(recall, precision, label=f"{name} (AP={ap:.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.grid(True, alpha=0.3)
ax.set_title(f"Per-class PR — {vec_tag}")
ax.legend(fontsize=8); plt.tight_layout()

os.makedirs("./outputs/metrics", exist_ok=True)
plt.savefig(f"./outputs/metrics/{vec_tag}_per_class_pr.png", dpi=200)
plt.show()

summary = pd.DataFrame([{
    "vector": vec_tag,
    "acc_argmax": acc_arg, "f1_macro_argmax": f1m_arg, "f1_micro_argmax": f1u_arg, "f1_weighted_argmax": f1w_arg,
    "acc_thr": acc_thr,     "f1_macro_thr": f1m_thr,   "f1_micro_thr": f1u_thr,   "f1_weighted_thr": f1w_thr,
    **{f"AP_{k}": v for k, v in ap_per_class.items()}
}])
summary_path = f"./outputs/metrics/{vec_tag}_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"\nSaved metrics summary → {summary_path}")


