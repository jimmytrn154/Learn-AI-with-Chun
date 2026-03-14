#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CMMD2 Baseline Runner (Monte Carlo CV, 50 runs)
- Modalities: Image embeddings (from JSON) + Radiomics (+ optional Text via Bio_ClinicalBERT)
- Final classifier: RandomForest
- Splits: provided monte_carlo_splits.csv with columns run_0..run_49 and values in {train, val, test}

This file is intentionally organized like your original pipeline: all key paths/models/columns are
declared at the top, and the rest is pure logic.

Author: ChatGPT (GPT-5.2 Thinking)
"""

# =========================
# CONFIG (EDIT THESE FIRST)
# =========================

# ---- Paths
# ---- Paths
MAIN_CSV_PATH          = "../input_csv/cmmd2_malignant_mavric_ready.csv"
EMB_JSON_PATH          = "../data/radiomics/cmmd_embeddings_full.json"   # full embeddings JSON (no truncation)
RADIOMICS_CSV_PATH     = "../data/radiomics/final_radiomics_features.csv"
MONTE_CARLO_SPLITS_CSV = "../input_csv/monte_carlo_splits.csv"
OUTPUT_DIR             = "/home/24chuong.ta/anaconda3/chun/Replicate_CMMD_Baseline/result"

# ---- Modalities toggles
USE_IMAGE    = False
USE_RAD      = False
USE_TEXT     = True   # set True to include ClinicalBERT embeddings


RADIOMICS_FEATURES = [
  "original_shape2D_Elongation",
    "original_shape2D_Perimeter",
    "original_shape2D_MinorAxisLength",
    "original_shape2D_MajorAxisLength",
    "original_shape2D_Sphericity",
    "original_shape2D_MeshSurface",
]


# ---- Label + IDs
LABEL_COL_CANDIDATES = ["Subtype", "subtype", "label", "Label"]
# If your MAIN CSV already has a unique sample id column, set it here (else leave None to auto-build).
MAIN_SAMPLE_ID_COL = None

# ---- Join keys (preferred in MAIN CSV)
PATIENT_COL_CANDIDATES     = ["PatientID", "patient_id", "patient", "patientid"]
LATERALITY_COL_CANDIDATES  = ["Laterality", "laterality", "side", "breast_side"]
VIEW_COL_CANDIDATES        = ["View", "view", "projection"]

# ---- How to parse image_id strings in embeddings JSON
# Example: "D2-0032_R_MLO" -> PatientID="D2-0032", Laterality="R", View="MLO"
EMB_IMAGE_ID_FIELD = "image_id"
EMB_VECTOR_FIELD   = "embedding_vector"

# ---- Radiomics keys
# If radiomics has (PatientID, Laterality, View) columns, they'll be used.
# Otherwise, if it has a single ID-like column (e.g., image_id), set it here:
RADIOMICS_ID_COL_CANDIDATES = ["image_id", "ImageID", "id", "ID"]

# ---- Text construction (IMPORTANT: avoid label leakage)
# These are structured columns from MAIN CSV that are safe to use (do NOT include text_report containing "Subtype: ...").
DEFAULT_TEXT_COLS = [
    "Breast density", "Mass", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9",
    "Calcification", "Unnamed: 13", "Unnamed: 14"
] # , "BI-RADS\nCategory" can add this later

# ---- Text encoder (ClinicalBERT)
ENCODER_NAME = "emilyalsentzer/Bio_ClinicalBERT"
TEXT_POOLING = "cls"   # "cls" or "mean"
TEXT_MAX_LEN = 192
TEXT_BATCH_SIZE = 16
TEXT_DEVICE = "cuda"   # "cuda" or "cpu" (will fall back to cpu if cuda not available)

# ---- Preprocessing / Dimensionality reduction
# If you don't want PCA, set *_PCA_DIM = None.
IMG_L2_NORM = True
IMG_PCA_DIM = 64

RAD_PCA_DIM = None      # often radiomics already compact; set to e.g. 64 if needed
TEXT_PCA_DIM = 96     # you can set to 64/128 if you want after BERT

FUSED_PCA_DIM = 64      # PCA on concatenated processed features (optional). Set None to disable.
ADD_PRESENCE_FLAGS = True  # add has_img/has_rad/has_text to fused vector

# ---- Monte Carlo split settings
RUN_PREFIX = "run_"     # columns named run_0 .. run_49
EXPECTED_RUNS = 50
# Whether to train on train+val and test on test (recommended for Monte Carlo evaluation)
TRAIN_ON_TRAINVAL = True

# ---- Classifier
RF_N_ESTIMATORS = 600
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 1
RF_RANDOM_STATE = 42
RF_N_JOBS = -1

# ---- Output details
SAVE_PER_RUN_CONFUSION_MATRIX = True
SAVE_PER_RUN_CLASSIFICATION_REPORT = True
VERBOSE = True

# =========================
# END CONFIG
# =========================


import os
import json
import re
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# optional plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_image_id(image_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse embeddings JSON 'image_id' into (PatientID, Laterality, View).
    Default expects: "{patient}_{L|R}_{VIEW}" e.g., "D2-0032_R_MLO"
    """
    if not isinstance(image_id, str):
        return None, None, None
    s = image_id.strip()
    parts = s.split("_")
    if len(parts) >= 3:
        patient = parts[0]
        lat = parts[1]
        view = parts[2]
        return patient, lat, view
    # fallback: try regex "patient_(L|R)_(CC|MLO|...)" or similar
    m = re.match(r"(.+)_([LR])_([A-Za-z0-9]+)", s)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None, None, None


def build_sample_id(patient: str, laterality: str, view: str) -> str:
    return f"{_safe_str(patient)}_{_safe_str(laterality)}_{_safe_str(view)}"


def load_main_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # label
    label_col = _pick_first_existing(df.columns.tolist(), LABEL_COL_CANDIDATES)
    if label_col is None:
        raise ValueError(f"Could not find label column in MAIN CSV. Tried: {LABEL_COL_CANDIDATES}")
    df = df.copy()
    df["_label_"] = df[label_col].astype(str).str.strip()

    # find keys
    pcol = _pick_first_existing(df.columns.tolist(), PATIENT_COL_CANDIDATES)
    lcol = _pick_first_existing(df.columns.tolist(), LATERALITY_COL_CANDIDATES)
    vcol = _pick_first_existing(df.columns.tolist(), VIEW_COL_CANDIDATES)
    if pcol is None or lcol is None or vcol is None:
        raise ValueError(
            "Could not find Patient/Laterality/View columns in MAIN CSV.\n"
            f"Patient candidates: {PATIENT_COL_CANDIDATES}\n"
            f"Laterality candidates: {LATERALITY_COL_CANDIDATES}\n"
            f"View candidates: {VIEW_COL_CANDIDATES}\n"
            f"Found: patient={pcol}, laterality={lcol}, view={vcol}"
        )

    df["_patient_"] = df[pcol].astype(str).str.strip()
    df["_lat_"] = df[lcol].astype(str).str.strip()
    df["_view_"] = df[vcol].astype(str).str.strip()

    # sample id
    if MAIN_SAMPLE_ID_COL and MAIN_SAMPLE_ID_COL in df.columns:
        df["_sid_"] = df[MAIN_SAMPLE_ID_COL].astype(str).str.strip()
    else:
        df["_sid_"] = df.apply(lambda r: build_sample_id(r["_patient_"], r["_lat_"], r["_view_"]), axis=1)

    # drop unknown-ish labels (mirrors your previous filtering idea, but gentle)
    bad = {"unknown", "na", "nan", "none", ""}
    df = df[~df["_label_"].str.lower().isin(bad)].reset_index(drop=True)
    return df


def load_embeddings_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for obj in data:
        image_id = obj.get(EMB_IMAGE_ID_FIELD, None)
        vec = obj.get(EMB_VECTOR_FIELD, None)
        if image_id is None or vec is None:
            continue
        patient, lat, view = parse_image_id(image_id)
        if patient is None:
            continue
        rows.append({
            "_sid_": build_sample_id(patient, lat, view),
            "_image_id_": image_id,
            "_img_vec_": np.asarray(vec, dtype=np.float32),
        })
    if not rows:
        raise ValueError("No embeddings loaded from JSON. Check EMB_IMAGE_ID_FIELD / EMB_VECTOR_FIELD.")
    df = pd.DataFrame(rows)
    # ensure same dim
    dims = df["_img_vec_"].apply(lambda x: int(x.shape[0])).unique().tolist()
    if len(dims) != 1:
        raise ValueError(f"Embeddings have inconsistent dimensions: {dims[:10]} ...")
    return df


def load_radiomics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = df.columns.tolist()

    # Try (PatientID, Laterality, View) first
    pcol = _pick_first_existing(cols, PATIENT_COL_CANDIDATES)
    lcol = _pick_first_existing(cols, LATERALITY_COL_CANDIDATES)
    vcol = _pick_first_existing(cols, VIEW_COL_CANDIDATES)

    df2 = df.copy()

    if pcol and lcol and vcol:
        df2["_sid_"] = df2.apply(lambda r: build_sample_id(r[pcol], r[lcol], r[vcol]), axis=1)
    else:
        # fall back to an ID column
        idcol = _pick_first_existing(cols, RADIOMICS_ID_COL_CANDIDATES)
        if idcol is None:
            raise ValueError(
                "Could not infer radiomics join keys. Provide (PatientID, Laterality, View) OR an id column.\n"
                f"Tried id candidates: {RADIOMICS_ID_COL_CANDIDATES}"
            )
        df2["_sid_"] = df2[idcol].astype(str).str.strip()
        # If that id is actually like D2-0032_R_MLO, normalize to sid:
        # (If it's already sid, this is idempotent enough)
        df2["_sid_"] = df2["_sid_"].apply(lambda s: build_sample_id(*parse_image_id(s)) if "_" in s and len(s.split("_"))>=3 else s)

    # keep only numeric features (excluding keys)
    drop_like = {pcol, lcol, vcol, "_sid_"}
    if "Subtype" in df2.columns:
        drop_like.add("Subtype")
    if "subtype" in df2.columns:
        drop_like.add("subtype")

    feature_cols = []
    for c in df2.columns:
        if c in drop_like:
            continue
        if pd.api.types.is_numeric_dtype(df2[c]):
            feature_cols.append(c)

    if not feature_cols:
        # attempt coercion
        numeric = df2.copy()
        for c in df2.columns:
            if c in drop_like:
                continue
            numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
        feature_cols = [c for c in numeric.columns if c not in drop_like and pd.api.types.is_numeric_dtype(numeric[c])]
        df2 = numeric

    if not feature_cols:
        raise ValueError("No numeric radiomics feature columns found.")

    out = df2[["_sid_"] + feature_cols].copy()
    return out


def build_text_strings(df_main: pd.DataFrame, text_cols: List[str]) -> pd.Series:
    existing = [c for c in text_cols if c in df_main.columns]
    if not existing:
        raise ValueError(f"None of the configured text columns exist in MAIN CSV. Tried: {text_cols}")
    def row_to_text(r):
        parts = []
        for c in existing:
            val = _safe_str(r.get(c, ""))
            if val:
                parts.append(f"{c}: {val}")
        return " | ".join(parts)
    return df_main.apply(row_to_text, axis=1)


def embed_texts_clinicalbert(texts: List[str], encoder_name: str, pooling: str,
                            max_len: int, batch_size: int, device_pref: str) -> np.ndarray:
    """
    Returns: (N, D) float32 embeddings.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cpu"
    if device_pref == "cuda" and torch.cuda.is_available():
        device = "cuda"

    tok = AutoTokenizer.from_pretrained(encoder_name)
    mdl = AutoModel.from_pretrained(encoder_name)
    mdl.eval()
    mdl.to(device)

    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            last = out.last_hidden_state  # (B, T, H)

            if pooling == "cls":
                vec = last[:, 0, :]  # (B, H)
            elif pooling == "mean":
                attn = enc.get("attention_mask", None)
                if attn is None:
                    vec = last.mean(dim=1)
                else:
                    mask = attn.unsqueeze(-1).type_as(last)  # (B,T,1)
                    summed = (last * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp(min=1e-6)
                    vec = summed / denom
            else:
                raise ValueError("TEXT_POOLING must be 'cls' or 'mean'.")

            all_vecs.append(vec.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(all_vecs, axis=0)


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def fit_transform_scaler_pca(X_train: np.ndarray, X_all: np.ndarray, pca_dim: Optional[int], prefix: str):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xall = scaler.transform(X_all)
    pca = None
    if pca_dim is not None:
        pca = PCA(n_components=pca_dim, random_state=RF_RANDOM_STATE)
        Xtr = pca.fit_transform(Xtr)
        Xall = pca.transform(Xall)
    return Xtr, Xall, scaler, pca


def run():
    _ensure_dir(OUTPUT_DIR)

    if VERBOSE:
        print("Loading MAIN CSV...")
    main = load_main_csv(MAIN_CSV_PATH)

    # Load MC splits
    if VERBOSE:
        print("Loading Monte Carlo splits...")
    splits = pd.read_csv(MONTE_CARLO_SPLITS_CSV)
    if "_sid_" not in splits.columns:
        # try infer: first column might be ID
        first = splits.columns[0]
        if first.lower() in ["id", "ID".lower(), "sample_id", "sid", "image_id"]:
            splits = splits.rename(columns={first: "_sid_"})
        else:
            # assume first column is an index-like id
            splits = splits.rename(columns={first: "_sid_"})

    # run_cols = [c for c in splits.columns if c.startswith(RUN_PREFIX)]
    run_cols = ['run_0', 'run_1', 'run_2', 'run_3', 'run_4']
    if EXPECTED_RUNS and len(run_cols) < EXPECTED_RUNS:
        # still proceed, but warn
        print(f"[WARN] Found {len(run_cols)} run columns, expected {EXPECTED_RUNS}. Columns: {run_cols[:5]}...")

    splits["_sid_"] = splits["_sid_"].astype(str).str.strip()

    # Merge splits onto main
    # Splits may be patient-level (PatientID only) or sample-level (Patient_Lat_View).
    split_ids = set(splits["_sid_"].astype(str).tolist())
    main_sid_set = set(main["_sid_"].astype(str).tolist())
    main_pid_set = set(main["_patient_"].astype(str).tolist())

    match_sid = len(split_ids & main_sid_set)
    match_pid = len(split_ids & main_pid_set)

    if VERBOSE:
        print(f"[INFO] Split ID overlap: with main _sid_ = {match_sid}, with main PatientID = {match_pid}")

    if match_pid > match_sid:
        if VERBOSE:
            print("[INFO] Splits look patient-level; merging splits on PatientID (main _patient_).")
        splits = splits.rename(columns={"_sid_": "_patient_"})
        df = main.merge(splits, on="_patient_", how="inner")
    else:
        if VERBOSE:
            print("[INFO] Splits look sample-level; merging splits on _sid_.")
        df = main.merge(splits, on="_sid_", how="inner")

    if df.empty:
        ex_main_sid = main["_sid_"].astype(str).head(3).tolist()
        ex_main_pid = main["_patient_"].astype(str).head(3).tolist()
        ex_split = splits[splits.columns[0]].astype(str).head(3).tolist()
        raise ValueError(
            "After merging MAIN with splits, there are 0 rows. Check ID formats.\n"
            f"Examples main _sid_: {ex_main_sid}\n"
            f"Examples main PatientID: {ex_main_pid}\n"
            f"Examples splits IDs: {ex_split}\n"
            "Tip: ensure splits first column matches either PatientID (e.g., D2-0032) or sample id (e.g., D2-0032_R_MLO)."
        )

    # Load embeddings JSON
    emb_df = None
    if USE_IMAGE:
        if VERBOSE:
            print("Loading embeddings JSON...")
        emb_df = load_embeddings_json(EMB_JSON_PATH)
        df = df.merge(emb_df[["_sid_", "_img_vec_"]], on="_sid_", how="left")
        # set missing to zeros
        img_dim = int(emb_df["_img_vec_"].iloc[0].shape[0])
        df["_has_img_"] = df["_img_vec_"].apply(lambda v: 0 if v is None or (isinstance(v, float) and np.isnan(v)) else 1)
        df["_img_vec_"] = df["_img_vec_"].apply(lambda v: np.zeros((img_dim,), dtype=np.float32) if not isinstance(v, np.ndarray) else v)
    else:
        df["_has_img_"] = 0

    # Load radiomics
    rad_df = None
    # Load radiomics
    rad_df = None
    if USE_RAD:
        if VERBOSE:
            print("Loading radiomics...")
        rad_df = load_radiomics(RADIOMICS_CSV_PATH)

        # choose which radiomics columns to use
        if RADIOMICS_FEATURES is not None and len(RADIOMICS_FEATURES) > 0:
            rad_feature_cols = [c for c in RADIOMICS_FEATURES if c in rad_df.columns]
        else:
            rad_feature_cols = [c for c in rad_df.columns if c != "_sid_"]

        rad_df = rad_df[["_sid_"] + rad_feature_cols].copy()
        df = df.merge(rad_df, on="_sid_", how="left")

        # compute has_rad BEFORE filling NaNs
        df["_has_rad_"] = df[rad_feature_cols].notna().all(axis=1).astype(int)

        # coerce + fill
        for c in rad_feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[rad_feature_cols] = df[rad_feature_cols].fillna(0.0)

        # now safe to print debug
        print("Radiomics features used:", len(rad_feature_cols))
        print("Has radiomics ratio:", df["_has_rad_"].mean())
    else:
        df["_has_rad_"] = 0


    # Text embeddings (compute once globally; transforms per run will still be fit on train only)
    if USE_TEXT:
        if VERBOSE:
            print("Building text strings...")
        df["_text_"] = build_text_strings(df, DEFAULT_TEXT_COLS)
        if VERBOSE:
            print(f"Embedding {len(df)} texts with {ENCODER_NAME}...")
        text_vecs = embed_texts_clinicalbert(
            texts=df["_text_"].tolist(),
            encoder_name=ENCODER_NAME,
            pooling=TEXT_POOLING,
            max_len=TEXT_MAX_LEN,
            batch_size=TEXT_BATCH_SIZE,
            device_pref=TEXT_DEVICE
        )
        df["_text_vec_"] = list(text_vecs)
        df["_has_text_"] = (df["_text_"].str.len() > 0).astype(int)
        text_dim = text_vecs.shape[1]
        # empty text -> zeros (still keep shape)
        df["_text_vec_"] = df.apply(lambda r: np.zeros((text_dim,), dtype=np.float32) if r["_has_text_"]==0 else r["_text_vec_"], axis=1)
    else:
        df["_has_text_"] = 0

    # Labels
    y_all = df["_label_"].astype(str).values
    classes = sorted(pd.unique(y_all).tolist())

    # Prepare index mapping for fast selection
    sid_all = df["_sid_"].astype(str).tolist()
    sid_to_idx = {sid: i for i, sid in enumerate(sid_all)}

    # Run columns
    run_cols = [c for c in df.columns if c.startswith(RUN_PREFIX)]
    if not run_cols:
        raise ValueError(f"No run columns found (expected columns like {RUN_PREFIX}0). Check split file merge.")

    # Store metrics per run
    mc_rows = []

    # Prebuild raw modality arrays (lists of vectors to numpy matrices) for efficiency
    # Image matrix
    if USE_IMAGE:
        X_img = np.vstack(df["_img_vec_"].values).astype(np.float32)
        if IMG_L2_NORM:
            X_img = l2_normalize(X_img)
    else:
        X_img = None

    # Radiomics matrix
    if USE_RAD:
        rad_feature_cols = [c for c in df.columns if c in (rad_df.columns.tolist() if rad_df is not None else []) and c != "_sid_"]
        X_rad = df[rad_feature_cols].to_numpy(dtype=np.float32)
    else:
        X_rad = None

    # Text matrix
    if USE_TEXT:
        X_text = np.vstack(df["_text_vec_"].values).astype(np.float32)
    else:
        X_text = None

    presence_cols = []
    if ADD_PRESENCE_FLAGS:
        presence_cols = ["_has_img_", "_has_rad_", "_has_text_"]
        for pc in presence_cols:
            if pc not in df.columns:
                df[pc] = 0
        X_presence = df[presence_cols].to_numpy(dtype=np.float32)
    else:
        X_presence = None

    # Helper to build fused matrix after per-run processing
    def build_fused(X_img_proc_all, X_rad_proc_all, X_text_proc_all, X_presence_all):
        parts = []
        if USE_RAD:
            parts.append(X_rad_proc_all)
        if USE_TEXT:
            parts.append(X_text_proc_all)
        if USE_IMAGE:
            parts.append(X_img_proc_all)
        if ADD_PRESENCE_FLAGS:
            parts.append(X_presence_all)
        if not parts:
            raise ValueError("No modalities selected. Set at least one of USE_IMAGE/USE_RAD/USE_TEXT.")
        return np.concatenate(parts, axis=1)

    # Iterate runs
    for run_col in run_cols:
        run_name = run_col
        run_dir = os.path.join(OUTPUT_DIR, "per_run", run_name)
        _ensure_dir(run_dir)

        split_vals = df[run_col].astype(str).str.lower().fillna("").values

        idx_train = np.where(split_vals == "train")[0]
        idx_val   = np.where(split_vals == "val")[0]
        idx_test  = np.where(split_vals == "test")[0]

        if TRAIN_ON_TRAINVAL:
            idx_fit = np.concatenate([idx_train, idx_val])
        else:
            idx_fit = idx_train

        if len(idx_fit) == 0 or len(idx_test) == 0:
            print(f"[WARN] {run_name}: empty fit or test split. fit={len(idx_fit)}, test={len(idx_test)}. Skipping.")
            continue

        y_fit = y_all[idx_fit]
        y_test = y_all[idx_test]

        # Per-modality scaling/PCA fit on training split only
        # Image
        if USE_IMAGE:
            X_img_fit = X_img[idx_fit]
            X_img_fit_proc, X_img_all_proc, img_scaler, img_pca = fit_transform_scaler_pca(
                X_train=X_img_fit, X_all=X_img, pca_dim=IMG_PCA_DIM, prefix="img"
            )
        else:
            X_img_all_proc = None

        # Radiomics
        if USE_RAD:
            X_rad_fit = X_rad[idx_fit]
            X_rad_fit_proc, X_rad_all_proc, rad_scaler, rad_pca = fit_transform_scaler_pca(
                X_train=X_rad_fit, X_all=X_rad, pca_dim=RAD_PCA_DIM, prefix="rad"
            )
        else:
            X_rad_all_proc = None

        # Text
        if USE_TEXT:
            X_text_fit = X_text[idx_fit]
            X_text_fit_proc, X_text_all_proc, text_scaler, text_pca = fit_transform_scaler_pca(
                X_train=X_text_fit, X_all=X_text, pca_dim=TEXT_PCA_DIM, prefix="text"
            )
        else:
            X_text_all_proc = None

        # Fused
        X_fused_all = build_fused(X_img_all_proc, X_rad_all_proc, X_text_all_proc, X_presence if ADD_PRESENCE_FLAGS else None)
        X_fused_fit = X_fused_all[idx_fit]

        if FUSED_PCA_DIM is not None:
            fused_scaler = StandardScaler()
            X_fused_fit_s = fused_scaler.fit_transform(X_fused_fit)
            X_fused_all_s = fused_scaler.transform(X_fused_all)
            fused_pca = PCA(n_components=FUSED_PCA_DIM, random_state=RF_RANDOM_STATE)
            X_fused_fit_final = fused_pca.fit_transform(X_fused_fit_s)
            X_fused_all_final = fused_pca.transform(X_fused_all_s)
        else:
            # still scale for RF stability? optional; RF doesn't need it, but keep consistent:
            fused_scaler = None
            fused_pca = None
            X_fused_all_final = X_fused_all
            X_fused_fit_final = X_fused_fit

        X_test_final = X_fused_all_final[idx_test]

        # Train classifier
        clf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            class_weight="balanced",
            random_state=RF_RANDOM_STATE,
            n_jobs=RF_N_JOBS,
        )
        clf.fit(X_fused_fit_final, y_fit)
        y_pred = clf.predict(X_test_final)

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")

        # Save report
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=classes)

        if SAVE_PER_RUN_CLASSIFICATION_REPORT:
            rep_path = os.path.join(run_dir, "classification_report.json")
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)

        if SAVE_PER_RUN_CONFUSION_MATRIX:
            fig = plt.figure(figsize=(8, 7))
            plt.imshow(cm, interpolation="nearest")
            plt.title(f"Confusion Matrix ({run_name})")
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45, ha="right")
            plt.yticks(tick_marks, classes)
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            fig_path = os.path.join(run_dir, "confusion_matrix.png")
            plt.savefig(fig_path, dpi=180, bbox_inches="tight")
            plt.close(fig)

        mc_rows.append({
            "run": run_name,
            "n_fit": int(len(idx_fit)),
            "n_test": int(len(idx_test)),
            "accuracy": float(acc),
            "macro_f1": float(f1m),
        })

        if VERBOSE:
            print(f"{run_name}: acc={acc:.4f}, macro_f1={f1m:.4f} (fit={len(idx_fit)}, test={len(idx_test)})")

    # Save aggregate metrics
    mc_df = pd.DataFrame(mc_rows).sort_values("run")
    mc_df.to_csv(os.path.join(OUTPUT_DIR, "montecarlo_metrics.csv"), index=False)

    summary = {}
    if not mc_df.empty:
        summary = {
            "n_runs_completed": int(len(mc_df)),
            "accuracy_mean": float(mc_df["accuracy"].mean()),
            "accuracy_std": float(mc_df["accuracy"].std(ddof=1)) if len(mc_df) > 1 else 0.0,
            "macro_f1_mean": float(mc_df["macro_f1"].mean()),
            "macro_f1_std": float(mc_df["macro_f1"].std(ddof=1)) if len(mc_df) > 1 else 0.0,
            "config": {
                "USE_IMAGE": USE_IMAGE,
                "USE_RAD": USE_RAD,
                "USE_TEXT": USE_TEXT,
                "ENCODER_NAME": ENCODER_NAME if USE_TEXT else None,
                "TEXT_POOLING": TEXT_POOLING if USE_TEXT else None,
                "IMG_PCA_DIM": IMG_PCA_DIM,
                "RAD_PCA_DIM": RAD_PCA_DIM,
                "TEXT_PCA_DIM": TEXT_PCA_DIM,
                "FUSED_PCA_DIM": FUSED_PCA_DIM,
                "TRAIN_ON_TRAINVAL": TRAIN_ON_TRAINVAL,
                "RF_N_ESTIMATORS": RF_N_ESTIMATORS,
                "RF_MAX_DEPTH": RF_MAX_DEPTH,
            }
        }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if VERBOSE and summary:
        print("=== Summary ===")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
