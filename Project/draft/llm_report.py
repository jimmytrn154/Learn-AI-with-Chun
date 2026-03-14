#!/usr/bin/env python3
"""
Generate LLM report text per patient from:
  - radiomics features (6 MAVRIC-style)
  - selected clinical/text columns (8 cols)

Outputs:
  1) reports CSV with columns: PatientID, llm_report_text
  2) optional merged CSV: original metadata + llm_report_text

Safety / validity:
  - Do NOT include ground-truth labels/subtype in the prompt.
  - De-identify: we never send PatientID, paths, or identifiers to Gemini.
  - Deterministic-ish: temperature default 0.0
  - Cache results to avoid re-generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------
# Defaults (match your pipeline)
# ---------------------------

RADIOMICS_FEATURES_DEFAULT = [
    "original_shape2D_Elongation",
    "original_shape2D_Perimeter",
    "original_shape2D_MinorAxisLength",
    "original_shape2D_MajorAxisLength",
    "original_shape2D_Sphericity",
    "original_shape2D_MeshSurface",
]

TEXT_COLS_DEFAULT = [
    "Breast density", "Mass", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9",
    "Calcification", "Unnamed: 13", "Unnamed: 14"
]

SYSTEM_INSTRUCTIONS = (
    "You are a neutral radiology summarizer. "
    "You will be given numeric imaging descriptors (radiomics) and categorical imaging fields. "
    "Write a short, de-identified imaging summary. "
    "Do NOT predict or mention breast cancer subtype, diagnosis, malignancy status, or any label. "
    "Do NOT include patient identifiers, dates, IDs, site names, or file paths. "
    "Output EXACTLY 4 lines with these headings:\n"
    "Findings: ...\n"
    "Shape summary: ...\n"
    "Morphology cues: ...\n"
    "Overall impression: ...\n"
)

# ---------------------------
# Gemini callers (SDK + REST)
# ---------------------------

def _try_import_google_genai():
    try:
        from google import genai  # type: ignore
        return genai
    except Exception:
        return None


def call_gemini_sdk(prompt: str, model: str, api_key: Optional[str], temperature: float, max_output_tokens: int) -> str:
    """
    Uses the official google-genai SDK. Quickstart shows:
      from google import genai
      client = genai.Client()
      client.models.generate_content(model="gemini-2.5-flash", contents="...")
    :contentReference[oaicite:1]{index=1}
    """
    genai = _try_import_google_genai()
    if genai is None:
        raise RuntimeError("google-genai not installed. Run: pip install -U google-genai")

    # If api_key is None, SDK reads env var GEMINI_API_KEY
    # :contentReference[oaicite:2]{index=2}
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        client = genai.Client()

    # Keep it simple & compatible: send a single text prompt.
    # If you want advanced config, you can extend this later.
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    text = getattr(resp, "text", None)
    if not text:
        # best-effort extraction
        text = str(resp)
    return text.strip()


def call_gemini_rest(prompt: str, model: str, api_key: str, temperature: float, max_output_tokens: int) -> str:
    """
    REST endpoint in docs:
      POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
      header x-goog-api-key: $GEMINI_API_KEY
    :contentReference[oaicite:3]{index=3}
    """
    import requests

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    data = r.json()

    # candidates[0].content.parts[0].text per typical generateContent response
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return json.dumps(data)[:2000]


# ---------------------------
# Caching
# ---------------------------

@dataclass
class CacheEntry:
    input_hash: str
    model: str
    text: str


def load_cache(cache_path: str) -> Dict[str, CacheEntry]:
    """
    JSONL cache: one line per PatientID:
      {"PatientID":"...", "input_hash":"...", "model":"...", "text":"..."}
    """
    cache: Dict[str, CacheEntry] = {}
    if not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = str(obj["PatientID"])
            cache[pid] = CacheEntry(
                input_hash=str(obj["input_hash"]),
                model=str(obj["model"]),
                text=str(obj["text"]),
            )
    return cache


def append_cache(cache_path: str, patient_id: str, entry: CacheEntry) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "PatientID": patient_id,
            "input_hash": entry.input_hash,
            "model": entry.model,
            "text": entry.text,
        }, ensure_ascii=False) + "\n")


# ---------------------------
# Prompt building (no PatientID sent)
# ---------------------------

def stable_hash_payload(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def safe_str(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = str(v).strip()
    if s.lower() == "nan":
        return ""
    return s


def build_payload(row: pd.Series, rad_vec: Dict[str, float], text_cols: List[str]) -> Dict[str, Any]:
    clinical_fields = {}
    for c in text_cols:
        if c in row.index:
            clinical_fields[c] = safe_str(row[c])
        else:
            clinical_fields[c] = ""
    payload = {
        "radiomics": rad_vec,
        "clinical_fields": clinical_fields,
    }
    return payload


def build_prompt(payload: Dict[str, Any]) -> str:
    # Keep numbers readable and consistent
    rad = payload["radiomics"]
    rad_fmt = {k: (None if rad[k] is None else float(rad[k])) for k in rad.keys()}

    # Remove empty fields to reduce noise
    cf = {k: v for k, v in payload["clinical_fields"].items() if safe_str(v) != ""}

    prompt = (
        SYSTEM_INSTRUCTIONS
        + "\nINPUT (JSON):\n"
        + json.dumps(
            {"radiomics": rad_fmt, "clinical_fields": cf},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n\nWrite the 4-line summary now."
    )
    return prompt


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_csv", required=True, help="Main metadata CSV containing PatientID and text columns.")
    ap.add_argument("--radiomics_csv", required=True, help="Radiomics CSV with PatientID and feature columns.")
    ap.add_argument("--patient_col", default="PatientID", help="Patient ID column name in both files.")
    ap.add_argument("--radiomics_features", default="|".join(RADIOMICS_FEATURES_DEFAULT))
    ap.add_argument("--text_cols", default="|".join(TEXT_COLS_DEFAULT))

    ap.add_argument("--out_reports_csv", default="patient_llm_reports.csv")
    ap.add_argument("--out_merged_csv", default="", help="If set, writes main_csv merged with llm_report_text.")

    ap.add_argument("--cache_jsonl", default="cache/llm_reports_cache.jsonl")

    ap.add_argument("--backend", choices=["sdk", "rest"], default="sdk")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model id. (See Gemini docs.)")
    ap.add_argument("--api_key", default="", help="If empty, uses env var GEMINI_API_KEY (SDK/REST).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=256)

    ap.add_argument("--sleep_s", type=float, default=0.2, help="Delay between calls (helps rate limits).")
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--dry_run", action="store_true", help="Build prompts but don't call Gemini.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only process first N patients.")

    args = ap.parse_args()

    rad_feats = [x.strip() for x in args.radiomics_features.split("|") if x.strip()]
    text_cols = [x.strip() for x in args.text_cols.split("|") if x.strip()]

    api_key = args.api_key.strip() or os.environ.get("GEMINI_API_KEY", "").strip()

    if args.backend == "rest" and not api_key and not args.dry_run:
        raise ValueError("REST backend requires GEMINI_API_KEY set (env) or --api_key provided.")

    # Load and merge
    main_df = pd.read_csv(args.main_csv)
    rad_df = pd.read_csv(args.radiomics_csv)

    if args.patient_col not in main_df.columns:
        raise ValueError(f"main_csv missing patient_col={args.patient_col}")
    if args.patient_col not in rad_df.columns:
        raise ValueError(f"radiomics_csv missing patient_col={args.patient_col}")

    # Keep only needed columns
    needed_main_cols = [args.patient_col] + [c for c in text_cols if c in main_df.columns]
    missing_text = [c for c in text_cols if c not in main_df.columns]
    if missing_text:
        print(f"[WARN] Missing text cols in main_csv (will be treated as empty): {missing_text}")

    main_df_small = main_df[needed_main_cols].copy()

    missing_rad = [c for c in rad_feats if c not in rad_df.columns]
    if missing_rad:
        raise ValueError(f"radiomics_csv missing required radiomics features: {missing_rad}")

    rad_df_small = rad_df[[args.patient_col] + rad_feats].copy()

    # Merge (left join: keep all patients in main)
    df = main_df_small.merge(rad_df_small, on=args.patient_col, how="left")
    df[rad_feats] = df[rad_feats].astype(np.float32)
    df[rad_feats] = df[rad_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Unique patient list (one report per patient)
    patients = df[args.patient_col].astype(str).tolist()
    if args.limit > 0:
        patients = patients[: args.limit]

    # Load cache
    cache = load_cache(args.cache_jsonl)
    print(f"[INFO] Loaded cache entries: {len(cache)}")

    outputs = []
    n_new = 0

    # Choose caller
    def call_model(prompt: str) -> str:
        if args.dry_run:
            return "[DRY_RUN]"
        if args.backend == "sdk":
            return call_gemini_sdk(prompt, args.model, api_key if api_key else None, args.temperature, args.max_output_tokens)
        else:
            return call_gemini_rest(prompt, args.model, api_key, args.temperature, args.max_output_tokens)

    for i, pid in enumerate(patients):
        row = df.iloc[i]

        # Build radiomics dict
        rad_vec = {f: float(row[f]) for f in rad_feats}

        payload = build_payload(row, rad_vec, text_cols)
        input_hash = stable_hash_payload(payload)

        # Use cache if identical
        if pid in cache and cache[pid].input_hash == input_hash and cache[pid].model == args.model:
            text = cache[pid].text
        else:
            prompt = build_prompt(payload)

            # Retry with backoff
            text = ""
            for attempt in range(args.max_retries):
                try:
                    text = call_model(prompt)
                    break
                except Exception as e:
                    wait = (2 ** attempt) + random.random()
                    print(f"[WARN] pid={pid} attempt={attempt+1}/{args.max_retries} error={e} -> sleep {wait:.1f}s")
                    time.sleep(wait)

            if text == "":
                text = "[ERROR] generation_failed"

            entry = CacheEntry(input_hash=input_hash, model=args.model, text=text)
            append_cache(args.cache_jsonl, pid, entry)
            cache[pid] = entry
            n_new += 1

            time.sleep(args.sleep_s)

        outputs.append({"PatientID": pid, "llm_report_text": text})

        if args.dry_run and i < 2:
            print("\n--- DRY RUN PROMPT EXAMPLE ---")
            print(build_prompt(payload)[:1500])
            print("--- END ---\n")

        if (i + 1) % 50 == 0:
            print(f"[INFO] processed {i+1}/{len(patients)} (new={n_new})")

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(args.out_reports_csv, index=False)
    print(f"[OK] Wrote reports CSV: {args.out_reports_csv}  (new generations: {n_new})")

    if args.out_merged_csv.strip():
        merged = main_df.merge(out_df, on=args.patient_col, how="left")
        merged.to_csv(args.out_merged_csv, index=False)
        print(f"[OK] Wrote merged CSV: {args.out_merged_csv}")

    print("[DONE]")


if __name__ == "__main__":
    main()
