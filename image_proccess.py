#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py
Aplicación sencilla para:
- Preprocesamiento de imagen (OpenCV)
- Layout analysis (layoutparser)
- OCR/HTR (TrOCR via HuggingFace + fallback pytesseract)
- RAG básico: indexado con sentence-transformers + faiss
- Síntesis final: generación condicionada en pasajes recuperados (OpenAI o HF)
- Export a Markdown (Obsidian-friendly)

Uso:
  python app.py --input_dir images/ --docs_dir docs/ --output_dir output/

Requiere:
  pip install -r requirements.txt
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import logging
import json
import textwrap

# Visión / imagen
import cv2
import numpy as np

# OCR / HTR (HuggingFace)
from PIL import Image

# Huggingface transformers (TrOCR)
from transformers import VisionEncoderDecoderModel, AutoProcessor, AutoTokenizer, pipeline

# sentence-transformers + faiss
from sentence_transformers import SentenceTransformer
import faiss

# Optional: pytesseract fallback
import pytesseract

# Optional: layoutparser
try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except Exception:
    LAYOUTPARSER_AVAILABLE = False

# OpenAI (optional, for generation)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# --------------------
# Config (ajusta aquí)
# --------------------
TROCR_MODEL = "microsoft/trocr-base-handwritten"  # modelo TrOCR para manuscrito
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"  # para generación si no se usa OpenAI
USE_OPENAI = False  # si True, requiere OPENAI_API_KEY en env
MAX_CHUNK = 512  # tamaño de chunk para indexado
# --------------------

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

# --------------------
# Image preprocessing
# --------------------
def preprocess_image(path, out_path=None, deskew=True):
    """Preprocesa imagen: gris, blur, adaptative threshold, deskew (opcional)"""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    # Adaptive threshold
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,11,2)
    proc = th
    # Deskew (simple)
    if deskew:
        coords = np.column_stack(np.where(proc > 0))
        angle = 0.0
        try:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = proc.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            proc = cv2.warpAffine(proc, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass
    if out_path:
        cv2.imwrite(str(out_path), proc)
    return proc

# --------------------
# Layout analysis
# --------------------
"""
def detect_layout(path_image):
    
    if not LAYOUTPARSER_AVAILABLE:
        logging.warning("layoutparser no disponible - se procesará la imagen completa como una región")
        h, w = Image.open(path_image).size[::-1]
        return [{"type":"text", "bbox":[0,0,w,h]}]
    image = Image.open(path_image).convert("RGB")
    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                     extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                                     label_map={0: "text", 1: "title", 2: "list", 3: "table", 4: "figure"})
    layout = model.detect(image)
    regions = []
    for b in layout:
        regions.append({"type": b.type, "bbox": [int(b.block.x_1), int(b.block.y_1), int(b.block.x_2), int(b.block.y_2)]})
    return regions
"""
def detect_layout(path_image):
    from PIL import Image
    img = Image.open(path_image)
    w, h = img.size
    return [{"type": "text", "bbox": [0, 0, w, h]}]
# --------------------
# HTR / OCR (TrOCR via HF) 
# --------------------
class TrocrRecognizer:
    def __init__(self, model_name=TROCR_MODEL, device=-1):
        logging.info(f"Cargando TrOCR: {model_name} (puede tardar)")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            # si GPU disponible, ajustar device map
            self.device = device
            self.model.to("cuda" if device == 0 else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.max_length = 128
            self.gen_kwargs = {"max_length": self.max_length, "num_beams": 4}
            self.available = True
        except Exception as e:
            logging.error("No se pudo cargar TrOCR: %s", e)
            self.available = False

    def recognize(self, pil_image):
        if not self.available:
            return None
        try:
            pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
            if self.device == 0:
                pixel_values = pixel_values.to("cuda")
            outputs = self.model.generate(pixel_values, **self.gen_kwargs)
            text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return text.strip()
        except Exception as e:
            logging.error("Error en TrOCR recognition: %s", e)
            return None

def ocr_fallback(pil_image):
    """Fallback simple con pytesseract"""
    try:
        text = pytesseract.image_to_string(pil_image, lang='eng')
        return text.strip()
    except Exception as e:
        logging.error("pytesseract error: %s", e)
        return ""

# --------------------
# RAG index (chunk docs + embeddings + FAISS)
# --------------------
def chunk_text(text, max_tokens=MAX_CHUNK):
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        cur.append(w)
        cur_len += 1
        if cur_len >= max_tokens:
            chunks.append(" ".join(cur))
            cur = []
            cur_len = 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks

class RagIndex:
    def __init__(self, model_name=EMBEDDING_MODEL, index_path=None):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.index_path = index_path

    def build_from_folder(self, docs_folder):
        texts = []
        for p in Path(docs_folder).glob("**/*"):
            if p.is_file() and p.suffix.lower() in [".txt", ".md"]:
                txt = p.read_text(encoding='utf-8', errors='ignore')
                texts.append((str(p), txt))
            # optionally parse pdf -> text externally or with libraries (not implemented here)
        # chunk
        chunks = []
        meta = []
        for fname, txt in texts:
            for c in chunk_text(txt):
                chunks.append(c)
                meta.append({"source": fname})
        if not chunks:
            logging.warning("No docs found to index in %s", docs_folder)
            self.index = None
            return
        embs = self.model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embs)
        index.add(embs)
        self.index = index
        self.texts = chunks
        self.meta = meta
        logging.info("Index created with %d chunks", len(chunks))

    def query(self, text, top_k=5):
        if self.index is None:
            return []
        q_emb = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            results.append(self.texts[idx])
        return results

# --------------------
# Generation / Synthesis
# --------------------
def generate_with_openai(prompt, max_tokens=512, temperature=0.0):
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed or available")
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = os.environ["OPENAI_API_KEY"]
    resp = openai.ChatCompletion.create(
        model="gpt-4o" if False else "gpt-4o-mini" if False else "gpt-4o-mini" , # placeholder; user configure
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.0
    )
    return resp["choices"][0]["message"]["content"].strip()

def generate_with_hf(prompt, model_name=SUMMARIZATION_MODEL, max_length=512):
    # simple summarization/generation via HuggingFace pipeline
    gen = pipeline("summarization", model=model_name, truncation=True)
    out = gen(prompt, max_length=max_length, min_length=30)
    return out[0]["summary_text"].strip()

def synthesize_notes(extracted_regions_texts, retrieved_contexts, title="Notes"):
    """Crea prompt y genera síntesis final (summary + structured markdown)"""
    header = f"Generate a structured Markdown note for Obsidian. Title: {title}\n\n"
    header += "Extracted texts (raw) follow. Then context docs from knowledge base follow. Produce sections, clean text, mark formulas as LaTeX if present, and produce a short summary and suggested tags.\n\n"
    prompt = header
    prompt += "=== EXTRACTED ===\n"
    for i, t in enumerate(extracted_regions_texts):
        prompt += f"[region_{i}]\n{t}\n\n"
    prompt += "\n=== CONTEXT ===\n"
    for c in retrieved_contexts:
        prompt += c + "\n\n"
    prompt += "\n=== END ===\nPlease produce Markdown output only."
    # choose generator
    if USE_OPENAI and OPENAI_AVAILABLE:
        return generate_with_openai(prompt)
    else:
        # fallback: use HF summarization to create a short synthesis (limited)
        # We will produce a naive consolidation: concat and summarize
        big = prompt[:4000]  # limit
        return generate_with_hf(big)

# --------------------
# Export utilities
# --------------------
def save_markdown(text, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    logging.info("Markdown saved to %s", out_path)

# --------------------
# Main flow
# --------------------
def process_image_file(path_image, trocr, rag_index, out_dir):
    path_image = Path(path_image)
    out_dir = Path(out_dir)
    base = out_dir / path_image.stem
    ensure_dirs(base)
    # preprocess
    preproc_path = base / f"{path_image.stem}_preproc.png"
    proc = preprocess_image(path_image, preproc_path)
    # layout
    regions = detect_layout(preproc_path)
    logging.info("Detected %d regions", len(regions))
    # recognize each region
    extracted_texts = []
    for i, r in enumerate(regions):
        x1,y1,x2,y2 = r["bbox"]
        pil = Image.open(preproc_path).convert("RGB")
        cropped = pil.crop((x1,y1,x2,y2))
        # try HTR
        text = None
        if trocr and trocr.available:
            text = trocr.recognize(cropped)
        if not text:
            text = ocr_fallback(cropped)
        extracted_texts.append(text or "")
        # save region image
        region_img_path = base / f"region_{i}.png"
        cropped.save(region_img_path)
    # RAG: query index with concatenation of extracted_texts
    query_text = "\n".join(extracted_texts)[:1000]
    retrieved = rag_index.query(query_text, top_k=6) if rag_index and rag_index.index is not None else []
    # synthesize
    md = synthesize_notes(extracted_texts, retrieved, title=path_image.stem)
    out_md = base / f"{path_image.stem}.md"
    save_markdown(md, out_md)
    return out_md

def build_cli():
    parser = argparse.ArgumentParser("quick_paper_pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Carpeta con imágenes a procesar")
    parser.add_argument("--docs_dir", type=str, default="docs", help="Carpeta con docs (txt/md) para RAG")
    parser.add_argument("--output_dir", type=str, default="output", help="Carpeta de salida")
    parser.add_argument("--use_openai", action="store_true", help="Usar OpenAI para síntesis (necesita API Key)")
    parser.add_argument("--no_trocr", action="store_true", help="No cargar TrOCR (usar pytesseract)")
    args = parser.parse_args()
    return args

def main():
    args = build_cli()
    ensure_dirs(args.output_dir)
    global USE_OPENAI
    USE_OPENAI = args.use_openai and OPENAI_AVAILABLE
    # init recognizer
    trocr = None
    if not args.no_trocr:
        trocr = TrocrRecognizer(TROCR_MODEL, device=-1)
        if not trocr.available:
            logging.warning("TrOCR no disponible -> se usará pytesseract como fallback")
            trocr = None
    # build rag index
    rag = RagIndex(EMBEDDING_MODEL)
    rag.build_from_folder(args.docs_dir)
    # process images
    for p in Path(args.input_dir).glob("*"):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff"]:
            logging.info("Procesando %s", p)
            md_path = process_image_file(p, trocr, rag, args.output_dir)
            logging.info("Resultado: %s", md_path)

if __name__ == "__main__":
    main() # tamaño de chunk para indexado
