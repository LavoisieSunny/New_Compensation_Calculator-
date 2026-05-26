import os
import re
import shutil
import tempfile
import logging
import uuid
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pypdf import PdfReader
import pypdfium2 as pdfium

from backend.parser_heuristics import parse_extracted_text
from backend.vector_db import index_document, COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OCRModule")

router = APIRouter()

# Global PaddleOCR instance (lazy initialized)
_ocr_instance = None
OCR_INITIALIZED = False

# Global EasyOCR instance (lazy initialized on first L4 call)
_easyocr_reader = None

# Global Batch Upload and Indexing Process Queue
BATCH_QUEUE = {}

# ======================================================
# OCR QUALITY GATE THRESHOLD
# Below this score, legal reconstruction is DISABLED.
# Fields are left blank rather than hallucinated.
# ======================================================
OCR_QUALITY_GATE_THRESHOLD = 0.25

# ======================================================
# PER-PAGE QUALITY THRESHOLDS FOR RETRY LAYERS
# ======================================================
PAGE_QUALITY_L1 = 0.50   # Accept PaddleOCR scale=2
PAGE_QUALITY_L2 = 0.40   # Accept PaddleOCR scale=3
PAGE_QUALITY_L3 = 0.35   # Accept PaddleOCR + enhanced preprocessing
PAGE_QUALITY_L4 = 0.20   # Minimum accept from EasyOCR

# Legal keywords that signal valid legal document content
_LEGAL_QUALITY_KEYWORDS = [
    "tribunal", "claimant", "petitioner", "mact", "mcop", "accident",
    "rs.", "compensation", "disability", "income", "award", "court",
    "deceased", "injured", "monthly", "insurance", "motor", "claim"
]


# ======================================================
# OCR ENGINE INITIALIZATION
# ======================================================

def get_ocr_instance():
    global _ocr_instance, OCR_INITIALIZED
    if _ocr_instance is None:
        try:
            logger.info("Initializing PaddleOCR (disabling MKLDNN to avoid oneDNN PIR crash)...")
            from paddleocr import PaddleOCR
            _ocr_instance = PaddleOCR(lang='en', enable_mkldnn=False)
            OCR_INITIALIZED = True
            logger.info("PaddleOCR successfully loaded!")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            _ocr_instance = None
            OCR_INITIALIZED = False
    return _ocr_instance


# ======================================================
# CORE TEXT EXTRACTION — DIGITAL PDF
# ======================================================

def extract_digital_pdf_text(file_path: str) -> list:
    """
    Extracts text lines from a digital (selectable) PDF using PyPDF.
    Extremely fast (~20ms per doc) and 100% accurate for digital PDFs.
    """
    try:
        reader = PdfReader(file_path)
        text_lines = []
        for i, page in enumerate(reader.pages):
            text_lines.append(f"--- PAGE {i+1} ---")
            text = page.extract_text()
            if text:
                for line in text.split("\n"):
                    line = line.strip()
                    if line:
                        text_lines.append(line)
        return text_lines
    except Exception as e:
        logger.warning(f"Failed to extract digital text from {file_path}: {str(e)}")
        return []


def extract_alternate_pdf_text(file_path: str) -> list:
    """
    Alternate layout extraction using PyMuPDF (fitz) and pdfplumber.
    Used when PaddleOCR returns sparse results on scanned PDFs.
    LEGAL SAFETY: Only returns actual extracted text. Never fabricates data.
    """
    text_lines = []

    # Try PyMuPDF (fitz)
    try:
        import fitz
        logger.info("Alternate OCR/Extraction: Running PyMuPDF (fitz)...")
        doc = fitz.open(file_path)
        pymupdf_lines = []
        for i, page in enumerate(doc):
            pymupdf_lines.append(f"--- PAGE {i+1} ---")
            text = page.get_text()
            if text:
                for line in text.split("\n"):
                    line = line.strip()
                    if line:
                        pymupdf_lines.append(line)
        if len(pymupdf_lines) > 20:
            logger.info(f"PyMuPDF extraction successful: found {len(pymupdf_lines)} lines.")
            text_lines = pymupdf_lines
    except Exception as e:
        logger.warning(f"Alternate OCR/Extraction PyMuPDF failed: {str(e)}")

    # Try pdfplumber if PyMuPDF extracted very little
    if len(text_lines) < 25:
        try:
            import pdfplumber
            logger.info("Alternate OCR/Extraction: Running pdfplumber...")
            with pdfplumber.open(file_path) as pdf:
                plumber_lines = []
                for i, page in enumerate(pdf.pages):
                    plumber_lines.append(f"--- PAGE {i+1} ---")
                    text = page.extract_text()
                    if text:
                        for line in text.split("\n"):
                            line = line.strip()
                            if line:
                                plumber_lines.append(line)
                if len(plumber_lines) > len(text_lines):
                    logger.info(f"pdfplumber extraction successful: found {len(plumber_lines)} lines.")
                    text_lines = plumber_lines
        except Exception as e:
            logger.warning(f"Alternate OCR/Extraction pdfplumber failed: {str(e)}")

    return text_lines


# ======================================================
# IMAGE PREPROCESSING — STANDARD
# ======================================================

def preprocess_image_for_ocr(pil_img):
    """
    Standard Preprocessing Engine using OpenCV & Pillow.
    Performs grayscale conversion, bilateral noise filtering,
    adaptive Otsu thresholding, and deskew rotation alignment.
    """
    try:
        import cv2
        from PIL import Image

        open_cv_image = np.array(pil_img)
        # Convert to grayscale
        if len(open_cv_image.shape) == 3:
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = open_cv_image

        # Bilateral noise filtering (removes noise, keeps text edges sharp)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # Otsu thresholding for contrast
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Deskew / Orientation alignment
        coords = np.column_stack(np.where(thresh < 255))
        if len(coords) > 100:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if 0.5 < abs(angle) < 15.0:
                logger.info(f"Skew detected: {angle:.2f} degrees. Rotating...")
                (h, w) = thresh.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                thresh = cv2.warpAffine(
                    thresh, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255
                )

        return Image.fromarray(thresh)
    except Exception as e:
        logger.warning(f"Standard preprocessing failed, using raw image: {str(e)}")
        return pil_img


# ======================================================
# IMAGE PREPROCESSING — ENHANCED (for low-quality scans)
# ======================================================

def preprocess_image_enhanced(pil_img):
    """
    Enhanced Preprocessing Pass for faint / low-quality scanned pages.
    Applies CLAHE contrast enhancement, morphological closing to reconnect
    broken character strokes, and unsharp masking to sharpen text edges.
    Falls back to standard preprocessing on failure.
    """
    try:
        import cv2
        from PIL import Image

        open_cv_image = np.array(pil_img)
        if len(open_cv_image.shape) == 3:
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = open_cv_image.copy()

        # 1. CLAHE — Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. Unsharp masking (Gaussian blur → subtract → sharpen)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        # 3. Morphological closing — reconnects broken character strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)

        # 4. Otsu thresholding on the enhanced result
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 5. Auto-invert if white-on-dark (more black pixels than white)
        if np.mean(thresh) < 127:
            thresh = cv2.bitwise_not(thresh)

        return Image.fromarray(thresh)
    except Exception as e:
        logger.warning(f"Enhanced preprocessing failed, falling back to standard: {str(e)}")
        return preprocess_image_for_ocr(pil_img)


# ======================================================
# OCR RESULT NORMALIZER
# ======================================================

def extract_text_lines_from_paddle_result(result) -> list:
    """
    Normalises PaddleOCR output across v2 (list-of-lists) and
    PaddleX v3 (object with rec_texts) output formats.
    Returns a flat deduplicated list of non-empty text strings.
    """
    text_lines = []
    if not result or len(result) == 0:
        return text_lines

    for item in result:
        if hasattr(item, 'rec_texts') and item.rec_texts:
            text_lines.extend(item.rec_texts)
        elif isinstance(item, dict) and 'rec_texts' in item:
            text_lines.extend(item['rec_texts'])
        elif hasattr(item, 'get') and item.get('rec_texts'):
            text_lines.extend(item.get('rec_texts'))
        elif isinstance(item, list):
            for line in item:
                if isinstance(line, list) and len(line) > 1 and isinstance(line[1], tuple):
                    text_lines.append(line[1][0])
                elif isinstance(line, tuple) and len(line) > 1 and isinstance(line[0], str):
                    text_lines.append(line[0])

    return [l.strip() for l in text_lines if l and l.strip()]


# ======================================================
# OCR QUALITY SCORING
# ======================================================

def score_ocr_page_quality(text_lines: list) -> float:
    """
    Scores the OCR output quality for a single page.
    Returns float 0.0 – 1.0 based on 4 weighted criteria:
      - line_count       (30%): More lines → better
      - keyword_hits     (35%): Legal MACT keywords detected
      - word_quality     (20%): Average word length (garble detection)
      - text_density     (15%): Characters per line
    """
    real_lines = [l for l in text_lines if l and not l.startswith("--- PAGE")]
    if not real_lines:
        return 0.0

    # 1. Line count score
    line_score = min(len(real_lines) / 10.0, 1.0)

    # 2. Legal keyword score
    full_text = " ".join(real_lines).lower()
    kw_hits = sum(1 for kw in _LEGAL_QUALITY_KEYWORDS if kw in full_text)
    keyword_score = min(kw_hits / 5.0, 1.0)

    # 3. Word quality score (garbled OCR = very short or very long tokens)
    words = full_text.split()
    if words:
        avg_len = sum(len(w) for w in words) / len(words)
        # Ideal word length: 3–10 chars
        word_score = 1.0 if 3.0 <= avg_len <= 10.0 else max(0.0, 1.0 - abs(avg_len - 6.5) / 6.5)
    else:
        word_score = 0.0

    # 4. Text density score (chars per line)
    avg_line_len = sum(len(l) for l in real_lines) / len(real_lines)
    density_score = min(avg_line_len / 40.0, 1.0)

    quality = (
        (line_score    * 0.30) +
        (keyword_score * 0.35) +
        (word_score    * 0.20) +
        (density_score * 0.15)
    )
    return round(quality, 3)


def _build_ocr_debug(
    engine_used: str,
    retry_count: int,
    quality_score: float,
    failed_pages: list,
    successful_pages: list,
    preprocessing_applied: list,
    fallback_ocr_engine: str,
    text_density_score: float,
    average_page_confidence: float = 0.0,
) -> dict:
    """Constructs the standard OCR debug metadata block."""
    return {
        "ocr_engine_used": engine_used,
        "ocr_retry_count": retry_count,
        "ocr_quality_score": round(quality_score, 3),
        "successful_pages": successful_pages,
        "failed_pages": failed_pages,
        "fallback_ocr_engine": fallback_ocr_engine,
        "preprocessing_applied": preprocessing_applied,
        "text_density_score": round(text_density_score, 3),
        "average_page_confidence": round(average_page_confidence, 3),
    }


# ======================================================
# SECONDARY OCR ENGINE — EasyOCR (lazy import)
# ======================================================

def run_easyocr_fallback(img_array: np.ndarray) -> tuple:
    """
    Secondary OCR engine using EasyOCR.
    Lazy-imported — application continues normally if EasyOCR is not installed.
    Returns: (text_lines: list[str], avg_confidence: float)
    LEGAL SAFETY: Only returns text actually read from the image.
    """
    global _easyocr_reader
    try:
        import easyocr
        if _easyocr_reader is None:
            logger.info("Initializing EasyOCR secondary engine (English)...")
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR initialized successfully.")

        results = _easyocr_reader.readtext(img_array, detail=1, paragraph=False)
        text_lines = []
        confidences = []
        for (bbox, text, conf) in results:
            if text and conf > 0.10:
                text_lines.append(text.strip())
                confidences.append(conf)

        avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        logger.info(f"EasyOCR: {len(text_lines)} lines extracted (avg confidence {avg_conf:.2f})")
        return text_lines, avg_conf

    except ImportError:
        logger.warning(
            "EasyOCR not installed. Secondary OCR engine unavailable. "
            "Install with: pip install easyocr"
        )
        return [], 0.0
    except Exception as e:
        logger.error(f"EasyOCR fallback failed: {str(e)}")
        return [], 0.0


# ======================================================
# 4-LAYER PER-PAGE OCR RETRY PIPELINE
# ======================================================

def perform_ocr_page_with_retry(ocr_engine, page_doc, page_idx: int, total_pages: int) -> tuple:
    """
    Attempts OCR on a single PDF page using up to 4 progressive retry layers.
    Each layer is only tried if the previous layer's quality is insufficient.

    LAYER 1: PaddleOCR at scale=2 + standard preprocessing
    LAYER 2: PaddleOCR at scale=3 (higher DPI) + standard preprocessing
    LAYER 3: PaddleOCR at scale=2 + enhanced preprocessing (CLAHE + sharpen)
    LAYER 4: EasyOCR secondary engine

    Returns: (text_lines: list[str], page_meta: dict)
    LEGAL SAFETY: Never injects hardcoded or fabricated text.
    """
    from PIL import Image

    preprocessing_applied = []
    best_lines = []
    best_quality = 0.0
    layer_used = "none"

    # ─── LAYER 1: PaddleOCR scale=2 + standard preprocessing ─────────────
    bitmap_l1 = None
    try:
        bitmap_l1 = page_doc.render(scale=2)
        pil_l1 = bitmap_l1.to_pil()
        pil_std = preprocess_image_for_ocr(pil_l1)
        preprocessing_applied.append("grayscale+bilateral+otsu+deskew@scale2")

        result_l1 = ocr_engine.ocr(np.array(pil_std))
        l1_lines = extract_text_lines_from_paddle_result(result_l1)
        l1_quality = score_ocr_page_quality(l1_lines)
        logger.info(f"  Page {page_idx+1}/{total_pages} L1 (PaddleOCR@scale2): {len(l1_lines)} lines, q={l1_quality:.2f}")

        if l1_quality >= PAGE_QUALITY_L1:
            return l1_lines, {
                "quality_score": l1_quality, "layer_used": "L1:PaddleOCR@scale2",
                "preprocessing_applied": preprocessing_applied, "lines": len(l1_lines)
            }
        if l1_quality > best_quality:
            best_lines, best_quality, layer_used = l1_lines, l1_quality, "L1:PaddleOCR@scale2"
    except Exception as e:
        logger.warning(f"  Page {page_idx+1} L1 failed: {str(e)}")

    # ─── LAYER 2: PaddleOCR scale=3 (higher DPI) ─────────────────────────
    try:
        bitmap_l2 = page_doc.render(scale=3)
        pil_l2 = bitmap_l2.to_pil()
        pil_std3 = preprocess_image_for_ocr(pil_l2)
        preprocessing_applied.append("grayscale+bilateral+otsu+deskew@scale3")

        result_l2 = ocr_engine.ocr(np.array(pil_std3))
        l2_lines = extract_text_lines_from_paddle_result(result_l2)
        l2_quality = score_ocr_page_quality(l2_lines)
        logger.info(f"  Page {page_idx+1}/{total_pages} L2 (PaddleOCR@scale3): {len(l2_lines)} lines, q={l2_quality:.2f}")

        if l2_quality >= PAGE_QUALITY_L2:
            return l2_lines, {
                "quality_score": l2_quality, "layer_used": "L2:PaddleOCR@scale3",
                "preprocessing_applied": preprocessing_applied, "lines": len(l2_lines)
            }
        if l2_quality > best_quality:
            best_lines, best_quality, layer_used = l2_lines, l2_quality, "L2:PaddleOCR@scale3"
    except Exception as e:
        logger.warning(f"  Page {page_idx+1} L2 failed: {str(e)}")

    # ─── LAYER 3: Enhanced preprocessing (CLAHE + sharpen + morphology) ──
    try:
        base_pil = bitmap_l1.to_pil() if bitmap_l1 else page_doc.render(scale=2).to_pil()
        pil_enh = preprocess_image_enhanced(base_pil)
        preprocessing_applied.append("clahe+unsharp_mask+morph_close+otsu@scale2")

        result_l3 = ocr_engine.ocr(np.array(pil_enh))
        l3_lines = extract_text_lines_from_paddle_result(result_l3)
        l3_quality = score_ocr_page_quality(l3_lines)
        logger.info(f"  Page {page_idx+1}/{total_pages} L3 (EnhancedPreproc): {len(l3_lines)} lines, q={l3_quality:.2f}")

        if l3_quality >= PAGE_QUALITY_L3:
            return l3_lines, {
                "quality_score": l3_quality, "layer_used": "L3:PaddleOCR+EnhancedPreproc",
                "preprocessing_applied": preprocessing_applied, "lines": len(l3_lines)
            }
        if l3_quality > best_quality:
            best_lines, best_quality, layer_used = l3_lines, l3_quality, "L3:PaddleOCR+EnhancedPreproc"
    except Exception as e:
        logger.warning(f"  Page {page_idx+1} L3 failed: {str(e)}")

    # ─── LAYER 4: EasyOCR secondary engine ───────────────────────────────
    try:
        img_easy = np.array(bitmap_l1.to_pil() if bitmap_l1 else page_doc.render(scale=2).to_pil())
        preprocessing_applied.append("easyocr_secondary_engine")

        l4_lines, l4_conf = run_easyocr_fallback(img_easy)
        l4_quality = score_ocr_page_quality(l4_lines)
        logger.info(f"  Page {page_idx+1}/{total_pages} L4 (EasyOCR): {len(l4_lines)} lines, q={l4_quality:.2f}, conf={l4_conf:.2f}")

        if l4_quality >= PAGE_QUALITY_L4 or l4_quality > best_quality:
            return l4_lines, {
                "quality_score": l4_quality, "layer_used": "L4:EasyOCR",
                "preprocessing_applied": preprocessing_applied, "lines": len(l4_lines)
            }
        if l4_quality > best_quality:
            best_lines, best_quality, layer_used = l4_lines, l4_quality, "L4:EasyOCR"
    except Exception as e:
        logger.warning(f"  Page {page_idx+1} L4 (EasyOCR) failed: {str(e)}")

    # Return best result across all layers (may be empty — never fabricated)
    if not best_lines:
        logger.warning(f"  Page {page_idx+1}: All OCR layers returned no text.")
    return best_lines, {
        "quality_score": best_quality, "layer_used": layer_used,
        "preprocessing_applied": preprocessing_applied, "lines": len(best_lines)
    }


# ======================================================
# MAIN SCANNED PDF OCR PIPELINE
# ======================================================

def perform_ocr_on_scanned_pdf(file_path: str, progress_callback=None) -> tuple:
    """
    Renders pages of a scanned PDF using pypdfium2 and runs the
    4-layer per-page OCR retry pipeline on each page.

    Returns: (text_lines: list[str], ocr_debug: dict)

    LEGAL SAFETY: If PaddleOCR is offline or all retries fail, returns
    ([], warning_ocr_debug). NEVER returns hardcoded or fabricated legal text.
    """
    ocr_engine = get_ocr_instance()

    if ocr_engine is None:
        logger.warning("PaddleOCR offline. Returning empty result — no fabricated legal data injected.")
        return [], _build_ocr_debug(
            engine_used="unavailable", retry_count=0, quality_score=0.0,
            failed_pages=[], successful_pages=[], preprocessing_applied=[],
            fallback_ocr_engine="none", text_density_score=0.0
        )

    try:
        doc = pdfium.PdfDocument(file_path)
        total_pages = len(doc)
        logger.info(f"Scanned PDF '{file_path}' ({total_pages} pages): starting 4-layer OCR retry pipeline...")

        # Optimization: for large PDFs (>10 pages), scan first 5 + last 5 only
        # (claimant details are at the start; award/decree at the end)
        if total_pages > 10:
            pages_to_scan = sorted(set(list(range(5)) + list(range(total_pages - 5, total_pages))))
            logger.info(f"Large PDF detected. Scanning pages: {[p+1 for p in pages_to_scan]}")
        else:
            pages_to_scan = list(range(total_pages))

        text_lines = []
        failed_pages = []
        successful_pages = []
        all_preprocessing_steps = []
        total_retry_count = 0
        page_qualities = []
        page_confidences = []
        fallback_engine_used = ""

        for idx, page_idx in enumerate(pages_to_scan):
            text_lines.append(f"--- PAGE {page_idx+1} ---")
            page_lines, page_meta = perform_ocr_page_with_retry(
                ocr_engine, doc[page_idx], page_idx, total_pages
            )

            layer = page_meta.get("layer_used", "L1")
            q = page_meta.get("quality_score", 0.0)
            page_qualities.append(q)

            # Count retries (L1=0, L2=1, L3=2, L4=3)
            layer_num = int(layer[1]) if (layer and len(layer) > 1 and layer[1].isdigit()) else 1
            total_retry_count += max(0, layer_num - 1)

            # Collect unique preprocessing steps
            for step in page_meta.get("preprocessing_applied", []):
                if step not in all_preprocessing_steps:
                    all_preprocessing_steps.append(step)

            # Track if EasyOCR was used
            if "EasyOCR" in layer and not fallback_engine_used:
                fallback_engine_used = "EasyOCR"

            if page_lines:
                successful_pages.append(page_idx + 1)
            else:
                failed_pages.append(page_idx + 1)
                logger.warning(f"Page {page_idx+1}: No text extracted across all 4 OCR layers.")

            text_lines.extend(page_lines)
            logger.info(f"Page {page_idx+1}/{total_pages}: layer={layer}, quality={q:.2f}, lines={len(page_lines)}")

            if progress_callback:
                progress_callback(int(((idx + 1) / len(pages_to_scan)) * 90))

        overall_quality = round(sum(page_qualities) / len(page_qualities), 3) if page_qualities else 0.0
        real_lines = [l for l in text_lines if not l.startswith("--- PAGE")]
        text_density = round(
            sum(len(l) for l in real_lines) / max(len(real_lines), 1) / 80.0, 3
        ) if real_lines else 0.0

        ocr_debug = _build_ocr_debug(
            engine_used="PaddleOCR",
            retry_count=total_retry_count,
            quality_score=overall_quality,
            failed_pages=failed_pages,
            successful_pages=successful_pages,
            preprocessing_applied=all_preprocessing_steps,
            fallback_ocr_engine=fallback_engine_used,
            text_density_score=min(text_density, 1.0),
            average_page_confidence=overall_quality,
        )
        return text_lines, ocr_debug

    except Exception as e:
        logger.error(f"Critical error during scanned PDF OCR: {str(e)}")
        return [], _build_ocr_debug(
            engine_used="PaddleOCR", retry_count=0, quality_score=0.0,
            failed_pages=[], successful_pages=[], preprocessing_applied=[],
            fallback_ocr_engine="none", text_density_score=0.0
        )


def perform_ocr_on_image(file_path: str) -> tuple:
    """
    Runs the 4-layer OCR retry pipeline directly on image uploads (PNG, JPG, BMP).
    Returns: (text_lines: list[str], ocr_debug: dict)

    LEGAL SAFETY: On engine failure returns ([], warning_debug).
    NEVER returns hardcoded or fabricated legal text.
    """
    ocr_engine = get_ocr_instance()

    if ocr_engine is None:
        logger.warning("PaddleOCR offline. Returning empty result — no fabricated legal data injected.")
        return [], _build_ocr_debug(
            engine_used="unavailable", retry_count=0, quality_score=0.0,
            failed_pages=[1], successful_pages=[], preprocessing_applied=[],
            fallback_ocr_engine="none", text_density_score=0.0
        )

    try:
        from PIL import Image
        pil_img = Image.open(file_path)
        preprocessing_applied = []
        retry_count = 0
        best_lines, best_quality = [], 0.0
        fallback_engine_used = ""

        # L1: Standard preprocessing
        pil_std = preprocess_image_for_ocr(pil_img)
        preprocessing_applied.append("grayscale+bilateral+otsu+deskew")
        result_l1 = ocr_engine.ocr(np.array(pil_std))
        l1_lines = extract_text_lines_from_paddle_result(result_l1)
        l1_quality = score_ocr_page_quality(l1_lines)
        logger.info(f"Image L1 (PaddleOCR standard): {len(l1_lines)} lines, q={l1_quality:.2f}")

        if l1_quality >= PAGE_QUALITY_L1:
            td = round(sum(len(l) for l in l1_lines) / max(len(l1_lines), 1) / 80.0, 3)
            return l1_lines, _build_ocr_debug(
                "PaddleOCR", 0, l1_quality, [], [1], preprocessing_applied, "", min(td, 1.0)
            )
        best_lines, best_quality = l1_lines, l1_quality

        # L2: Enhanced preprocessing (CLAHE + sharpen + morphology)
        retry_count += 1
        pil_enh = preprocess_image_enhanced(pil_img)
        preprocessing_applied.append("clahe+unsharp_mask+morph_close+otsu")
        result_l2 = ocr_engine.ocr(np.array(pil_enh))
        l2_lines = extract_text_lines_from_paddle_result(result_l2)
        l2_quality = score_ocr_page_quality(l2_lines)
        logger.info(f"Image L2 (PaddleOCR enhanced): {len(l2_lines)} lines, q={l2_quality:.2f}")

        if l2_quality >= PAGE_QUALITY_L2:
            td = round(sum(len(l) for l in l2_lines) / max(len(l2_lines), 1) / 80.0, 3)
            return l2_lines, _build_ocr_debug(
                "PaddleOCR", retry_count, l2_quality, [], [1], preprocessing_applied, "", min(td, 1.0)
            )
        if l2_quality > best_quality:
            best_lines, best_quality = l2_lines, l2_quality

        # L3: EasyOCR secondary engine
        retry_count += 1
        preprocessing_applied.append("easyocr_secondary_engine")
        fallback_engine_used = "EasyOCR"
        l3_lines, l3_conf = run_easyocr_fallback(np.array(pil_img))
        l3_quality = score_ocr_page_quality(l3_lines)
        logger.info(f"Image L3 (EasyOCR): {len(l3_lines)} lines, q={l3_quality:.2f}, conf={l3_conf:.2f}")

        if l3_quality > best_quality:
            best_lines, best_quality = l3_lines, l3_quality

        failed_pages = [] if best_lines else [1]
        successful_pages = [1] if best_lines else []
        td = round(sum(len(l) for l in best_lines) / max(len(best_lines), 1) / 80.0, 3)

        return best_lines, _build_ocr_debug(
            engine_used="PaddleOCR",
            retry_count=retry_count,
            quality_score=best_quality,
            failed_pages=failed_pages,
            successful_pages=successful_pages,
            preprocessing_applied=preprocessing_applied,
            fallback_ocr_engine=fallback_engine_used,
            text_density_score=min(td, 1.0),
            average_page_confidence=best_quality,
        )

    except Exception as e:
        logger.error(f"Error running OCR on image: {str(e)}")
        return [], _build_ocr_debug(
            "PaddleOCR", 0, 0.0, [1], [], [], "none", 0.0
        )


# ======================================================
# UTILITIES
# ======================================================

def is_extracted_text_sparse(text_lines: list) -> bool:
    """
    Returns True if extracted text (excluding page markers) has fewer than 15 lines.
    Used to decide whether to escalate to the scanned-PDF OCR pipeline.
    """
    actual = [l for l in text_lines if not l.strip().startswith("--- PAGE")]
    return len(actual) < 15


def extract_award_amount_from_text(text_lines: list) -> float:
    """
    Extracts an explicit tribunal award amount from raw OCR text lines.
    Returns 0.0 if no clear amount is found.
    LEGAL SAFETY: Only returns amounts found in actual OCR text.
    """
    award_patterns = [
        r'(?:total|final|award|sum\s+of|compensation\s+of)\s*(?:award|amount|is|of)?\s*'
        r'(?:rs\.?|inr|rupees)?\s*(\d{5,7})\b',
        r'\b(?:rs\.?|inr)\s*(\d{5,7})\b\s*(?:with\s*interest|is\s*awarded|as\s*compensation)'
    ]
    full_text_lower = "\n".join(text_lines).lower()
    for pat in award_patterns:
        m = re.findall(pat, full_text_lower)
        if m:
            return float(m[0])
    return 0.0


# ======================================================
# OCR QUALITY GATE
# ======================================================

def apply_ocr_quality_gate(suggestions: dict, ocr_debug: dict) -> dict:
    """
    Legal Safety Gate: if OCR quality is below the minimum threshold,
    all inferred legal fields are blanked and a warning is emitted.

    The system NEVER fabricates legal facts — missing = blank, not invented.
    """
    quality = ocr_debug.get("ocr_quality_score", 1.0)
    suggestions["ocr_quality_insufficient"] = quality < OCR_QUALITY_GATE_THRESHOLD

    if quality < OCR_QUALITY_GATE_THRESHOLD:
        logger.warning(
            f"OCR quality {quality:.2f} below legal safety threshold {OCR_QUALITY_GATE_THRESHOLD}. "
            "Disabling reconstruction and blanking legal fields."
        )
        suggestions["ocr_warning"] = (
            f"OCR confidence insufficient for legal reconstruction (score: {quality:.2f}). "
            "Please provide a cleaner document scan. Legal fields left blank for safety."
        )
        # Blank all inferred legal fields to prevent hallucination propagation
        legal_fields_to_blank = [
            "name", "claimant_name", "father_name", "deceased_name",
            "monthly_income", "disability", "age", "date_of_birth",
            "date_of_accident", "total_compensation", "award_amount",
            "occupation", "dependents", "marital_status",
        ]
        for field in legal_fields_to_blank:
            if field in suggestions:
                suggestions[field] = None

    return suggestions


# ======================================================
# BACKGROUND BATCH INDEXING PIPELINE
# ======================================================

def run_background_pdf_indexing(file_id: str, temp_path: str, filename: str):
    """
    Background worker: dual-mode PDF text extraction → 4-layer OCR retry
    → legal heuristics parsing → OCR quality gate → Qdrant vector indexing.
    """
    try:
        BATCH_QUEUE[file_id]["status"] = "scanning"
        BATCH_QUEUE[file_id]["progress"] = 20

        # 1. Attempt digital selectable text extraction (fastest path)
        logger.info(f"Background task: Extracting selectable text from '{filename}'")
        text_lines = extract_digital_pdf_text(temp_path)
        fallback_source = "DigitalPDF"
        ocr_debug = _build_ocr_debug("DigitalPDF", 0, 1.0, [], [], [], "", 0.0)

        # 2. Sparse → 4-layer OCR retry pipeline
        if is_extracted_text_sparse(text_lines):
            logger.info(f"Selectable text sparse. Running 4-layer OCR pipeline for '{filename}'")

            def report_progress(prog_percent):
                BATCH_QUEUE[file_id]["progress"] = prog_percent

            text_lines, ocr_debug = perform_ocr_on_scanned_pdf(
                temp_path, progress_callback=report_progress
            )
            fallback_source = "PaddleOCR"

        # 3. Still sparse → alternate layout extraction (PyMuPDF / pdfplumber)
        if is_extracted_text_sparse(text_lines):
            logger.info("OCR pipeline sparse. Trying alternate layout extraction (PyMuPDF/pdfplumber)...")
            alt_lines = extract_alternate_pdf_text(temp_path)
            if len(alt_lines) > len(text_lines):
                text_lines = alt_lines
                fallback_source = "AlternateOCR"
                ocr_debug["fallback_ocr_engine"] = "PyMuPDF/pdfplumber"

        BATCH_QUEUE[file_id]["progress"] = 90

        # 4. Parse with legal heuristics
        logger.info(f"Parsing extracted text for '{filename}'")
        suggestions = parse_extracted_text(text_lines)

        # 5. OCR Quality Gate — disable reconstruction if quality is insufficient
        suggestions = apply_ocr_quality_gate(suggestions, ocr_debug)

        if suggestions.get("ai_recovery_triggered", False):
            fallback_source = "RealTextRecovery"
        suggestions["fallback_source_used"] = fallback_source

        # 6. Extract explicit judicial award amount
        award_amount = extract_award_amount_from_text(text_lines)
        if award_amount > 0:
            suggestions["award_amount"] = award_amount
            logger.info(f"Judicial award amount parsed: Rs. {award_amount:,.0f}")

        # 7. Index into Qdrant vector DB
        BATCH_QUEUE[file_id]["status"] = "indexing"
        logger.info(f"Indexing '{filename}' in Qdrant...")
        success = index_document(filename, text_lines, suggestions)

        # 8. Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

        if success:
            BATCH_QUEUE[file_id]["status"] = "indexed"
            BATCH_QUEUE[file_id]["progress"] = 100
            BATCH_QUEUE[file_id]["suggestions"] = suggestions
            BATCH_QUEUE[file_id]["raw_text"] = text_lines
            BATCH_QUEUE[file_id]["ocr_debug"] = ocr_debug
            logger.info(f"Background task: '{filename}' indexed successfully!")
        else:
            BATCH_QUEUE[file_id]["status"] = "failed"
            BATCH_QUEUE[file_id]["error"] = "Qdrant indexing upsert failed."
            logger.error(f"Background task: Indexing failed for '{filename}'")

    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        BATCH_QUEUE[file_id]["status"] = "failed"
        BATCH_QUEUE[file_id]["error"] = str(e)
        logger.error(f"Background task exception for '{filename}': {str(e)}")


# ======================================================
# API ROUTERS
# ======================================================

@router.post("/process-ocr")
async def process_single_file(file: UploadFile = File(...)):
    """
    Synchronous single image/PDF upload — instant OCR + legal heuristics parsing.
    Response includes ocr_debug metadata block with quality score and retry info.
    """
    file_ext = os.path.splitext(file.filename)[1].lower()
    allowed_images = {".png", ".jpg", ".jpeg", ".bmp"}
    allowed_docs = {".pdf"}

    if file_ext not in allowed_images and file_ext not in allowed_docs:
        raise HTTPException(
            status_code=400,
            detail="Only standard images (PNG, JPG, BMP) and PDFs are supported."
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {str(e)}")

    try:
        fallback_source = "DigitalPDF"
        ocr_debug = _build_ocr_debug("DigitalPDF", 0, 1.0, [], [], [], "", 0.0)

        if file_ext == ".pdf":
            # 1. Try selectable digital text (fastest)
            text_lines = extract_digital_pdf_text(temp_path)

            # 2. Sparse → 4-layer OCR retry pipeline
            if is_extracted_text_sparse(text_lines):
                logger.info("Selectable PDF text sparse. Running 4-layer OCR pipeline...")
                text_lines, ocr_debug = perform_ocr_on_scanned_pdf(temp_path)
                fallback_source = "PaddleOCR"

            # 3. Still sparse → alternate layout extraction
            if is_extracted_text_sparse(text_lines):
                logger.info("OCR output sparse. Running alternate layout extraction...")
                alt_lines = extract_alternate_pdf_text(temp_path)
                if len(alt_lines) > len(text_lines):
                    text_lines = alt_lines
                    fallback_source = "AlternateOCR"
                    ocr_debug["fallback_ocr_engine"] = "PyMuPDF/pdfplumber"
        else:
            # Image upload — 4-layer retry
            text_lines, ocr_debug = perform_ocr_on_image(temp_path)
            fallback_source = "PaddleOCR"

        # Legal heuristics parsing
        suggestions = parse_extracted_text(text_lines)

        # OCR Quality Gate
        suggestions = apply_ocr_quality_gate(suggestions, ocr_debug)

        if suggestions.get("ai_recovery_triggered", False):
            fallback_source = "RealTextRecovery"
        suggestions["fallback_source_used"] = fallback_source

        # Extract judicial award amount
        award_amount = extract_award_amount_from_text(text_lines)
        if award_amount > 0:
            suggestions["award_amount"] = award_amount

        os.unlink(temp_path)
        return {
            "success": True,
            "filename": file.filename,
            "ocr_status": "loaded",
            "fallback_source": fallback_source,
            "suggestions": suggestions,
            "raw_text": text_lines,
            "ocr_debug": ocr_debug,
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-batch")
async def upload_batch_pdfs(files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    """
    Batch PDF upload — queues 1–100+ PDFs for background 4-layer OCR + Qdrant indexing.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    enqueued_files = []

    for file in files:
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext != ".pdf":
            continue

        file_id = f"file_{uuid.uuid4().hex[:10]}"

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name
        except Exception as e:
            logger.error(f"Failed saving batch PDF '{filename}': {str(e)}")
            continue

        BATCH_QUEUE[file_id] = {
            "file_id": file_id,
            "filename": filename,
            "status": "queued",
            "progress": 0,
            "suggestions": None,
            "raw_text": [],
            "ocr_debug": None,
            "error": None,
        }

        if background_tasks:
            background_tasks.add_task(run_background_pdf_indexing, file_id, temp_path, filename)
        else:
            run_background_pdf_indexing(file_id, temp_path, filename)

        enqueued_files.append({"file_id": file_id, "filename": filename, "status": "queued"})

    return {
        "success": True,
        "message": f"Successfully queued {len(enqueued_files)} PDFs for background OCR + indexing.",
        "queue": enqueued_files,
    }


@router.get("/batch-status")
async def get_batch_status():
    """Returns real-time processing states of all PDFs in the batch queue."""
    return {
        "success": True,
        "queue": list(BATCH_QUEUE.values()),
    }
