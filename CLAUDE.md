# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDFCrunch is a forensic PDF analysis tool designed to extract visible and hidden content from PDF documents. It processes PDFs to uncover potential content masking techniques, performs OCR on embedded images, and classifies extracted data.

## Core Architecture

The application is built as a single-file monolithic script (`main.py`) with a pipeline architecture:

1. **File Selection & Archive Handling** - Interactive file dialog supporting PDFs and archives (zip/7z)
2. **PDF Processing Pipeline** - Each PDF goes through two analysis phases:
   - Content layer analysis (detecting redactions, overlays, rasterization, hidden text)
   - Image extraction and OCR processing
3. **OCR Classification** - Images are classified as "text_only" or "mixed" based on visual characteristics, affecting word extraction limits

## Key Dependencies

- **PyMuPDF (fitz)**: Core PDF manipulation and extraction
- **easyocr**: OCR processing for extracted images
- **py7zr**: 7z archive extraction
- **Pillow + numpy**: Image processing and classification

## Running the Application

```bash
python main.py
```

This launches a GUI file picker. Select PDF files or archives (zip/7z) containing PDFs.

## Installation

```bash
pip install -r requirements.txt
```

Note: easyocr downloads language models on first run (~100MB for English).

## Code Structure

**File Selection Flow:**
- `select_files()` → `build_pdf_list()` → `extract_archive()` (if needed) → `process_pdfs()`

**Per-PDF Processing:**
- `analyze_content_layers()`: Detects redactions, overlays, rasterized pages, and white/hidden text
- `extract_images_from_pdf()`: Extracts images to `{pdf_name}/` subdirectory
- `process_image_with_ocr()`: Runs OCR with `classify_image_type()` determining output verbosity

## Content Masking Detection

The `analyze_content_layers()` function identifies several obfuscation techniques:
- Redaction annotations (type 12)
- Excessive drawing objects (threshold: >10)
- Rasterized pages (single large image covering >50% page area with <50 chars text)
- White text on white backgrounds (color code 0xFFFFFF)

## Image Classification Logic

`classify_image_type()` uses heuristics:
- **text_only**: color_variance < 100 AND unique_values < 50, OR color_variance < 300 AND unique_values < 20
- **mixed**: Everything else
- Affects OCR output: text_only shows 20 words, mixed shows 10 words

## Output Structure

For each PDF `example.pdf`:
- Creates directory `example/` containing extracted images
- Images named: `page_{N}_img_{M}.{ext}`
- Console output shows content layer findings and OCR results tagged with `[OCR-TEXT]` or `[OCR-MIXED]`
