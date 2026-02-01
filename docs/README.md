# PDFCrunch Wiki

Welcome to the PDFCrunch documentation! PDFCrunch is a forensic PDF analysis tool designed to extract visible and hidden content from PDF documents.

## Overview

PDFCrunch processes PDF files to:
- Extract readable text directly from PDFs
- Perform OCR on embedded images when needed
- Detect persons in images using YOLOv8 Pose
- Classify images as text-heavy or mixed content
- Support batch processing with multiprocessing

## Key Features

- **Smart OCR**: Only runs OCR when PDF lacks extractable text (>50 words threshold)
- **Person Detection**: Uses YOLOv8 Pose to detect persons with minimal keypoints (censored images)
- **Multiprocessing**: Parallel PDF processing with queue-based YOLO scanning
- **Archive Support**: Handles zip and 7z archives containing PDFs
- **Edge Density Classification**: Fast image classification (8-30% edge density = text)

## Quick Start

```bash
# Install PaddlePaddle GPU first (see README)
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

This launches a file picker. Select PDFs or archives (zip/7z).

## Key Dependencies

- **PyMuPDF (fitz)**: PDF manipulation and text extraction
- **PaddleOCR**: OCR engine for text recognition
- **Ultralytics YOLO**: YOLOv8 Pose model for person detection
- **py7zr**: 7z archive extraction
- **Pillow + OpenCV**: Image processing and edge detection

## Output Structure

For each PDF `example.pdf`:
```
example/
├── example.pdf                    # Original PDF (moved here)
├── example_preview.txt            # Text preview (first 50 words)
├── page_1_img_1.png              # Extracted images
├── page_2_img_1.jpg
└── ...

detected_persons/                  # Shared folder for all PDFs
├── example_page_1_img_2.png      # Images with detected persons
├── another_page_3_img_1.png
└── ...
```

## Navigation

- **[Architecture](./architecture.md)** - Process flow and technical design
- **[Function Reference](./function-reference.md)** - Complete API documentation
- **[Output Structure](./output-structure.md)** - File organization details

## Performance

- **Multiprocessing**: Uses all CPU cores for parallel PDF processing
- **Queue-based YOLO**: Single-threaded YOLO consumer (thread-safe)
- **Intelligent OCR skipping**: ~90% faster for text-based PDFs
- **Edge density only**: Removed connected components analysis for 3x speedup

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (recommended)
- CUDA 12.0 or 12.6
