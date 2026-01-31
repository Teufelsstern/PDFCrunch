# PDFCrunch

A tool for extracting visible content from PDF documents, classifying the extracted data, and storing it in a format optimized for scanning large volumes of information.

## Overview

PDFCrunch processes PDF files to extract their visible content, performs data classification on the extracted information, and organizes the results in a structure that enables efficient search and analysis across large datasets.

## Features

- Extract visible text and content from PDF documents
- Classify extracted data based on content patterns
- Optimize storage for high-volume data scanning
- Support for batch processing of multiple PDF files

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (recommended for OCR performance)
- CUDA 12.0 or 12.6

## Installation

### 1. Install PaddlePaddle GPU

For CUDA 12.6:
```bash
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

For CUDA 12.0:
```bash
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
```

**Note:** The PaddlePaddle installation is large (~500MB) and may take several minutes to download.

### 2. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "from paddleocr import PaddleOCR; print('PaddleOCR ready')"
```

## Usage

Details coming soon.

## License

TBD
