# PDFCrunch

A tool for extracting visible content from PDF documents, classifying the extracted data, and storing it in a format optimized for scanning large volumes of information.

## Overview

PDFCrunch processes PDF files to extract their visible content, performs data classification on the extracted information, and organizes the results in a structure that enables efficient search and analysis across large datasets.

## Disclaimer

Please take note of the provided AGPL license. You alone are responsible to always backup the data you intend to work with. This tool is not meant to delete any data in any use case. It cannot however be guaranteed not to do so in any shape or form.
You are also solely responsible to not use this Tool for any Data you do not have permission to do so.

## Features

- Extract visible text and content from PDF documents
- Classify extracted data based on content patterns
- Optimize storage for high-volume data scanning
- Support for batch processing of multiple PDF files

## Expected Behavior
**The PaddlePaddle messages are expected behavior and cannot be suppressed/ignored for now - At least I don't know how. Just ignore them at program start.**  
This concerns:  
- "Connectivity check to the model hoster has been skipped because `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` is enabled."  
- "INFORMATION: Es konnten keine Dateien mit dem angegebenen Muster gefunden werden."  
  
This tool accepts either zip/7z files of PDFs or multiple PDFs. It will then extract those if applicable before scanning the files for text and images. 
For now all images will be extracted and moved, together with a preview of the PDF's text into a subfolder named like the PDF. After processing, the PDF file is also moved into said folder to keep a clean structure.
This is to possibly be toggleable in the future.

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

```bash
python main.py
```

This launches a GUI file picker. Select PDF files or archives (zip/7z) containing PDFs.

## Extensibility

**Text Processing Hook**: All extracted text from each PDF is available via the `all_text` variable (returned by `process_pdf_content()` at line 535 in `main.py`). This can be used for:
- Keyword extraction
- Sentiment analysis
- Custom text parsing
- Data classification
- Full-text search indexing

Currently, the text is extracted but not processed beyond preview generation. Modify `_process_pdf_worker()` (line 562) to add custom text processing logic.

## Documentation

For detailed documentation, see the [docs](./docs/) folder:

- **[Architecture](./docs/architecture.md)** - Process flow and system design
- **[Function Reference](./docs/function-reference.md)** - Complete API documentation
- **[Output Structure](./docs/output-structure.md)** - File organization and naming

## License

GNU Affero General Public License v3.0
