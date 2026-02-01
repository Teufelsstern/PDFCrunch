# Output Structure

Detailed explanation of how PDFCrunch organizes output files and directories.

## Directory Organization

```
working_directory/
├── detected_persons/              # Shared across all PDFs
│   ├── document1_page_2_img_1.png
│   ├── document1_page_5_img_2.png
│   ├── report_page_1_img_1.png
│   └── scan_page_10_img_3.png
│
├── document1/                     # Per-PDF directory
│   ├── document1.pdf             # Original PDF (moved here)
│   ├── document1_preview.txt     # Text preview
│   ├── page_1_img_1.png         # Extracted images
│   ├── page_2_img_1.jpg
│   └── page_5_img_2.png
│
├── report/
│   ├── report.pdf
│   ├── report_preview.txt
│   └── page_1_img_1.png
│
└── scan/
    ├── scan.pdf
    ├── scan_preview.txt
    ├── page_1_img_1.jpg
    └── page_10_img_3.png
```

## File Naming Conventions

### Per-PDF Directories

- **Name:** PDF filename without extension
- **Example:** `document.pdf` → `document/`
- **Contents:** Original PDF, preview text, extracted images

### Extracted Images

**Format:** `page_{PAGE}_img_{INDEX}.{EXT}`

- `PAGE`: PDF page number (1-indexed)
- `INDEX`: Image index on that page (1-indexed)
- `EXT`: Original image extension (png, jpg, jpeg, etc.)

**Examples:**
- `page_1_img_1.png` - First image on page 1
- `page_2_img_3.jpg` - Third image on page 2
- `page_10_img_1.jpeg` - First image on page 10

### Detected Persons

**Format:** `{PDF_NAME}_page_{PAGE}_img_{INDEX}.png`

- `PDF_NAME`: PDF filename without extension
- `PAGE`: PDF page number (1-indexed)
- `INDEX`: Image index on that page (1-indexed)
- Always saved as `.png`

**Examples:**
- `report_page_1_img_1.png` - Person found in first image of page 1 in report.pdf
- `scan_page_10_img_3.png` - Person found in third image of page 10 in scan.pdf

### Preview Text Files

**Format:** `{PDF_NAME}_preview.txt`

**Contents:**
```
Extracted Text Preview (1,234 words total):
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat...
```

**Two types:**

1. **Text-based PDFs (>50 words extractable):**
   - Shows first 50 words of extracted text
   - Total word count in header
   - No OCR results

2. **Image-based PDFs (≤50 words extractable):**
   - Shows OCR results per image
   - Format: `  - page {N} img {M}: {text}`
   - Example:
   ```
     - page 1 img 1: This is the extracted text from the first image
     - page 2 img 1: Another piece of text from page two
   ```

## Processing Output

### Console Output

```
PDFCrunch - PDF Content Extraction Tool
========================================

Selected 3 file(s)

Using 8 parallel workers
Models (PaddleOCR, YOLO) will be initialized in each worker process...

Processing 3 PDF file(s):

Initializing models...
Models initialized

[1/3] ✓ document1 (5 pages) | Text: Yes | Person: No
Progress: 1/3
[2/3] ✓ report (12 pages) | Text: No | Person: Yes
Progress: 2/3
[3/3] ✓ scan (8 pages) | Text: Yes | Person: Yes
Progress: 3/3

✓ Completed processing 3 PDFs (25 pages total) in 1m 23s
```

### Progress Indicators

- **`[X/Y]`**: Current PDF / Total PDFs
- **`✓ {name} ({N} pages)`**: PDF name and page count
- **`Text: Yes/No`**: Whether text was found (extractable or OCR)
- **`Person: Yes/No`**: Whether person was detected in any image
- **`Progress: X/Y`**: Overall progress (redundant with `[X/Y]`)

### Timing

- **Format:** `{minutes}m {seconds}s`
- **Example:** `15m 42s`
- **Includes:** All processing time (initialization, extraction, OCR, YOLO)

## Special Cases

### PDFs Without Images

**Directory created:** Yes (empty except PDF and optional preview)

```
document_no_images/
├── document_no_images.pdf
└── document_no_images_preview.txt
```

### PDFs Without Text

**No preview file created** if:
- No extractable text (≤50 words)
- No OCR results from images
- All images classified as non-text

```
scan_no_text/
├── scan_no_text.pdf
├── page_1_img_1.jpg
└── page_2_img_1.png
```

### Fully Redacted Images

**Not saved** - Skip processing entirely if >50% black pixels

### Archives

**Extraction location:** Same directory as archive

```
archive.zip
archive/                    # Extracted contents
├── pdf1.pdf
├── pdf2.pdf
└── subfolder/
    └── pdf3.pdf

pdf1/                      # Processed PDFs
├── pdf1.pdf
└── ...
pdf2/
├── pdf2.pdf
└── ...
subfolder_pdf3/            # Flattened path
├── pdf3.pdf
└── ...
```

**Note:** PDFs from archives follow the same structure as regular PDFs.

## Storage Considerations

### Image Formats

- **Original format preserved** for extracted images
- **PNG only** for detected persons (consistency)

### File Sizes

**Typical sizes:**
- Preview text: 1-10 KB
- Extracted images: Varies (original size preserved)
- Detected persons: 100 KB - 2 MB (PNG compression)

**Large datasets:**
- 1,000 PDFs with 5 images each
- ~5,000 extracted images
- ~500 detected persons (assuming 10% hit rate)
- Total storage: 2-10 GB (depends on image sizes)

## Data Privacy

**Detected persons directory:**
- Contains all images with persons from ALL PDFs
- Easy to review/audit person detections
- Filename includes source PDF for traceability
- Consider encrypting this directory for sensitive data

**Preview text files:**
- May contain sensitive information
- First 50 words or OCR results
- Keep in per-PDF directories for organization
- Easy to delete selectively

## Workflow Example

```bash
# Before processing
my_pdfs/
├── document1.pdf
├── report.pdf
└── scan.pdf

# After processing
my_pdfs/
├── detected_persons/
│   ├── document1_page_2_img_1.png
│   ├── report_page_1_img_1.png
│   └── scan_page_10_img_3.png
│
├── document1/
│   ├── document1.pdf
│   ├── document1_preview.txt
│   └── page_2_img_1.png
│
├── report/
│   ├── report.pdf
│   ├── report_preview.txt
│   └── page_1_img_1.png
│
└── scan/
    ├── scan.pdf
    ├── scan_preview.txt
    └── page_10_img_3.png
```
