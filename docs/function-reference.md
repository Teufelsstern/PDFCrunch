# Function Reference

Complete API documentation for all functions in PDFCrunch.

## Table of Contents

- [File Selection & Archive Handling](#file-selection--archive-handling)
- [Image Analysis](#image-analysis)
- [PDF Processing](#pdf-processing)
- [Person Detection](#person-detection)
- [Multiprocessing Infrastructure](#multiprocessing-infrastructure)
- [Utilities](#utilities)

---

## File Selection & Archive Handling

### `select_files()`

Opens a GUI file dialog for selecting PDF files or archives.

```python
def select_files() -> list[str]
```

**Returns:**
- `list[str]`: List of selected file paths

**Behavior:**
- Supports PDF, ZIP, and 7Z files
- Uses tkinter file dialog (topmost window)
- Returns empty list if user cancels

---

### `extract_archive()`

Extracts archive to a folder with the archive's name.

```python
def extract_archive(archive_path: Path) -> list[Path]
```

**Parameters:**
- `archive_path` (Path): Path to ZIP or 7Z archive

**Returns:**
- `list[Path]`: List of PDF file paths found in extracted archive

**Raises:**
- `ValueError`: If archive format is not ZIP or 7Z

---

### `get_pdf_files_from_directory()`

Recursively finds all PDF files in a directory.

```python
def get_pdf_files_from_directory(directory: Path) -> list[Path]
```

**Parameters:**
- `directory` (Path): Directory to search

**Returns:**
- `list[Path]`: List of all PDF file paths found

---

### `build_pdf_list()`

Builds a list of PDF file paths from selected files/archives.

```python
def build_pdf_list(selected_paths: list[str]) -> list[Path]
```

**Parameters:**
- `selected_paths` (list[str]): Paths from file dialog

**Returns:**
- `list[Path]`: Unified list of all PDF paths (extracted or direct)

**Logic:**
- ZIP/7Z files → extract → collect PDFs
- PDF files → add directly

---

## Image Analysis

### `is_text_scan()`

Checks if image is a text scan using edge density analysis.

```python
def is_text_scan(img_array: np.ndarray) -> Tuple[bool, bool]
```

**Parameters:**
- `img_array` (np.ndarray): Image as numpy array (RGB or grayscale)

**Returns:**
- `tuple[bool, bool]`: `(is_text_scan, is_fully_redacted)`
  - `is_text_scan`: True if image contains predominantly text
  - `is_fully_redacted`: True if >50% black pixels

**Algorithm:**
1. Convert to grayscale if needed
2. Check for redaction (>50% black pixels)
3. Canny edge detection (thresholds 50, 150)
4. Calculate edge density
5. Text if 8-30% edge density and >10%

**Thresholds:**
- Edge density < 8% → not text (too sparse)
- Edge density > 30% → not text (too complex)
- Edge density > 10% → text document

---

### `find_text_region()`

Finds the top text region in an image based on dark pixel density.

```python
def find_text_region(img_array: np.ndarray, skip_amount: int = 0) -> tuple[int, int] | None
```

**Parameters:**
- `img_array` (np.ndarray): Image as numpy array
- `skip_amount` (int, optional): Number of rows to skip from top. Default: 0

**Returns:**
- `tuple[int, int] | None`: `(top, bottom)` coordinates or None if no region found

**Algorithm:**
1. Convert to grayscale
2. Count dark pixels per row (threshold < 128)
3. Find first row with >10% dark pixels
4. Return region: top = row - 5%, bottom = row + 15%

---

### `ocr_image_region()`

Runs OCR on image with optional region detection for performance.

```python
def ocr_image_region(
    img_array: np.ndarray,
    reader: PaddleOCR,
    use_find_region: bool = True,
    debug: bool = False,
    min_words: int = 10,
) -> str
```

**Parameters:**
- `img_array` (np.ndarray): Image as numpy array
- `reader` (PaddleOCR): PaddleOCR instance
- `use_find_region` (bool, optional): Use region detection. Default: True
- `debug` (bool, optional): Print debug info. Default: False
- `min_words` (int, optional): Minimum words before stopping. Default: 10

**Returns:**
- `str`: Extracted text (space-joined)

**Behavior:**
- Iterates up to 5 regions
- Stops when min_words reached
- Converts grayscale to RGB for PaddleOCR
- Returns empty string if no text found

---

## PDF Processing

### `process_pdf_content()`

Main PDF content processing function: extracts text, classifies images, generates previews.

```python
def process_pdf_content(
    pdf_path: Path,
    reader: PaddleOCR,
    persons_dir: Path,
    words_per_page: int = 50,
) -> tuple[bool, bool, str, int]
```

**Parameters:**
- `pdf_path` (Path): Path to PDF file
- `reader` (PaddleOCR): PaddleOCR instance (initialized once, reused)
- `persons_dir` (Path): Shared directory for detected persons
- `words_per_page` (int, optional): Words for preview. Default: 50

**Returns:**
- `tuple[bool, bool, str, int]`: `(text_found, person_found, extracted_text, page_count)`

**Logic:**
1. Extract all readable text from PDF
2. If >50 words: create preview, skip OCR, only run YOLO
3. If <=50 words: run OCR on text-like images, run YOLO on non-text
4. Save preview to `{pdf_name}_preview.txt`
5. Return status and page count

---

### `process_single_image()`

Processes a single image: classifies, runs OCR if needed, detects persons.

```python
def process_single_image(args: tuple[Any, ...]) -> tuple[str | None, bool]
```

**Parameters:**
- `args` (tuple): Tuple of `(image_bytes, image_ext, img_array, page_num, img_index, output_dir, persons_dir, pdf_name, reader, words_per_page, skip_ocr)`

**Returns:**
- `tuple[str | None, bool]`: `(ocr_preview_text, person_detected)`

**Logic:**
1. Check if image is fully redacted → skip
2. Save image to disk
3. If `skip_ocr=True`: only run YOLO
4. If `skip_ocr=False`:
   - Check `is_text_scan()`
   - If text: run OCR, check word count
   - If <=10 words or not text: run YOLO

---

### `save_image_to_disk()`

Saves image bytes to disk with standardized naming.

```python
def save_image_to_disk(
    output_dir: Path,
    image_bytes: bytes,
    image_ext: str,
    page_num: int,
    img_index: int
) -> Path
```

**Parameters:**
- `output_dir` (Path): Directory to save image in
- `image_bytes` (bytes): Raw image bytes
- `image_ext` (str): Image file extension (png, jpg, etc.)
- `page_num` (int): PDF page number (1-indexed)
- `img_index` (int): Image index on page (1-indexed)

**Returns:**
- `Path`: Path to saved image file

**Naming Convention:**
- `page_{page_num}_img_{img_index}.{ext}`
- Example: `page_1_img_2.png`

---

## Person Detection

### `yolo_scan()`

Scans image for persons using YOLOv8 Pose model.

```python
def yolo_scan(
    img_array: np.ndarray,
    output_dir: Path,
    pdf_name: str,
    page_num: int,
    img_index: int,
    model: YOLO,
) -> bool
```

**Parameters:**
- `img_array` (np.ndarray): Image as numpy array
- `output_dir` (Path): Directory to save detected images (`detected_persons/`)
- `pdf_name` (str): Name of PDF file (without extension)
- `page_num` (int): PDF page number (1-indexed)
- `img_index` (int): Image index on page (1-indexed)
- `model` (YOLO): YOLOv8 Pose model instance

**Returns:**
- `bool`: True if person detected, False otherwise

**Detection Logic:**
1. Convert grayscale to RGB if needed
2. Run YOLO with confidence=0.5
3. For each person detection (class 0):
   - Check keypoints (17 per person)
   - Count visible keypoints (confidence > 0.3)
   - If >0 visible keypoints: person detected
4. Save image if person detected

**Naming Convention:**
- `{pdf_name}_page_{page_num}_img_{img_index}.png`
- Example: `report_page_3_img_1.png`

---

### `_yolo_scan_queued()`

Queues YOLO task and waits for result (producer-consumer pattern).

```python
def _yolo_scan_queued(
    img_array: np.ndarray,
    persons_dir: Path,
    pdf_name: str,
    page_num: int,
    img_index: int,
) -> bool
```

**Parameters:**
- Same as `yolo_scan()` except no model parameter

**Returns:**
- `bool`: True if person detected, False otherwise (or timeout)

**Behavior:**
1. Generate unique task ID (UUID)
2. Put task in shared queue
3. Wait for result (max 60 seconds)
4. Return result from shared dict
5. Remove result from dict

**Thread Safety:**
- Workers (producers) add tasks to queue
- YOLO consumer thread (consumer) processes tasks
- Prevents YOLO thread-safety issues

---

## Multiprocessing Infrastructure

### `process_pdfs()`

Orchestrates multiprocessing for PDF batch processing.

```python
def process_pdfs(pdf_paths: list[Path]) -> None
```

**Parameters:**
- `pdf_paths` (list[Path]): List of PDF file paths to process

**Returns:**
- `None`

**Architecture:**
1. Create shared `detected_persons/` directory
2. Initialize YOLO model in main process
3. Start YOLO consumer thread
4. Create `Manager.Queue()` and `Manager.dict()`
5. Spawn worker processes (min(CPU cores, PDF count))
6. Collect results and print progress
7. Print completion stats (total PDFs, pages, time)

**Progress Output:**
```
[1/100] ✓ document (5 pages) | Text: Yes | Person: No
[2/100] ✓ report (12 pages) | Text: No | Person: Yes
...
✓ Completed processing 100 PDFs (1,234 pages total) in 5m 42s
```

---

### `_initialize_worker()`

Initializes models in each worker process.

```python
def _initialize_worker(
    yolo_queue: Any,
    yolo_results: DictProxy,
    total_pages_counter: ValueProxy
) -> None
```

**Parameters:**
- `yolo_queue` (Manager.Queue): Shared queue for YOLO tasks
- `yolo_results` (DictProxy): Shared dict for YOLO results
- `total_pages_counter` (ValueProxy): Shared page counter

**Returns:**
- `None`

**Behavior:**
1. Set environment variables (suppress logging)
2. Initialize PaddleOCR (per worker, stdout/stderr redirected)
3. Store shared queue, results dict, counter as globals

**Why per-worker PaddleOCR?**
- PaddleOCR is CPU-bound
- Each worker needs independent instance
- No shared state between workers

---

### `_process_pdf_worker()`

Worker function for processing a single PDF in multiprocessing.

```python
def _process_pdf_worker(args: tuple[Path, Path]) -> tuple[str, bool, bool, int]
```

**Parameters:**
- `args` (tuple[Path, Path]): `(pdf_path, persons_dir)`

**Returns:**
- `tuple[str, bool, bool, int]`: `(pdf_name, text_found, person_found, page_count)`

**Process:**
1. Call `process_pdf_content()`
2. Update global page counter
3. Create output directory
4. Move PDF to output directory
5. Move preview.txt if exists
6. Return status to main process

---

### `_yolo_consumer_thread()`

Consumer thread that processes YOLO tasks from queue.

```python
def _yolo_consumer_thread(
    yolo_queue: Any,
    yolo_results: DictProxy,
    yolo_model: YOLO
) -> None
```

**Parameters:**
- `yolo_queue` (Manager.Queue): Shared queue for YOLO tasks
- `yolo_results` (DictProxy): Shared dict to store results
- `yolo_model` (YOLO): YOLOv8 Pose model instance

**Returns:**
- `None`

**Infinite Loop:**
1. Get task from queue (timeout 1s)
2. If task is `None` (poison pill): break
3. Run `yolo_scan()` with task data
4. Store result in shared dict with task ID
5. Continue

**Error Handling:**
- Empty queue: continue
- Exceptions: print error, continue

---

## Utilities

### `inspect_pdf_structure()`

Analyzes and prints PDF structure and content (debugging utility).

```python
def inspect_pdf_structure(pdf_path: Path) -> None
```

**Parameters:**
- `pdf_path` (Path): Path to PDF file

**Returns:**
- `None` (prints to stdout)

**Output:**
- Metadata (title, author, creator, dates)
- Document info (pages, encryption, PDF version)
- First page details (dimensions, text, images, links)
- Total image count

---

### `analyze_content_layers()`

Analyzes document for potential content masking techniques.

```python
def analyze_content_layers(pdf_path: Path) -> list[Path]
```

**Parameters:**
- `pdf_path` (Path): Path to PDF file

**Returns:**
- `list[Path]`: List of findings (as strings, not paths - return type is wrong)

**Detects:**
- Redaction annotations (type 12)
- Excessive drawing objects (>10)
- White text on white background (color 0xFFFFFF)

**Note:** This function is not used in the main processing pipeline. It's a utility for forensic analysis.

---

### `main()`

Main application entry point.

```python
def main() -> None
```

**Returns:**
- `None`

**Flow:**
1. Print welcome message
2. Call `select_files()`
3. Build PDF list with `build_pdf_list()`
4. Print worker count
5. Call `process_pdfs()`
6. Exit

**Invocation:**
```bash
python main.py
```
