import os
import warnings

# Suppress all warnings globally
warnings.filterwarnings("ignore")

# Suppress PaddleOCR/PaddleX logging BEFORE importing
os.environ["DISABLE_AUTO_LOGGING_CONFIG"] = "1"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["PADDLEX_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["HUB_OFFLINE"] = "1"  # Disable connectivity checks

# Suppress PaddlePaddle C++ backend logging (glog)
os.environ["FLAGS_logtostderr"] = "0"
os.environ["FLAGS_stderrthreshold"] = "3"  # 3 = FATAL only
os.environ["GLOG_minloglevel"] = "3"  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL

# Save original stdout/stderr for later use
import sys

_original_stdout = sys.stdout
_original_stderr = sys.stderr

import zipfile
import py7zr
import fitz
from paddleocr import PaddleOCR
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
from PIL import Image
from typing import Tuple, Any
from multiprocessing.managers import DictProxy, ValueProxy
from ultralytics import YOLO
import cv2
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager
from queue import Empty
import threading
import logging

# Suppress PaddleOCR and PaddleX logging
logging.getLogger("paddleocr").setLevel(logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddlex").setLevel(logging.ERROR)
logging.getLogger("PaddleX").setLevel(logging.ERROR)


def select_files() -> list[str]:
    """Open file dialog and return selected file paths."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_paths = filedialog.askopenfilenames(
        title="Select PDF files or archive (zip/7z)",
        filetypes=[
            ("All supported", "*.pdf;*.zip;*.7z"),
            ("PDF files", "*.pdf"),
            ("Archives", "*.zip;*.7z"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()
    return list(file_paths)


def extract_archive(archive_path: Path) -> list[Path]:
    """Extract archive to a folder with the archive's name and return extracted PDF paths."""
    archive_path = Path(archive_path)
    extract_dir = archive_path.parent / archive_path.stem
    extract_dir.mkdir(exist_ok=True)

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.suffix.lower() == ".7z":
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path=extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

    return get_pdf_files_from_directory(extract_dir)


def get_pdf_files_from_directory(directory: Path) -> list[Path]:
    """Recursively find all PDF files in a directory."""
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(Path(root) / file)
    return pdf_files


def build_pdf_list(selected_paths: list[str]) -> list[Path]:
    """Build a list of PDF file paths from selected files/archives."""
    pdf_list = []

    for path in selected_paths:
        path = Path(path)

        if path.suffix.lower() in [".zip", ".7z"]:
            extracted_pdfs = extract_archive(path)
            pdf_list.extend(extracted_pdfs)
        elif path.suffix.lower() == ".pdf":
            pdf_list.append(path)

    return pdf_list


def is_text_scan(img_array: np.ndarray) -> Tuple[bool, bool]:
    """Check if image is a text scan using edge density and connected components analysis.

    Args:
        img_array: Image as numpy array

    Returns:
        Tuple of (is_text_scan, is_fully_redacted):
        - is_text_scan: True if image contains predominantly text
        - is_fully_redacted: True if image is mostly blacked out (>50% black pixels)
    """
    if len(img_array.shape) == 3:
        grayscale = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        grayscale = img_array.astype(np.uint8)
    height, width = grayscale.shape
    total_pixels = height * width

    black_pixels = np.sum(grayscale < 20)
    if black_pixels / total_pixels > 0.5:
        return False, True  # Not text, is redacted

    edges = cv2.Canny(grayscale, 50, 150)
    edge_density = np.sum(edges > 0) / total_pixels
    if edge_density < 0.08 or edge_density > 0.30:
        return False, False  # Too few or too many edges for substantial text

    is_text = edge_density > 0.10

    # # Connected components analysis (COMMENTED OUT FOR PERFORMANCE)
    # # Threshold image to binary
    # _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    # # Find connected components
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    #     cv2.bitwise_not(binary), connectivity=8
    # )
    #
    # # Analyze components (skip background label 0)
    # text_like_components = 0
    # min_component_size = 15  # Minimum pixels for a component (stricter)
    # max_component_size = total_pixels * 0.08  # Max 8% of image (stricter)
    #
    # for i in range(1, num_labels):
    #     area = stats[i, cv2.CC_STAT_AREA]
    #     width_comp = stats[i, cv2.CC_STAT_WIDTH]
    #     height_comp = stats[i, cv2.CC_STAT_HEIGHT]
    #
    #     # Filter by size
    #     if area < min_component_size or area > max_component_size:
    #         continue
    #
    #     # Calculate aspect ratio
    #     if height_comp > 0:
    #         aspect_ratio = width_comp / height_comp
    #     else:
    #         continue
    #
    #     # Text characters typically have aspect ratios between 0.2 and 4.0 (stricter)
    #     if 0.2 <= aspect_ratio <= 4.0:
    #         text_like_components += 1
    #
    # # If we have many text-like components relative to image size,
    # # it's likely a text scan (higher threshold = less sensitive)
    # component_density = text_like_components / (total_pixels / 1000)  # per 1000 pixels
    #
    # is_text = component_density > 2.0 and edge_density > 0.10

    return is_text, False


def find_text_region(
    img_array: np.ndarray, skip_amount: int = 0
) -> tuple[int, int] | None:
    """Find the top text region in an image based on dark pixel density.

    Args:
        img_array: Image as numpy array
        skip_amount: Number of rows to skip from top before searching

    Returns:
        (top, bottom) coordinates or None if no text region found
    """
    if len(img_array.shape) == 3:
        grayscale = np.mean(img_array, axis=2)
    else:
        grayscale = img_array

    height, width = grayscale.shape
    dark_threshold = 128
    dark_pixels_per_row = np.sum(grayscale < dark_threshold, axis=1)
    black_percent = np.sum(dark_pixels_per_row) / (height * width)
    significant_threshold = width * 0.1
    blackout_threshold = 0.5

    skip_amount = max(0, min(skip_amount, len(dark_pixels_per_row) - 1))

    if black_percent > blackout_threshold:
        return None

    for i, count in enumerate(dark_pixels_per_row[skip_amount:], start=skip_amount):
        if count > significant_threshold:
            top = max(0, int(i - height * 0.05))
            bottom = min(height, int(i + height * 0.15))
            return top, bottom
    return None


def ocr_image_region(
    img_array: np.ndarray,
    reader: PaddleOCR,
    use_find_region: bool = True,
    debug: bool = False,
    min_words: int = 10,
) -> str:
    """Run OCR on image with optional region detection for performance.

    Args:
        img_array: Image as numpy array
        reader: PaddleOCR instance
        use_find_region: If True, try to find text region first for faster OCR
        debug: If True, print debug information about region detection
        min_words: Minimum words to extract before stopping region search

    Returns:
        Extracted text as string
    """
    all_text_results = []
    skip_amount = 0
    max_iterations = 5

    if use_find_region:
        for iteration in range(max_iterations):
            region = find_text_region(img_array, skip_amount=skip_amount)
            if not region:
                break

            top, bottom = region
            region_height = bottom - top
            img_height = img_array.shape[0]
            cropped = img_array[top:bottom, :]

            if len(cropped.shape) == 2:
                cropped = np.stack([cropped] * 3, axis=-1)

            result = reader.predict(cropped)

            texts = []
            if result and len(result) > 0 and "rec_texts" in result[0]:
                texts = result[0]["rec_texts"]

            if texts:
                all_text_results.extend(texts)
                word_count = len(" ".join(all_text_results).split())
                if word_count >= min_words:
                    break

            skip_amount = bottom + 1
            if skip_amount >= img_height:
                break

    return " ".join(all_text_results)


def _yolo_scan_queued(
    img_array: np.ndarray,
    persons_dir: Path,
    pdf_name: str,
    page_num: int,
    img_index: int,
) -> bool:
    """Queue YOLO task and wait for result.

    Args:
        img_array: Image as numpy array
        persons_dir: Directory to save detected images
        pdf_name: PDF name
        page_num: Page number
        img_index: Image index

    Returns:
        True if person detected, False otherwise
    """
    global _yolo_queue, _yolo_results
    import uuid
    import time

    task_id = str(uuid.uuid4())

    _yolo_queue.put(
        {
            "id": task_id,
            "img_array": img_array,
            "persons_dir": persons_dir,
            "pdf_name": pdf_name,
            "page_num": page_num,
            "img_index": img_index,
        }
    )

    max_wait = 60
    start_time = time.time()
    while task_id not in _yolo_results:
        if time.time() - start_time > max_wait:
            return False
        time.sleep(0.01)
    result = _yolo_results.pop(task_id)
    return result


def process_single_image(args: tuple[Any, ...]) -> tuple[str | None, bool]:
    """Process a single image.

    Args:
        args: Tuple of (image_bytes, image_ext, img_array, page_num, img_index,
                       output_dir, persons_dir, pdf_name, reader, words_per_page, skip_ocr)

    Returns:
        Tuple of (ocr_preview_text or None, person_detected)
    """
    (
        image_bytes,
        image_ext,
        img_array,
        page_num,
        img_index,
        output_dir,
        persons_dir,
        pdf_name,
        reader,
        words_per_page,
        skip_ocr,
    ) = args

    is_text, is_fully_redacted = is_text_scan(img_array)
    if is_fully_redacted:
        return None, False

    save_image_to_disk(output_dir, image_bytes, image_ext, page_num, img_index)
    ocr_preview = None
    person_detected = False
    if skip_ocr:
        person_detected = _yolo_scan_queued(
            img_array, persons_dir, pdf_name, page_num, img_index
        )
    else:
        if is_text:
            ocr_text = ocr_image_region(
                img_array, reader, use_find_region=False, debug=False
            )
            ocr_words = ocr_text.split()[:words_per_page]

            if ocr_words:
                embed_preview = " ".join(ocr_words)
                ocr_preview = f"  - page {page_num} img {img_index}: {embed_preview}\n"

            if len(ocr_words) <= 10:
                person_detected = _yolo_scan_queued(
                    img_array, persons_dir, pdf_name, page_num, img_index
                )
        else:
            person_detected = _yolo_scan_queued(
                img_array, persons_dir, pdf_name, page_num, img_index
            )

    return ocr_preview, person_detected


def process_pdf_content(
    pdf_path: Path,
    reader: PaddleOCR,
    persons_dir: Path,
    words_per_page: int = 50,
) -> tuple[bool, bool, str, int]:
    """Unified PDF content processing: extract text, classify images, generate previews.

    Args:
        pdf_path: Path to PDF file
        reader: PaddleOCR instance (initialized once, reused across PDFs)
        persons_dir: Shared directory for saving images with detected persons
        words_per_page: Number of words to extract for preview (default: 50)

    Returns:
        Tuple of (text_found, person_found, extracted_text, page_count)
    """
    from io import BytesIO

    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem
    output_file = pdf_path.parent / f"{pdf_name}_preview.txt"
    output_dir = pdf_path.parent / pdf_name
    doc = fitz.open(pdf_path)

    all_text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_text = page.get_text().strip()
        if page_text:
            all_text += page_text + "\n"

    all_words = all_text.split()
    has_readable_text = len(all_words) > 50
    page_contents = []
    person_found = False
    image_tasks = []

    if has_readable_text:
        preview_words = all_words[:words_per_page]
        preview_text = " ".join(preview_words)
        page_contents.append(
            f"Extracted Text Preview ({len(all_words)} words total):\n{preview_text}\n"
        )

        for page_num in range(doc.page_count):
            page = doc[page_num]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                pil_image = Image.open(BytesIO(image_bytes))
                img_array = np.array(pil_image)

                image_tasks.append(
                    (
                        image_bytes,
                        image_ext,
                        img_array,
                        page_num + 1,
                        img_index + 1,
                        output_dir,
                        persons_dir,
                        pdf_name,
                        reader,
                        words_per_page,
                        True,
                    )
                )
    else:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                pil_image = Image.open(BytesIO(image_bytes))
                img_array = np.array(pil_image)
                image_tasks.append(
                    (
                        image_bytes,
                        image_ext,
                        img_array,
                        page_num + 1,
                        img_index + 1,
                        output_dir,
                        persons_dir,
                        pdf_name,
                        reader,
                        words_per_page,
                        False,
                    )
                )
    page_count = doc.page_count
    doc.close()
    if image_tasks:
        for task in image_tasks:
            ocr_preview, person_detected = process_single_image(task)
            if ocr_preview:
                page_contents.append(ocr_preview)
            if person_detected:
                person_found = True

    if page_contents:
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(page_contents)

    text_found = has_readable_text or bool(page_contents)
    return text_found, person_found, all_text, page_count


_worker_reader: PaddleOCR | None = None
_yolo_queue: Any = None
_yolo_results: DictProxy | None = None
_total_pages_counter: ValueProxy | None = None


def _initialize_worker(
    yolo_queue: Any, yolo_results: DictProxy, total_pages_counter: ValueProxy
) -> None:
    """Initialize models in each worker process."""
    global _worker_reader, _yolo_queue, _yolo_results, _total_pages_counter
    import sys
    import os

    os.environ["DISABLE_AUTO_LOGGING_CONFIG"] = "1"
    os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "1"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
    os.environ["PADDLEX_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
    os.environ["HUB_OFFLINE"] = "1"

    import logging

    logging.getLogger("paddleocr").setLevel(logging.ERROR)
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    logging.getLogger("paddlex").setLevel(logging.ERROR)
    logging.getLogger("PaddleX").setLevel(logging.ERROR)

    if _worker_reader is None:
        devnull = open(os.devnull, "w")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull

        try:
            _worker_reader = PaddleOCR(use_textline_orientation=True, lang="en")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull.close()

    _yolo_queue = yolo_queue
    _yolo_results = yolo_results
    _total_pages_counter = total_pages_counter


def _process_pdf_worker(args: tuple[Path, Path]) -> tuple[str, bool, bool, int]:
    """Worker function for processing a single PDF in multiprocessing.

    Args:
        args: Tuple of (pdf_path, persons_dir)

    Returns:
        Tuple of (pdf_name, text_found, person_found, page_count)
    """
    pdf_path, persons_dir = args
    global _worker_reader, _total_pages_counter
    text_found, person_found, extracted_text, page_count = process_pdf_content(
        pdf_path, _worker_reader, persons_dir
    )
    if _total_pages_counter is not None:
        _total_pages_counter.value += page_count

    pdf_name = pdf_path.stem
    output_dir = pdf_path.parent / pdf_name
    output_dir.mkdir(exist_ok=True)
    pdf_destination = output_dir / pdf_path.name
    shutil.move(str(pdf_path), str(pdf_destination))
    preview_file = pdf_path.parent / f"{pdf_name}_preview.txt"
    if preview_file.exists():
        preview_destination = output_dir / preview_file.name
        shutil.move(str(preview_file), str(preview_destination))

    # Return status to main process for printing (Windows spawn doesn't show worker prints)
    return pdf_name, text_found, person_found, page_count


def _yolo_consumer_thread(
    yolo_queue: Any, yolo_results: DictProxy, yolo_model: YOLO
) -> None:
    """Consumer thread that processes YOLO tasks from queue.

    Args:
        yolo_queue: Queue with YOLO tasks
        yolo_results: Dict to store results
        yolo_model: YOLO model instance
    """
    while True:
        try:
            task = yolo_queue.get(timeout=1)
            if task is None:  # Poison pill
                break

            result = yolo_scan(
                task["img_array"],
                task["persons_dir"],
                task["pdf_name"],
                task["page_num"],
                task["img_index"],
                yolo_model,
            )

            yolo_results[task["id"]] = result

        except Empty:
            continue
        except Exception as e:
            print(f"Error in YOLO consumer: {e}", flush=True)


def process_pdfs(pdf_paths: list[Path]) -> None:
    """Process the list of PDF files using multiprocessing.

    Args:
        pdf_paths: List of PDF file paths
    """
    import time

    start_time = time.time()
    if pdf_paths:
        persons_dir = pdf_paths[0].parent / "detected_persons"
    else:
        persons_dir = Path("detected_persons")

    total = len(pdf_paths)
    print(f"\nProcessing {total} PDF file(s):\n")

    manager = Manager()
    yolo_queue = manager.Queue()
    yolo_results = manager.dict()
    total_pages_counter = manager.Value("i", 0)
    print("Initializing models...", flush=True)

    import os

    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull

    try:
        import warnings

        warnings.filterwarnings("ignore")
        yolo_model = YOLO("yolov8n-pose.pt", verbose=False)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

    consumer_thread = threading.Thread(
        target=_yolo_consumer_thread,
        args=(yolo_queue, yolo_results, yolo_model),
        daemon=True,
    )
    consumer_thread.start()
    print("Models initialized\n", flush=True)
    tasks = [(pdf_path, persons_dir) for pdf_path in pdf_paths]
    max_workers = min(cpu_count(), len(pdf_paths))

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_initialize_worker,
        initargs=(yolo_queue, yolo_results, total_pages_counter),
    ) as executor:
        future_to_pdf = {
            executor.submit(_process_pdf_worker, task): task[0] for task in tasks
        }
        completed = 0
        for future in as_completed(future_to_pdf):
            completed += 1
            pdf_name, text_found, person_found, page_count = future.result()
            text_status = "Yes" if text_found else "No"
            person_status = "Yes" if person_found else "No"
            print(
                f"[{completed}/{total}] ✓ {pdf_name} ({page_count} pages) | Text: {text_status} | Person: {person_status}",
                flush=True,
            )

    yolo_queue.put(None)  # Poison pill
    consumer_thread.join(timeout=10)
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    total_pages = total_pages_counter.value

    print(
        f"\n✓ Completed processing {total} PDFs ({total_pages} pages total) in {minutes}m {seconds}s"
    )


def inspect_pdf_structure(pdf_path: Path) -> None:
    """Demonstrate PyMuPDF's extraction capabilities by analyzing PDF structure and content."""
    doc = fitz.open(pdf_path)
    output = []
    metadata = doc.metadata
    meta_items = [
        (k, metadata.get(k, ""))
        for k in [
            "title",
            "author",
            "subject",
            "creator",
            "producer",
            "creationDate",
            "modDate",
        ]
    ]
    if any(v for _, v in meta_items):
        output.append("\nMETADATA:")
        for label, value in zip(
            [
                "Title",
                "Author",
                "Subject",
                "Creator",
                "Producer",
                "Created",
                "Modified",
            ],
            [v for _, v in meta_items],
        ):
            if value:
                output.append(f"  {label}: {value}")
    if doc.is_encrypted or doc.metadata.get("format"):
        output.append("\nDOCUMENT INFO:")
        output.append(f"  Pages: {doc.page_count}")
        if doc.is_encrypted:
            output.append(f"  Encrypted: True")
        if doc.metadata.get("format"):
            output.append(f"  PDF Version: {doc.metadata.get('format')}")
    if doc.page_count > 0:
        page = doc[0]
        text = page.get_text()
        image_list = page.get_images()
        links = page.get_links()

        if text or image_list or links:
            output.append(f"\nFIRST PAGE DETAILS:")
            output.append(
                f"  Dimensions: {page.rect.width:.1f} x {page.rect.height:.1f} pts"
            )
            if text:
                output.append(f"  Text: {len(text)} chars - {text[:200].strip()}...")
            if image_list:
                output.append(f"  Embedded images: {len(image_list)}")
            if links:
                output.append(f"  Links: {len(links)}")

    total_images = sum(len(doc[i].get_images()) for i in range(doc.page_count))
    if total_images:
        output.append(f"\nTotal images: {total_images}")

    doc.close()

    if output:
        print(f"\n{'='*60}")
        print(f"PDF Analysis: {Path(pdf_path).name}")
        print(f"{'='*60}")
        print("\n".join(output))


def save_image_to_disk(
    output_dir: Path, image_bytes: bytes, image_ext: str, page_num: int, img_index: int
) -> Path:
    """Save image bytes to disk with standardized naming.

    Args:
        output_dir: Directory to save image in
        image_bytes: Raw image bytes
        image_ext: Image file extension
        page_num: PDF page number (1-indexed)
        img_index: Image index on page (1-indexed)

    Returns:
        Path to saved image file
    """
    output_dir.mkdir(exist_ok=True)
    image_filename = f"page_{page_num}_img_{img_index}.{image_ext}"
    image_path = output_dir / image_filename
    with open(image_path, "wb") as img_file:
        img_file.write(image_bytes)
    return image_path


def yolo_scan(
    img_array: np.ndarray,
    output_dir: Path,
    pdf_name: str,
    page_num: int,
    img_index: int,
    model: YOLO,
) -> bool:
    """Scan image for persons using YOLOv8 Pose model.

    Args:
        img_array: Image as numpy array
        output_dir: Directory to save detected images
        pdf_name: Name of the PDF file (without extension)
        page_num: PDF page number (1-indexed)
        img_index: Image index on page (1-indexed)
        model: YOLOv8 Pose model instance

    Returns:
        True if person detected, False otherwise
    """
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    results = model(img_array, conf=0.5, verbose=False)

    person_detected = False
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for idx, box in enumerate(result.boxes):
                person_class = int(box.cls[0])
                if person_class == 0:
                    if (
                        result.keypoints is not None
                        and len(result.keypoints.data) > idx
                    ):
                        keypoints = result.keypoints.data[idx]
                        visible_keypoints = (keypoints[:, 2] > 0.3).sum().item()

                        if visible_keypoints > 0:
                            person_detected = True
                            break
                    else:
                        continue

    if person_detected:
        output_dir.mkdir(exist_ok=True)
        output_filename = f"{pdf_name}_page_{page_num}_img_{img_index}.png"
        output_path = output_dir / output_filename

        if len(img_array.shape) == 2:
            pil_image = Image.fromarray(img_array)
        else:
            pil_image = Image.fromarray(img_array)

        pil_image.save(output_path)

    return person_detected


def analyze_content_layers(pdf_path: Path) -> list[Path]:
    """Analyze document for potential content masking techniques."""
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    findings = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        annots = page.annots()
        redaction_count = 0
        if annots:
            for annot in annots:
                if annot.type[0] == 12:
                    redaction_count += 1
        if redaction_count > 0:
            print(f"\n  Content Layer Analysis:")
            finding = (
                f"Page {page_num + 1}: Found {redaction_count} redaction annotation(s)"
            )
            findings.append(finding)
            print(f"    [FOUND] {finding}")
        drawings = page.get_drawings()
        if len(drawings) > 10:
            finding = (
                f"Page {page_num + 1}: High number of drawing objects ({len(drawings)})"
            )
            findings.append(finding)
            print(f"    [FOUND] {finding}")

        text_dict = page.get_text("dict")
        white_text_count = 0
        black_text_count = 0

        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        color = span.get("color")
                        if color is not None:
                            if color == 0xFFFFFF:
                                white_text_count += 1
                            elif color == 0x000000:
                                black_text_count += 1

        if white_text_count > 0:
            finding = f"Page {page_num + 1}: Found {white_text_count} white text span(s) (potentially hidden on white background)"
            findings.append(finding)
            print(f"    [FOUND] {finding}")

    doc.close()

    return findings


def main() -> None:
    """Main application entry point."""
    print("PDFCrunch - PDF Content Extraction Tool")
    print("=" * 40)

    selected_files = select_files()

    if not selected_files:
        print("No files selected. Exiting.")
        return

    print(f"\nSelected {len(selected_files)} file(s)")

    pdf_list = build_pdf_list(selected_files)

    if not pdf_list:
        print("No PDF files found in selection. Exiting.")
        return

    print(f"\nUsing {min(cpu_count(), len(pdf_list))} parallel workers")
    print("Models (PaddleOCR, YOLO) will be initialized in each worker process...\n")

    process_pdfs(pdf_list)


if __name__ == "__main__":
    main()
