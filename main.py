import os
import zipfile
import py7zr
import fitz
from paddleocr import PaddleOCR
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
from PIL import Image
from typing import Tuple
from ultralytics import YOLO
import cv2
import shutil


def select_files():
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
    # Convert to grayscale
    if len(img_array.shape) == 3:
        grayscale = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        grayscale = img_array.astype(np.uint8)

    height, width = grayscale.shape
    total_pixels = height * width

    # Quick check for fully blacked out images
    black_pixels = np.sum(grayscale < 20)
    if black_pixels / total_pixels > 0.5:
        return False, True  # Not text, is redacted

    # Edge detection using Canny
    edges = cv2.Canny(grayscale, 50, 150)
    edge_density = np.sum(edges > 0) / total_pixels

    # Text documents typically have 8-25% edge density (stricter range)
    if edge_density < 0.08 or edge_density > 0.30:
        return False, False  # Too few or too many edges for substantial text

    # Connected components analysis
    # Threshold image to binary
    _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(binary), connectivity=8
    )

    # Analyze components (skip background label 0)
    text_like_components = 0
    min_component_size = 15  # Minimum pixels for a component (stricter)
    max_component_size = total_pixels * 0.08  # Max 8% of image (stricter)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width_comp = stats[i, cv2.CC_STAT_WIDTH]
        height_comp = stats[i, cv2.CC_STAT_HEIGHT]

        # Filter by size
        if area < min_component_size or area > max_component_size:
            continue

        # Calculate aspect ratio
        if height_comp > 0:
            aspect_ratio = width_comp / height_comp
        else:
            continue

        # Text characters typically have aspect ratios between 0.2 and 4.0 (stricter)
        if 0.2 <= aspect_ratio <= 4.0:
            text_like_components += 1

    # If we have many text-like components relative to image size,
    # it's likely a text scan (higher threshold = less sensitive)
    component_density = text_like_components / (total_pixels / 1000)  # per 1000 pixels

    is_text = component_density > 2.0 and edge_density > 0.10

    return is_text, False


def find_text_region(img_array: np.ndarray, skip_amount: int = 0) -> tuple[int, int] | None:
    """Find the top text region in an image based on dark pixel density.

    Args:
        img_array: Image as numpy array
        skip_amount: Number of rows to skip from top before searching

    Returns:
        (top, bottom) coordinates or None if no text region found
    """
    # Convert to grayscale
    if len(img_array.shape) == 3:
        grayscale = np.mean(img_array, axis=2)
    else:
        grayscale = img_array

    height, width = grayscale.shape

    # Count dark pixels per row (< 128 is considered dark/text)
    dark_threshold = 128
    dark_pixels_per_row = np.sum(grayscale < dark_threshold, axis=1)
    black_percent = np.sum(dark_pixels_per_row) / (height * width)

    # Find first row with significant dark pixels (> 10% of width)
    significant_threshold = width * 0.1
    # Document probably blacked out (>50% of all pixels are black)
    blackout_threshold = 0.5

    # Ensure skip_amount is within bounds
    skip_amount = max(0, min(skip_amount, len(dark_pixels_per_row) - 1))

    if black_percent > blackout_threshold:
        return None
    for i, count in enumerate(dark_pixels_per_row[skip_amount:], start=skip_amount):
        if count > significant_threshold:
            # Found first text row - define region around it
            top = max(0, int(i - height * 0.05))
            bottom = min(height, int(i + height * 0.15))  # 10% below
            return top, bottom
    return None


def ocr_image_region(img_array: np.ndarray, reader: PaddleOCR, use_find_region: bool = True, debug: bool = False, min_words: int = 10) -> str:
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
    max_iterations = 5  # Prevent infinite loops

    if use_find_region:
        for iteration in range(max_iterations):
            region = find_text_region(img_array, skip_amount=skip_amount)
            if not region:
                if debug and iteration == 0:
                    print(f"      [find_region] No region found")
                break

            top, bottom = region
            region_height = bottom - top
            img_height = img_array.shape[0]

            if debug:
                print(f"      [find_region] Iteration {iteration + 1}: rows {top}-{bottom} (height: {region_height}px, {region_height/img_height*100:.1f}% of image)")

            cropped = img_array[top:bottom, :]

            # PaddleOCR expects RGB, convert if needed
            if len(cropped.shape) == 2:  # Grayscale
                cropped = np.stack([cropped] * 3, axis=-1)

            # PaddleOCR 3.4.0 returns: [{'rec_texts': [...], 'rec_scores': [...], ...}]
            result = reader.predict(cropped)

            # Extract text from PaddleOCR result
            texts = []
            if result and len(result) > 0 and 'rec_texts' in result[0]:
                texts = result[0]['rec_texts']  # List of recognized text strings

            if texts:
                all_text_results.extend(texts)
                word_count = len(" ".join(all_text_results).split())

                if debug:
                    print(f"      [find_region] Found {len(texts)} text lines, total words: {word_count}")

                if word_count >= min_words:
                    break

            # Continue search below this region
            skip_amount = bottom + 1

            if skip_amount >= img_height:
                break

    return " ".join(all_text_results)


def process_pdf_content(pdf_path: Path, reader: PaddleOCR, yolo_model: YOLO, persons_dir: Path, words_per_page: int = 30) -> tuple[bool, bool]:
    """Unified PDF content processing: extract text, classify images, generate previews.

    Args:
        pdf_path: Path to PDF file
        reader: PaddleOCR instance (initialized once, reused across PDFs)
        yolo_model: YOLOv8 Pose model instance for person detection
        persons_dir: Shared directory for saving images with detected persons
        words_per_page: Number of words to extract per page for preview

    Returns:
        Tuple of (text_found, person_found)
    """
    from io import BytesIO

    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem
    output_file = pdf_path.parent / f"{pdf_name}_preview.txt"
    output_dir = pdf_path.parent / pdf_name
    doc = fitz.open(pdf_path)

    page_contents = []
    total_images = 0
    text_found = False
    person_found = False

    for page_num in range(doc.page_count):
        page = doc[page_num]

        # 1. Get normal text for preview
        normal_text = page.get_text().strip()
        normal_words = normal_text.split()

        if len(normal_words) >= 10:
            preview_words = normal_words[:words_per_page]
            preview_text = " ".join(preview_words)
            page_contents.append(f"Page {page_num + 1}: {preview_text}\n")
            text_found = True

        # 2. Process all embedded images uniformly
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Load image
            pil_image = Image.open(BytesIO(image_bytes))
            img_array = np.array(pil_image)

            # Check if image contains text
            is_text, is_fully_redacted = is_text_scan(img_array)

            # Skip fully redacted images
            if is_fully_redacted:
                continue

            # Save image to disk
            save_image_to_disk(output_dir, image_bytes, image_ext, page_num + 1, img_index + 1)
            total_images += 1

            if is_text:
                # Contains text → run OCR
                ocr_text = ocr_image_region(img_array, reader, use_find_region=False, debug=False)
                ocr_words = ocr_text.split()[:words_per_page]

                if ocr_words:
                    embed_preview = " ".join(ocr_words)
                    page_contents.append(f"  - page {page_num + 1} img {img_index + 1}: {embed_preview}\n")
                    text_found = True
                if len(ocr_words) > 10:
                    continue
                # Run YOLO if less than 10 words found

            # No or not much text → run YOLO directly
            detected = yolo_scan(img_array, persons_dir, pdf_name, page_num + 1, img_index + 1, yolo_model)
            if detected:
                person_found = True

    doc.close()

    # Write preview file
    if page_contents:
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(page_contents)

    return text_found, person_found


def process_pdfs(pdf_paths: list[Path], reader: PaddleOCR, yolo_model: YOLO) -> None:
    """Process the list of PDF files.

    Args:
        pdf_paths: List of PDF file paths
        reader: PaddleOCR instance (initialized once, reused across PDFs)
        yolo_model: YOLOv8 Pose model instance for person detection
    """
    # Create shared directory for all detected persons
    if pdf_paths:
        persons_dir = pdf_paths[0].parent / "detected_persons"
    else:
        persons_dir = Path("detected_persons")

    total = len(pdf_paths)
    print(f"\nProcessing {total} PDF file(s):\n")

    for idx, pdf_path in enumerate(pdf_paths, 1):
        text_found, person_found = process_pdf_content(pdf_path, reader, yolo_model, persons_dir)

        # Format output
        pdf_name = pdf_path.stem
        text_status = "Yes" if text_found else "No"
        person_status = "Yes" if person_found else "No"

        print(f"[{idx}/{total}] {pdf_name}")
        print(f"  Text: {text_status}  |  Person: {person_status}\n")

        # Move PDF and preview.txt into extraction folder
        output_dir = pdf_path.parent / pdf_name
        if output_dir.exists():
            # Move PDF
            pdf_destination = output_dir / pdf_path.name
            shutil.move(str(pdf_path), str(pdf_destination))

            # Move preview.txt if it exists
            preview_file = pdf_path.parent / f"{pdf_name}_preview.txt"
            if preview_file.exists():
                preview_destination = output_dir / preview_file.name
                shutil.move(str(preview_file), str(preview_destination))


def inspect_pdf_structure(pdf_path: Path) -> None:
    """Demonstrate PyMuPDF's extraction capabilities by analyzing PDF structure and content."""
    doc = fitz.open(pdf_path)
    output = []

    # Collect metadata
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

    # Document info
    if doc.is_encrypted or doc.metadata.get("format"):
        output.append("\nDOCUMENT INFO:")
        output.append(f"  Pages: {doc.page_count}")
        if doc.is_encrypted:
            output.append(f"  Encrypted: True")
        if doc.metadata.get("format"):
            output.append(f"  PDF Version: {doc.metadata.get('format')}")

    # First page analysis
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

    # Total images
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
    # Run YOLO inference with confidence threshold
    results = model(img_array, conf=0.5, verbose=False)

    # Check if any person detected with sufficient keypoints
    person_detected = False
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            # Check if any detection is a person (class 0 in COCO)
            for idx, box in enumerate(result.boxes):
                person_class = int(box.cls[0])
                if person_class == 0:  # Person class
                    # Verify keypoints (YOLOv8-Pose has 17 keypoints per person)
                    if result.keypoints is not None and len(result.keypoints.data) > idx:
                        keypoints = result.keypoints.data[idx]  # Shape: (17, 3) - x, y, confidence
                        # Count visible keypoints (confidence > 0.5)
                        visible_keypoints = (keypoints[:, 2] > 0.5).sum().item()

                        if visible_keypoints > 2:  # At least 3 keypoints visible
                            person_detected = True
                            break
                    else:
                        # No keypoints available, skip this detection
                        continue

    # Save image if person detected
    if person_detected:
        output_dir.mkdir(exist_ok=True)
        output_filename = f"{pdf_name}_page_{page_num}_img_{img_index}.png"
        output_path = output_dir / output_filename

        # Convert to PIL Image and save
        if len(img_array.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(img_array)
        else:  # RGB/RGBA
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

        # Check for redaction annotations
        annots = page.annots()
        redaction_count = 0
        if annots:
            for annot in annots:
                if annot.type[0] == 12:  # Redaction annotation type
                    redaction_count += 1
        if redaction_count > 0:
            print(f"\n  Content Layer Analysis:")
            finding = (
                f"Page {page_num + 1}: Found {redaction_count} redaction annotation(s)"
            )
            findings.append(finding)
            print(f"    [FOUND] {finding}")

        # Check for drawings/overlays (rectangles, paths)
        drawings = page.get_drawings()
        if len(drawings) > 10:  # Arbitrary threshold for suspicious overlay count
            finding = (
                f"Page {page_num + 1}: High number of drawing objects ({len(drawings)})"
            )
            findings.append(finding)
            print(f"    [FOUND] {finding}")

        # Check for suspicious text colors (white text, or text matching likely backgrounds)
        text_dict = page.get_text("dict")
        white_text_count = 0
        black_text_count = 0

        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        color = span.get("color")
                        if color is not None:
                            # Color is stored as integer RGB
                            if color == 0xFFFFFF:  # White
                                white_text_count += 1
                            elif color == 0x000000:  # Black
                                black_text_count += 1

        if white_text_count > 0:
            finding = f"Page {page_num + 1}: Found {white_text_count} white text span(s) (potentially hidden on white background)"
            findings.append(finding)
            print(f"    [FOUND] {finding}")

    doc.close()

    return findings


def main():
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

    # Initialize PaddleOCR once for all PDFs
    print("\nInitializing PaddleOCR...")
    reader = PaddleOCR(use_textline_orientation=True, lang='en')
    print("PaddleOCR ready!")

    # Initialize YOLOv8 Pose model for person detection
    print("Initializing YOLOv8 Pose model...")
    yolo_model = YOLO('yolov8n-pose.pt')
    print("YOLOv8 ready!\n")

    process_pdfs(pdf_list, reader, yolo_model)


if __name__ == "__main__":
    main()
