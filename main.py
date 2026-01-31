import os
import zipfile
import py7zr
import fitz
import easyocr
import numpy as np
import torch
from pathlib import Path
from tkinter import Tk, filedialog
from PIL import Image
from typing import Tuple

# Configuration
USE_GPU = True  # Set to True to enable GPU acceleration for OCR (requires CUDA runtime)

# Check if CUDA is actually available
if USE_GPU and not torch.cuda.is_available():
    print("WARNING: GPU requested but CUDA not available. Falling back to CPU.")
    print(f"PyTorch version: {torch.__version__}")
    print("Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    USE_GPU = False


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
    """Check if image is a text scan (≥90% black/white pixels).

    Args:
        img_array: Image as numpy array

    Returns:
        True if image is likely a text scan (90%+ black/white pixels)
    """
    # Convert to grayscale
    if len(img_array.shape) == 3:
        grayscale = np.mean(img_array, axis=2)
    else:
        grayscale = img_array

    # Count black, white, and gray pixels
    black_pixels = np.sum(grayscale < 50)  # Very dark
    white_pixels = np.sum(grayscale > 200)  # Very light
    total_pixels = grayscale.size

    # Calculate percentage of black/white pixels
    bw_percentage = (black_pixels + white_pixels) / total_pixels
    b_percentage = (black_pixels) / total_pixels

    return bw_percentage >= 0.90, b_percentage >= 0.5


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
        print(f"Page probably blacked out ({black_percent*100:.1f}% black pixels), won't read.")
        return None
    for i, count in enumerate(dark_pixels_per_row[skip_amount:], start=skip_amount):
        if count > significant_threshold:
            # Found first text row - define region around it
            top = i
            bottom = min(height, int(i + height * 0.05))  # 10% below
            return top, bottom
    return None


def ocr_image_region(img_array: np.ndarray, reader: easyocr.Reader, use_find_region: bool = True, debug: bool = False, min_words: int = 10) -> str:
    """Run OCR on image with optional region detection for performance.

    Args:
        img_array: Image as numpy array
        reader: EasyOCR reader instance
        use_find_region: If True, try to find text region first for faster OCR
        debug: If True, print debug information about region detection
        min_words: Minimum words to extract before stopping region search

    Returns:
        Extracted text as string
    """
    all_ocr_results = []
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
            ocr_results = reader.readtext(cropped, detail=0)

            if ocr_results:
                all_ocr_results.extend(ocr_results)
                word_count = len(" ".join(all_ocr_results).split())

                if debug:
                    print(f"      [find_region] Found {len(ocr_results)} text blocks, total words: {word_count}")

                if word_count >= min_words:
                    break

            # Continue search below this region
            skip_amount = bottom + 1

            if skip_amount >= img_height:
                break

    return " ".join(all_ocr_results)


def process_pdf_content(pdf_path: Path, words_per_page: int = 30) -> None:
    """Unified PDF content processing: extract text, classify images, generate previews.

    Args:
        pdf_path: Path to PDF file
        words_per_page: Number of words to extract per page for preview
    """
    from io import BytesIO

    pdf_path = Path(pdf_path)
    output_file = pdf_path.parent / f"{pdf_path.stem}_preview.txt"
    output_dir = pdf_path.parent / pdf_path.stem
    doc = fitz.open(pdf_path)

    # Initialize OCR reader (lazy, only when needed)
    reader = None

    print(f"  Processing content...")

    page_contents = []
    total_real_images = 0

    for page_num in range(doc.page_count):
        page = doc[page_num]

        # 1. Get normal text
        normal_text = page.get_text().strip()
        normal_words = normal_text.split()

        # 2. Get embedded images and classify
        image_list = page.get_images()
        text_scans = []
        real_images = []

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Load image and classify
            pil_image = Image.open(BytesIO(image_bytes))
            img_array = np.array(pil_image)
            is_text, is_fully_redacted = is_text_scan(img_array)
            if is_fully_redacted: # Skip pages that are probably fully blacked out
                continue
            if is_text:
                text_scans.append((img_array, img_index + 1))
            real_images.append((image_bytes, image_ext, img_index + 1))

        # 3. Save real images
        for image_bytes, image_ext, img_index in real_images:
            save_image_to_disk(output_dir, image_bytes, image_ext, page_num + 1, img_index)
            total_real_images += 1

        # 4. Generate preview
        if len(normal_words) >= 10:
            # Case A: Normal text available
            preview_words = normal_words[:words_per_page]
            preview_text = " ".join(preview_words)
            page_contents.append(f"Page {page_num + 1}: {preview_text}\n")

            # OCR embedded text scans
            if text_scans:
                if reader is None:
                    reader = easyocr.Reader(["en"], gpu=USE_GPU, verbose=False)

                for img_array, img_index in text_scans:
                    print(f"    OCR embed{img_index} on page {page_num + 1}:")
                    ocr_text = ocr_image_region(img_array, reader, use_find_region=True, debug=True)
                    ocr_words = ocr_text.split()[:words_per_page]
                    if ocr_words:
                        embed_preview = " ".join(ocr_words)
                        page_contents.append(f"  - embed{img_index}: {embed_preview}\n")

        else:
            # Case B: No normal text → OCR whole page
            if reader is None:
                reader = easyocr.Reader(["en"], gpu=USE_GPU, verbose=False)

            # Render page as image
            pix = page.get_pixmap(dpi=300)
            page_img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            print(f"    OCR full page {page_num + 1}:")
            ocr_text = ocr_image_region(page_img_array, reader, use_find_region=True, debug=True)
            ocr_words = ocr_text.split()[:words_per_page]
            if ocr_words:
                preview_text = " ".join(ocr_words)
                page_contents.append(f"Page {page_num + 1}: {preview_text}\n")

    doc.close()

    # Write preview file
    if page_contents:
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(page_contents)
        print(f"  Saved page previews to: {output_file}")
    else:
        print(f"  No text found in any page, skipping preview file")

    # Report saved images
    if total_real_images > 0:
        print(f"  Saved {total_real_images} real images (excluded text scans) to: {output_dir}")


def process_pdfs(pdf_paths: list[Path]) -> None:
    """Process the list of PDF files."""
    print(f"\nProcessing {len(pdf_paths)} PDF file(s):")
    for pdf_path in pdf_paths:
        print(f"  - {pdf_path}")
        process_pdf_content(pdf_path)
        analyze_content_layers(pdf_path)


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


def analyze_content_layers(pdf_path: Path) -> list[Path]:
    """Analyze document for potential content masking techniques."""
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    findings = []

    print(f"\n  Content Layer Analysis:")

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

        # Check if page is fully rasterized (single large image covering page)
        images = page.get_images()
        text = page.get_text().strip()
        if len(images) == 1 and len(text) < 50:
            image_info = images[0]
            xref = image_info[0]
            base_image = doc.extract_image(xref)
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            page_area = page.rect.width * page.rect.height
            image_area_estimate = width * height

            # If image is large and text is minimal, likely rasterized
            if image_area_estimate > (page_area * 0.5):
                finding = f"Page {page_num + 1}: Appears to be rasterized (single large image, minimal extractable text)"
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

    if not findings:
        print(f"    No content masking indicators detected")

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

    process_pdfs(pdf_list)


if __name__ == "__main__":
    main()
