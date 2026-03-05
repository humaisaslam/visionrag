"""
pdf_extractor.py
----------------
Responsible for:
1. Extracting all text from a PDF (page by page)
2. Extracting all embedded images and saving them to data/images/
3. Returning the full text and list of saved image paths
"""

import fitz  # PyMuPDF
import os


def extract_from_pdf(pdf_path: str, images_output_dir: str = "data/images") -> dict:
    """
    Main function to extract text and images from a PDF.

    Args:
        pdf_path: Path to the input PDF file
        images_output_dir: Folder where extracted images will be saved

    Returns:
        A dict with:
            - 'text': full extracted text as a single string
            - 'image_paths': list of file paths to saved images
    """

    # Make sure the images output folder exists
    os.makedirs(images_output_dir, exist_ok=True)

    # Open the PDF
    doc = fitz.open(pdf_path)

    full_text = ""
    image_paths = []
    image_counter = 0

    for page_number in range(len(doc)):
        page = doc[page_number]

        # --- Extract Text ---
        page_text = page.get_text()
        if page_text.strip():  # only add if there's actual content
            full_text += f"\n\n--- Page {page_number + 1} ---\n\n"
            full_text += page_text

        # --- Extract Images ---
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]  # image reference number in the PDF

            # Get the raw image data
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]  # e.g. png, jpeg

            # Build a unique filename for this image
            image_filename = f"page{page_number + 1}_img{image_counter + 1}.{image_ext}"
            image_save_path = os.path.join(images_output_dir, image_filename)

            # Save the image to disk
            with open(image_save_path, "wb") as img_file:
                img_file.write(image_bytes)

            image_paths.append(image_save_path)
            image_counter += 1

    doc.close()

    print(f"[pdf_extractor] Extracted text: {len(full_text)} characters")
    print(f"[pdf_extractor] Extracted images: {len(image_paths)} images saved to '{images_output_dir}'")

    return {
        "text": full_text,
        "image_paths": image_paths
    }