"""
image_captioner.py
------------------
Responsible for:
1. Taking a list of image file paths
2. Skipping tiny/irrelevant images (logos, icons, decorative elements)
3. Sending images to Moondream via Ollama IN PARALLEL for speed
4. Returning a list of captions paired with their source image path
"""

import requests
import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
VISION_MODEL = "llava-phi3:latest"
MIN_IMAGE_SIZE_BYTES = 10_000   # skip images smaller than 10KB (logos, icons)
MAX_WORKERS = 3              

CAPTION_SYSTEM_PROMPT = """You are an expert document analyst specializing in extracting information from images found inside PDF documents.
Your job is to produce a rich, structured description of the image that will later be used to answer questions about the document.

## Your analysis must cover:

1. **Type** — What kind of image is this? (photograph, diagram, flowchart, bar chart, line graph, pie chart, table, schematic, screenshot, map, equation, etc.)

2. **Content** — Describe everything visible in detail:
   - For charts/graphs: describe axes, units, data trends, peaks, patterns, and the key insight
   - For tables: transcribe all rows, columns, headers, and values
   - For diagrams/schematics: describe components, connections, flow direction, labels
   - For photographs: describe the subject, setting, notable elements
   - For text/equations: transcribe the full text or equation exactly 

3. **Labels & Annotations** — List every label, legend entry, axis title, unit, annotation, or caption visible

4. **Key Takeaway** — In 1-2 sentences, what is the main point or finding this image communicates?

5. **Context Clues** — Note any figure numbers, titles, or captions that appear near or within the image

## Rules:
- Be exhaustive — a retrieval system will use your description to answer user questions
- Never say "I cannot read" or skip content — do your best to transcribe everything
- If numbers or values are present, include them exactly
- Use structured formatting (bullet points, sections) for clarity
"""

CAPTION_PROMPT = """Analyze this image extracted from a PDF document and provide a complete, structured description following your analysis framework.
Cover: image type, all visible content, labels and annotations, key takeaway, and any figure numbers or titles.
Be exhaustive and precise — your description will be used to answer questions about this document."""


def is_image_worth_captioning(image_path: str) -> bool:
    """
    Returns False for tiny images that are likely logos, icons,
    or decorative elements not worth captioning.
    """
    size = os.path.getsize(image_path)
    return size >= MIN_IMAGE_SIZE_BYTES


def encode_image_to_base64(image_path: str) -> str:
    """Reads an image file and encodes it as a base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def caption_single_image(image_path: str) -> dict:
    """
    Sends one image to MiniCPM-V and returns a result dict.
    Uses /api/chat endpoint which MiniCPM-V requires for vision.

    Returns:
        Dict with 'image_path' and 'caption' (empty string on failure)
    """
    image_b64 = encode_image_to_base64(image_path)

    # MiniCPM-V works best with /api/chat endpoint using message format
    chat_payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": CAPTION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": CAPTION_PROMPT,
                "images": [image_b64]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024
        }
    }

    try:
        # Try /api/chat first (works with MiniCPM-V)
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=chat_payload,
            timeout=500  # vision models need more time
        )
        response.raise_for_status()
        data = response.json()

        # /api/chat returns message.content
        caption = ""
        if "message" in data:
            caption = data["message"].get("content", "").strip()

        # fallback: try response field just in case
        if not caption:
            caption = data.get("response", "").strip()

        if caption:
            return {"image_path": image_path, "caption": caption}

        # If still empty, log the raw response for debugging
        print(f"[image_captioner] WARNING: Empty caption for {os.path.basename(image_path)}. Raw response: {str(data)[:200]}")
        return {"image_path": image_path, "caption": ""}

    except requests.exceptions.Timeout:
        print(f"[image_captioner] WARNING: Timeout on {os.path.basename(image_path)}, skipping.")
        return {"image_path": image_path, "caption": ""}

    except requests.exceptions.RequestException as e:
        print(f"[image_captioner] ERROR on {os.path.basename(image_path)}: {e}")
        return {"image_path": image_path, "caption": ""}


def caption_all_images(image_paths: list) -> list:
    """
    Captions all images in parallel, skipping tiny ones.

    Args:
        image_paths: List of image file paths (from pdf_extractor)

    Returns:
        List of dicts with 'image_path' and 'caption'
    """
    if not image_paths:
        print("[image_captioner] No images to caption.")
        return []

    # Filter out tiny images first
    worthwhile = [p for p in image_paths if is_image_worth_captioning(p)]
    skipped = len(image_paths) - len(worthwhile)

    if skipped > 0:
        print(f"[image_captioner] Skipping {skipped} tiny images (logos/icons).")

    if not worthwhile:
        print("[image_captioner] No images worth captioning after filtering.")
        return []

    print(f"[image_captioner] Captioning {len(worthwhile)} images with {MAX_WORKERS} parallel workers...")

    captions = []
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(caption_single_image, path): path for path in worthwhile}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result["caption"]:
                captions.append(result)
                print(f"[image_captioner] {completed}/{len(worthwhile)} done — {os.path.basename(result['image_path'])} ({len(result['caption'])} chars)")
            else:
                print(f"[image_captioner] {completed}/{len(worthwhile)} skipped (empty response).")

    print(f"[image_captioner] Captioned {len(captions)}/{len(worthwhile)} images successfully.")
    return captions