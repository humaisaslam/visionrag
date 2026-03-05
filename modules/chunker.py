"""
chunker.py
----------
Responsible for:
1. Taking the full PDF text and splitting it into overlapping chunks
2. Taking image captions and treating each as its own chunk
3. Attaching metadata (source, page_num, image path) to each chunk
4. Returning a unified list of chunks ready for embedding
"""

import re

# --- Configuration ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def chunk_pdf_text(full_text: str) -> list:
    """
    Chunks raw PDF text. Tries to detect page breaks in the text
    (common markers: 'Page X', form feed char \x0c) to assign page_num.
    Falls back to estimating page number by chunk position if not found.
    """
    # Split into pages if pdf_extractor used form feed (\x0c) as page separator
    pages = full_text.split("\x0c")
    # Filter empty pages
    pages = [p for p in pages if p.strip()]

    result = []
    chunk_index = 0

    if len(pages) > 1:
        # We have real page breaks
        for page_num, page_text in enumerate(pages, start=1):
            raw_chunks = split_text_into_chunks(page_text)
            for chunk in raw_chunks:
                result.append({
                    "text": chunk,
                    "source": "pdf_text",
                    "metadata": {
                        "chunk_index": chunk_index,
                        "page_num": page_num,
                    }
                })
                chunk_index += 1
    else:
        # No page breaks — try to detect "Page N" markers in text
        # and split there, otherwise chunk without page info
        page_pattern = re.compile(r'(?:^|\n)(?:Page|PAGE)\s+(\d+)', re.MULTILINE)
        markers = list(page_pattern.finditer(full_text))

        if markers:
            segments = []
            for i, match in enumerate(markers):
                start = match.start()
                end = markers[i + 1].start() if i + 1 < len(markers) else len(full_text)
                segments.append((int(match.group(1)), full_text[start:end]))

            for page_num, segment_text in segments:
                raw_chunks = split_text_into_chunks(segment_text)
                for chunk in raw_chunks:
                    result.append({
                        "text": chunk,
                        "source": "pdf_text",
                        "metadata": {
                            "chunk_index": chunk_index,
                            "page_num": page_num,
                        }
                    })
                    chunk_index += 1
        else:
            # No page info available — chunk normally, page_num = "?"
            raw_chunks = split_text_into_chunks(full_text)
            for chunk in raw_chunks:
                result.append({
                    "text": chunk,
                    "source": "pdf_text",
                    "metadata": {
                        "chunk_index": chunk_index,
                        "page_num": "?",
                    }
                })
                chunk_index += 1

    print(f"[chunker] PDF text split into {len(result)} chunks.")
    return result


def chunk_image_captions(captions: list) -> list:
    """
    Converts image captions into chunks.
    image_path format: 'data/images/page12_img14.jpeg' → page_num = 12
    """
    result = []

    for item in captions:
        caption_text = item["caption"]
        image_path = item["image_path"]

        # Extract page number from filename e.g. page12_img3.jpeg → 12
        page_num = "?"
        match = re.search(r'page(\d+)', image_path)
        if match:
            page_num = int(match.group(1))

        base_meta = {
            "image_path": image_path,
            "page_num": page_num,
        }

        if len(caption_text) <= CHUNK_SIZE:
            result.append({
                "text": caption_text,
                "source": "image_caption",
                "metadata": base_meta
            })
        else:
            sub_chunks = split_text_into_chunks(caption_text)
            for i, sub in enumerate(sub_chunks):
                result.append({
                    "text": sub,
                    "source": "image_caption",
                    "metadata": {**base_meta, "chunk_index": i}
                })

    print(f"[chunker] Image captions produced {len(result)} chunks.")
    return result


def chunk_all(full_text: str, captions: list) -> list:
    text_chunks = chunk_pdf_text(full_text)
    caption_chunks = chunk_image_captions(captions)
    all_chunks = text_chunks + caption_chunks
    print(f"[chunker] Total chunks ready for embedding: {len(all_chunks)}")
    return all_chunks