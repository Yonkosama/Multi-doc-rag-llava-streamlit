from __future__ import annotations

from typing import List, Tuple, Dict, Any
import os
import base64

from unstructured.partition.pdf import partition_pdf


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def process_pdf(
    file_path: str,
    image_output_dir: str = "uploaded_images",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract text, tables, and images (as base64) from a PDF file.
    Saves images to disk for reference and returns their base64 too.
    Returns: (texts, tables, images_b64)
    """
    _ensure_dir(image_output_dir)

    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    texts: List[str] = []
    tables: List[str] = []
    images_b64: List[str] = []

    def _save_b64_image(image_b64: str, idx: int) -> str:
        base_name = os.path.basename(file_path)
        name_no_ext = os.path.splitext(base_name)[0]
        out_path = os.path.join(image_output_dir, f"{name_no_ext}_img_{idx}.png")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        return out_path

    image_index = 0

    for chunk in chunks:
        chunk_type = str(type(chunk))

        if "CompositeElement" in chunk_type:
            orig_elements = getattr(chunk.metadata, "orig_elements", [])
            for el in orig_elements:
                el_type = str(type(el))
                if "Table" in el_type:
                    tables.append(str(el))
                elif "Image" in el_type:
                    img_b64 = el.metadata.image_base64
                    images_b64.append(img_b64)
                    _save_b64_image(img_b64, image_index)
                    image_index += 1
                else:
                    texts.append(str(el))

        elif "Table" in chunk_type:
            tables.append(str(chunk))

        elif "Image" in chunk_type:
            img_b64 = chunk.metadata.image_base64
            images_b64.append(img_b64)
            _save_b64_image(img_b64, image_index)
            image_index += 1

        else:
            texts.append(str(chunk))

    return texts, tables, images_b64

