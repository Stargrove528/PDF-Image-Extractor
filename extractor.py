import fitz  # type: ignore
import os
import logging
from datetime import datetime
from io import BytesIO
from PIL import Image, ImageCms

# Keep originals for these formats only when they DO NOT have transparency.
SKIP_CONVERSION = ["jpeg"]


def setup_logging():
    log_filename = datetime.now().strftime("image_extraction_%Y%m%d_%H%M%S.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Logging initialized.")


def _tuple_has_mask(img_tuple):
    """Check if page.get_images(full=True) tuple reports a soft-mask xref at index 1."""
    try:
        return bool(img_tuple[1])
    except Exception:
        return False


def _open_pillow_image(image_bytes):
    """Open bytes with Pillow and normalize without flattening."""
    img = Image.open(BytesIO(image_bytes))
    # Normalize CMYK -> RGB (PNG can’t store CMYK)
    if img.mode == "CMYK":
        try:
            srgb = ImageCms.createProfile("sRGB")
            img = ImageCms.profileToProfile(img, None, srgb, outputMode="RGB")
        except Exception:
            img = img.convert("RGB")
    return img


def _resize_mask(mask_img, size):
    """Resize mask to match base image size, keeping it smooth."""
    if mask_img.mode not in ("L", "1"):
        mask_img = mask_img.convert("L")
    if mask_img.size == size:
        return mask_img
    # Use bilinear for soft masks; nearest can create holes on scaled masks
    return mask_img.resize(size, Image.BILINEAR)


def _choose_mask_orientation(base_rgb, mask_gray):
    """
    Try mask as-is and inverted, then pick the one that:
    1) Has a reasonable visible-coverage (not ~all transparent/opaque), and
    2) Has brighter visible pixels (avoids the 'black bg, cutout transparent' result).
    """
    from statistics import mean

    def apply_and_score(m):
        rgba = base_rgb.copy().convert("RGBA")
        rgba.putalpha(m)

        A = rgba.getchannel("A")
        alpha = list(A.getdata())
        total = len(alpha) if alpha else 1

        # visible pixels = alpha > 8
        vis_idx = [i for i, a in enumerate(alpha) if a > 8]
        visible_ratio = len(vis_idx) / total

        # mean alpha (overall)
        mean_alpha = sum(alpha) / total

        # mean brightness of visible pixels (Y’ ≈ 0.2126 R + 0.7152 G + 0.0722 B)
        if vis_idx:
            R, G, B = [list(ch.getdata()) for ch in rgba.split()[:3]]
            bright = [0.2126 * R[i] + 0.7152 * G[i] + 0.0722 * B[i] for i in vis_idx]
            mean_brightness = mean(bright)
        else:
            mean_brightness = 0.0

        # Score: prefer reasonable coverage and brighter visible area
        coverage_penalty = 0.0
        if visible_ratio < 0.02 or visible_ratio > 0.98:
            coverage_penalty = 200.0
        score = (mean_brightness + 0.3 * mean_alpha) - coverage_penalty
        return rgba, score, visible_ratio, mean_brightness, mean_alpha

    # Try as-is
    rgba_a, score_a, cov_a, bright_a, alpha_a = apply_and_score(mask_gray)
    # Try inverted
    inv = Image.eval(mask_gray, lambda p: 255 - p)
    rgba_b, score_b, cov_b, bright_b, alpha_b = apply_and_score(inv)

    logging.info(
        f"Mask orientation scores: as_is(score={score_a:.1f}, cov={cov_a:.2%}, "
        f"bright={bright_a:.1f}, mean_alpha={alpha_a:.1f}) "
        f"vs inverted(score={score_b:.1f}, cov={cov_b:.2%}, "
        f"bright={bright_b:.1f}, mean_alpha={alpha_b:.1f})"
    )

    return rgba_a if score_a >= score_b else rgba_b


def _save_png_from_base_and_mask(base_bytes, mask_bytes, out_path):
    """
    Build a PNG with alpha from base + mask bytes.
    If base already has alpha (e.g., PNG/JPX with alpha) just save it.
    """
    base = _open_pillow_image(base_bytes)

    # If base already has an alpha channel, preserve it
    if base.mode in ("RGBA", "LA"):
        base.save(out_path, "PNG")
        return

    # Ensure RGB base for alpha attachment
    if base.mode not in ("RGB", "L"):
        base = base.convert("RGB")
    if base.mode == "L":
        base = base.convert("RGB")

    if not mask_bytes:
        # No mask -> opaque PNG
        base.save(out_path, "PNG")
        return

    mask = Image.open(BytesIO(mask_bytes))
    if mask.mode not in ("L", "1"):
        mask = mask.convert("L")

    mask = _resize_mask(mask, base.size)
    rgba = _choose_mask_orientation(base, mask)
    rgba.save(out_path, "PNG")


def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extract images from a PDF and save as PNG, preserving transparency.

    Rules:
    - If an image has a soft mask (SMask) -> combine base + mask in Pillow to RGBA PNG.
    - If the image has internal alpha (JPX/PNG) -> keep alpha (via Pixmap probe).
    - If image is in SKIP_CONVERSION and has no transparency -> save raw as-is.
    - Otherwise -> convert to opaque PNG.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = fitz.open(pdf_path)
    logging.info(f"Opened PDF file: {pdf_path}")

    count = 0

    for pno in range(len(doc)):
        page = doc[pno]
        logging.info(f"Processing page {pno + 1}/{len(doc)}...")

        for idx, it in enumerate(page.get_images(full=True)):
            xref = it[0]
            tuple_mask = _tuple_has_mask(it)

            # Pull base image bytes
            base = doc.extract_image(xref)
            base_bytes = base["image"]
            ext = base["ext"].lower()

            # Find soft mask bytes if any (from dict or tuple)
            smask_xref = base.get("smask") or (it[1] if len(it) > 1 else None)
            mask_bytes = None
            if smask_xref:
                try:
                    mask_bytes = doc.extract_image(smask_xref)["image"]
                except Exception as e:
                    logging.warning(f"Could not extract soft mask for xref={xref}: {e}")

            # Probe internal alpha via Pixmap (fast + reliable for detection)
            pix_alpha = False
            try:
                probe = fitz.Pixmap(doc, xref)
                pix_alpha = bool(probe.alpha)
                probe = None
            except Exception:
                pass

            logging.info(
                f"Found image p{pno+1} i{idx+1}: xref={xref}, ext={ext}, "
                f"tuple_has_mask={tuple_mask}, smask_xref={smask_xref}, pix_alpha={pix_alpha}"
            )

            raw_name = f"image_{pno + 1}_{idx + 1}.{ext}"
            raw_path = os.path.join(output_dir, raw_name)

            # Case A: Skip conversion ONLY if format is in SKIP_CONVERSION and there is NO transparency
            if ext in SKIP_CONVERSION and not (tuple_mask or smask_xref or pix_alpha):
                with open(raw_path, "wb") as f:
                    f.write(base_bytes)
                logging.info(f"Saved original (no transparency): {raw_path}")
                count += 1
                continue

            # We will output PNG
            png_path = os.path.splitext(raw_path)[0] + ".png"

            try:
                if tuple_mask or smask_xref:
                    # Case B: Explicit soft mask present -> manual Pillow merge
                    _save_png_from_base_and_mask(base_bytes, mask_bytes, png_path)
                    logging.info(f"Saved PNG (Pillow soft-mask merge): {png_path}")

                elif pix_alpha:
                    # Case C: No soft mask, but internal alpha -> let Pixmap write PNG with alpha
                    pix = fitz.Pixmap(doc, xref)
                    # Normalize colorspace while preserving alpha
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGBA if pix.alpha else fitz.csRGB, pix)
                    else:
                        if pix.alpha:
                            pix = fitz.Pixmap(fitz.csRGBA, pix)
                        elif pix.colorspace not in (fitz.csRGB, fitz.csGRAY):
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(png_path)
                    pix = None
                    logging.info(f"Saved PNG (Pixmap internal alpha): {png_path}")

                else:
                    # Case D: No transparency -> convert to opaque PNG
                    base_img = _open_pillow_image(base_bytes)
                    if base_img.mode not in ("RGB", "L", "RGBA", "LA"):
                        base_img = base_img.convert("RGB")
                    if base_img.mode in ("L",):
                        base_img = base_img.convert("RGB")
                    base_img.save(png_path, "PNG")
                    logging.info(f"Saved PNG (opaque): {png_path}")

            except Exception as e:
                # If anything fails, save raw for inspection
                with open(raw_path, "wb") as f:
                    f.write(base_bytes)
                logging.error(f"Failed to create PNG for xref={xref}: {e}. Raw saved: {raw_path}")

            count += 1

    logging.info(f"Extraction complete! {count} images saved to {output_dir}")


# Create / update requirements
requirements = """
PyMuPDF
Pillow
"""
with open("requirements.txt", "w") as req_file:
    req_file.write(requirements)


if __name__ == "__main__":
    setup_logging()
    pdf_file_path = "YOUR PDF FILE NAME HERE"  # Replace with your PDF file path
    output_directory = "images2"  # Replace with your desired output directory
    extract_images_from_pdf(pdf_file_path, output_directory)
    print("Requirements saved to 'requirements.txt'. Use 'pip install -r requirements.txt' to install dependencies.")
