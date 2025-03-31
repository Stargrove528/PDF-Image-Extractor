import fitz  # type: ignore
import os
import logging
from datetime import datetime
from PIL import Image, ImageOps, ImageCms

# Specify image formats to skip conversion (e.g., png, jpeg)
# Choices for 'image_ext': 'png', 'jpeg', 'jpx', etc.
SKIP_CONVERSION = ["jpeg"]  # Add extensions you want to save as is

def setup_logging():
    """
    Set up logging to a file with the current date and time.
    """
    log_filename = datetime.now().strftime("image_extraction_%Y%m%d_%H%M%S.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Logging initialized.")

def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extract all images from a PDF file and save them to a specified directory.

    :param pdf_path: Path to the input PDF file.
    :param output_dir: Directory where extracted images will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    logging.info(f"Opened PDF file: {pdf_path}")

    image_counter = 0

    # Iterate through each page in the PDF
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        logging.info(f"Processing page {page_number + 1}/{len(pdf_document)}...")

        # Extract images from the page
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]  # XREF of the image

            # Extract the image bytes
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_info = {
                "width": base_image.get("width"),
                "height": base_image.get("height"),
                "colorspace": base_image.get("colorspace")
            }
            logging.info(f"Found image on page {page_number + 1}: XREF={xref}, Format={image_ext}, Info={image_info}")

            # Save the image to the output directory
            image_filename = f"image_{page_number + 1}_{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            # Skip conversion if the image type is in SKIP_CONVERSION
            if image_ext in SKIP_CONVERSION:
                logging.info(f"Skipping conversion for {image_path} (format: {image_ext})")
                image_counter += 1
                continue

            # Convert images to .webp
            try:
                with Image.open(image_path) as img:
                    if img.mode == "CMYK":
                        # Convert CMYK to RGB using ICC profiles if available
                        cmyk_conversion_failed = False
                        try:
                            srgb_profile = ImageCms.createProfile("sRGB")
                            img = ImageCms.profileToProfile(img, None, srgb_profile, outputMode="RGB")
                            logging.info(f"Successfully converted CMYK to RGB using ICC profile for {image_path}.")
                        except Exception as conversion_error:
                            cmyk_conversion_failed = True
                            logging.warning(f"Failed to use ICC profile for {image_path}: {conversion_error}. Performing basic CMYK to RGB conversion.")
                            img = img.convert("RGB")
                            logging.info(f"Performed fallback CMYK to RGB conversion for {image_path}.")

                        # If conversion fails and image appears inverted, attempt to invert colors
                        if cmyk_conversion_failed:
                            img = ImageOps.invert(img)
                            logging.info(f"Applied color inversion to {image_path} as a correction step after failed ICC profile conversion.")

                    # Detect and correct negative images by analyzing pixel data
                    if img.mode in ["RGB", "L"]:
                        inverted_img = ImageOps.invert(img.convert("RGB"))
                        if list(img.getdata())[:10] == list(inverted_img.getdata())[:10]:
                            img = inverted_img
                            logging.info(f"Corrected inverted colors for {image_path}.")

                    webp_path = os.path.splitext(image_path)[0] + ".webp"
                    img.save(webp_path, format="WEBP")
                    logging.info(f"Converted {image_path} to {webp_path}")
                os.remove(image_path)  # Remove the original file
                image_path = webp_path
            except Exception as e:
                logging.error(f"Failed to convert {image_path}: {e}")

            logging.info(f"Saved image: {image_path}")
            image_counter += 1

    logging.info(f"Extraction complete! {image_counter} images saved to {output_dir}")

# Create a requirements file
requirements = """
PyMuPDF
Pillow
"""
with open("requirements.txt", "w") as req_file:
    req_file.write(requirements)

# Example usage
if __name__ == "__main__":
    setup_logging()
    pdf_file_path = "Field_Catalogue.pdf"  # Replace with your PDF file path
    output_directory = "images"  # Replace with your desired output directory
    extract_images_from_pdf(pdf_file_path, output_directory)
    print("Requirements saved to 'requirements.txt'. Use 'pip install -r requirements.txt' to install dependencies.")
