"""
Convert input images to proper format to run in use_model.py.

Run: 
python convert_images.py --input test_images --output test_images_converted
"""
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
import argparse

register_heif_opener()
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff", ".heic", ".heif"
}

def convert_images(input_dir, output_dir, quality=95):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    failed = 0

    image_paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print(f"Found {len(image_paths)} candidate images.\n")

    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                output_path = output_dir / f"{image_path.stem}.jpg"

                if output_path.exists():
                    print(f"[SKIP] Output already exists: {output_path.name}")
                    skipped += 1
                    continue

                img.save(
                    output_path,
                    format="JPEG",
                    quality=quality,
                    optimize=True
                )
                print(
                    f"[OK] {image_path.name} -> {output_path.name} "
                    f"({img.width}x{img.height})"
                )
                converted += 1

        except (UnidentifiedImageError, OSError, ValueError) as e:
            print(f"[FAILED] {image_path.name}: {e}")
            failed += 1

    print("\n" + "=" * 45)
    print("IMAGE CONVERSION SUMMARY")
    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Output:    {output_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert supported images into RGB JPEG files."
    )
    parser.add_argument(
        "--input",
        default="test_images",
        help="Input image directory"
    )
    parser.add_argument(
        "--output",
        default="test_images_converted",
        help="Output directory"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (1-100)"
    )
    args = parser.parse_args()
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")

    convert_images(args.input, args.output, args.quality)
