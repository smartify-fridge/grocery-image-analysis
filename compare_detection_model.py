"""
Compare YOLO and Grounding DINO on the SAME grocery-ingredient test set.

Install:
  python -m pip install ultralytics transformers torch torchvision pillow pyyaml numpy pandas

Run:
  python compare_detection_model.py --images test_images
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLOWorld
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

DEFAULT_CLASSES = Path("class_names.json")
DEFAULT_YOLO_MODEL = "yolov8s-worldv2.pt"
DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_OUTPUT = Path("comparison_output")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_classes(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Class file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        try:
            data = [
                data[key]
                for key in sorted(data, key=lambda x: int(x))
            ]
        except (ValueError, TypeError):
            raise ValueError(
                "Dictionary keys in classes.json must be "
                "integer indices such as 0, 1, 2."
            )
    if not isinstance(data, list) or not data:
        raise ValueError(
            "classes.json must contain a non-empty list or dictionary."
        )
    classes = []
    seen = set()
    for name in data:
        name = str(name).strip().lower()
        name = name.replace("_", " ").replace("-", " ")
        name = re.sub(r"\s+", " ", name)
        if name and name not in seen:
            classes.append(name)
            seen.add(name)
    if not classes:
        raise ValueError("No valid ingredient classes found.")
    print(f"Loaded {len(classes)} ingredient classes.")
    return classes

def load_yolo(classes, model_path):
    model = YOLOWorld(model_path)
    model.set_classes(classes)
    print("YOLO-World loaded successfully.")

    return model

def load_grounding_dino(model_id, device):
    print(f"Model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    print("Grounding DINO loaded successfully.")

    return processor, model

def detect_yolo(image_path, model, confidence_threshold):
    start_time = time.perf_counter()
    results = model.predict(
        source=str(image_path),
        conf=confidence_threshold,
        verbose=False
    )
    elapsed = time.perf_counter() - start_time
    detections = []
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            coordinates = box.xyxy[0].cpu().tolist()
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = result.names[class_id]
            detections.append({
                "box": tuple(coordinates),
                "label": str(class_name),
                "confidence": confidence,
            })

    return detections, elapsed

def split_prompt_batches(classes, batch_size):
    return [
        classes[i:i + batch_size]
        for i in range(0, len(classes), batch_size)
    ]

def detect_grounding_dino(
    image,
    processor,
    model,
    classes,
    device,
    box_threshold,
    text_threshold,
    batch_size,
):
    start_time = time.perf_counter()
    width, height = image.size
    all_detections = []
    prompt_batches = split_prompt_batches(classes, batch_size)
    for batch_number, labels in enumerate(
        prompt_batches,
        start=1
    ):
        print(
            f"Grounding DINO batch "
            f"{batch_number}/{len(prompt_batches)}"
        )
        # Each batch contains ingredient names.
        inputs = processor(
            images=image,
            text=[labels],
            return_tensors="pt"
        )
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)],
            text_labels=[labels],
        )
        result = results[0]
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        detected_labels = result.get("text_labels")
        if detected_labels is None:
            detected_labels = result.get("labels")
        for index, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(float, box)

            # Clip boxes to image boundaries.
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            if detected_labels is None:
                detected_label = "unknown"
            else:
                detected_label = detected_labels[index]
                if isinstance(detected_label, torch.Tensor):
                    detected_label = detected_label.item()

                # Some Transformers versions return integer labels.
                if isinstance(detected_label, int):
                    if 0 <= detected_label < len(labels):
                        detected_label = labels[detected_label]
                    else:
                        detected_label = str(detected_label)

            all_detections.append({
                "box": (x1, y1, x2, y2),
                "label": str(detected_label),
                "confidence": float(score),
            })
        del inputs, outputs, results
    elapsed = time.perf_counter() - start_time

    return all_detections, elapsed

def get_font():
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except OSError:
        return ImageFont.load_default()

def draw_detections(image, detections, color):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = get_font()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = detection["label"]
        confidence = detection["confidence"]
        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=3
        )
        text = f"{label} ({confidence:.2f})"
        text_x = int(x1)
        text_y = max(0, int(y1) - 22)
        draw.text(
            (text_x, text_y),
            text,
            fill=color,
            font=font,
            stroke_width=2,
            stroke_fill="black"
        )

    return output

def create_comparison_image(
    original_image,
    yolo_detections,
    dino_detections,
    image_stem,
    output_dir,
):
    yolo_image = draw_detections(
        original_image,
        yolo_detections,
        "lime"
    )
    dino_image = draw_detections(
        original_image,
        dino_detections,
        "red"
    )
    width, height = original_image.size
    # Two images side by side.
    comparison = Image.new(
        "RGB",
        (width * 2, height + 50),
        "white"
    )
    comparison.paste(yolo_image, (0, 50))
    comparison.paste(dino_image, (width, 50))
    draw = ImageDraw.Draw(comparison)
    font = get_font()
    draw.text(
        (10, 10),
        f"YOLO-World: {len(yolo_detections)} detections",
        fill="black",
        font=font
    )
    draw.text(
        (width + 10, 10),
        f"Grounding DINO: {len(dino_detections)} detections",
        fill="black",
        font=font
    )
    output_path = output_dir / "side_by_side" / (image_stem + "_comparison.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path, quality=95)

    return output_path

def save_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "model",
        "detections",
        "inference_time_seconds",
        "detected_labels",
    ]
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

def get_image_paths(input_path, max_images=None):
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_path]
        return []
    if not input_path.is_dir():
        raise FileNotFoundError(
            f"Image folder or file not found: {input_path}"
        )
    image_paths = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.stem.endswith("_comparison")
    )
    if max_images is not None:
        image_paths = image_paths[:max_images]

    return image_paths

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare YOLO-World and Grounding DINO "
            "for ingredient object detection."
        )
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Path to a refrigerator image or image folder."
    )
    parser.add_argument(
        "--classes",
        default=str(DEFAULT_CLASSES),
        help="Path to classes.json."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output folder."
    )
    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL,
        help="YOLO-World pretrained model."
    )
    parser.add_argument(
        "--dino-model",
        default=DEFAULT_DINO_MODEL,
        help="Grounding DINO model ID."
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold."
    )
    parser.add_argument(
        "--dino-box-threshold",
        type=float,
        default=0.30,
        help="Grounding DINO box threshold."
    )
    parser.add_argument(
        "--dino-text-threshold",
        type=float,
        default=0.25,
        help="Grounding DINO text threshold."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of ingredient prompts per DINO batch."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process."
    )
    args = parser.parse_args()
    # Validate settings.
    for name, value in [
        ("YOLO confidence", args.yolo_conf),
        ("DINO box threshold", args.dino_box_threshold),
        ("DINO text threshold", args.dino_text_threshold),
    ]:
        if not 0 <= value <= 1:
            parser.error(f"{name} must be between 0 and 1.")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    # Set device.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("YOLO-WORLD VS GROUNDING DINO")
    print("=" * 60)
    print(f"Device: {device}")

    if device == "cpu":
        print("Warning: CPU inference may be slow.")

    classes = load_classes(args.classes)
    image_paths = get_image_paths(
        args.images,
        args.max_images
    )

    if not image_paths:
        parser.error("No supported images found.")

    print(f"\nImages to process: {len(image_paths)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo = load_yolo(classes, args.yolo_model)
    processor, dino = load_grounding_dino(args.dino_model, device)
    csv_rows = []

    for image_index, image_path in enumerate(
        image_paths,
        start=1
    ):
        print("\n" + "=" * 60)
        print(
            f"Image {image_index}/{len(image_paths)}: "
            f"{image_path.name}"
        )
        print("=" * 60)

        image = Image.open(image_path).convert("RGB")
        print("\nRunning YOLO-World...")
        yolo_detections, yolo_time = detect_yolo(
            image_path,
            yolo,
            args.yolo_conf
        )
        print(f"YOLO detections: {len(yolo_detections)}")
        print(f"YOLO inference time: {yolo_time:.3f} seconds")

        print("\nRunning Grounding DINO...")
        dino_detections, dino_time = detect_grounding_dino(
            image,
            processor,
            dino,
            classes,
            device,
            args.dino_box_threshold,
            args.dino_text_threshold,
            args.batch_size
        )
        print(
            f"Grounding DINO detections: "
            f"{len(dino_detections)}"
        )
        print(
            f"Grounding DINO inference time: "
            f"{dino_time:.3f} seconds"
        )
      
        image_stem = image_path.stem
        yolo_annotated = draw_detections(
            image,
            yolo_detections,
            "lime"
        )
        dino_annotated = draw_detections(
            image,
            dino_detections,
            "red"
        )
        yolo_dir = output_dir / "yolo"
        dino_dir = output_dir / "grounding_dino"

        yolo_dir.mkdir(parents=True, exist_ok=True)
        dino_dir.mkdir(parents=True, exist_ok=True)
        yolo_annotated.save(
            yolo_dir / f"{image_stem}_yolo.jpg",
            quality=95
        )
        dino_annotated.save(
            dino_dir / f"{image_stem}_dino.jpg",
            quality=95
        )

        comparison_path = create_comparison_image(
            image,
            yolo_detections,
            dino_detections,
            image_stem,
            output_dir
        )
        print(f"Saved comparison: {comparison_path}")

        csv_rows.append({
            "image": image_path.name,
            "model": "YOLO-World",
            "detections": len(yolo_detections),
            "inference_time_seconds": round(yolo_time, 4),
            "detected_labels": "; ".join(
                d["label"] for d in yolo_detections
            ),
        })

        csv_rows.append({
            "image": image_path.name,
            "model": "Grounding DINO",
            "detections": len(dino_detections),
            "inference_time_seconds": round(dino_time, 4),
            "detected_labels": "; ".join(
                d["label"] for d in dino_detections
            ),
        })

    csv_path = output_dir / "detection_summary.csv"
    save_csv(csv_rows, csv_path)

    print("\n" + "=" * 60)
    print("COMPARISON FINISHED")
    print("=" * 60)
    print(f"Images processed: {len(image_paths)}")
    print(f"Results folder: {output_dir.resolve()}")
    print(f"CSV summary: {csv_path.resolve()}")
    print("\nYOLO-World total detections:",
          sum(
              row["detections"]
              for row in csv_rows
              if row["model"] == "YOLO-World"
          ))
    print("Grounding DINO total detections:",
          sum(
              row["detections"]
              for row in csv_rows
              if row["model"] == "Grounding DINO"
          ))

if __name__ == "__main__":
    main()
