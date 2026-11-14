"""
Command to run: python use_model.py --images test_images_converted --crop-mode mask
"""
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
import numpy as np

# lets ops the Mac GPU doesn't support fall back to the CPU; must be set before importing torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
from keras.utils import img_to_array
from mobilenet import MobileNet
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
# 86-class model trained on combined_dataset/; the class list must match the weights
DETECTOR_CLASSES = "detector_prompts.json"
CLASSIFIER_CLASSES = "trained_model/class_names.json"
CLASSIFIER_WEIGHTS = "trained_model/combined_mobilenet.weights.h5"
# these weights were trained on a Mac GPU, where the head's ReLU was skipped (see mobilenet.py);
# use --head-relu for models trained after that fix
CLASSIFIER_HEAD_RELU = False
PROJECT_DIR = Path(__file__).resolve().parent
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CONFIG_FILE = (
    PROJECT_DIR
    / "sam2_repo"
    / "sam2"
    / "configs"
    / "sam2.1"
    / "sam2.1_hiera_t.yaml"
)
SAM2_CHECKPOINT = str(PROJECT_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt")
OUTPUT_DIR = Path("model_output")
IMG_SIZE = 192
DINO_BOX_THRESHOLD = 0.30
DINO_TEXT_THRESHOLD = 0.25
SAM2_SCORE_THRESHOLD = 0.60
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.30
# Duplicate filtering. Every prompt batch is a separate DINO pass, so the same
# item usually comes back several times under different prompt words.
PREFILTER_IOU_THRESHOLD = 0.90
BOX_NMS_THRESHOLD = 0.50
MASK_NMS_THRESHOLD = 0.60
CONTAINMENT_THRESHOLD = 0.85
GROUP_BOX_MIN_ITEMS = 2
MIN_MASK_AREA_RATIO = 0.001
MAX_MASK_AREA_RATIO = 0.80
CLASS_BATCH_SIZE = 20
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

def load_class_mapping(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)

    if isinstance(names, dict):
        try:
            names = [
                names[key]
                for key in sorted(
                    names,
                    key=lambda x: int(x)
                )
            ]

        except (ValueError, TypeError):
            raise ValueError(
                f"{path} must use numeric dictionary keys."
            )
    if not isinstance(names, list) or not names:
        raise ValueError(
            f"{path} must contain a non-empty list."
        )
    names = [str(name).strip() for name in names]
    if any(not name for name in names):
        raise ValueError(
            f"{path} contains an empty class name."
        )

    return {i: name for i, name in enumerate(names)}

def load_classifier(weights_path, classes_path, head_relu=CLASSIFIER_HEAD_RELU):
    print("\nLoading your trained MobileNetV2...")
    label_map = load_class_mapping(classes_path)
    num_classes = len(label_map)
    # Same architecture as your old code.
    classifier, _ = MobileNet.build(
        3,
        IMG_SIZE,
        IMG_SIZE,
        num_classes,
        head_relu=head_relu
    )

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Classifier weights not found: {weights_path}"
        )

    classifier.load_weights(str(weights_path))
    if int(classifier.output_shape[-1]) != num_classes:
        raise ValueError(
            "Classifier output size does not match "
            "the number of classes in class_names.json.\n"
            f"Model outputs: {classifier.output_shape[-1]}\n"
            f"Class names: {num_classes}"
        )
    print("MobileNetV2 loaded successfully.")
    print("Classifier classes:", num_classes)
    print("Input size:", IMG_SIZE, "x", IMG_SIZE)
    print("Weights:", weights_path)

    return classifier, label_map

def load_grounding_dino():
    print("\nLoading Grounding DINO...")
    processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID)
    model.to(DEVICE)
    model.eval()
    print("Grounding DINO loaded successfully.")
    return processor, model

def load_sam2():
    print("\nLoading SAM 2...")
    if not SAM2_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"SAM 2 config not found: {SAM2_CONFIG_FILE}"
        )

    if not Path(SAM2_CHECKPOINT).exists():
        raise FileNotFoundError(
            f"SAM 2 checkpoint not found: {SAM2_CHECKPOINT}"
        )

    print(f"Config found: {SAM2_CONFIG_FILE}")
    print(f"Checkpoint found: {SAM2_CHECKPOINT}")

    model = build_sam2(
        SAM2_CONFIG,
        SAM2_CHECKPOINT,
        device=DEVICE
    )
    predictor = SAM2ImagePredictor(model)
    print("SAM 2 loaded successfully.")

    return predictor

def get_image_paths(folder):
    folder = Path(folder)
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ])

def clean_prompt(name):
    name = str(name).lower().strip()
    name = name.replace("_", " ")
    name = name.replace("-", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip(" .")

def make_prompt_chunks(classes, batch_size):
    prompts = []
    seen = set()

    for name in classes.values():
        prompt = clean_prompt(name)

        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)

    return [
        prompts[i:i + batch_size]
        for i in range(0, len(prompts), batch_size)
    ]

def detect_with_dino(
    image,
    processor,
    detector,
    prompt_chunks,
    box_threshold,
    text_threshold
):
    width, height = image.size
    detections = []

    for batch_index, labels in enumerate(prompt_chunks):
        print(
            f"DINO batch {batch_index + 1}/"
            f"{len(prompt_chunks)}"
        )

        inputs = processor(
            images=image,
            text=[labels],
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = detector(**inputs)

        try:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(height, width)],
                text_labels=[labels]
            )

        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(height, width)]
            )
        result = results[0]
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        detected_labels = result.get("text_labels")
        if detected_labels is None:
            detected_labels = result.get(
                "labels",
                ["unknown"] * len(boxes)
            )

        for box, score, label in zip(
            boxes,
            scores,
            detected_labels
        ):
            if isinstance(label, torch.Tensor):
                label = str(label.item())
            x1, y1, x2, y2 = map(float, box)
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "detector_label": str(label),
                "detector_confidence": float(score),
                "box": [x1, y1, x2, y2]
            })
        del inputs, outputs, results
    print("Raw DINO detections:", len(detections))
    return detections

def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )
    area_a = (
        max(0, a[2] - a[0]) *
        max(0, a[3] - a[1])
    )
    area_b = (
        max(0, b[2] - b[0]) *
        max(0, b[3] - b[1])
    )
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0

def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

def box_containment(a, b):
    """Fraction of the smaller box that lies inside the other box."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    smaller = min(box_area(a), box_area(b))

    return intersection / smaller if smaller > 0 else 0

def remove_near_identical_boxes(detections, threshold):
    """
    Drop boxes that are near copies of a more confident box, whatever
    prompt word found them. Only catches near-identical boxes, so it is
    safe to run before SAM 2 and saves segmenting the same item many times.
    """
    ranked = sorted(
        detections,
        key=lambda x: x["detector_confidence"],
        reverse=True
    )
    selected = []

    for candidate in ranked:
        if all(
            box_iou(candidate["box"], kept["box"]) < threshold
            for kept in selected
        ):
            selected.append(candidate)

    return selected

def remove_group_boxes(detections, containment, iou_threshold, min_items):
    """
    Drop boxes drawn around a whole shelf or the whole fridge: any box
    that contains min_items or more separate smaller detections.
    """
    final = []

    for candidate in detections:
        inside = [
            other for other in detections
            if other is not candidate
            and box_area(other["box"]) < box_area(candidate["box"])
            and box_containment(other["box"], candidate["box"]) >= containment
        ]
        separate_items = remove_near_identical_boxes(inside, iou_threshold)

        if len(separate_items) >= min_items:
            print(
                f"  Removed group box: {candidate['detector_label']} "
                f"(contains {len(separate_items)} items)"
            )
            continue

        final.append(candidate)

    return final

def segment_with_sam2(
    image,
    detections,
    predictor,
    score_threshold,
    min_area,
    max_area
):
    image_array = np.array(image.convert("RGB"))
    height, width = image_array.shape[:2]
    image_area = height * width
    predictor.set_image(image_array)
    results = []
    for index, detection in enumerate(detections):
        print(
            f"SAM 2: {index + 1}/{len(detections)} "
            f"{detection['detector_label']}"
        )
        box = np.array(
            detection["box"],
            dtype=np.float32
        )
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=False
        )
        mask = masks[0].astype(bool)
        sam_score = float(scores[0])
        mask_area = int(mask.sum())
        area_ratio = mask_area / image_area

        if sam_score < score_threshold:
            print("  Rejected: low SAM 2 score")
            continue

        if area_ratio < min_area:
            print("  Rejected: mask too small")
            continue

        if area_ratio > max_area:
            print("  Rejected: mask too large")
            continue

        results.append({
            **detection,
            "mask": mask,
            "sam_score": sam_score,
            "mask_area": mask_area,
            "mask_area_ratio": area_ratio
        })

    return results

def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(
        mask_a,
        mask_b
    ).sum()
    union = np.logical_or(
        mask_a,
        mask_b
    ).sum()

    return intersection / union if union > 0 else 0

def mask_containment(mask_a, mask_b):
    """Fraction of the smaller mask that lies inside the other mask."""
    smaller = min(mask_a.sum(), mask_b.sum())
    if smaller == 0:
        return 0

    return np.logical_and(mask_a, mask_b).sum() / smaller

def is_duplicate(candidate, kept, box_threshold, mask_threshold, containment):
    if box_iou(candidate["box"], kept["box"]) >= box_threshold:
        return True

    if mask_iou(candidate["mask"], kept["mask"]) >= mask_threshold:
        return True

    # e.g. the greens inside a bag, found once as the bag and once as the greens
    if mask_containment(candidate["mask"], kept["mask"]) >= containment:
        return True

    # A tight and a loose box around the same item overlap less, but one
    # sits inside the other and the classifier gives them the same label.
    return (
        candidate["predicted_label"] == kept["predicted_label"]
        and box_containment(candidate["box"], kept["box"]) >= containment
    )

def remove_duplicate_predictions(predictions, box_threshold, mask_threshold, containment):
    """
    Keep one prediction per item. Runs after classification so the crop
    the classifier is most confident about is the one that survives.
    Sets each prediction's status to kept, duplicate or rejected.
    """
    accepted = sorted(
        (p for p in predictions if p["accepted"]),
        key=lambda x: x["classifier_confidence"],
        reverse=True
    )
    kept = []

    for candidate in accepted:
        match = next(
            (
                existing for existing in kept
                if is_duplicate(candidate, existing, box_threshold, mask_threshold, containment)
            ),
            None
        )

        if match is None:
            candidate["status"] = "kept"
            kept.append(candidate)
        else:
            candidate["status"] = "duplicate"
            candidate["duplicate_of"] = match["crop_name"]

    for prediction in predictions:
        if not prediction["accepted"]:
            prediction["status"] = "rejected"

    return kept

def prepare_crop(image, detection, crop_padding, crop_mode):
    x1, y1, x2, y2 = detection["box"]
    width, height = image.size
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * crop_padding
    pad_y = box_height * crop_padding
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None, None
    crop = image.crop(
        (left, top, right, bottom)
    ).convert("RGB")

    # Optional segmentation-mask crop.
    # Default is rectangular crop.
    if crop_mode == "mask":
        mask = detection["mask"]

        local_mask = mask[
            top:bottom,
            left:right
        ]

        crop_array = np.array(crop)
        # White background outside the segmented ingredient.
        crop_array[~local_mask] = 255
        crop = Image.fromarray(crop_array)

    return crop, (left, top, right, bottom)

def classify_crop(classifier, crop, label_map):
    # Same preprocessing as your old code.
    crop = crop.convert("RGB")
    crop = crop.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR
    )
    image_array = img_to_array(crop).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    probabilities = classifier.predict(
        image_array,
        verbose=0
    )[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    predicted_label = label_map[predicted_index]

    return predicted_label, confidence

def safe_filename(value):
    value = re.sub(
        r"[^\w.-]+",
        "_",
        str(value)
    )
    return value.strip("._") or "ingredient"

def classify_detections(
    image,
    detections,
    classifier,
    label_map,
    output_folder,
    image_stem,
    confidence_threshold,
    crop_padding,
    crop_mode
):
    crop_folder = (
        output_folder
        / "crops"
        / image_stem
    )
    predictions = []

    for index, detection in enumerate(detections):
        print(
            f"\nClassifying crop {index + 1}/"
            f"{len(detections)}"
        )

        crop, crop_box = prepare_crop(
            image,
            detection,
            crop_padding,
            crop_mode
        )

        if crop is None:
            continue

        predicted_label, classifier_score = classify_crop(classifier, crop, label_map)
        accepted = (
            classifier_score >= confidence_threshold
        )

        crop_name = (
            f"{index + 1:03d}_"
            f"{safe_filename(predicted_label)}_"
            f"{classifier_score:.2f}.jpg"
        )

        prediction = {
            "predicted_label": predicted_label,
            "classifier_confidence": classifier_score,
            "accepted": accepted,
            "detector_label": detection["detector_label"],
            "detector_confidence": detection["detector_confidence"],
            "sam_score": detection["sam_score"],
            "box": crop_box,
            "mask": detection["mask"],
            "mask_area": detection["mask_area"],
            "mask_area_ratio": detection["mask_area_ratio"],
            "crop": crop,
            "crop_name": crop_name,
            "crop_path": str(crop_folder / crop_name),
            "status": "",
            "duplicate_of": ""
        }
        predictions.append(prediction)

        if accepted:
            print(
                f"  ACCEPTED: {predicted_label} "
                f"({classifier_score:.1%})"
            )

        else:
            print(
                f"  REJECTED: {predicted_label} "
                f"({classifier_score:.1%})"
            )

    return predictions

def save_crops(predictions, save_all_crops):
    for prediction in predictions:
        if prediction["status"] != "kept" and not save_all_crops:
            continue

        path = Path(prediction["crop_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction["crop"].save(path, quality=95)

def save_annotated_image(image, predictions, output_path):
    image_array = np.array(
        image.convert("RGB")
    ).astype(np.float32)
    overlay = image_array.copy()

    # Display masks only for kept predictions (accepted, not duplicates).
    for prediction in predictions:
        if prediction["status"] != "kept":
            continue

        mask = prediction["mask"]
        seed = sum(
            ord(c)
            for c in prediction["predicted_label"]
        )

        rng = np.random.default_rng(seed)
        color = rng.integers(
            50,
            256,
            size=3
        )

        overlay[mask] = (
            0.45 * overlay[mask]
            + 0.55 * color
        )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    result = Image.fromarray(overlay)
    draw = ImageDraw.Draw(result)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for prediction in predictions:
        if prediction["status"] != "kept":
            continue

        x1, y1, x2, y2 = prediction["box"]
        label = prediction["predicted_label"]
        confidence = prediction["classifier_confidence"]
        text = f"{label} ({confidence:.1%})"
        seed = sum(ord(c) for c in label)
        rng = np.random.default_rng(seed)
        color = tuple(
            int(x)
            for x in rng.integers(50, 256, size=3)
        )

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=3
        )

        text_y = max(0, y1 - 22)
        text_bbox = draw.textbbox(
            (x1, text_y),
            text,
            font=font
        )

        draw.rectangle(text_bbox, fill=color)
        draw.text(
            (x1, text_y),
            text,
            fill="white",
            font=font
        )

    result.save(output_path, quality=95)

def process_image(
    image_path,
    processor,
    detector,
    prompt_chunks,
    predictor,
    classifier,
    label_map,
    output_folder,
    args
):
    print("\n" + "=" * 65)
    print("Processing:", image_path.name)
    print("=" * 65)
    image = Image.open(image_path).convert("RGB")
    image_stem = image_path.stem
    dino_start = time.perf_counter()
    dino_detections = detect_with_dino(
        image,
        processor,
        detector,
        prompt_chunks,
        args.box_threshold,
        args.text_threshold
    )
    dino_time = time.perf_counter() - dino_start

    if args.keep_duplicates:
        box_filtered = dino_detections
        group_filtered = dino_detections
    else:
        box_filtered = remove_near_identical_boxes(dino_detections, args.prefilter_iou)
        print("\nAfter removing near-identical boxes:", len(box_filtered))
        group_filtered = remove_group_boxes(
            box_filtered,
            args.containment,
            args.box_nms,
            GROUP_BOX_MIN_ITEMS
        )
        print("After removing group boxes:", len(group_filtered))

    sam_start = time.perf_counter()
    segmented = segment_with_sam2(
        image,
        group_filtered,
        predictor,
        args.sam_threshold,
        args.min_area,
        args.max_area
    )
    sam_time = time.perf_counter() - sam_start
    print( "\nAfter SAM 2 filtering:", len(segmented))
    classification_start = time.perf_counter()
    predictions = classify_detections(
        image=image,
        detections=segmented,
        classifier=classifier,
        label_map=label_map,
        output_folder=output_folder,
        image_stem=image_stem,
        confidence_threshold=args.classifier_threshold,
        crop_padding=args.crop_padding,
        crop_mode=args.crop_mode
    )
    classification_time = time.perf_counter() - classification_start
    accepted = [
        p for p in predictions
        if p["accepted"]
    ]

    if args.keep_duplicates:
        for prediction in predictions:
            prediction["status"] = "kept" if prediction["accepted"] else "rejected"
        kept = accepted
    else:
        kept = remove_duplicate_predictions(
            predictions,
            args.box_nms,
            args.mask_nms,
            args.containment
        )
    print(f"\nItems found: {len(kept)} ({len(accepted) - len(kept)} duplicates removed)")
    for prediction in kept:
        print(f"  {prediction['predicted_label']} ({prediction['classifier_confidence']:.1%})")
    save_crops(predictions, args.save_all_crops)

    annotated_folder = output_folder / "annotated"
    annotated_folder.mkdir(
        parents=True,
        exist_ok=True
    )
    annotated_path = (annotated_folder / f"{image_stem}_classified.jpg")
    save_annotated_image(image, predictions, annotated_path)
    print("\nSaved annotated image:", annotated_path)
    rows = []
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["box"]
        rows.append({
            "image": image_path.name,
            "detector_label": prediction["detector_label"],
            "detector_confidence": prediction["detector_confidence"],
            "sam2_score": prediction["sam_score"],
            "classifier_label": prediction["predicted_label"],
            "classifier_confidence": prediction["classifier_confidence"],
            "accepted": prediction["accepted"],
            "status": prediction["status"],
            "duplicate_of": prediction["duplicate_of"],
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "mask_area_pixels": prediction["mask_area"],
            "mask_area_ratio": prediction["mask_area_ratio"],
            "crop_path": prediction["crop_path"]
        })

    summary = {
        "image": image_path.name,
        "dino_detections": len(dino_detections),
        "after_box_nms": len(box_filtered),
        "after_group_filter": len(group_filtered),
        "after_sam2": len(segmented),
        "classified_crops": len(predictions),
        "accepted_classifications": len(accepted),
        "items_found": len(kept),
        "items": "; ".join(p["predicted_label"] for p in kept),
        "dino_time_seconds": round(dino_time, 3),
        "sam2_time_seconds": round(sam_time, 3),
        "classification_time_seconds": round(
            classification_time, 3
        ),
        "annotated_image": str(annotated_path)
    }

    return rows, summary

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Grounding DINO + SAM 2 + trained MobileNetV2"
        )
    )

    parser.add_argument(
        "--images",
        default="test_images"
    )

    parser.add_argument(
        "--detector-classes",
        default=DETECTOR_CLASSES
    )

    parser.add_argument(
        "--classifier-classes",
        default=CLASSIFIER_CLASSES
    )

    parser.add_argument(
        "--weights",
        default=CLASSIFIER_WEIGHTS
    )

    parser.add_argument(
        "--head-relu",
        action="store_true",
        default=CLASSIFIER_HEAD_RELU,
        help="Use for weights trained after the tensorflow-metal ReLU fix (e.g. on the RC server)."
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR)
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=0
    )

    parser.add_argument(
        "--box-threshold",
        type=float,
        default=DINO_BOX_THRESHOLD
    )

    parser.add_argument(
        "--text-threshold",
        type=float,
        default=DINO_TEXT_THRESHOLD
    )

    parser.add_argument(
        "--sam-threshold",
        type=float,
        default=SAM2_SCORE_THRESHOLD
    )

    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=CLASSIFIER_CONFIDENCE_THRESHOLD
    )

    parser.add_argument(
        "--prefilter-iou",
        type=float,
        default=PREFILTER_IOU_THRESHOLD,
        help="Before SAM 2, drop boxes overlapping a more confident box by at least this much."
    )

    parser.add_argument(
        "--box-nms",
        type=float,
        default=BOX_NMS_THRESHOLD,
        help="Two kept items whose boxes overlap by at least this much are the same item."
    )

    parser.add_argument(
        "--mask-nms",
        type=float,
        default=MASK_NMS_THRESHOLD,
        help="Two kept items whose SAM 2 masks overlap by at least this much are the same item."
    )

    parser.add_argument(
        "--containment",
        type=float,
        default=CONTAINMENT_THRESHOLD,
        help="A box this much inside another counts as inside it (for group boxes and same-label duplicates)."
    )

    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Skip duplicate filtering and keep every accepted detection (the old raw output)."
    )

    parser.add_argument(
        "--min-area",
        type=float,
        default=MIN_MASK_AREA_RATIO
    )

    parser.add_argument(
        "--max-area",
        type=float,
        default=MAX_MASK_AREA_RATIO
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=CLASS_BATCH_SIZE
    )

    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--crop-mode",
        choices=["box", "mask"],
        default="box",
        help="Use rectangular box crop or SAM 2 masked crop."
    )

    parser.add_argument(
        "--save-all-crops",
        action="store_true"
    )
    args = parser.parse_args()

    for name in [
        "box_threshold",
        "text_threshold",
        "sam_threshold",
        "classifier_threshold",
        "prefilter_iou",
        "box_nms",
        "mask_nms",
        "containment"
    ]:
        value = getattr(args, name)

        if not 0 <= value <= 1:
            parser.error(f"{name} must be between 0 and 1.")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    if args.crop_padding < 0:
        parser.error("--crop-padding cannot be negative.")

    if not 0 < args.min_area <= args.max_area <= 1:
        parser.error(
            "Area thresholds must satisfy "
            "0 < min-area <= max-area <= 1."
        )

    print("=" * 65)
    print("GROUNDING DINO + SAM 2 + MOBILENETV2")
    print("=" * 65)
    print("Device:", DEVICE)

    if DEVICE == "cpu":
        print("WARNING: CPU inference will be slow.")

    image_folder = Path(args.images)
    if not image_folder.exists():
        raise FileNotFoundError(
            f"Image folder not found: {image_folder}"
        )

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    detector_classes = load_class_mapping(
        args.detector_classes
    )

    classifier_classes = load_class_mapping(
        args.classifier_classes
    )

    print(
        "\nGrounding DINO prompt classes:",
        len(detector_classes)
    )

    print(
        "MobileNetV2 classifier classes:",
        len(classifier_classes)
    )

    if args.head_relu:
        MobileNet.fix_metal_relu()

    classifier, label_map = load_classifier(
        args.weights,
        args.classifier_classes,
        args.head_relu
    )
    processor, detector = load_grounding_dino()
    predictor = load_sam2()
    prompt_chunks = make_prompt_chunks(detector_classes, args.batch_size)
    print("\nGrounding DINO prompt batches:", len(prompt_chunks))
    image_paths = get_image_paths(image_folder)

    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        raise RuntimeError(
            "No supported images found."
        )
    print("\nImages to process:", len(image_paths))

    all_rows = []
    all_summaries = []

    for index, image_path in enumerate(image_paths):
        print(f"\nImage {index + 1}/{len(image_paths)}")

        try:
            rows, summary = process_image(
                image_path=image_path,
                processor=processor,
                detector=detector,
                prompt_chunks=prompt_chunks,
                predictor=predictor,
                classifier=classifier,
                label_map=label_map,
                output_folder=output_folder,
                args=args
            )
            all_rows.extend(rows)
            all_summaries.append(summary)

        except Exception as error:
            print(f"ERROR processing {image_path.name}: {error}")

    detection_csv = output_folder / "detections.csv"
    columns = [
        "image",
        "detector_label",
        "detector_confidence",
        "sam2_score",
        "classifier_label",
        "classifier_confidence",
        "accepted",
        "status",
        "duplicate_of",
        "x1",
        "y1",
        "x2",
        "y2",
        "mask_area_pixels",
        "mask_area_ratio",
        "crop_path"
    ]

    with open(
        detection_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_csv = output_folder / "summary.csv"
    summary_columns = [
        "image",
        "dino_detections",
        "after_box_nms",
        "after_group_filter",
        "after_sam2",
        "classified_crops",
        "accepted_classifications",
        "items_found",
        "items",
        "dino_time_seconds",
        "sam2_time_seconds",
        "classification_time_seconds",
        "annotated_image"
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary_columns
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    total_detections = sum(
        x["dino_detections"]
        for x in all_summaries
    )
    total_crops = sum(
        x["classified_crops"]
        for x in all_summaries
    )
    total_accepted = sum(
        x["accepted_classifications"]
        for x in all_summaries
    )
    total_items = sum(
        x["items_found"]
        for x in all_summaries
    )

    print("\n" + "=" * 65)
    print("PROCESSING COMPLETED")
    print("=" * 65)
    print("Images processed:", len(all_summaries))
    print("Raw DINO detections:", total_detections)
    print("Crops classified:", total_crops)
    print("Accepted classifications:", total_accepted)
    print("Items found (after removing duplicates):", total_items)
    print("Detection results:", detection_csv)
    print("Summary:", summary_csv)
    print("Output folder:", output_folder)


if __name__ == "__main__":
    main()