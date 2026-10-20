import os
import glob
import json
import argparse
import numpy as np
from PIL import Image
from keras.utils import img_to_array
from ultralytics import YOLO
from lenet import LeNet
from mobilenet import MobileNet
from pathlib import Path

CLASS_NAMES_PATH = "./class_names.json"

def load_class_mapping(path):
    """Load the class names used when training the combined dataset."""

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Class mapping not found: {path}\n"
            "Save class_names.json from your training pipeline first."
        )

    with open(path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if isinstance(class_names, dict):
        # Supports mappings such as {"0": "Apple", "1": "Banana"}.
        try:
            class_names = [
                class_names[str(i)]
                for i in range(len(class_names))
            ]
        except KeyError as error:
            raise ValueError(
                "Class mapping dictionary must have consecutive "
                "indices starting at 0."
            ) from error

    if (
        not isinstance(class_names, list)
        or not class_names
        or not all(isinstance(name, str) for name in class_names)
    ):
        raise ValueError(
            "class_names.json must contain a non-empty list "
            "of class names or an index-to-name dictionary."
        )

    if len(set(class_names)) != len(class_names):
        raise ValueError(
            "Class names contain duplicates. Check the training mapping."
        )

    return {i: name for i, name in enumerate(class_names)}


def load_classifier(model_type, class_names_path, weights_path):
    """Build the classifier with the combined dataset's class count."""

    label_map = load_class_mapping(class_names_path)
    num_classes = len(label_map)

    img_size = 192 if model_type == "mobilenet" else 64
    channels = 3

    if model_type == "mobilenet":
        model, _ = MobileNet.build(
            channels, img_size, img_size, num_classes
        )
    else:
        model = LeNet.build(
            channels, img_size, img_size, num_classes
        )

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Classifier weights not found: {weights_path}"
        )

    model.load_weights(weights_path)

    # Check that the model output dimension matches the mapping.
    output_classes = int(model.output_shape[-1])

    if output_classes != num_classes:
        raise ValueError(
            f"Model outputs {output_classes} classes, but the "
            f"mapping contains {num_classes} classes."
        )

    print(f"Loaded {model_type} classifier.")
    print(f"Number of classes: {num_classes}")
    print(f"Classifier input size: {img_size} x {img_size}")
    print(f"Weights: {weights_path}")

    return model, img_size, label_map


def load_detector():
    """Load the pretrained YOLO detector."""

    print("\nLoading YOLO11n detector...")
    return YOLO("yolo11n.pt")


def get_image_paths(input_path):
    """Accept one image or a directory of images."""

    input_path = os.path.abspath(
        os.path.expanduser(input_path.strip().strip('"'))
    )

    extensions = (
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    )

    if os.path.isfile(input_path):
        if input_path.lower().endswith(extensions):
            return [input_path]
        return []

    if os.path.isdir(input_path):
        paths = [
            path
            for path in glob.glob(
                os.path.join(input_path, "*")
            )
            if os.path.isfile(path)
            and path.lower().endswith(extensions)
        ]
        return sorted(paths)

    return []


def classify_crop(classifier, crop, img_size, label_map):
    """Classify one detected object."""

    crop = crop.convert("RGB")

    # Keep preprocessing consistent with training.
    crop = crop.resize(
        (img_size, img_size),
        Image.Resampling.BILINEAR,
    )

    image_array = img_to_array(crop).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = classifier.predict(
        image_array, verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    label = label_map[predicted_index]

    return label, confidence


def detect_and_classify(
    image_path,
    detector,
    classifier,
    img_size,
    label_map,
    detector_confidence=0.25,
    classifier_confidence=0.30,
    save_visualization=False,
):
    """Detect objects, classify crops, and return unique labels."""

    with Image.open(image_path) as source:
        original = source.convert("RGB")

    width, height = original.size

    results = detector.predict(
        source=np.asarray(original),
        conf=detector_confidence,
        imgsz=1024,
        save=False,
        verbose=False,
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
        if save_visualization:
            save_annotated_image(result, image_path)
        return {}

    best_predictions = {}

    for box in result.boxes:
        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().numpy().astype(int)
        )

        # Add padding around the detection.
        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.08)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = original.crop((x1, y1, x2, y2))

        label, confidence = classify_crop(
            classifier, crop, img_size, label_map
        )

        if confidence < classifier_confidence:
            continue

        # Keep the strongest prediction per category.
        if (
            label not in best_predictions
            or confidence > best_predictions[label]["confidence"]
        ):
            best_predictions[label] = {
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
            }

    if not best_predictions:
        print("No predictions passed the confidence threshold.")
    else:
        print("\nPredicted grocery categories:")

        for label, prediction in sorted(
            best_predictions.items(),
            key=lambda item: item[1]["confidence"],
            reverse=True,
        ):
            print(
                f"- {label} "
                f"({prediction['confidence']:.1%})"
            )

    if save_visualization:
        save_annotated_image(result, image_path)

    return best_predictions


def save_annotated_image(result, image_path):
    """Save YOLO's bounding-box visualization."""

    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]

    output_path = (
        os.path.splitext(image_path)[0] + "_detected.jpg"
    )

    Image.fromarray(annotated_rgb).save(output_path)
    print(f"Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect objects with YOLO and classify them using "
            "a trained grocery classifier."
        )
    )

    parser.add_argument(
        "-m", "--model",
        choices=["lenet", "mobilenet"],
        default="mobilenet",
        help="Classifier architecture.",
    )

    parser.add_argument(
        "-i", "--image",
        required=True,
        help="Path to an image or directory of images.",
    )

    parser.add_argument(
        "--classes",
        default=CLASS_NAMES_PATH,
        help="Path to the training class mapping JSON.",
    )

    parser.add_argument(
        "--weights",
        default=None,
        help="Optional explicit path to classifier weights.",
    )

    parser.add_argument(
        "--detector-confidence",
        type=float,
        default=0.25,
        help="Minimum YOLO confidence.",
    )

    parser.add_argument(
        "--classifier-confidence",
        type=float,
        default=0.30,
        help="Minimum grocery classifier confidence.",
    )

    parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Save YOLO bounding-box images.",
    )

    args = parser.parse_args()

    for name in (
        "detector_confidence",
        "classifier_confidence",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")

    if args.weights is None:
        prefix = (
            "combined"
            if args.model == "mobilenet"
            else "grocery"
        )
        weights_path = f"{prefix}_mobilenet.weights.h5" 
    else:
        weights_path = args.weights

    try:
        classifier, img_size, label_map = load_classifier(
            args.model,
            args.classes,
            weights_path,
        )
        detector = load_detector()

    except Exception as error:
        parser.error(str(error))

    image_paths = get_image_paths(args.image)

    if not image_paths:
        parser.error(
            f"No supported images found at: {args.image}"
        )

    print(f"\nFound {len(image_paths)} image(s).")

    successful = 0
    failed = 0

    for image_path in image_paths:
        print("\n" + "=" * 50)
        print(f"Image: {os.path.basename(image_path)}")
        print("=" * 50)

        try:
            detect_and_classify(
                image_path=image_path,
                detector=detector,
                classifier=classifier,
                img_size=img_size,
                label_map=label_map,
                detector_confidence=args.detector_confidence,
                classifier_confidence=args.classifier_confidence,
                save_visualization=args.save_visualization,
            )
            successful += 1

        except Exception as error:
            print(f"Failed to process image: {error}")
            failed += 1

    print("\nProcessing summary")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()